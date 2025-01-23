import numpy as np
from numpy.typing import NDArray

from .other import _pad_trim


def _polyphase_decomposition(
    in_sig: NDArray[np.float64],
    number_polyphase_components: int,
    flip: bool = False,
) -> tuple[NDArray[np.float64], int]:
    """Converts input signal array with shape (time samples, channels) into
    its polyphase representation with shape (time samples, polyphase
    components, channels).

    Parameters
    ----------
    in_sig : NDArray[np.float64]
        Input signal array to be rearranged in polyphase representation. It
        should have the shape (time samples, channels).
    number_polyphase_components : int
        Number of polyphase components to be used.
    flip : bool, optional
        When `True`, axis of polyphase components is flipped. Needed for
        downsampling with FIR filter because of the reversing of filter
        coefficients. Default: `False`.

    Returns
    -------
    poly : NDArray[np.float64]
        Rearranged vector with polyphase representation. New shape is
        (time samples, polyphase components, channels).
    padding : int
        Amount of padded elements in the beginning of array.

    """
    # Dimensions of vector
    if in_sig.ndim == 1:
        in_sig = in_sig[..., None]
    assert (
        in_sig.ndim == 2
    ), "Vector should have exactly two dimensions: (time samples, channels)"
    # Rename for practical purposes
    n = number_polyphase_components
    # Pad zeros in the beginning to avoid remainder
    remainder = in_sig.shape[0] % n
    padding = n - remainder
    if remainder != 0:
        in_sig = _pad_trim(
            in_sig, in_sig.shape[0] + padding, axis=0, in_the_end=False
        )
    # Here (time samples, polyphase, channels)
    poly = np.zeros((in_sig.shape[0] // n, n, in_sig.shape[1]))
    for ind in range(n):
        poly[:, ind, :] = in_sig[ind::n, :]
    if flip:
        poly = np.flip(poly, axis=1)
    return poly, padding


def _polyphase_reconstruction(
    poly: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Returns the reconstructed input signal array from its polyphase
    representation, possibly with a different length if padded was needed for
    reconstruction. Polyphase representation shape is assumed to be
    (time samples, polyphase components, channels).

    Parameters
    ----------
    poly : NDArray[np.float64]
        Array with 3 dimensions (time samples, polyphase components, channels)
        as polyphase respresentation of signal.

    Returns
    -------
    in_sig : NDArray[np.float64]
        Rearranged vector with shape (time samples, channels).

    """
    # If squeezed array with one channel is passed
    if poly.ndim == 2:
        poly = poly[..., None]
    assert poly.ndim == 3, (
        "Invalid shape. The dimensions must be (time samples, polyphase "
        + "components, channels)"
    )
    n = poly.shape[1]
    in_sig = np.zeros((poly.shape[0] * n, poly.shape[2]))
    for ind in range(n):
        in_sig[ind::n, :] = poly[:, ind, :]
    return in_sig
