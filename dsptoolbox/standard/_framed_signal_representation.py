import numpy as np
from scipy.signal import windows
from numpy.typing import NDArray

from .._general_helpers import _pad_trim, _compute_number_frames
from ._standard_backend import _get_window_envelope


def _get_framed_signal(
    time_data: NDArray[np.float64],
    window_length_samples: int,
    step_size: int,
    keep_last_frames: bool = True,
) -> NDArray[np.float64]:
    """This function turns a signal into (possibly) overlaping time frames.
    The original data gets copied.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Signal with shape (time samples, channels).
    window_length_samples : int
        Window length in samples.
    step_size : int
        Step size (also called hop length) in samples.
    keep_last_frames : bool, optional
        When `True`, the last frames (probably with zero-padding) are kept.
        Otherwise, no frames with zero padding are included. Default: `True`.

    Returns
    -------
    time_data_framed : NDArray[np.float64]
        Framed signal with shape (time samples, frames, channels).

    Notes
    -----
    - Perfect reconstruction from this representation can be achieved when the
      signal is zero-padded at the edges where the window does not yet meet
      the COLA condition. Otherwise, these sections might be distorted.

    """
    assert time_data.ndim == 2, "Time data should have exactly two dimensions."
    # Force casting to integers
    if type(window_length_samples) is not int:
        window_length_samples = int(window_length_samples)
    if type(step_size) is not int:
        step_size = int(step_size)

    # Start Parameters
    n_frames, padding_samp = _compute_number_frames(
        window_length_samples,
        step_size,
        time_data.shape[0],
        zero_padding=keep_last_frames,
    )
    td = _pad_trim(time_data, time_data.shape[0] + padding_samp)
    td_framed = np.zeros(
        (window_length_samples, n_frames, td.shape[1]), dtype=np.float64
    )

    # Create time frames
    start = 0
    for n in range(n_frames):
        td_framed[:, n, :] = td[
            start : start + window_length_samples, :
        ].copy()
        start += step_size

    return td_framed


def _reconstruct_framed_signal(
    td_framed: NDArray[np.float64],
    step_size: int,
    window: str | NDArray[np.float64] | None = None,
    original_signal_length: int | None = None,
    safety_threshold: float = 1e-4,
) -> NDArray[np.float64]:
    """Gets and returns a framed signal into its vector representation.

    Parameters
    ----------
    td_framed : NDArray[np.float64]
        Framed signal with shape (time samples, frames, channels).
    step_size : int
        Step size in samples between frames (also known as hop length).
    window : str, NDArray[np.float64], optional
        Window (if applies). Pass `None` to avoid using a window during
        reconstruction. Default: `None`.
    original_signal_length : int, optional
        When different than `None`, the output is padded or trimmed to this
        length. Default: `None`.
    safety_threshold : float, optional
        When reconstructing the signal with a window, very small values can
        lead to instabilities. This safety threshold avoids dividing with
        samples beneath this value. Default: 1e-4.

        Dividing by 1e-4 is the same as amplifying by 80 dB.

    Returns
    -------
    td : NDArray[np.float64]
        Reconstructed signal with shape (time samples, channels).

    """
    assert (
        td_framed.ndim == 3
    ), "Framed signal must contain exactly three dimensions"
    if window is not None:
        if type(window) is str:
            window = windows.get_window(window, td_framed.shape[0])
        elif type(window) is NDArray[np.float64]:
            assert window.ndim == 1, "Window must be a 1D-array"
            assert (
                window.shape[0] == td_framed.shape[0]
            ), "Window length does not match signal length"
        td_framed *= window[:, np.newaxis, np.newaxis]

    total_length = int(
        step_size * td_framed.shape[1]
        + td_framed.shape[0] * (1 - step_size / td_framed.shape[0])
    )
    td = np.zeros((total_length, td_framed.shape[-1]))

    start = 0
    for i in range(td_framed.shape[1]):
        td[start : start + td_framed.shape[0], :] += td_framed[:, i, :]
        start += step_size

    if window is not None:
        envelope = _get_window_envelope(
            window, total_length, step_size, td_framed.shape[1], True
        )
        if safety_threshold is not None:
            envelope = np.clip(envelope, a_min=safety_threshold, a_max=None)
        non_zero = envelope > np.finfo(td.dtype).tiny
        td[non_zero, ...] /= envelope[non_zero, np.newaxis]

    if original_signal_length is not None:
        td = _pad_trim(td, original_signal_length)
    return td
