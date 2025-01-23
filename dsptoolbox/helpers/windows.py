import numpy as np
from numpy.typing import NDArray
from scipy.signal import windows

from ..standard.enums import Window


def calculate_tukey_like_window(
    points,
    window_length: int,
    window_type: Window | list[Window],
    at_start: bool,
    inverse: bool,
) -> NDArray[np.float64]:
    """Creates a custom window with given indexes

    Parameters
    ----------
    points: array_like
        Vector containing 4 points for the construction of the custom
        window.
    window_length: int
        Length of the window.
    window_type: Window, list[Window]
        Type of window to use. Select from scipy.signal.windows. It can be a
        tuple with the window type and extra parameters or a list with two
        window types.
    at_start: bool
        Creates a half rising window at the start as well.
    inverse: bool
        When `True`, the window is inversed so that the middle section
        contains 0.

    Returns
    -------
    window_full: NDArray[np.float64]
        Custom window.

    """
    assert len(points) == 4, "For the custom window 4 points are needed"
    if type(window_type) is Window:
        left_window_type = window_type.to_scipy_format()
        right_window_type = window_type.to_scipy_format()
    if type(window_type) is list:
        assert len(window_type) == 2, "There must be exactly two window types"
        left_window_type = window_type[0].to_scipy_format()
        right_window_type = window_type[1].to_scipy_format()

    idx_start_stop_f = [int(i) for i in points]

    len_low_flank = idx_start_stop_f[1] - idx_start_stop_f[0]

    if at_start:
        low_flank = windows.get_window(
            left_window_type, len_low_flank * 2, fftbins=True
        )[:len_low_flank]
    else:
        low_flank = np.ones(len_low_flank)

    len_high_flank = idx_start_stop_f[3] - idx_start_stop_f[2]
    high_flank = windows.get_window(
        right_window_type, len_high_flank * 2, fftbins=True
    )[len_high_flank:]

    zeros_low = np.zeros(idx_start_stop_f[0])
    ones_mid = np.ones(idx_start_stop_f[2] - idx_start_stop_f[1])
    zeros_high = np.zeros(window_length - idx_start_stop_f[3])
    window_full = np.concatenate(
        (zeros_low, low_flank, ones_mid, high_flank, zeros_high)
    )
    if inverse:
        window_full = 1 - window_full
    return window_full


def _gaussian_window_sigma(window_length: int, alpha: float = 2.5) -> float:
    """Compute the standard deviation sigma for a gaussian window according to
    its length and `alpha`.

    Parameters
    ----------
    window_length : int
        Total window length.
    alpha : float, optional
        Alpha parameter for defining how wide the shape of the gaussian. The
        greater alpha is, the narrower the window becomes. Default: 2.5.

    Returns
    -------
    float
        Standard deviation.

    """
    return (window_length - 1) / (2 * alpha)


def gaussian_window(
    length: int, alpha: float, symmetric: bool, offset: int = 0
):
    """Produces a gaussian window as defined in [1] and [2].

    Parameters
    ----------
    length : int
        Length for the window.
    alpha : float
        Parameter to define window width. It is inversely proportional to the
        standard deviation.
    symmetric : bool
        Define if the window should be symmetric or not.
    offset : int, optional
        The offset moves the middle point of the window to the passed value.
        Default: 0.

    Returns
    -------
    w : NDArray[np.float64]
        Gaussian window.

    References
    ----------
    - [1]: https://www.mathworks.com/help/signal/ref/gausswin.html.
    - [2]: https://de.wikipedia.org/wiki/Fensterfunktion.

    """
    if not symmetric:
        length += 1

    n = np.arange(length)
    half = (length - 1) / 2
    w = np.exp(-0.5 * (alpha * ((n - offset) - half) / half) ** 2)

    if not symmetric:
        return w[:-1]
    return w
