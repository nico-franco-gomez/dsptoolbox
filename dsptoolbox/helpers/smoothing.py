import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import lfilter, lfilter_zi, oaconvolve, windows

from .windows import _gaussian_window_sigma


def _fractional_octave_smoothing(
    vector: NDArray[np.float64],
    bin_spacing_octaves: float | None = None,
    num_fractions: int = 3,
    window_type="hann",
    window_vec: NDArray[np.float64] | None = None,
    clip_values: bool = False,
) -> NDArray[np.float64]:
    """Smoothes a vector using interpolation to a logarithmic scale. Usually
    done for smoothing of frequency data. This implementation is taken from
    the pyfar package, see references.

    Parameters
    ----------
    vector : NDArray[np.float64]
        Vector to be smoothed. It is assumed that the first axis is to
        be smoothed.
    bin_spacing_octaves : float, None, optional
        Spacing between frequency bins in octaves. If None, it is assumed that
        the vector is linearly spaced. Default: None.
    num_fractions : int, optional
        Fraction of octave to be smoothed across. Default: 3 (third band).
    window_type : str, optional
        Type of window to be used. See `scipy.signal.windows.get_window` for
        valid types. If the window is `'gaussian'`, the parameter passed will
        be interpreted as alpha and not sigma. Default: `'hann'`.
    window_vec : NDArray[np.float64], optional
        Window vector to be used as a window. `window_type` should be set to
        `None` if this direct window is going to be used. Default: `None`.
    clip_values : bool, optional
        When `True`, negative values are clipped to 0. Default: `False`.

    Returns
    -------
    vec_final : NDArray[np.float64]
        Vector after smoothing.

    References
    ----------
    - Tylka, Joseph & Boren, Braxton & Choueiri, Edgar. (2017). A Generalized
      Method for Fractional-Octave Smoothing of Transfer Functions that
      Preserves Log-Frequency Symmetry. Journal of the Audio Engineering
      Society. 65. 239-245. 10.17743/jaes.2016.0053.
    - https://github.com/pyfar/pyfar

    """
    lin_spaced = bin_spacing_octaves is None

    if lin_spaced:
        # Linear and logarithmic frequency vector
        N = len(vector)
        l1 = np.arange(N, dtype=np.float64)
        k_log = (N) ** (l1 / (N - 1))
        l1 += 1.0
        beta = np.log2(k_log[1])
        # Interpolate to logarithmic scale
        vector = PchipInterpolator(l1, vector, axis=0)(k_log)
    else:
        beta = bin_spacing_octaves

    # Smooth
    # Window length always odd, so that delay can be easily compensated
    n_window = int(1 / (num_fractions * beta) + 0.5)  # Round
    n_window += 1 - n_window % 2  # Ensure odd length

    # Generate window
    if window_type is not None:
        assert (
            window_vec is None
        ), "When window type is passed, no window vector should be added"
        if "gauss" in window_type[0]:
            window_type = (
                "gaussian",
                _gaussian_window_sigma(n_window, window_type[1]),
            )
        window = windows.get_window(window_type, n_window, fftbins=False)
    else:
        assert (
            window_type is None
        ), "When using a window as a vector, window type should be None"
        window = window_vec

    # Dimension handling
    one_dim = False
    if vector.ndim == 1:
        one_dim = True
        vector = vector[..., None]

    # Normalize window
    window /= window.sum()

    # Smoothe by convolving with window (output is centered)
    n_window_half = n_window // 2
    smoothed = oaconvolve(
        np.pad(
            vector,
            ((n_window_half, n_window_half - (1 - n_window % 2)), (0, 0)),
            mode="edge",
        ),
        window[..., None],
        mode="valid",
        axes=0,
    )
    if one_dim:
        smoothed = smoothed.squeeze()

    # Interpolate back to linear scale
    if lin_spaced:
        smoothed = interp1d(
            k_log,
            smoothed,
            kind="linear",
            copy=False,
            assume_sorted=True,
            axis=0,
        )(l1)

    # Avoid any negative values
    if clip_values:
        smoothed = np.clip(smoothed, a_min=0, a_max=None)
    return smoothed


def _get_smoothing_factor_ema(
    relaxation_time_s: float, sampling_rate_hz: int, accuracy: float = 0.95
):
    """This computes the smoothing factor needed for a single-pole IIR,
    or exponential moving averager. The returned value (alpha) should be used
    as follows::

        y[n] = alpha * x[n] + (1-alpha)*y[n-1]

    Parameters
    ----------
    relaxation_time_s : float
        Time for the step response to stabilize around the given value
        (with the given accuracy).
    sampling_rate_hz : int
        Sampling rate to be used.
    accuracy : float, optional
        Accuracy with which the value of the step response can differ from
        1 after the relaxation time. This must be between ]0, 1[.
        Default: 0.95.

    Returns
    -------
    alpha : float
        Smoothing value for the exponential smoothing.

    Notes
    -----
    - The formula coincides with the one presented in
      https://en.wikipedia.org/wiki/Exponential_smoothing, but it uses an
      extra factor for accuracy.

    """
    factor = np.log(1 - accuracy)
    return 1 - np.exp(factor / relaxation_time_s / sampling_rate_hz)


def _time_smoothing(
    x: NDArray[np.float64],
    sampling_rate_hz: int,
    ascending_time_s: float,
    descending_time_s: float | None = None,
) -> NDArray[np.float64]:
    """Smoothing for a time series with independent ascending and descending
    times using an exponential moving average. It works on 1D and 2D arrays.
    The smoothing is always applied along the longest axis.

    If no descending time is provided, `ascending_time_s` is used for both
    increasing and decreasing values.

    Parameters
    ----------
    x : NDArray[np.float64]
        Vector to apply smoothing to.
    sampling_rate_hz : int
        Sampling rate of the time series `x`.
    ascending_time_s : float
        Corresponds to the needed time for achieving a 95% accuracy of the
        step response when the samples are increasing in value. Pass 0. in
        order to avoid any smoothing for rising values.
    descending_time_s : float, None, optional
        As `ascending_time_s` but for descending values. If None,
        `ascending_time_s` is applied. Default: None.

    Returns
    -------
    NDArray[np.float64]
        Smoothed time series.

    """
    onedim = x.ndim == 1
    x = np.atleast_2d(x)
    if x.shape[0] < x.shape[1]:
        reverse_axis = True
        x = x.T
    else:
        reverse_axis = False

    assert x.ndim < 3, "This function is only available for 2D arrays"
    assert ascending_time_s >= 0.0, "Attack time must be at least 0"
    ascending_factor = (
        _get_smoothing_factor_ema(ascending_time_s, sampling_rate_hz)
        if ascending_time_s > 0.0
        else 1.0
    )

    if descending_time_s is None:
        b, a = [ascending_factor], [1, -(1 - ascending_factor)]
        zi = lfilter_zi(b, a)
        y = lfilter(
            b,
            a,
            x,
            axis=0,
            zi=np.asarray([zi * x[0, ch] for ch in range(x.shape[1])]).T,
        )[0]
        if reverse_axis:
            y = y.T
        if onedim:
            return y.squeeze()
        return y

    assert descending_time_s >= 0.0, "Release time must at least 0"
    assert not (
        ascending_time_s == 0.0 and descending_time_s == ascending_time_s
    ), "These times will not apply any smoothing"

    descending_factor = (
        _get_smoothing_factor_ema(descending_time_s, sampling_rate_hz)
        if descending_time_s > 0.0
        else 1.0
    )

    y = np.zeros_like(x)
    y[0, :] = x[0, :]

    for n in np.arange(1, x.shape[0]):
        for ch in range(x.shape[1]):
            smoothing_factor = (
                ascending_factor
                if x[n, ch] > y[n - 1, ch]
                else descending_factor
            )
            y[n, ch] = (
                smoothing_factor * x[n, ch]
                + (1.0 - smoothing_factor) * y[n - 1, ch]
            )

    if reverse_axis:
        y = y.T
    if onedim:
        return y.squeeze()
    return y
