"""
Backend for transfer functions methods
"""

import numpy as np
from scipy.signal import get_window, lfilter
from .._general_helpers import (
    _find_nearest,
    _calculate_window,
    _pad_trim,
    _get_chirp_rate,
)


def _spectral_deconvolve(
    num_fft: np.ndarray,
    denum_fft: np.ndarray,
    freqs_hz,
    time_signal_length: int,
    mode="regularized",
    start_stop_hz=None,
) -> np.ndarray:
    assert num_fft.shape == denum_fft.shape, "Shapes do not match"
    assert len(freqs_hz) == len(num_fft), "Frequency vector does not match"

    if mode == "regularized":
        # Regularized division
        ids = _find_nearest(start_stop_hz, freqs_hz)
        eps = _calculate_window(ids, len(freqs_hz), inverse=True) * 10 ** (
            30 / 20
        )
        denum_reg = denum_fft.conj() / (np.abs(denum_fft) ** 2 + eps)
        new_time_data = np.fft.irfft(num_fft * denum_reg, n=time_signal_length)
    elif mode == "window":
        ids = _find_nearest(start_stop_hz, freqs_hz)
        window = _calculate_window(ids, len(freqs_hz), inverse=False)
        window += 10 ** (-200 / 10)
        num_fft_n = num_fft * window
        new_time_data = np.fft.irfft(
            np.divide(num_fft_n, denum_fft), n=time_signal_length
        )
    elif mode == "standard":
        new_time_data = np.fft.irfft(
            np.divide(num_fft, denum_fft), n=time_signal_length
        )
    else:
        raise ValueError(
            f"{mode} is not supported. Choose window"
            + ", regularized or standard"
        )
    return new_time_data


def _window_this_ir_tukey(
    vec,
    total_length: int,
    window_type: str = "hann",
    constant_percentage: float = 0.75,
    at_start: bool = True,
    offset_samples: int = 0,
    left_to_right_flank_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function finds the index of the impulse and trims or windows it
    accordingly. Window used and the start sample are returned.

    It is defined to place the impulse at the start of the constant area
    of the tukey window. However, an offset for delaying the impulse can be
    passed. Flanks can be any type.

    """
    start_sample = 0

    # Expected flank length
    flank_length_total = int((1 - constant_percentage) * total_length)
    left_flank_length = int(
        flank_length_total * 0.5 * left_to_right_flank_ratio
    )
    right_flank_length = flank_length_total - left_flank_length

    # Maximum
    impulse_index = np.argmax(np.abs(vec))

    # If offset and impulse index are outside or inside
    if impulse_index - offset_samples < 0:
        vec = np.pad(vec, ((-(impulse_index - offset_samples), 0)))
        start_sample += -(impulse_index - offset_samples)
    else:
        impulse_index -= offset_samples

    # If left flank is longer than the amount of samples expected
    if impulse_index - left_flank_length < 0:
        vec = np.pad(vec, ((-(impulse_index - left_flank_length), 0)))
        start_sample += -(impulse_index - left_flank_length)
    else:
        vec = vec[impulse_index - left_flank_length :]
        start_sample = impulse_index - left_flank_length

    # If total length is larger than actual length
    if len(vec) < total_length:
        vec = np.pad(vec, ((0, total_length - len(vec))))
    else:
        vec = vec[:total_length]

    points = [
        0,
        left_flank_length,
        total_length - right_flank_length,
        total_length,
    ]
    window = _calculate_window(
        points, total_length, window_type, at_start=at_start
    )
    return vec * window, window, start_sample


def _min_phase_ir_from_real_cepstrum(time_data: np.ndarray):
    """Returns minimum-phase version of a time series using the real cepstrum
    method.

    Parameters
    ----------
    time_data : `np.ndarray`
        Time series to compute the minimum phase version from. It is assumed
        to have shape (time samples, channels).

    Returns
    -------
    min_phase_time_data : `np.ndarray`
        New time series.

    """
    return np.real(
        np.fft.ifft(
            _get_minimum_phase_spectrum_from_real_cepstrum(time_data), axis=0
        )
    )


def _get_minimum_phase_spectrum_from_real_cepstrum(time_data: np.ndarray):
    """Returns minimum-phase version of a time series using the real cepstrum
    method.

    Parameters
    ----------
    time_data : `np.ndarray`
        Time series to compute the minimum phase version from. It is assumed
        to have shape (time samples, channels).

    Returns
    -------
    `np.ndarray`
        New spectrum with minimum phase.

    """
    # Real cepstrum
    y = np.real(
        np.fft.ifft(np.log(np.abs(np.fft.fft(time_data, axis=0))), axis=0)
    )

    # Window in the cepstral domain, like obtaining hilbert transform
    w = np.zeros(y.shape[0])
    w[0] = 1
    w[: len(w) // 2 - 1] = 2
    # If length is even, nyquist is exactly in the middle
    if len(w) % 2 == 0:
        w[len(w) // 2] = 1

    # Windowing in cepstral domain and back to spectral domain
    return np.exp(np.fft.fft(y * w[..., None], axis=0))


def _window_this_ir(
    vec, total_length: int, window_type: str = "hann", window_parameter=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function windows an impulse response by placing the peak exactly
    in the middle of the window. It trims or pads at the end if needed. The
    windowed IR, window and the start sample are passed.

    Returns
    -------
    td : `np.ndarray`
        Windowed vector.
    w : `np.ndarray`
        Generated window.
    ind_low_td : int
        Sample position of the start.

    """
    if window_parameter is not None and type(window_type) is str:
        window_type = (window_type, window_parameter)
    peak_ind = np.argmax(np.abs(vec))
    half_length = total_length // 2

    # If Peak is in the second half
    flipping = False
    if peak_ind > half_length:
        vec = vec[::-1]
        flipping = True
        peak_ind = len(vec) - peak_ind - 1

    w = get_window(window_type, half_length * 2 + 1, False)

    if peak_ind - half_length < 0:
        ind_low_td = 0
        ind_low_w = half_length - peak_ind
    else:
        ind_low_td = peak_ind - half_length
        ind_low_w = 0

    if peak_ind + half_length + 1 > len(vec):
        ind_up_td = len(vec)
        ind_up_w = peak_ind + half_length + 1 - len(vec)
    else:
        ind_up_td = peak_ind + half_length + 1
        ind_up_w = len(w)

    w = w[ind_low_w:ind_up_w]
    td = vec[ind_low_td:ind_up_td] * w

    if flipping:
        td = td[::-1]
        w = w[::-1]

    # Length adaptation
    if len(td) != total_length:
        td = _pad_trim(td, total_length)
        w = _pad_trim(w, total_length)
    return td, w, ind_low_td


def _warp_time_series(td: np.ndarray, warping_factor: float):
    """Warp or unwarp a time series.

    Parameters
    ----------
    td : `np.ndarray`
        Time series with shape (time samples, channels).
    warping_factor : float
        The warping factor to use.

    Returns
    -------
    warped_td : `np.ndarray`
        Time series in the (un)warped domain.

    """
    warped_td = np.zeros_like(td)

    dirac = np.zeros(td.shape[0])
    dirac[0] = 1

    b = np.array([-warping_factor, 1])
    a = np.array([1, -warping_factor])

    warped_td = dirac[..., None] * td[0, :]

    # Print progress to console
    ns = [
        int(0.25 * td.shape[0]),
        int(0.5 * td.shape[0]),
        int(0.75 * td.shape[0]),
    ]

    for n in np.arange(1, td.shape[0]):
        dirac = lfilter(b, a, dirac)
        warped_td += dirac[..., None] * td[n, :]
        if n in ns:
            print(f"Warped: {ns.pop(0)}% of signal")
    return warped_td


def _get_harmonic_times(
    chirp_range_hz: list,
    chirp_length_seconds: float,
    n_harmonics: int,
    time_offset_seconds: float = 0.0,
) -> np.ndarray:
    """Get the time at which each harmonic IR occur relative to the fundamental
    IR in a measurement with an exponential chirp. This is computed according
    to [1]. If the fundamental happens at time `t=0`, all harmonics will be at
    a negative time.

    Parameters
    ----------
    chirp_range_hz : list of length 2
        The frequency range of the chirp.
    chirp_length_seconds : float
        Length of chirp in seconds (without zero-padding).
    n_harmonics : int
        Number of harmonics to analyze.
    time_offset_seconds : float, optional
        Time at which the fundamental occurs. Default: 0.

    Returns
    -------
    np.ndarray
        Array with the times for each harmonic in ascending order. The values
        are given in seconds.

    References
    ----------
    - [1]: Weinzierl, S. Handbuch der Audiotechnik. Chapter 21.

    """
    rate = _get_chirp_rate(chirp_range_hz, chirp_length_seconds)
    return time_offset_seconds - np.log2(np.arange(n_harmonics) + 2) / rate
