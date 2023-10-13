"""
Backend for transfer functions methods
"""
import numpy as np
from scipy.signal import get_window
from .._general_helpers import _find_nearest, _calculate_window, _pad_trim


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
        outside = 30
        inside = 10 ** (-200 / 20)
        eps = _calculate_window(ids, len(freqs_hz), inverse=True)
        eps += inside
        eps *= outside
        denum_reg = denum_fft.conj() / (denum_fft.conj() * denum_fft + eps)
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
    exp2_trim: int = 13,
    constant_percentage: float = 0.75,
    at_start: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function finds the index of the impulse and trims or windows it
    accordingly. Window used and the start sample are returned.

    It is defined to place the impulse at the start of the constant area
    of the tukey window. However, flanks can be any type.

    """
    start_sample = 0
    # Trimming
    if exp2_trim is not None:
        # Padding
        if 2**exp2_trim >= len(vec):
            # Padding
            vec = np.hstack([vec, np.zeros(total_length - len(vec))])
            length = np.argmax(abs(vec))
        else:
            # Selecting
            assert (
                constant_percentage > 0 and constant_percentage < 1
            ), "Constant percentage must be between 0 and 1"
            length = int((1 - constant_percentage) * 2**exp2_trim) // 2
            ind_max = np.argmax(abs(vec))
            if ind_max - length < 0:
                length = ind_max
            start_sample = ind_max - length
            vec = vec[start_sample : start_sample + 2**exp2_trim]
    else:
        length = np.argmax(abs(vec))
    points = [0, length, total_length - length, total_length]
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
    min_phase_time_data : `np.ndarray`
        New time series.

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

    """
    if window_parameter is not None and isinstance(window_type, str):
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
