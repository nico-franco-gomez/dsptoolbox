"""
Backend for transfer functions methods
"""

import numpy as np
from scipy.signal import get_window, hilbert
from scipy.fft import next_fast_len
from scipy.stats import pearsonr
from warnings import warn
from numpy.typing import NDArray

from .._general_helpers import (
    _find_nearest,
    _calculate_window,
    _pad_trim,
    _get_chirp_rate,
)
from ..tools import to_db, time_smoothing


def _spectral_deconvolve(
    num_fft: NDArray[np.complex128],
    denum_fft: NDArray[np.complex128],
    freqs_hz,
    time_signal_length: int,
    mode="regularized",
    start_stop_hz=None,
) -> NDArray[np.complex128]:
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
    window_type: str | tuple | list = "hann",
    constant_percentage: float = 0.75,
    at_start: bool = True,
    offset_samples: int = 0,
    left_to_right_flank_ratio: float = 1.0,
    adaptive_window: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """This function finds the index of the impulse and trims or windows it
    accordingly. Window used and the start sample are returned.

    It is defined to place the impulse at the start of the constant area
    of the tukey window. However, an offset for delaying the impulse can be
    passed. Flanks can be any type.

    """
    # Start sample for vector
    start_sample = 0

    # Expected flank length
    flank_length_total = int((1 - constant_percentage) * total_length)
    left_flank_length = int(
        flank_length_total * 0.5 * left_to_right_flank_ratio
    )
    right_flank_length = flank_length_total - left_flank_length

    # Maximum
    impulse_index = np.argmax(np.abs(vec))

    if not adaptive_window:
        # If offset and impulse index are outside or inside
        padding_left = 0
        if impulse_index - offset_samples < 0:
            pad_length = -(impulse_index - offset_samples)
            vec = np.pad(vec, ((pad_length, 0)))
            start_sample += pad_length
            padding_left += pad_length
        else:
            impulse_index -= offset_samples

        # If left flank is longer than the amount of samples expected
        if impulse_index - left_flank_length < 0:
            pad_length = -(impulse_index - left_flank_length)
            vec = np.pad(vec, ((pad_length, 0)))
            start_sample += pad_length
            padding_left += pad_length
        else:
            vec = vec[impulse_index - left_flank_length :]
            start_sample = impulse_index - left_flank_length

        # If total length is larger than actual length
        padding_right = 0
        if len(vec) < total_length:
            pad_length = total_length - len(vec)
            vec = np.pad(vec, ((0, pad_length)))
            padding_right += pad_length
        else:
            vec = vec[:total_length]
    else:
        # Left flank adaptation
        if impulse_index - offset_samples - left_flank_length < 0:
            left_flank_length = max(0, impulse_index - offset_samples)
        else:
            start_sample = impulse_index - offset_samples - left_flank_length
            vec = vec[start_sample:]

        # Right flank adaptation
        if len(vec) > total_length:
            vec = vec[:total_length]

        padding_after_adaptation = 0
        if len(vec) < total_length:
            padding_after_adaptation = total_length - len(vec)
            total_length = len(vec)

        if (
            left_flank_length + offset_samples
            > total_length - right_flank_length
        ):
            right_flank_length = (
                total_length - left_flank_length - offset_samples - 1
            )

    points = [
        0,
        left_flank_length,
        total_length - right_flank_length,
        total_length,
    ]
    window = _calculate_window(
        points, total_length, window_type, at_start=at_start
    )

    if not adaptive_window:
        window[:padding_left] = 0
        if padding_right != 0:
            window[-padding_right:] = 0
    else:
        vec = np.pad(vec, ((0, padding_after_adaptation)))
        window = np.pad(window, ((0, padding_after_adaptation)))

    return vec * window, window, start_sample


def _window_this_ir(
    vec, total_length: int, window_type: str = "hann", window_parameter=None
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """This function windows an impulse response by placing the peak exactly
    in the middle of the window. It trims or pads at the end if needed. The
    windowed IR, window and the start sample are passed.

    Returns
    -------
    td : NDArray[np.float64]
        Windowed vector.
    w : NDArray[np.float64]
        Generated window.
    ind_low_td : int
        Sample position of the start.

    """
    if window_parameter is not None and type(window_type) is str:
        window_type = (window_type, window_parameter)
    peak_ind = np.argmax(np.abs(vec))
    half_length = total_length // 2
    centered_impulse_and_even = (
        peak_ind + half_length == len(vec) and len(vec) % 2 == 0
    )

    # If Peak is in the second half
    flipping = peak_ind > half_length
    if flipping:
        vec = vec[::-1]
        peak_ind = len(vec) - peak_ind - 1

    w = get_window(window_type, half_length * 2 + 1, False)

    # Define start index for time data and window
    if peak_ind - half_length < 0:
        ind_low_td = 0
        ind_low_w = half_length - peak_ind
    else:
        ind_low_td = peak_ind - half_length
        ind_low_w = 0

    # Pad vector if necessary
    if total_length - ind_low_td > len(vec):
        vec = np.pad(vec, ((0, total_length + ind_low_td - len(vec))))

    # Get second half
    if peak_ind + half_length + 1 > len(vec) and not centered_impulse_and_even:
        ind_up_td = len(vec)
        ind_up_w = peak_ind + half_length + 1 - len(vec)
    else:
        ind_up_td = peak_ind + half_length + 1
        ind_up_w = len(w) - (1 if centered_impulse_and_even else 0)

    # Get time data and window
    w = w[ind_low_w:ind_up_w]
    td = vec[ind_low_td:ind_up_td] * w

    # Final length adaptation (ensure length)
    if len(td) != total_length:
        td = _pad_trim(td, total_length)
        w = _pad_trim(w, total_length)

    # Flip back if needed
    if flipping:
        td = td[::-1]
        w = w[::-1]

    return td, w, ind_low_td


def _get_harmonic_times(
    chirp_range_hz: list,
    chirp_length_s: float,
    n_harmonics: int,
    time_offset_seconds: float = 0.0,
) -> NDArray[np.float64]:
    """Get the time at which each harmonic IR occur relative to the fundamental
    IR in a measurement with an exponential chirp. This is computed according
    to [1]. If the fundamental happens at time `t=0`, all harmonics will be at
    a negative time.

    Parameters
    ----------
    chirp_range_hz : list of length 2
        The frequency range of the chirp.
    chirp_length_s : float
        Length of chirp in seconds (without zero-padding).
    n_harmonics : int
        Number of harmonics to analyze.
    time_offset_seconds : float, optional
        Time at which the fundamental occurs. Default: 0.

    Returns
    -------
    NDArray[np.float64]
        Array with the times for each harmonic in ascending order. The values
        are given in seconds.

    References
    ----------
    - [1]: Weinzierl, S. Handbuch der Audiotechnik. Chapter 21.

    """
    rate = _get_chirp_rate(chirp_range_hz, chirp_length_s)
    return time_offset_seconds - np.log2(np.arange(n_harmonics) + 2) / rate


def _trim_ir(
    time_data: NDArray[np.float64],
    fs_hz: int,
    offset_start_s: float,
    safety_distance_to_noise_floor_db: float = 10.0,
) -> tuple[int, int, int]:
    """
    Obtain the starting and stopping index curve using the smooth (exponential)
    envelope of the energy time curve. Non-overlapping windows are checked, so
    that the first window to grow its average energy after the impulse is
    taken as the end.

    This function returns the start and stop indices relative to the original
    time data, but the impulse index relative to the new vector.

    Returns
    -------
    start : int
    stop : int
    impulse : int

    """

    # Start index
    impulse_index = int(np.argmax(np.abs(time_data)))
    offset_start_samples = int(offset_start_s * fs_hz + 0.5)
    start_index = int(np.max([0, impulse_index - 1 - offset_start_samples]))
    impulse_index -= start_index

    # ETC (from impulse until end)
    etc = to_db(
        hilbert(
            time_data[start_index + impulse_index :],
            N=next_fast_len(
                len(time_data) - start_index - impulse_index, False
            ),
        ),
        True,
    )

    # Smoothing of ETC
    envelope = time_smoothing(etc, fs_hz, 20e-3, None)

    # Ensure that energy is always decaying by checking it in non-overlapping
    # windows. When the energy of a window is larger than the previous one,
    # the end of the IR has been surpassed. Do this for different window sizes
    # and check the different correlation coefficients of energy decay with
    # time for the selected ending points. If one is below -0.95, then take it
    # as the final one. If not, then look for all the ones below -0.9 and
    # average them. If None, then take the ones below -0.7 and average them
    # with the highest weight. If all are above -0.7, method failed -> no
    # trimming

    window_lengths = (
        np.array([10, 30, 50, 70, 90]) * 1e-3 * fs_hz + 0.5
    ).astype(int)
    end = np.zeros(len(window_lengths))
    x = np.arange(len(envelope))
    corr_coeff = np.zeros(len(window_lengths))
    for ind, window_length in enumerate(window_lengths):
        current_start_position = 0
        current_window_mean_db = 0

        for _ in range(len(envelope) // window_length):
            new_window_mean_db = np.mean(
                envelope[
                    current_start_position : current_start_position
                    + window_length
                ]
            )
            if current_window_mean_db <= new_window_mean_db:
                break
            current_window_mean_db = new_window_mean_db
            current_start_position += window_length

        # End in the center of the next window
        end_with_current_window = min(
            (current_start_position * 2 + window_length) // 2, len(envelope)
        )
        corr_coeff[ind] = pearsonr(
            x[:end_with_current_window],
            envelope[:end_with_current_window],
        )[0]
        end[ind] = end_with_current_window

    select = np.argmin(corr_coeff)
    if corr_coeff[select] <= -0.95:
        end_point = int(end[select])
    elif np.any(corr_coeff <= -0.9):
        inds = corr_coeff <= -0.9
        end_point = int(np.mean(end[inds]))
    elif np.any(corr_coeff <= -0.7):
        inds = corr_coeff <= -0.7
        end_point = int(
            np.mean(np.hstack([np.ones(9) * end[select], end[inds]]))
        )
    else:
        warn("No satisfactory estimation for trimming the rir could be made")
        end_point = int(np.mean(np.hstack([np.ones(5) * len(envelope), end])))

    stop = end_point + start_index + impulse_index
    if safety_distance_to_noise_floor_db != 0.0:
        end_point = __find_index_above_noise_floor(
            envelope[:end_point],
            to_db(np.var(time_data[stop:]), False),
            np.abs(safety_distance_to_noise_floor_db),
        )
        stop = end_point + start_index + impulse_index

    return start_index, stop, impulse_index


def __find_index_above_noise_floor(
    envelope: NDArray[np.float64],
    noise_floor_db: float,
    distance_to_noise_floor_db: float,
):
    """Get a safety distance from the noise floor using a polynomial fit of
    the IR power density in dB."""
    polynomial = (
        np.polynomial.Polynomial.fit(
            np.arange(len(envelope)),
            envelope,
            1,
        )
        .convert()
        .coef
    )

    if polynomial[1] > 0.0:
        return len(envelope)

    new_stop_index = int(
        ((noise_floor_db + distance_to_noise_floor_db) - polynomial[0])
        / polynomial[1]
        + 0.5
    )

    min_retain_length_percentage = 75.0
    return np.clip(
        new_stop_index,
        int(len(envelope) * min_retain_length_percentage / 100.0 + 0.5),
        len(envelope),
    )
