import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, ifft, next_fast_len

from .latency import _fractional_latency, _remove_ir_latency_from_phase


def _get_minimum_phase_spectrum_from_real_cepstrum(
    time_data: NDArray[np.float64], padding_factor: int
):
    """Returns minimum-phase version of a time series using the real cepstrum
    method.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Time series to compute the minimum phase version from. It is assumed
        to have shape (time samples, channels).
    padding_factor : int
        Zero-padding to a length corresponding to
        `current_length * padding_factor` can be done, in order to avoid time
        aliasing errors.

    Returns
    -------
    NDArray[np.float64]
        New spectrum with minimum phase.

    """
    fft_length = next_fast_len(
        max(time_data.shape[0] * padding_factor, time_data.shape[0])
    )
    # Real cepstrum
    y = np.real(
        ifft(np.log(np.abs(fft(time_data, n=fft_length, axis=0))), axis=0)
    )

    # Window in the cepstral domain, like obtaining hilbert transform
    # If length is even, nyquist is exactly in the middle
    N = y.shape[0]
    if N % 2 == 0:
        # 0 is dc, N//2 is nyquist
        y[1 : N // 2, ...] *= 2.0
        y[N // 2 + 1 :, ...] = 0.0
    else:
        # 0 is dc, N//2 is before nyquist (N+1)//2 is after nyquist
        y[1 : (N + 1) // 2, ...] *= 2.0
        y[(N + 1) // 2 :, ...] = 0.0

    # Back to spectral domain
    return np.exp(fft(y, axis=0))


def _min_phase_ir_from_real_cepstrum(
    time_data: NDArray[np.float64], padding_factor: int
):
    """Returns minimum-phase version of a time series using the real cepstrum
    method.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Time series to compute the minimum phase version from. It is assumed
        to have shape (time samples, channels).
    padding_factor : int, optional
        Zero-padding to a length corresponding to
        `current_length * padding_factor` can be done, in order to avoid time
        aliasing errors. Default: 8.

    Returns
    -------
    min_phase_time_data : NDArray[np.float64]
        New time series.

    """
    return np.real(
        np.fft.ifft(
            _get_minimum_phase_spectrum_from_real_cepstrum(
                time_data, padding_factor
            ),
            axis=0,
        )
    )


def _remove_ir_latency_from_phase_min_phase(
    freqs: NDArray[np.float64],
    phase: NDArray[np.float64],
    time_data: NDArray[np.float64],
    sampling_rate_hz: int,
    padding_factor: int,
):
    """
    Remove the impulse delay from a phase response.

    Parameters
    ----------
    freqs : NDArray[np.float64]
        Frequency vector.
    phase : NDArray[np.float64]
        Phase vector.
    time_data : NDArray[np.float64]
        Corresponding time signal.
    sampling_rate_hz : int
        Sample rate.
    padding_factor : int
        Padding factor used to obtain the minimum phase equivalent.

    Returns
    -------
    new_phase : NDArray[np.float64]
        New phase response without impulse delay.

    """
    min_ir = _min_phase_ir_from_real_cepstrum(time_data, padding_factor)
    return _remove_ir_latency_from_phase(
        freqs,
        phase,
        _fractional_latency(time_data, min_ir, 1),
        sampling_rate_hz,
    )
