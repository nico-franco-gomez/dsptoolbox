from warnings import warn
from scipy.stats import pearsonr
import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate, hilbert

from .spectrum_utilities import _wrap_phase


def _get_fractional_impulse_peak_index(
    time_data: NDArray[np.float64], polynomial_points: int = 1
):
    """
    Obtain the index for the peak in subsample precision using the root
    of the analytical function.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Time vector with shape (time samples, channels).
    polynomial_points : int, optional
        Number of points to take for the polynomial interpolation and root
        finding of the analytic part of the time series. Default: 1.

    Returns
    -------
    latency_samples : NDArray[np.float64]
        Latency of impulses (in samples). It has shape (channels).

    """
    n_channels = time_data.shape[1]
    delay_samples = np.argmax(np.abs(time_data), axis=0).astype(int)

    # Take only the part of the time vector with the peaks and some safety
    # samples (±200)
    time_data = time_data[: np.max(delay_samples) + 200, :]
    start_offset = max(np.min(delay_samples) - 200, 0)
    time_data = time_data[start_offset:, :]
    delay_samples -= start_offset

    h = hilbert(time_data, axis=0).imag
    x = np.arange(-polynomial_points + 1, polynomial_points + 1)

    latency_samples = np.zeros(n_channels)

    for ch in range(n_channels):
        # ===== Ensure that delay_samples is before the peak in each channel
        selection = h[delay_samples[ch] : delay_samples[ch] + 2, ch]
        move_back_one_sample = selection[0] * selection[1] > 0
        delay_samples[ch] -= int(move_back_one_sample)
        if h[delay_samples[ch], ch] * h[delay_samples[ch] + 1, ch] > 0:
            latency_samples[ch] = delay_samples[ch] + int(move_back_one_sample)
            warn(
                f"Fractional latency detection failed for channel {ch}. "
                + "Integer latency is"
                + " returned"
            )
            continue
        # =====

        # Fit polynomial
        pol = np.polyfit(
            x,
            h[
                delay_samples[ch]
                - polynomial_points
                + 1 : delay_samples[ch]
                + polynomial_points
                + 1,
                ch,
            ],
            deg=2 * polynomial_points - 1,
        )

        # Find roots
        roots = np.roots(pol)
        # Get only root between 0 and 1
        roots = roots[
            # Real roots
            (roots == roots.real)
            # Range
            & (roots <= 1)
            & (roots >= 0)
        ].real
        try:
            fractional_delay_samples = roots[0]
        except IndexError as e:
            print(e)
            warn(
                f"Fractional latency detection failed for channel {ch}. "
                + "Integer latency is"
                + " returned"
            )
            latency_samples[ch] = delay_samples[ch] + int(move_back_one_sample)
            continue

        latency_samples[ch] = delay_samples[ch] + fractional_delay_samples
    return latency_samples + start_offset


def _fractional_latency(
    td1: NDArray[np.float64],
    td2: NDArray[np.float64] | None = None,
    polynomial_points: int = 1,
):
    """This function computes the sub-sample latency between two signals using
    Zero-Crossing of the analytic (hilbert transformed) correlation function.
    The number of polynomial points taken around the correlation maximum can be
    set, although some polynomial orders might fail to compute the root. In
    that case, integer latency will be returned for the respective channel.

    Parameters
    ----------
    td1 : `np.ndaray`
        Delayed version of the signal.
    td2 : NDArray[np.float64], optional
        Original version of the signal. If `None` is passed, the latencies
        are computed between the first channel of td1 and every other.
        Default: `None`.
    polynomial_points : int, optional
        This corresponds to the number of points taken around the root in order
        to fit a polynomial. Accuracy might improve with higher orders but
        it could also lead to ill-conditioned polynomials. In case root finding
        is not successful, integer latency values are returned. Default: 1.

    Returns
    -------
    lags : NDArray[np.float64]
        Fractional delays. It has shape (channel). In case td2 was `None`, its
        length is `channels - 1`.

    References
    ----------
    - N. S. M. Tamim and F. Ghani, "Hilbert transform of FFT pruned cross
      correlation function for optimization in time delay estimation," 2009
      IEEE 9th Malaysia International Conference on Communications (MICC),
      Kuala Lumpur, Malaysia, 2009, pp. 809-814,
      doi: 10.1109/MICC.2009.5431382.

    """
    if td2 is None:
        td2_ = td1[:, 0][..., None]
        td1_ = np.atleast_2d(td1[:, 1:])
        xcor = correlate(td2_, td1_)
    else:
        xcor = np.zeros((td1.shape[0] + td2.shape[0] - 1, td2.shape[1]))
        for i in range(td2.shape[1]):
            xcor[:, i] = correlate(td2[:, i], td1[:, i])
    inds = _get_fractional_impulse_peak_index(xcor, polynomial_points)
    return td1.shape[0] - inds - 1


def _remove_ir_latency_from_phase(
    freqs: NDArray[np.float64],
    phase: NDArray[np.float64],
    latency_samples: NDArray,
    sampling_rate_hz: int,
):
    """
    Remove the impulse delay from a phase response.

    Parameters
    ----------
    freqs : NDArray[np.float64]
        Frequency vector.
    phase : NDArray[np.float64]
        Phase vector.
    latency_samples : NDArray
        Latency per channel to remove in samples.
    sampling_rate_hz : int
        Sampling rate in Hz.

    Returns
    -------
    new_phase : NDArray[np.float64]
        New phase response without impulse delay.

    """
    assert latency_samples.ndim == 1
    assert len(latency_samples) == phase.shape[1]
    delays_s = latency_samples / sampling_rate_hz
    return _wrap_phase(phase + 2 * np.pi * freqs[:, None] * delays_s[None, :])


def _remove_ir_latency_from_phase_peak(
    freqs: NDArray[np.float64],
    phase: NDArray[np.float64],
    time_data: NDArray[np.float64],
    sampling_rate_hz: int,
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

    Returns
    -------
    new_phase : NDArray[np.float64]
        New phase response without impulse delay.

    """
    return _remove_ir_latency_from_phase(
        freqs,
        phase,
        _get_fractional_impulse_peak_index(time_data),
        sampling_rate_hz,
    )


def _get_correlation_of_latencies(
    time_data: NDArray[np.float64],
    other_time_data: NDArray[np.float64],
    latencies: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute the pearson correlation coefficient of each channel between
    `time_data` and `other_time_data` in order to obtain an estimation on the
    quality of the latency computation.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Original time data. This is the "undelayed" version if the latency
        is positive. It must have either one channel or a matching number
        of channels with `other_time_data`.
    other_time_data : NDArray[np.float64]
        "Delayed" time data, when the latency is positive.
    latencies : NDArray[np.int_]
        Computed latencies for each channel.

    Returns
    -------
    NDArray[np.float64]
        Correlation coefficient for each channel.

    """
    one_channel = time_data.shape[1] == 1

    correlations = np.zeros(len(latencies))

    for ch in range(len(latencies)):
        if latencies[ch] > 0:
            undelayed = time_data[:, 0] if one_channel else time_data[:, ch]
            delayed = other_time_data[:, ch]
        else:
            undelayed = other_time_data[:, ch]
            delayed = time_data[:, 0] if one_channel else time_data[:, ch]

        # Remove delay samples
        delayed = delayed[abs(latencies[ch]) :]

        # Get effective length
        length_to_check = min(len(delayed), len(undelayed))

        delayed = delayed[:length_to_check]
        undelayed = undelayed[:length_to_check]
        correlations[ch] = pearsonr(delayed, undelayed)[0]
    return correlations
