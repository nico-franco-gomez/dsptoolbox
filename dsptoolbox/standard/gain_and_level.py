import numpy as np
from numpy.typing import NDArray

from ..classes import Filter, FilterBank, MultiBandSignal, Signal
from ..helpers.gain_and_level import _rms, from_db, to_db
from ._framed_signal_representation import _get_framed_signal
from .enums import BiquadEqType


def true_peak_level(
    signal: Signal | MultiBandSignal,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes true-peak level of a signal using the standardized method
    by the Rec. ITU-R BS.1770-4. See references.

    Parameters
    ----------
    signal : `Signal` or `MultiBandSignal`
        Signal for which to compute the true-peak level.

    Returns
    -------
    true_peak_levels : NDArray[np.float64]
        True-peak levels (in dBTP) as an array with shape (channels) or
        (band, channels) in case that the input signal is `MultiBandSignal`.
    peak_levels : NDArray[np.float64]
        Peak levels (in dBFS) as an array with shape (channels) or
        (band, channels) in case that the input signal is `MultiBandSignal`.

    References
    ----------
    - https://www.itu.int/rec/R-REC-BS.1770

    """
    if isinstance(signal, Signal):
        sig = signal.copy()
        # Reduce gain by 12.04 dB
        down_factor = from_db(-12.04, True)
        up_factor = 1 / down_factor
        sig.constrain_amplitude = False
        sig.time_data = sig.time_data * down_factor
        # Resample by 4
        sig_over = sig.resample(sig.sampling_rate_hz * 4)
        true_peak_levels = to_db(
            np.max(np.abs(sig_over.time_data), axis=0) * up_factor, True
        )
        peak_levels = to_db(np.max(np.abs(sig.time_data), axis=0) * up_factor, True)
        return true_peak_levels, peak_levels
    elif isinstance(signal, MultiBandSignal):
        true_peak_levels = np.empty((signal.number_of_bands, signal.number_of_channels))
        peak_levels = np.empty_like(true_peak_levels)
        for ind, b in enumerate(signal.bands):
            true_peak_levels[ind, :], peak_levels[ind, :] = true_peak_level(b)
        return true_peak_levels, peak_levels
    else:
        raise TypeError("Passed signal must be of type Signal or MultiBandSignal")


def rms(sig: Signal | MultiBandSignal, in_dbfs: bool = True) -> NDArray[np.float64]:
    """Returns Root Mean Squared (RMS) value for each channel.

    Parameters
    ----------
    sig : `Signal` or `MultiBandSignal`
        Signal for which to compute the RMS values. It can be a
        `MultiBandSignal` as well.
    in_dbfs : bool, optional
        When `True`, RMS values are returned in dBFS. Default: `True`.

    Returns
    -------
    rms_values : NDArray[np.float64]
        Array with RMS values. If a `Signal` is passed, it has shape
        (channel). If a `MultiBandSignal` is passed, its shape is
        (bands, channel).

    Notes
    -----
    - The RMS value includes any DC component of the signal. If only the AC
      power is relevant, or the signal is known to carry an offset, use
      `Signal.detrend()` before calling this function.

    """
    if isinstance(sig, Signal):
        rms = _rms(sig.time_data)
    elif isinstance(sig, MultiBandSignal):
        rms = np.zeros((sig.number_of_bands, sig.number_of_channels))
        for ind, b in enumerate(sig):
            rms[ind, :] = _rms(b.time_data)
    else:
        raise TypeError(
            "Passed signal should be either a Signal or " + "MultiBandSignal type"
        )
    if in_dbfs:
        rms = 20.0 * np.log10(rms)
    return np.atleast_1d(rms)


def lufs_integrated(s: Signal) -> float:
    """Compute integrated loudness using Loudness Units relative to Full Scale (LUFS or
    LKFS) of the signal. The computation is done according to [1].

    Parameters
    ----------
    s : dsp.Signal
        Signal from which to compute the loudness. The signal can have up to 5 channels
        where the following order is always assumed: (Left, Right, Center, Left
        Surround, Right Surround). This is relevant for the channel-specific gain
        compensation.

    Returns
    -------
    float
        Loudness in LUFS-i

    Notes
    -----
    - [1] provides exact filter coefficients for a sampling rate of 48 kHz. In this
      implementation, the corresponding biquad filter parameters were extracted and
      can be used for any sampling rate provided it is not too low (frequency warping).

    References
    ----------
    - [1]: Recommendation ITU-R BS.1770-5 (11/2023). Algorithms to measure audio
      programme loudness and true-peak audio level

    """
    assert s.number_of_channels <= 5, "Not implemented for more channels than 5"
    fs_hz = s.sampling_rate_hz

    # Constants in algorithm
    k_filter = FilterBank(
        [
            # Acoustic shadowing of the head: Highshelf
            Filter.biquad(
                eq_type=BiquadEqType.Highshelf,
                frequency_hz=1500,
                gain_db=4.0,
                q=2**0.5 / 2.0,
                sampling_rate_hz=fs_hz,
            ),
            # RLB Weighting: Highpass
            Filter.biquad(
                eq_type=BiquadEqType.Highpass,
                frequency_hz=38.1,
                gain_db=0.0,
                q=0.5,
                sampling_rate_hz=fs_hz,
            ),
        ]
    ).merge_filters()
    Tg = 400e-3
    G = np.array([1.0, 1.0, 1.0, 1.41, 1.41])[: s.number_of_channels]
    Tg_samples = int(Tg * fs_hz + 0.5)
    overlap = 0.75
    step = int((1.0 - overlap) * Tg_samples + 0.5)
    GAMMA_A = -70
    DIFF_GAMMA_R = 10

    # Filter handling amplitude
    constrained = s.constrain_amplitude
    s.constrain_amplitude = False
    s_prefiltered = k_filter.filter_signal(s)
    s.constrain_amplitude = constrained

    # Compute blocks
    z_ji = np.mean(
        _get_framed_signal(s_prefiltered.time_data**2.0, Tg_samples, step, False),
        axis=0,
    )

    def gated_loudness(x: NDArray[np.float64]) -> float:
        return -0.691 + 10.0 * np.log10(x @ G)

    l_j = gated_loudness(z_ji)
    gamma_r = gated_loudness(np.mean(z_ji[l_j > GAMMA_A, :], axis=0)) - DIFF_GAMMA_R
    return gated_loudness(np.mean(z_ji[l_j > max(gamma_r, GAMMA_A), :], axis=0))


def crest_factor(
    sig: Signal | MultiBandSignal, in_db: bool = True, use_true_peak: bool = False
) -> NDArray[np.float64]:
    """Compute the crest factor of a signal, which is defined as the level
    difference between its peak and RMS value.

    Parameters
    ----------
    sig : Signal, MultiBandSignal
        Input signal.
    in_db : bool, optional
        When True, the output is given in dB. Otherwise, it is given in the
        amplitude form. Default: True.
    use_true_peak : bool, optional
        When `True`, the true peak value is used for computing the crest factor.
        Default: `False`.

    Returns
    -------
    NDArray[np.float64]
        Crest factors for each channel. If it the input is a MultiBandSignal,
        the shape is (band, channel).

    Notes
    -----
    - The RMS value includes any DC component of the signal. If only the AC
      power is relevant, use `Signal.detrend()` before calling this function.

    """
    if isinstance(sig, Signal):
        peak = (
            from_db(true_peak_level(sig)[0], True)
            if use_true_peak
            else np.max(np.abs(sig.time_data), axis=0)
        )
        crest = peak / _rms(sig.time_data)
    elif isinstance(sig, MultiBandSignal):
        crest = np.zeros((sig.number_of_bands, sig.number_of_channels))
        for ind, b in enumerate(sig):
            crest[ind, :] = crest_factor(b, in_db=False, use_true_peak=use_true_peak)
    else:
        raise TypeError(
            "Passed signal should be either a Signal or " + "MultiBandSignal type"
        )
    return np.atleast_1d(to_db(crest, True) if in_db else crest)
