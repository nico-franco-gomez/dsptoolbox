import numpy as np
from numpy.typing import NDArray

from ..classes import Signal, MultiBandSignal, FilterBank, Filter
from .._general_helpers import _normalize, _fade, _rms
from ..tools import from_db
from .resampling import resample
from .enums import FadeType


def normalize(
    sig: Signal | MultiBandSignal,
    norm_dbfs: float,
    peak_normalization: bool = True,
    each_channel: bool = False,
) -> Signal | MultiBandSignal:
    """Normalizes a signal to a given dBFS value. It either normalizes each
    channel or the signal as a whole.

    Parameters
    ----------
    sig : `Signal` or `MultiBandSignal`
        Signal to be normalized.
    norm_dbfs : float
        Value in dBFS to reach after normalization.
    peak_normalization : bool, optional
        When True, signal is normalized at peak. False uses RMS value.
        See notes. Default: True.
    each_channel : bool, optional
        When `True`, each channel on its own is normalized. When `False`,
        the peak or rms value across all channels is regarded.
        Default: `False`.

    Returns
    -------
    new_sig : `Signal` or `MultiBandSignal`
        Normalized signal.

    Notes
    -----
    - Normalization can be done for peak or RMS. The latter might generate a
      signal with samples above 0 dBFS if `signal.constrain_amplitude=False`.

    """
    if isinstance(sig, Signal):
        new_sig = sig.copy()
        new_sig.time_data = _normalize(
            new_sig.time_data, norm_dbfs, peak_normalization, each_channel
        )
    elif isinstance(sig, MultiBandSignal):
        new_sig = sig.copy()
        for ind in range(sig.number_of_bands):
            new_sig.bands[ind] = normalize(
                sig.bands[ind], norm_dbfs, peak_normalization, each_channel
            )
    else:
        raise TypeError(
            "Type of signal is not valid. Use either Signal or MultiBandSignal"
        )
    return new_sig


def fade(
    sig: Signal,
    fade_type: FadeType,
    length_fade_seconds: float | None = None,
    at_start: bool = True,
    at_end: bool = True,
) -> Signal:
    """Applies fading to signal.

    Parameters
    ----------
    sig : `Signal`
        Signal to apply fade to.
    fade_type : FadeType
        Type of fading to be applied.
    length_fade_seconds : float, optional
        Fade length in seconds. If `None`, 2.5% of the signal's length is used
        for the fade. Default: `None`.
    at_start : bool, optional
        When `True`, the start of signal is faded. Default: `True`.
    at_end : bool, optional
        When `True`, the ending of signal is faded. Default: `True`.

    Returns
    -------
    new_sig : `Signal`
        New Signal

    """
    assert (
        at_start or at_end
    ), "At least start or end of signal should be faded"
    if length_fade_seconds is None:
        length_fade_seconds = sig.time_vector_s[-1] * 0.025
    assert (
        length_fade_seconds < sig.time_vector_s[-1]
    ), "Fade length should not be longer than the signal itself"

    new_time_data = np.empty_like(sig.time_data)
    for n in range(sig.number_of_channels):
        vec = sig.time_data[:, n].copy()
        if at_start:
            new_time_data[:, n] = _fade(
                vec,
                length_fade_seconds,
                mode=fade_type,
                sampling_rate_hz=sig.sampling_rate_hz,
                at_start=True,
            )
        if at_end:
            new_time_data[:, n] = _fade(
                vec,
                length_fade_seconds,
                mode=fade_type,
                sampling_rate_hz=sig.sampling_rate_hz,
                at_start=False,
            )
    new_sig = sig.copy()
    new_sig.time_data = new_time_data
    return new_sig


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
        down_factor = 10 ** (-12.04 / 20)
        up_factor = 1 / down_factor
        sig.time_data *= down_factor
        # Resample by 4
        sig_over = resample(sig, sig.sampling_rate_hz * 4)
        true_peak_levels = 20 * np.log10(
            np.max(np.abs(sig_over.time_data), axis=0) * up_factor
        )
        peak_levels = 20 * np.log10(
            np.max(np.abs(sig.time_data), axis=0) * up_factor
        )
    elif isinstance(signal, MultiBandSignal):
        true_peak_levels = np.empty(
            (signal.number_of_bands, signal.number_of_channels)
        )
        peak_levels = np.empty_like(true_peak_levels)
        for ind, b in enumerate(signal.bands):
            true_peak_levels[ind, :], peak_levels[ind, :] = true_peak_level(b)
    else:
        raise TypeError(
            "Passed signal must be of type Signal or MultiBandSignal"
        )
    return true_peak_levels, peak_levels


def rms(
    sig: Signal | MultiBandSignal, in_dbfs: bool = True
) -> NDArray[np.float64]:
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

    """
    if isinstance(sig, Signal):
        rms = _rms(sig.time_data)
    elif isinstance(sig, MultiBandSignal):
        rms = np.zeros((sig.number_of_bands, sig.number_of_channels))
        for ind, b in enumerate(sig):
            rms[ind, :] = _rms(b.time_data)
    else:
        raise TypeError(
            "Passed signal should be either a Signal or "
            + "MultiBandSignal type"
        )
    if in_dbfs:
        rms = 20.0 * np.log10(rms)
    return np.atleast_1d(rms)


def apply_gain(
    target: Signal | MultiBandSignal | Filter | FilterBank,
    gain_db: float | NDArray[np.float64],
) -> Signal | MultiBandSignal | Filter | FilterBank:
    """Apply some gain to a signal or filters.

    If it is a Signal or MultiBandSignal, it can be done to the signal as a
    whole or per channel.

    If it is a Filter or a FilterBank, it can be applied to the different
    filters. When passing a single gain value to a FilterBank, this will be
    applied to all filters. See notes for details.

    Parameters
    ----------
    target : Signal, MultiBandSignal, Filter, FilterBank
        Target to apply gain to.
    gain_db : float, NDArray[np.float64]
        Gain in dB to be applied. If it is an array, it should have as many
        elements as there are channels in the signal or filters in the
        filter bank.

    Returns
    -------
    Signal, MultiBandSignal, Filter, FilterBank
        Signal or filter with new gain.

    Notes
    -----
    - If `constrain_amplitude=True` in the signal, the resulting time data
      might get rescaled after applying the gain.
    - When applying gain to a FilterBank, it should be regarded how it will be
      used. If the intended mode is "parallel", then a single gain value will
      modify each band. If "sequential", the gain value will be applied to the
      output signal for each filter. In the latter case, a single filter should
      get the gain modification.

    """
    if isinstance(target, Signal):
        gain_linear = from_db(np.atleast_1d(gain_db), True)
        if len(gain_linear) == 1:
            gain_linear = gain_linear[0]
        new_sig = target.copy()
        new_sig.time_data *= gain_linear
        if new_sig.time_data_imaginary is not None:
            new_sig.time_data_imaginary *= gain_linear
        return new_sig
    elif isinstance(target, MultiBandSignal):
        new_mb = target.copy()
        for ind in range(new_mb.number_of_bands):
            new_mb.bands[ind] = apply_gain(new_mb.bands[ind], gain_db)
        return new_mb
    elif isinstance(target, Filter):
        filter = target.copy()
        gain_linear = from_db(np.atleast_1d(gain_db), True)
        if len(gain_linear) == 1:
            gain_linear = gain_linear[0]
        if filter.has_zpk:
            filter.zpk[-1] *= gain_linear
        if filter.has_sos:
            filter.sos[-1, :3] *= gain_linear
        else:
            filter.ba[0] *= gain_linear
        return filter
    elif isinstance(target, FilterBank):
        gain = np.atleast_1d(gain_db)
        assert (
            len(gain) == 1 or len(gain) == target.number_of_filters
        ), "Incompatible number of gains"
        if len(gain) == 1:
            gain = np.repeat(gain, target.number_of_filters)
        new_fb = target.copy()
        for ind in range(new_fb.number_of_filters):
            new_fb.filters[ind] = apply_gain(new_fb.filters[ind], gain[ind])
        return new_fb
    else:
        raise TypeError("No valid type was passed")
