"""
Standard functions
------------------
This module contains a general collection of DSP functions that do not fall
under a same category.

"""
import numpy as np
import pickle
from scipy.signal import resample_poly, convolve, hilbert
from scipy.special import iv as bessel_first_mod
from fractions import Fraction
from warnings import warn

from dsptoolbox.classes.signal_class import Signal
from dsptoolbox.classes.multibandsignal import MultiBandSignal
from dsptoolbox.classes.filterbank import FilterBank
from dsptoolbox.classes.filter_class import Filter
from dsptoolbox._standard import (_latency,
                                  _center_frequencies_fractional_octaves_iec,
                                  _exact_center_frequencies_fractional_octaves,
                                  _kaiser_window_beta,
                                  _indices_above_threshold_dbfs,
                                  _detrend, _rms)
from dsptoolbox._general_helpers import (
    _pad_trim, _normalize, _fade, _check_format_in_path,
    _get_smoothing_factor_ema)
from dsptoolbox.transfer_functions import (
    min_phase_from_mag, lin_phase_from_mag)


def latency(in1: Signal, in2: Signal = None) -> np.ndarray:
    """Computes latency between two signals using the correlation method.
    If there is no second signal, the latency between the first and the other
    channels is computed. `in1` is to be understood as a the delayed version
    of `in2` for the latency to be positive. The other way around will give
    the same result but negative.

    Parameters
    ----------
    in1 : Signal
        First signal.
    in2 : Signal, optional
        Second signal. Default: `None`.

    Returns
    -------
    latency_per_channel_samples : `np.ndarray`
        Array with latency between two signals in samples per channel.

    """
    if in2 is not None:
        assert in1.number_of_channels == in2.number_of_channels, \
            'Channel number does not match'
        latency_per_channel_samples = _latency(in1.time_data, in2.time_data)
    else:
        assert in1.number_of_channels > 1, \
            'Signal should have at least two channels'
        latency_per_channel_samples = \
            np.zeros(in1.number_of_channels-1, dtype=int)
        for n in range(in1.number_of_channels-1):
            latency_per_channel_samples[n] = \
                _latency(in1.time_data[:, 0], in1.time_data[:, n+1])
    return latency_per_channel_samples


def pad_trim(signal: Signal | MultiBandSignal, desired_length_samples: int,
             in_the_end: bool = True) -> Signal | MultiBandSignal:
    """Returns a copy of the signal with padded or trimmed time data. If signal
    is `MultiBandSignal`, only `same_sampling_rate=True` is valid.

    Parameters
    ----------
    signal : `Signal` or `MultiBandSignal`
        Signal to be padded or trimmed.
    desired_length_samples : int
        Length of resulting signal.
    in_the_end : bool, optional
        Defines if padding or trimming should be done in the beginning or
        in the end of the signal. Default: `True`.

    Returns
    -------
    new_signal : `Signal` or `MultiBandSignal`
        New padded signal.

    """
    if type(signal) == Signal:
        new_time_data = \
            np.zeros((desired_length_samples, signal.number_of_channels))
        for n in range(signal.number_of_channels):
            new_time_data[:, n] = \
                _pad_trim(
                    signal.time_data[:, n],
                    desired_length_samples,
                    in_the_end=in_the_end)
        new_sig = signal.copy()
        if hasattr(new_sig, 'window'):
            del new_sig.window
        new_sig.time_data = new_time_data
    elif type(signal) == MultiBandSignal:
        assert signal.same_sampling_rate, \
            'Padding or trimming is not supported for multirate signals'
        new_sig = signal.copy()
        for ind, b in enumerate(signal.bands):
            new_sig.bands[ind] = pad_trim(
                b, desired_length_samples, in_the_end)
    else:
        raise TypeError('Signal must be of type Signal or MultiBandSignal')
    return new_sig


def merge_signals(in1: Signal | MultiBandSignal, in2: Signal | MultiBandSignal,
                  padding_trimming: bool = True, at_end: bool = True) -> \
        Signal | MultiBandSignal:
    """Merges two signals by appending the channels of the second one to the
    first. If the length of in2 is not the same, trimming or padding is
    applied to in2 when `padding_trimming=True`, otherwise an error
    is raised. `at_end=True` applies the padding/trimming at the end of signal.

    Parameters
    ----------
    in1 : `Signal` or `MultiBandSignal`
        First signal.
    in2 : `Signal` or `MultiBandSignal`
        Second signal.
    padding_trimming : bool, optional
        If the signals do not have the same length, the second one is padded
        or trimmed. When `True`, padding/trimming is done.
        Default: `True`.
    at_end : bool, optional
        When `True` and `padding_trimming=True`, padding or trimming is done
        at the end of signal. Otherwise it is done in the beginning.
        Default: `True`.

    Returns
    -------
    new_sig : `Signal`
        New merged signal.

    """
    if type(in1) == Signal:
        assert type(in2) == Signal, \
            'Both signals have to be type Signal'
        assert in1.sampling_rate_hz == in2.sampling_rate_hz, \
            'Sampling rates do not match'
        if in1.time_data.shape[0] != in2.time_data.shape[0]:
            if padding_trimming:
                in2 = pad_trim(in2, in1.time_data.shape[0], in_the_end=at_end)
            else:
                raise ValueError(
                    'Signals have different lengths and padding or trimming ' +
                    'is not activated')
        new_time_data = np.append(in1.time_data, in2.time_data, axis=1)
        new_sig = in1.copy()
        if hasattr(new_sig, 'window'):
            del new_sig.window
        new_sig.time_data = new_time_data
    elif type(in1) == MultiBandSignal:
        assert type(in2) == MultiBandSignal, \
            'Both signals should be multi band signals'
        assert in1.same_sampling_rate == in2.same_sampling_rate, \
            'Both Signals should have same settings regarding sampling rate'
        if in1.same_sampling_rate:
            assert in1.sampling_rate_hz == in2.sampling_rate_hz, \
                'Sampling rates do not match'
        assert in1.number_of_bands == in2.number_of_bands, \
            'Both signals should have the same number of bands'
        new_bands = []
        for n in range(in1.number_of_bands):
            new_bands.append(
                merge_signals(in1.bands[n], in2.bands[n],
                              padding_trimming,
                              at_end))
        new_sig = MultiBandSignal(
            new_bands,
            same_sampling_rate=in1.same_sampling_rate, info=in1.info)
    else:
        raise ValueError(
            'Signals have to be type of type Signal or MultiBandSignal')
    return new_sig


def merge_filterbanks(fb1: FilterBank, fb2: FilterBank) -> FilterBank:
    """Merges two filterbanks.

    Parameters
    ----------
    fb1 : `FilterBank`
        First filterbank.
    fb2 : `FilterBank`
        Second filterbank.

    Returns
    -------
    new_fb : `FilterBank`
        New filterbank with merged filters

    """
    assert fb1.same_sampling_rate == fb2.same_sampling_rate, \
        'Both filterbanks should have the same settings regarding ' +\
        'sampling rates'
    if fb1.same_sampling_rate:
        assert fb1.sampling_rate_hz == fb2.sampling_rate_hz, \
            'Sampling rates do not match'

    new_filters = fb1.filters
    for n in fb2.filters:
        new_filters.append(n)
    new_fb = FilterBank(new_filters, fb1.same_sampling_rate, fb1.info)
    return new_fb


def resample(sig: Signal, desired_sampling_rate_hz: int) -> Signal:
    """Resamples signal to the desired sampling rate using
    `scipy.signal.resample_poly` with an efficient polyphase representation.

    Parameters
    ----------
    sig : `Signal`
        Signal to be resampled.
    desired_sampling_rate_hz : int
        Sampling rate to convert the signal to.

    Returns
    -------
    new_sig : `Signal`
        Resampled signal.

    """
    if sig.sampling_rate_hz == desired_sampling_rate_hz:
        return sig.copy()
    ratio = Fraction(
        numerator=desired_sampling_rate_hz, denominator=sig.sampling_rate_hz)
    u, d = ratio.as_integer_ratio()
    new_time_data = resample_poly(sig.time_data, up=u, down=d, axis=0)
    new_sig = sig.copy()
    if hasattr(new_sig, 'window'):
        del new_sig.window
    new_sig.time_data = new_time_data
    new_sig.sampling_rate_hz = desired_sampling_rate_hz
    return new_sig


def fractional_octave_frequencies(num_fractions=1,
                                  frequency_range=(20, 20e3),
                                  return_cutoff=False) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray] | \
        tuple[np.ndarray, np.ndarray]:
    """Return the octave center frequencies according to the IEC 61260:1:2014
    standard. This implementation has been taken from the pyfar package. See
    references.

    For numbers of fractions other than `1` and `3`, only the
    exact center frequencies are returned, since nominal frequencies are not
    specified by corresponding standards.

    Parameters
    ----------
    num_fractions : int, optional
        The number of bands an octave is divided into. Eg., ``1`` refers to
        octave bands and ``3`` to third octave bands. The default is ``1``.
    frequency_range : array, tuple
        The lower and upper frequency limits, the default is
        ``frequency_range=(20, 20e3)``.

    Returns
    -------
    nominal : array, float
        The nominal center frequencies in Hz specified in the standard.
        Nominal frequencies are only returned for octave bands and third octave
        bands.
    exact : array, float
        The exact center frequencies in Hz, resulting in a uniform distribution
        of frequency bands over the frequency range.
    cutoff_freq : tuple, array, float
        The lower and upper critical frequencies in Hz of the bandpass filters
        for each band as a tuple corresponding to `(f_lower, f_upper)`.

    References
    ----------
    - The pyfar package: https://github.com/pyfar/pyfar

    """
    nominal = None

    f_lims = np.asarray(frequency_range)
    if f_lims.size != 2:
        raise ValueError(
            "You need to specify a lower and upper limit frequency.")
    if f_lims[0] > f_lims[1]:
        raise ValueError(
            "The second frequency needs to be higher than the first.")

    if num_fractions in [1, 3]:
        nominal, exact = _center_frequencies_fractional_octaves_iec(
            nominal, num_fractions)

        mask = (nominal >= f_lims[0]) & (nominal <= f_lims[1])
        nominal = nominal[mask]
        exact = exact[mask]

    else:
        exact = _exact_center_frequencies_fractional_octaves(
            num_fractions, f_lims)

    if return_cutoff:
        octave_ratio = 10**(3/10)
        freqs_upper = exact * octave_ratio**(1/2/num_fractions)
        freqs_lower = exact * octave_ratio**(-1/2/num_fractions)
        f_crit = (freqs_lower, freqs_upper)
        return nominal, exact, f_crit
    else:
        return nominal, exact


def normalize(sig: Signal | MultiBandSignal, peak_dbfs: float = -6,
              each_channel: bool = False) -> Signal | MultiBandSignal:
    """Normalizes a signal to a given peak value. It either normalizes each
    channel or the signal as a whole.

    Parameters
    ----------
    sig : `Signal` or `MultiBandSignal`
        Signal to be normalized.
    peak_dbfs : float, optional
        dBFS to which to normalize peak. Default: -6.
    each_channel : bool, optional
        When `True`, each channel on its own is normalized. When `False`,
        the peak value for all channels is regarded. Default: `False`.

    Returns
    -------
    new_sig : `Signal` or `MultiBandSignal`
        Normalized signal.

    """
    if type(sig) == Signal:
        new_sig = sig.copy()
        new_time_data = np.empty_like(sig.time_data)
        if each_channel:
            for n in range(sig.number_of_channels):
                new_time_data[:, n] = \
                    _normalize(sig.time_data[:, n], peak_dbfs)
        else:
            new_time_data = _normalize(sig.time_data, peak_dbfs)
        new_sig.time_data = new_time_data
    elif type(sig) == MultiBandSignal:
        new_sig = sig.copy()
        for ind in range(sig.number_of_bands):
            new_sig.bands[ind] = normalize(sig.bands[ind], peak_dbfs)
    else:
        raise TypeError(
            'Type of signal is not valid. Use either Signal or MultiBandSignal'
        )
    return new_sig


def load_pkl_object(path: str):
    """WARNING: This is not secure. Only unpickle data you know!
    Loads an object with all its attributes and methods.

    Parameters
    ----------
    path : str
        Path to the pickle object.

    Returns
    -------
    obj : object
        Unpacked pickle object.

    """
    obj = None
    path = _check_format_in_path(path, 'pkl')
    with open(path, 'rb') as inp:
        obj = pickle.load(inp)
    return obj


def fade(sig: Signal, type_fade: str = 'lin',
         length_fade_seconds: float = None, at_start: bool = True,
         at_end: bool = True) -> Signal:
    """Applies fading to signal.

    Parameters
    ----------
    sig : `Signal`
        Signal to apply fade to.
    type_fade : {'exp', 'lin', 'log'} str, optional
        Type of fading to be applied. Choose from `'exp'` (exponential),
        `'lin'` (linear) or `'log'` (logarithmic). Default: `'lin'`.
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
    type_fade = type_fade.lower()
    assert type_fade in ('lin', 'exp', 'log'), \
        'Type of fade is invalid'
    assert at_start or at_end, \
        'At least start or end of signal should be faded'
    if length_fade_seconds is None:
        length_fade_seconds = sig.time_vector_s[-1]*0.025
    assert length_fade_seconds < sig.time_vector_s[-1], \
        'Fade length should not be longer than the signal itself'

    new_time_data = np.empty_like(sig.time_data)
    for n in range(sig.number_of_channels):
        vec = sig.time_data[:, n]
        if at_start:
            new_time_data[:, n] = _fade(
                vec, length_fade_seconds,
                mode=type_fade,
                sampling_rate_hz=sig.sampling_rate_hz, at_start=True)
        if at_end:
            new_time_data[:, n] = _fade(
                vec, length_fade_seconds,
                mode=type_fade,
                sampling_rate_hz=sig.sampling_rate_hz, at_start=False)
    new_sig = sig.copy()
    new_sig.time_data = new_time_data
    return new_sig


def erb_frequencies(freq_range_hz=[20, 20000], resolution: float = 1,
                    reference_frequency_hz: float = 1000) -> np.ndarray:
    """Get frequencies that are linearly spaced on the ERB frequency scale.
    This implementation was taken and adapted from the pyfar package. See
    references.

    Parameters
    ----------
    freq_range : array-like, optional
        The upper and lower frequency limits in Hz between which the frequency
        vector is computed. Default: [20, 20e3].
    resolution : float, optional
        The frequency resolution in ERB units. 1 returns frequencies that are
        spaced by 1 ERB unit, a value of 0.5 would return frequencies that are
        spaced by 0.5 ERB units. Default: 1.
    reference_frequency : float, optional
        The reference frequency in Hz relative to which the frequency vector
        is constructed. Default: 1000.

    Returns
    -------
    frequencies : `np.ndarray`
        The frequencies in Hz that are linearly distributed on the ERB scale
        with a spacing given by `resolution` ERB units.

    References
    ----------
    - The pyfar package: https://github.com/pyfar/pyfar
    - B. C. J. Moore, An introduction to the psychology of hearing,
      (Leiden, Boston, Brill, 2013), 6th ed.
    - V. Hohmann, “Frequency analysis and synthesis using a gammatone
      filterbank,” Acta Acust. united Ac. 88, 433-442 (2002).
    - P. L. Søndergaard, and P. Majdak, “The auditory modeling toolbox,”
      in The technology of binaural listening, edited by J. Blauert
      (Heidelberg et al., Springer, 2013) pp. 33-56.

    """

    # check input
    if not isinstance(freq_range_hz, (list, tuple, np.ndarray)) \
            or len(freq_range_hz) != 2:
        raise ValueError("freq_range must be an array like of length 2")
    if freq_range_hz[0] > freq_range_hz[1]:
        freq_range_hz = [freq_range_hz[1], freq_range_hz[0]]
    if resolution <= 0:
        raise ValueError("Resolution must be larger than zero")

    # convert the frequency range and reference to ERB scale
    # (Hohmann 2002, Eq. 16)
    erb_range = 9.2645 * np.sign(freq_range_hz) * np.log(
        1 + np.abs(freq_range_hz) * 0.00437)
    erb_ref = 9.2645 * np.sign(reference_frequency_hz) * np.log(
        1 + np.abs(reference_frequency_hz) * 0.00437)

    # get the referenced range
    erb_ref_range = np.array([erb_ref - erb_range[0], erb_range[1] - erb_ref])

    # construct the frequencies on the ERB scale
    n_points = np.floor(erb_ref_range / resolution).astype(int)
    erb_points = np.arange(-n_points[0], n_points[1] + 1) * resolution \
        + erb_ref

    # convert to frequencies in Hz
    frequencies = 1 / 0.00437 * np.sign(erb_points) * (
        np.exp(np.abs(erb_points) / 9.2645) - 1)

    return frequencies


def ir_to_filter(signal: Signal, channel: int = 0,
                 phase_mode: str = 'direct') -> Filter:
    """This function takes in a signal with type `'ir'` or `'rir'` and turns
    the selected channel into an FIR filter. With `phase_mode` it is possible
    to use minimum phase or minimum linear phase.

    Parameters
    ----------
    signal : `Signal`
        Signal to be converted into a filter.
    channel : int, optional
        Channel of the signal to be used. Default: 0.
    phase_mode : {'direct', 'min', 'lin'} str, optional
        Phase of the FIR filter. Choose from `'direct'` (no changes to phase),
        `'min'` (minimum phase) or `'lin'` (minimum linear phase).
        Default: `'direct'`.

    Returns
    -------
    filt : `Filter`
        FIR filter object.

    """
    assert signal.signal_type in ('ir', 'rir', 'h1', 'h2', 'h3'), \
        f'{signal.signal_type} is not valid. Use one of ' +\
        '''('ir', 'rir', 'h1', 'h2', 'h3')'''
    assert channel < signal.number_of_channels, \
        f'Signal does not have a channel {channel}'
    phase_mode = phase_mode.lower()
    assert phase_mode in ('direct', 'min', 'lin'), \
        f'''{phase_mode} is not valid. Choose from ('direct', 'min', 'lin')'''

    # Choose channel
    signal = signal.get_channels(channel)

    # Change phase
    if phase_mode == 'min':
        f, sp = signal.get_spectrum()
        signal = min_phase_from_mag(np.abs(sp), signal.sampling_rate_hz)
    elif phase_mode == 'lin':
        f, sp = signal.get_spectrum()
        signal = lin_phase_from_mag(np.abs(sp), signal.sampling_rate_hz)
    b = signal.time_data[:, 0]
    a = [1]
    filt = Filter(
        'other', {'ba': [b, a]}, sampling_rate_hz=signal.sampling_rate_hz)
    return filt


def filter_to_ir(fir: Filter) -> Signal:
    """Takes in an FIR filter and converts it into an IR by taking its
    b coefficients.

    Parameters
    ----------
    fir : `Filter`
        Filter containing an FIR filter.

    Returns
    -------
    new_sig : `Signal`
        New IR signal.

    """
    assert fir.filter_type == 'fir', \
        'This is only valid is only available for FIR filters'
    b, _ = fir.get_coefficients(mode='ba')
    new_sig = Signal(
        None, b, sampling_rate_hz=fir.sampling_rate_hz, signal_type='ir',
        signal_id='IR from FIR filter')
    return new_sig


def true_peak_level(signal: Signal | MultiBandSignal) \
        -> tuple[np.ndarray, np.ndarray]:
    """Computes true-peak level of a signal using the standardized method
    by the Rec. ITU-R BS.1770-4. See references.

    Parameters
    ----------
    signal : `Signal` or `MultiBandSignal`
        Signal for which to compute the true-peak level.

    Returns
    -------
    true_peak_levels : `np.ndarray`
        True-peak levels (in dBTP) as an array with shape (channels) or
        (band, channels) in case that the input signal is `MultiBandSignal`.
    peak_levels : `np.ndarray`
        Peak levels (in dBFS) as an array with shape (channels) or
        (band, channels) in case that the input signal is `MultiBandSignal`.

    References
    ----------
    - https://www.itu.int/rec/R-REC-BS.1770

    """
    if type(signal) == Signal:
        sig = signal.copy()
        # Reduce gain by 12.04 dB
        down_factor = 10**(-12.04/20)
        up_factor = 1/down_factor
        sig.time_data *= down_factor
        # Resample by 4
        sig_over = resample(sig, sig.sampling_rate_hz*4)
        true_peak_levels = 20*np.log10(np.max(
            np.abs(sig_over.time_data), axis=0) * up_factor)
        peak_levels = 20*np.log10(np.max(
            np.abs(sig.time_data), axis=0) * up_factor)
    elif type(signal) == MultiBandSignal:
        true_peak_levels = \
            np.empty((signal.number_of_bands, signal.number_of_channels))
        peak_levels = np.empty_like(true_peak_levels)
        for ind, b in enumerate(signal.bands):
            true_peak_levels[ind, :], peak_levels[ind, :] = true_peak_level(b)
    else:
        raise TypeError(
            'Passed signal must be of type Signal or MultiBandSignal')
    return true_peak_levels, peak_levels


def fractional_delay(sig: Signal | MultiBandSignal, delay_seconds: float,
                     channels=None, keep_length: bool = False,
                     order: int = 30, side_lobe_suppression_db: float = 60) \
        -> Signal | MultiBandSignal:
    """Apply fractional time delay to a signal. This
    implementation is taken and adapted from the pyfar package. See references.

    Parameters
    ----------
    sig : `Signal` or `MultiBandSignal`
        Signal to be delayed.
    delay_seconds : float
        Delay in seconds.
    channels : int or array-like, optional
        Channels to be delayed. Pass `None` to delay all channels.
        Default: `None`.
    keep_length : bool, optional
        When `True`, the signal retains its original length and loses
        information for the latest samples. If only specific channels are to be
        delayed, and keep_length is set to `False`, the remaining channels are
        zero-padded in the end. Default: `False`.
    order : int, optional
        Order of the sinc filter, higher order yields better results at the
        expense of computation time. Default: 30.
    side_lobe_suppression_db : float, optional
        Side lobe suppresion in dB for the Kaiser window. Default: 60.

    Returns
    -------
    out_sig : `Signal` or `MultiBandSignal`
        Newly created Signal

    References
    ----------
    - The pyfar package: https://github.com/pyfar/pyfar
    - T. I. Laakso, V. Välimäki, M. Karjalainen, and U. K. Laine,
      'Splitting the unit delay,' IEEE Signal Processing Magazine 13,
      30-60 (1996). doi:10.1109/79.482137
    - A. V. Oppenheim and R. W. Schafer, Discrete-time signal processing,
      (Upper Saddle et al., Pearson, 2010), Third edition.

    """
    assert delay_seconds >= 0, \
        'Delay must be positive'
    if type(sig) == Signal:
        if delay_seconds == 0:
            return sig
        if sig.time_data_imaginary is not None:
            warn('Imaginary time data will be ignored in this function. ' +
                 'Delay it manually by creating another signal object, if ' +
                 'needed.')
        delay_samples = delay_seconds*sig.sampling_rate_hz
        assert delay_samples < sig.time_data.shape[0], \
            'Delay too large for the given signal'
        assert order + 1 < sig.time_data.shape[0], \
            'Filter order is longer than the signal itself'
        if channels is None:
            channels = np.arange(sig.number_of_channels)
        channels = np.atleast_1d(np.asarray(channels).squeeze())
        assert np.all(channels < sig.number_of_channels), \
            'There is at least an invalid channel number'
        assert len(np.unique(channels)) == len(channels), \
            'At least one channel is repeated'

        # =========== separate integer and fractional delay ===================
        delay_int = np.atleast_1d(delay_samples).astype(int)
        delay_frac = np.atleast_1d(delay_samples - delay_int)
        # force delay_frac >= 0 as required by Laakso et al. 1996 Eq. (2)
        mask = delay_frac < 0
        delay_int[mask] -= 1
        delay_frac[mask] += 1

        # =========== get sinc function =======================================
        if order % 2:
            M_opt = delay_frac.astype("int") - (order-1)/2
        else:
            M_opt = np.round(delay_frac) - order / 2
        # get matrix versions of the fractional shift and M_opt
        delay_frac_matrix = np.tile(
            delay_frac[..., np.newaxis],
            tuple(np.ones(delay_frac.ndim, dtype="int")) + (order + 1, ))
        M_opt_matrix = np.tile(
            M_opt[..., np.newaxis],
            tuple(np.ones(M_opt.ndim, dtype="int")) + (order + 1, ))
        # discrete time vector
        n = np.arange(order + 1) + M_opt_matrix - delay_frac_matrix
        sinc = np.sinc(n)

        # =========== get kaiser window =======================================
        #  beta parameter for side lobe rejection according to
        # Oppenheim (2010) Eq. (10.13)
        beta = _kaiser_window_beta(np.abs(side_lobe_suppression_db))

        # Kaiser window according to Oppenheim (2010) Eq. (10.12)
        alpha = order / 2
        L = np.arange(order + 1).astype("float") - delay_frac_matrix
        # required to counter operations on M_opt and make sure that the maxima
        # of the underlying continuous sinc function and Kaiser window appear
        # at the same time
        if order % 2:
            L += .5
        else:
            L[delay_frac_matrix > .5] += 1
        Z = beta * np.sqrt(
            np.array(1 - ((L - alpha) / alpha)**2, dtype="complex"))
        # suppress small imaginary parts
        kaiser = np.real(bessel_first_mod(0, Z)) / bessel_first_mod(0, beta)

        # =========== create and apply fractional delay filter ================
        # compute filter and match dimensions
        frac_delay_filter = (sinc * kaiser).squeeze()

        # Copy data
        new_time_data = sig.time_data

        # Create space for the filter in the end of signal
        new_time_data = _pad_trim(
            new_time_data,
            sig.time_data.shape[0] + len(frac_delay_filter) - 1)

        # Delay channels
        new_time_data[:, channels] = convolve(
            sig.time_data[:, channels], frac_delay_filter[..., None],
            mode='full')

        # =========== apply integer delay =====================================
        delay_int += M_opt.astype("int")
        delay_int = np.squeeze(delay_int)

        channels_not = np.setdiff1d(
            channels, np.arange(new_time_data.shape[1]))
        not_delayed = new_time_data[:, channels_not]
        delayed = new_time_data[:, channels]

        # Delay respective channels in the beginning and add zeros in the end
        # to the others
        delayed = _pad_trim(
            delayed, delay_int+new_time_data.shape[0], in_the_end=False)
        not_delayed = _pad_trim(
            not_delayed, delay_int+new_time_data.shape[0], in_the_end=True)

        new_time_data = _pad_trim(
            new_time_data, delay_int+new_time_data.shape[0], in_the_end=True)
        new_time_data[:, channels_not] = not_delayed
        new_time_data[:, channels] = delayed

        # =========== handle length ===========================================
        if keep_length:
            new_time_data = new_time_data[:sig.time_data.shape[0], :]

        # =========== give out object =========================================
        out_sig = sig.copy()
        if hasattr(out_sig, 'window'):
            del out_sig.window
        out_sig.time_data = new_time_data

    elif type(sig) == MultiBandSignal:
        new_bands = []
        out_sig = sig.copy()
        for b in sig.bands:
            new_bands.append(
                fractional_delay(b, delay_seconds, channels, keep_length))
        out_sig.bands = new_bands
    else:
        raise TypeError('Passed signal should be either type Signal or ' +
                        'MultiBandSignal')
    return out_sig


def activity_detector(signal: Signal, threshold_dbfs: float = -20,
                      channel: int = 0, relative_to_peak: bool = True,
                      pre_filter: Filter = None, attack_time_ms: float = 1,
                      release_time_ms: float = 25) \
        -> tuple[Signal, dict]:
    """This is a simple signal activity detector that uses a power threshold.
    It can be used relative to the signal's peak value or absolute. It is only
    applicable to one channel of the signal. This function returns the signal
    and a dictionary containing noise (as a signal) and
    the time indices corresponding to the bins that were found to surpass
    the threshold according to attack and release times.

    Prefiltering (for example with a bandpass filter) is possible when a
    `pre_filter` is passed.

    See Returns to gain insight into the returned dictionary and its keys.

    Parameters
    ----------
    signal : `Signal`
        Signal in which to detect activity.
    threshold_dbfs : float
        Threshold in dBFS to separate noise from activity.
    channel : int, optional
        Channel in which to perform the detection. Default: 0.
    relative_to_peak : bool, optional
        When `True`, the threshold value is relative to the signal's peak
        value. Otherwise, it is regarded as an absolute threshold.
        Default: `True`.
    pre_filter : `Filter`, optional
        Filter used for prefiltering the signal. It can be for instance a
        bandpass filter selecting the relevant frequencies in which the
        activity might be. Pass `None` to avoid any pre filtering.
        Default: `None`.
    attack_time_ms : float, optional
        Attack time (in ms). It corresponds to a lag time for detecting
        activity after surpassing the threshold. Default: 1.
    release_time_ms : float, optional
        Release time (in ms) for activity detector after signal has fallen
        below power threshold. Pass 0 to release immediately. Default: 25.

    Returns
    -------
    detected_sig : `Signal`
        Detected signal.
    others : dict
        Dictionary containing following keys:
        - `'noise'`: left-out noise in original signal (below threshold) as
          `Signal` object.
        - `'signal_indices'`: array of boolean that describes which indices
          of the original time series belong to signal and which to noise.
          `True` at index n means index n was passed to signal.
        - `'noise_indices'`: the inverse array to `'signal_indices'`.

    """
    assert type(channel) == int, \
        'Channel must be type integer. Function is not implemented for ' +\
        'multiple channels.'
    assert threshold_dbfs < 0, \
        'Threshold must be below zero'
    assert release_time_ms >= 0, \
        'Release time must be positive'
    assert attack_time_ms >= 0, \
        'Attack time must be positive'

    # Get channel
    signal = signal.get_channels(channel)

    # Pre-filtering
    if pre_filter is not None:
        assert type(pre_filter) == Filter, \
            'pre_filter must be of type Filter'
        signal_filtered = pre_filter.filter_signal(signal)
    else:
        signal_filtered = signal

    # Release samples
    attack_coeff = _get_smoothing_factor_ema(
        attack_time_ms/1e3, signal.sampling_rate_hz)
    release_coeff = _get_smoothing_factor_ema(
        release_time_ms/1e3, signal.sampling_rate_hz)

    # Get indices
    signal_indices = _indices_above_threshold_dbfs(
        signal_filtered.time_data, threshold_dbfs=threshold_dbfs,
        attack_smoothing_coeff=attack_coeff,
        release_smoothing_coeff=release_coeff, normalize=relative_to_peak)
    noise_indices = ~signal_indices

    # Separate signals
    detected_sig = signal.copy()
    noise = signal.copy()
    if hasattr(detected_sig, 'window'):
        del detected_sig.window
        del noise.window

    try:
        detected_sig.time_data = signal.time_data[signal_indices, 0]
    except ValueError as e:
        warn('No detected activity, threshold might be too high. Detected ' +
             'signal will be a vector filled with zeroes')
        print('Numpy error: ', e)
        detected_sig.time_data = np.zeros(500)

    try:
        noise.time_data = signal.time_data[noise_indices, 0]
    except ValueError as e:
        warn('No detected noise, threshold might be too low. Noise will be ' +
             'a vector filled with zeroes')
        print('Numpy error: ', e)
        noise.time_data = np.zeros(500)

    others = dict(
        noise=noise, signal_indices=signal_indices,
        noise_indices=noise_indices)
    return detected_sig, others


def detrend(sig: Signal | MultiBandSignal, polynomial_order: int = 0) \
        -> Signal | MultiBandSignal:
    """Returns the detrended signal.

    Parameters
    ----------
    sig : Signal
        Signal to detrend.
    polynomial_order : int, optional
        Polynomial order of the fitted polynomial that will be removed
        from time data. 0 is equal to mean removal. Default: 0.

    Returns
    -------
    detrended_sig : Signal
        Detrended signal.

    """
    if type(sig) == Signal:
        assert polynomial_order >= 0, \
            'Polynomial order should be positive'
        td = sig.time_data
        new_td = _detrend(td, polynomial_order)
        detrended_sig = sig.copy()
        detrended_sig.time_data = new_td
        return detrended_sig
    elif type(sig) == MultiBandSignal:
        detrended_sig = sig.copy()
        for n in range(sig.number_of_bands):
            detrended_sig.bands[n] = detrend(
                sig.bands[n], polynomial_order)
        return detrended_sig
    else:
        raise TypeError('Pass either a Signal or a MultiBandSignal')


def rms(sig: Signal | MultiBandSignal, in_dbfs: bool = True) -> np.ndarray:
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
    rms_values : `np.ndarray`
        Array with RMS values. If a `Signal` is passed, it has shape
        (channel). If a `MultiBandSignal` is passed, its shape is
        (bands, channel).

    """
    if type(sig) == Signal:
        rms = _rms(sig.time_data)
    elif type(sig) == MultiBandSignal:
        rms = np.zeros((sig.number_of_bands, sig.number_of_channels))
        for ind, b in enumerate(sig):
            rms[ind, :] = _rms(b.time_data)
    else:
        raise TypeError('Passed signal should be either a Signal or ' +
                        'MultiBandSignal type')
    if in_dbfs:
        rms = 20*np.log10(rms)
    return rms


class CalibrationData():
    """This is a class that takes in a calibration recording and can be used
    to calibrate other signals.

    """
    def __init__(self, calibration_data, type_of_calibration: str = '94db',
                 high_snr: bool = True):
        """Load a calibration sound file. It is expected that it contains
        a recorded harmonic tone of 1 kHz with either 94 dB or 114 dB SPL
        according to [1]. This class can later be used to calibrate a signal.

        Parameters
        ----------
        calibration_data : str, tuple or `Signal`
            Calibration recording. It can be a path (str), a tuple with entries
            (time_data, sampling_rate) or a `Signal` object.
        type_of_calibration : {'94db', '114db'} str, optional
            Type of calibration data. It must be either `'94db'` or `'114db'`.
            Default: `'94db'`.
        high_snr : bool, optional
            If the calibration is expected to have a high Signal-to-noise
            ratio, RMS value is computed directly through the time signal. This
            is done when set to `True`. If not, it might be more precise to
            take the spectrum of the signal and evaluate it at 1 kHz.
            This is recommended for systems where the SNR drops below 10 dB.
            Default: `True`.

        References
        ----------
        - [1]: DIN EN IEC 60942:2018-07.

        """
        if type(calibration_data) == str:
            calibration_data = Signal(calibration_data, None, None)
        elif type(calibration_data) == tuple:
            assert len(calibration_data) == 2, \
                'Tuple must have length 2'
            calibration_data = Signal(None, calibration_data[0],
                                      calibration_data[1])
        elif type(calibration_data) == Signal:
            pass
        else:
            raise TypeError(
                f'{type(calibration_data)} is not a valid type. Use '
                'either str, tuple or Signal')
        self.calibration_signal = calibration_data

        type_of_calibration = type_of_calibration.lower()
        assert type_of_calibration in ('94db', '114db'), \
            f'{type_of_calibration} is not valid. Use 94db or 114db'
        self.calibration_type = type_of_calibration

        self.high_snr = high_snr
        # State tracker
        self.__update = True

    def add_calibration_channel(self, new_channel):
        """Adds a new calibration channel to the calibration signal.

        Parameters
        ----------
        new_channel : str, tuple or `Signal`
            New calibration channel. It can be either a path (str), a tuple
            with entries (time_data, sampling_rate) or a `Signal` object.
            If the lengths are different, padding or trimming is done
            at the end of the new channel. This is supported, but not
            recommended since zero-padding might distort the real RMS value
            of the recorded signal.

        """
        if type(new_channel) == str:
            new_channel = Signal(new_channel, None, None)
        elif type(new_channel) == tuple:
            assert len(new_channel) == 2, \
                'Tuple must have length 2'
            new_channel = Signal(None, new_channel[0], new_channel[1])
        elif type(new_channel) == Signal:
            pass
        else:
            raise TypeError(f'{type(new_channel)} is not a valid type. Use '
                            'either str, tuple or Signal')
        self.calibration_signal = merge_signals(self.calibration_signal,
                                                new_channel)
        self.__update = True

    def _compute_calibration_factors(self):
        """Computes the calibration factors for each channel.

        """
        if self.__update:
            if self.high_snr:
                rms_channels = rms(self.calibration_signal, in_dbfs=False)
            else:
                rms_channels = self._get_rms_from_spectrum()
            factor = 94 if self.calibration_type == '94db' else 114
            p0 = 20e-6
            p_analytical = 10**(factor/20)*p0
            self.calibration_factors = p_analytical / rms_channels
            self.__update = False

    def _get_rms_from_spectrum(self):
        self.calibration_signal.set_spectrum_parameters(
            method='welch', scaling='power spectrum')
        f, sp = self.calibration_signal.get_spectrum()
        ind1k = np.argmin(np.abs(f - 1e3))
        return sp[ind1k, :]**0.5

    def calibrate_signal(self, signal: Signal | MultiBandSignal,
                         force_update: bool = False) \
            -> Signal | MultiBandSignal:
        """Calibrates the time data of a signal and returns it as a new object.
        It can also be a `MultiBandSignal`. If the calibration data only
        contains one channel, this factor is used for all channels of the
        signal. Otherwise, the number of channels must coincide.

        Parameters
        ----------
        signal : `Signal` or `MultiBandSignal`
            Signal to be calibrationrated.
        force_update : bool, optional
            When `True`, an update of the calibration data is forced. This
            might be necessary if the calibration signal or the parameters
            of the object have been manually changed. Default: `False`.

        Returns
        -------
        calibrated_signal : `Signal` or `MultiBandSignal`
            Calibrated signal with time data in Pascal. These values
            are no longer constrained to the range [-1, 1].

        """
        if force_update:
            self.__update = True
        self._compute_calibration_factors()
        if len(self.calibration_factors) > 1:
            assert signal.number_of_channels == \
                len(self.calibration_factors), \
                'Number of channels does not match'
            calibration_factors = self.calibration_factors
        else:
            calibration_factors = np.ones(signal.number_of_channels) * \
                self.calibration_factors

        if type(signal) == Signal:
            calibrated_signal = signal.copy()
            calibrated_signal.signal_id += ' – Calibrated (time data in Pa)'
            calibrated_signal.constrain_amplitude = False
            calibrated_signal.time_data *= calibration_factors
            calibrated_signal.calibrated_signal = True
        elif type(signal) == MultiBandSignal:
            calibrated_signal = signal.copy()
            for b in calibrated_signal:
                b.constrain_amplitude = False
                b.time_data *= calibration_factors
                b.signal_id += ' – Calibrated (time data in Pa)'
                b.calibrated_signal = True
        else:
            raise TypeError('signal has not a valid type. Use Signal or ' +
                            'MultiBandSignal')
        return calibrated_signal


def envelope(signal: Signal, mode: str = 'analytic',
             window_length_samples: int = None):
    """This function computes the envelope of a given signal by means of its
    hilbert transformation. It can also compute the RMS value over a certain
    window length (boxcar). The time signal is always detrended with a linear
    polynomial.

    Parameters
    ----------
    signal : `Signal`
        Time series for which to find the envelope.
    mode : str {'analytic', 'rms'}, optional
        Type of envelope. It either uses the hilbert transform to obtain the
        analytic signal or RMS values. Default: `'analytic'`.
    window_length_samples : int, optional
        Window length (boxcar) to average the RMS values. Cannot be `None`
        if `mode = 'rms'`. Default: `None`.

    Returns
    -------
    `np.ndarray`
        Signal envelope. It has the shape (time sample, channel).

    """
    mode = mode.lower()
    assert mode in ('analytic', 'rms'), \
        'Invalid mode. Use either analytic or rms.'

    signal = detrend(signal, 1)
    if mode == 'analytic':
        env = signal.time_data
        env = np.abs(hilbert(env, axis=0))
        return env
    else:
        assert window_length_samples > 0,\
            'Window length must be more than 1 sample'
        rms_vec = signal.time_data
        rms_vec = convolve(
            rms_vec**2,
            np.ones(window_length_samples)[..., None]/window_length_samples,
            mode='full')[:len(rms_vec), ...]
        rms_vec **= 0.5
        return rms_vec
