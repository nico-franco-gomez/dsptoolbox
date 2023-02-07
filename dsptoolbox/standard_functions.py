"""
Standard functions
------------------
This module contains a general collection of DSP functions that do not fall
under a same category.

"""
import numpy as np
import pickle
from scipy.signal import resample_poly, convolve
from scipy.special import iv as bessel_first_mod
from fractions import Fraction
from warnings import warn

from dsptoolbox.classes.signal_class import Signal
from dsptoolbox.classes.multibandsignal import MultiBandSignal
from dsptoolbox.classes.filterbank import FilterBank
from dsptoolbox.classes.filter_class import Filter
from dsptoolbox._standard import (_latency,
                                  _group_delay_direct,
                                  _minimum_phase,
                                  _center_frequencies_fractional_octaves_iec,
                                  _exact_center_frequencies_fractional_octaves,
                                  _kaiser_window_beta,
                                  _indexes_above_threshold_dbfs)
from dsptoolbox.classes._filter import _group_delay_filter
from dsptoolbox._general_helpers import _pad_trim, _normalize, _fade
from dsptoolbox.transfer_functions import (
    min_phase_from_mag, lin_phase_from_mag)


def latency(in1: Signal, in2: Signal = None) -> np.ndarray:
    """Computes latency between two signals using the correlation method.
    If there is no second signal, the latency between the first and the other
    channels is computed.

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


def group_delay(signal: Signal, method='matlab') \
        -> tuple[np.ndarray, np.ndarray]:
    """Computation of group delay.

    Parameters
    ----------
    signal : Signal
        Signal for which to compute group delay.
    method : str, optional
        `'direct'` uses gradient with unwrapped phase. `'matlab'` uses
        this implementation:
        https://www.dsprelated.com/freebooks/filters/Phase_Group_Delay.html.
        Default: `'matlab'`.

    Returns
    -------
    freqs : `np.ndarray`
        Frequency vector in Hz.
    group_delays : `np.ndarray`
        Matrix containing group delays in seconds with shape (gd, channel).

    """
    method = method.lower()
    assert method in ('direct', 'matlab'), \
        f'{method} is not valid. Use direct or matlab'

    signal.set_spectrum_parameters('standard')
    f, sp = signal.get_spectrum()
    if method == 'direct':
        group_delays = np.zeros((sp.shape[0], sp.shape[1]))
        for n in range(signal.number_of_channels):
            group_delays[:, n] = _group_delay_direct(sp[:, n], f[1]-f[0])
    else:
        group_delays = \
            np.zeros(
                (signal.time_data.shape[0]//2+1,
                 signal.time_data.shape[1]))
        for n in range(signal.number_of_channels):
            b = signal.time_data[:, n].copy()
            a = [1]
            _, group_delays[:, n] = \
                _group_delay_filter(
                    [b, a],
                    len(b)//2+1,
                    signal.sampling_rate_hz)
    return f, group_delays


def minimum_phase(signal: Signal) -> tuple[np.ndarray, np.ndarray]:
    """Gives back a matrix containing the minimal phase for every channel.

    Parameters
    ----------
    signal : Signal
        Signal for which to compute the minimal phase.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    min_phases : `np.ndarray`
        Minimal phases as matrix with shape (phase, channel).

    """
    assert signal.signal_type in ('rir', 'ir', 'h1', 'h2', 'h3'), \
        'Signal type must be rir or ir'
    signal.set_spectrum_parameters('standard')
    f, sp = signal.get_spectrum()

    min_phases = np.zeros((sp.shape[0], sp.shape[1]), dtype='float')
    for n in range(signal.number_of_channels):
        min_phases[:, n] = _minimum_phase(np.abs(sp[:, n]), unwrapped=False)
    return f, min_phases


def minimum_group_delay(signal: Signal) -> tuple[np.ndarray, np.ndarray]:
    """Computes minimum group delay of given signal.

    Parameters
    ----------
    signal : Signal
        Signal object for which to compute minimal group delay.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    min_gd : `np.ndarray`
        Minimal group delays in seconds as matrix with shape (gd, channel).

    References
    ----------
    - https://www.roomeqwizard.com/help/help_en-GB/html/minimumphase.html

    """
    f, min_phases = minimum_phase(signal)
    min_gd = np.zeros_like(min_phases)
    for n in range(signal.number_of_channels):
        min_gd[:, n] = _group_delay_direct(min_phases[:, n], f[1]-f[0])
    return f, min_gd


def excess_group_delay(signal: Signal) -> tuple[np.ndarray, np.ndarray]:
    """Computes excess group delay.

    Parameters
    ----------
    signal : Signal
        Signal object for which to compute minimal group delay.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    ex_gd : `np.ndarray`
        Excess group delays in seconds with shape (excess_gd, channel).

    References
    ----------
    - https://www.roomeqwizard.com/help/help_en-GB/html/minimumphase.html

    """
    f, min_gd = minimum_group_delay(signal)
    f, gd = group_delay(signal)
    ex_gd = gd - min_gd
    return f, ex_gd


def pad_trim(signal: Signal | MultiBandSignal, desired_length_samples: int,
             in_the_end: bool = True) -> Signal | MultiBandSignal:
    """Returns a copy of the signal with padded or trimmed time data. Only
    valid for same sampling rate MultiBandSignal.

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
    ratio = Fraction(
        numerator=desired_sampling_rate_hz, denominator=sig.sampling_rate_hz)
    u, d = ratio.as_integer_ratio()
    new_time_data = resample_poly(sig.time_data, up=u, down=d, axis=0)
    new_sig = sig.copy()
    new_sig.time_data = new_time_data
    new_sig.sampling_rate_hz = desired_sampling_rate_hz
    return new_sig


def fractional_octave_frequencies(
        num_fractions=1, frequency_range=(20, 20e3), return_cutoff=False):
    """Return the octave center frequencies according to the IEC 61260:1:2014
    standard.

    For numbers of fractions other than ``1`` and ``3``, only the
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
        for each band as a tuple corresponding to ``(f_lower, f_upper)``.

    References
    ----------
    - This implementation is taken from the pyfar package. See
      https://github.com/pyfar/pyfar

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
    Loads a optimization object with all its attributes and methods.

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
    assert path[-4:] == '.pkl', \
        'Only pickle objects can be loaded'
    path2file = path
    with open(path2file, 'rb') as inp:
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
    type_fade : str, optional
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
                    reference_frequency_hz=1000) -> np.ndarray:
    """Get frequencies that are linearly spaced on the ERB frequency scale.
    This implementation was taken and adapted from the pyfar package. See
    references.

    Parameters
    ----------
    freq_range : array-like
        The upper and lower frequency limits in Hz between which the frequency
        vector is computed.
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
    - B. C. J. Moore, An introduction to the psychology of hearing,
      (Leiden, Boston, Brill, 2013), 6th ed.
    - V. Hohmann, “Frequency analysis and synthesis using a gammatone
      filterbank,” Acta Acust. united Ac. 88, 433-442 (2002).
    - P. L. Søndergaard, and P. Majdak, “The auditory modeling toolbox,”
      in The technology of binaural listening, edited by J. Blauert
      (Heidelberg et al., Springer, 2013) pp. 33-56.
    - The pyfar package: https://github.com/pyfar/pyfar

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
    phase_mode : str, optional
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


def true_peak_level(sig: Signal | MultiBandSignal) \
        -> tuple[np.ndarray, np.ndarray]:
    """Computes true-peak level of a signal using the standardized method
    by the Rec. ITU-R BS.1770-4. See references.

    Parameters
    ----------
    sig : `Signal` or `MultiBandSignal`
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
    if type(sig) == Signal:
        # Reduce gain by 12.04 dB
        down_factor = 10**(-12.04/20)
        up_factor = 1/down_factor
        sig.time_data *= down_factor
        # Resample by 4
        sig_over = resample(sig, sig.sampling_rate_hz*4)
        true_peak_levels = np.empty(sig.number_of_channels)
        peak_levels = np.empty_like(true_peak_levels)
        # Find new peak value and add back 12.04 dB of gain
        for n in range(sig.number_of_channels):
            true_peak_levels[n] = \
                20*np.log10(
                    np.max(np.abs(sig_over.time_data[:, n])) * up_factor)
            peak_levels[n] = \
                20*np.log10(
                    np.max(np.abs(sig.time_data[:, n])) * up_factor)
    elif type(sig) == MultiBandSignal:
        true_peak_levels = \
            np.empty((sig.number_of_bands, sig.number_of_channels))
        peak_levels = np.empty_like(true_peak_levels)
        for ind, b in enumerate(sig.bands):
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
    assert side_lobe_suppression_db > 0, \
        'Side lobe suppression must be positive'
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

        # Delay all channels in the beginning
        new_time_data = _pad_trim(
            new_time_data,
            delay_int+new_time_data.shape[0], in_the_end=False)

        # =========== handle length ===========================================
        if keep_length:
            new_time_data = new_time_data[:sig.time_data.shape[0], :]

        # =========== give out object =========================================
        out_sig = sig.copy()
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


def activity_detector(signal: Signal, channel: int = 0,
                      threshold_dbfs: float = -20, pre_filter: Filter = None,
                      attack_time_ms: float = 0.5,
                      release_time_ms: float = 25) \
        -> tuple[Signal, dict]:
    """This is a simple signal activity detector that uses a power threshold
    relative to maximum peak level in a given signal (for one channel).
    It returns the signal and a dictionary containing noise (as a signal) and
    the time indexes corresponding to the bins that were found to surpass
    the threshold according to attack and release times.

    Prefiltering (for example with a bandpass filter) is possible when a
    `pre_filter` is passed.

    See Returns to gain insight into the returned dictionary and its keys.

    Parameters
    ----------
    signal : `Signal`
        Signal in which to detect activity.
    channel : int, optional
        Channel in which to perform the detection. Default: 0.
    threshold_dbfs : float, optional
        Threshold in dBFS (relative to the signal's maximum peak level) to
        separate noise from activity. Default: -20.
    pre_filter : `Filter`, optional
        Filter used for prefiltering the signal. It can be for instance a
        bandpass filter selecting the relevant frequencies in which the
        activity might be. Pass `None` to avoid any pre filtering.
        Default: `None`.
    attack_time_ms : float, optional
        Attack time (in ms). It corresponds to the time that the signal has
        to surpass the power threshold in order to be regarded as valid
        acitivity. Pass 0 to attack immediately. Default: 0.5.
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
        - `'signal_indexes'`: array of boolean that describes which indexes
          of the original time series belong to signal and which to noise.
          `True` at index n means index n was passed to signal.
        - `'noise_indexes'`: the inverse array to `'signal_indexes'`.

    """
    assert type(channel) == int, \
        'Channel must be type integer. Function is not implemented for ' +\
        'multiple channels.'
    assert threshold_dbfs < 0, \
        'Threshold must be below zero'
    assert release_time_ms >= 0, \
        'Release time must be positive'

    # Get relevant channel
    signal = signal.get_channels(channel)

    # Pre-filtering
    if pre_filter is not None:
        assert type(pre_filter) == Filter, \
            'pre_filter must be of type Filter'
        assert signal.sampling_rate_hz == pre_filter.sampling_rate_hz, \
            'Sampling rates for signal and pre filter do not match'
        signal_filtered = pre_filter.filter_signal(signal)
    else:
        signal_filtered = signal

    # Release samples
    release_time_samples = int(release_time_ms*signal.sampling_rate_hz*1e-3)
    attack_time_samples = int(attack_time_ms*signal.sampling_rate_hz*1e-3)

    # Get indexes
    signal_indexes = _indexes_above_threshold_dbfs(
        signal_filtered.time_data, threshold_dbfs=threshold_dbfs,
        attack_samples=attack_time_samples,
        release_samples=release_time_samples)
    noise_indexes = ~signal_indexes

    # Separate signals
    detected_sig = signal.copy()
    noise = signal.copy()
    detected_sig.time_data = signal.time_data[signal_indexes, 0]
    noise.time_data = signal.time_data[noise_indexes, 0]

    others = dict(
        noise=noise, signal_indexes=signal_indexes,
        noise_indexes=noise_indexes)
    return detected_sig, others
