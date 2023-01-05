"""
Standard functions in DSP processes
"""
import numpy as np
from scipy.signal import resample_poly
from dsptoolbox.classes.signal_class import Signal
from dsptoolbox.classes.multibandsignal import MultiBandSignal
from dsptoolbox.classes.filterbank import FilterBank
from dsptoolbox._standard import (_latency,
                                  _group_delay_direct,
                                  _minimal_phase,
                                  _center_frequencies_fractional_octaves_iec,
                                  _exact_center_frequencies_fractional_octaves)
from dsptoolbox.classes._filter import _group_delay_filter
from dsptoolbox._general_helpers import _pad_trim, _normalize, _fade
from fractions import Fraction
import pickle


def latency(in1: Signal, in2: Signal = None):
    """Computes latency between two signals using the correlation method.
    If there is no second signal, the latency between the first and the other
    channels of the is computed.

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
        latency_per_channel_samples = \
            np.zeros(in1.number_of_channels, dtype=int)
        for n in range(in1.number_of_channels-1):
            latency_per_channel_samples[n] = \
                _latency(in1.time_data[:, n], in1.time_data[:, n+1])
    return latency_per_channel_samples


def group_delay(signal: Signal, method='direct'):
    """Computation of group delay.

    Parameters
    ----------
    signal : Signal
        Signal for which to compute group delay.
    method : str, optional
        `'direct'` uses gradient with unwrapped phase. `'matlab'` uses
        this implementation:
        https://www.dsprelated.com/freebooks/filters/Phase_Group_Delay.html

    Returns
    -------
    freqs : `np.ndarray`
        Frequency vector in Hz.
    group_delays : `np.ndarray`
        Matrix containing group delays in seconds.

    """
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


def minimal_phase(signal: Signal):
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
        Minimal phases as matrix.

    """
    assert signal.signal_type in ('rir', 'ir', 'h1', 'h2', 'h3'), \
        'Signal type must be rir or ir'
    signal.set_spectrum_parameters('standard')
    f, sp = signal.get_spectrum()

    min_phases = np.zeros((sp.shape[0], sp.shape[1]), dtype='float')
    for n in range(signal.number_of_channels):
        min_phases[:, n] = _minimal_phase(np.abs(sp[:, n]), unwrapped=False)
    return f, min_phases


def minimal_group_delay(signal: Signal):
    """Computes minimal group delay

    Parameters
    ----------
    signal : Signal
        Signal object for which to compute minimal group delay.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    min_gd : `np.ndarray`
        Minimal group delays in seconds as matrix.

    """
    f, min_phases = minimal_phase(signal)
    min_gd = np.zeros_like(min_phases)
    for n in range(signal.number_of_channels):
        min_gd[:, n] = _group_delay_direct(min_phases[:, n], f[1]-f[0])
    return f, min_gd


def excess_group_delay(signal: Signal):
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
        Excess group delays in seconds.
    """
    f, min_gd = minimal_group_delay(signal)
    f, gd = group_delay(signal)
    ex_gd = gd - min_gd
    return f, ex_gd


def pad_trim(signal: Signal, desired_length_samples: int,
             in_the_end: bool = True):
    """Returns a copy of the signal with padded or trimmed time data.

    Parameters
    ----------
    signal : Signal
        Signal to be padded or trimmed.
    desired_length_samples : int
        Length of resulting signal.
    in_the_end : bool, optional
        Defines if padding or trimming should be done in the beginning or
        in the end of the signal. Default: `True`.

    Returns
    -------
    new_signal : Signal
        New padded signal.

    """
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
    return new_sig


def merge_signals(in1, in2, padding_trimming: bool = True,
                  at_end: bool = True):
    """Merges two signals by appending the channels of the second one to the
    first. If the length of in2 is not the same, trimming or padding is
    applied at the end.

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
        new_sig._generate_metadata()  # Bug with number of channels
    else:
        raise ValueError(
            'Signals have to be type of type Signal or MultiBandSignal')
    return new_sig


def merge_filterbanks(fb1: FilterBank, fb2: FilterBank):
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


def resample(sig: Signal, desired_sampling_rate_hz: int):
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


def normalize(sig, peak_dbfs: float = -6, each_channel: bool = False):
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
         at_end: bool = True):
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
        When `True`, the start of signal of faded. Default: `True`.
    at_end : bool, optional
        When `True`, the ending of signal of faded. Default: `True`.

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
