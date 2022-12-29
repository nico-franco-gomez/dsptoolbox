"""
Standard functions in DSP processes
"""
import numpy as np
from dsptoolbox.classes.signal_class import Signal
from dsptoolbox.classes.multibandsignal import MultiBandSignal
from dsptoolbox.classes.filterbank import FilterBank
from dsptoolbox._standard import (_latency,
                                  _group_delay_direct,
                                  _minimal_phase)
from dsptoolbox.classes._filter import _group_delay_filter
from dsptoolbox._general_helpers import _pad_trim
from copy import deepcopy


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

    new_sig = deepcopy(signal)
    new_sig.time_data = new_time_data
    return new_sig


def merge_signals(in1, in2, trimming_at_end: bool = True):
    """Merges two signals by appending the channels of the second one to the
    first. If the length of in2 is not the same, trimming or padding is
    applied at the end.

    Parameters
    ----------
    in1 : `Signal` or `MultiBandSignal`
        First signal.
    in2 : `Signal` or `MultiBandSignal`
        Second signal.
    trimming_at_end : bool, optional
        If the signals do not have the same length, the second one is padded
        or trimmed. When `True`, padding/trimming is done at the end.
        Default: `True`.
    
    Returns
    -------
    new_sig : `Signal`
        New merged signal.

    """
    
    if type(in1) == Signal:
        assert in1.sampling_rate_hz == in2.sampling_rate_hz, \
            'Sampling rates do not match'
        assert type(in2) == Signal, \
            'Both signals have to be type Signal'
        if in1.time_data.shape[0] != in2.time_data.shape[0]:
            in2 = pad_trim(in2, in1.time_data.shape[0], trimming_at_end)
        new_time_data = np.append(in1.time_data, in2.time_data, axis=1)
        new_sig = deepcopy(in1)
        new_sig.time_data = new_time_data
    elif type(in1) == MultiBandSignal:
        assert in1.same_sampling_rate == in2.same_sampling_rate, \
            'Both Signals should have same settings regarding sampling rate'
        if in1.same_sampling_rate:
            assert in1.sampling_rate_hz == in2.sampling_rate_hz, \
                'Sampling rates do not match'
        assert type(in2) == MultiBandSignal, \
            'Both signals should be multi band signals'
        assert in1.number_of_bands == in2.number_of_bands, \
            'Both signals should have the same number of bands'
        new_bands = []
        for n in range(in1.number_of_bands):
            new_bands.append(merge_signals(in1.bands[n], in2.bands[n]))
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
    fb1 : `FilterBank`
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
