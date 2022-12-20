"""
High-level methods for room acoustics functions
"""
import numpy as np
from scipy.signal import find_peaks, convolve
from .signal_class import Signal
from .standard_functions import group_delay
from .backend._room_acoustics import (_reverb,
                                      _complex_mode_identification,
                                      _sum_magnitude_spectra)
from .backend._general_helpers import _find_nearest, _normalize


__all__ = ['reverb_time', 'find_modes', 'convolve_rir_on_signal']


def reverb_time(signal: Signal, mode: str = 'T20'):
    """Computes reverberation time. T20, T30, T60 and EDT.

    Parameters
    ----------
    signal : Signal
        Signal for which to compute reverberation times. It must be type
        `'ir'` or `'rir'`.
    mode : str, optional
        Reverberation time mode. Options are `'T20'`, `'T30'`, `'T60'` or
        `'EDT'`. Default: `'T20'`.

    Returns
    -------
    reverberation_times : np.ndarray
        Reverberation times for each channel.

    References
    ----------
    - DIN 3382
    - ISO 3382-1:2009-10, Acoustics - Measurement of the reverberation time of
    rooms with reference to other acoustical parameters. pp. 22
    """
    assert signal.signal_type in ('ir', 'rir'), f'{signal.signal_type} is ' +\
        'not a valid signal type for reverb_time. It should be ir or rir'
    valid_modes = ('T20', 'T30', 'T60', 'EDT')
    valid_modes = (mode.casefold() for n in valid_modes)
    assert mode.casefold() in valid_modes, \
        f'{mode} is not valid. Use either one of ' +\
        'these: T20, T30, T60 or EDT'

    reverberation_times = np.zeros((signal.number_of_channels))
    for n in range(signal.number_of_channels):
        reverberation_times[n] = \
            _reverb(
                signal.time_data[:, n],
                signal.sampling_rate_hz,
                mode.casefold())
    return reverberation_times


def find_modes(signal: Signal, f_range_hz=[50, 200],
               proximity_effect=False, dist_hz=5):
    """This metod is NOT validated. It might not be sufficient to find all
    modes in the given range.

    Computes the room modes of a set of RIR using different criteria:
    Complex mode indication function, sum of magnitude responses and group
    delay peaks of the first RIR.

    Parameters
    ----------
    rir: array-like, matrix
        List or matrix containing RIR's. It is assumed that every RIR has
        the same length.
    f_range_hz: array-like, optional
        Vector setting range for mode search. Default: [50, 200].
    proximity_effect: bool, optional
        When `True`, only group delay criteria is used for finding modes
        up until 200 Hz. This is done since a gradient transducer will not
        easily see peaks in its magnitude response in low frequencies
        due to near-field effects.
        There is also an assessment that the modes are not dips of
        the magnitude response. Default: `False`.
    dist_hz: float, optional
        Minimum distance (in Hz) between modes. Default: 5.

    Returns
    -------
    f_modes: np.ndarray
        Vector containing frequencies where modes have been localized.

    References
    ----------
    http://papers.vibetech.com/Paper17-CMIF.pdf
    """
    assert len(f_range_hz) == 2, 'Range of frequencies must have a ' +\
        'minimum and a maximum value'

    assert signal.signal_type in ('rir', 'ir'), \
        f'{signal.signal_type} is not a valid signal type. It should ' +\
        'be either rir or ir'
    signal.set_spectrum_parameters('standard')
    f, sp = signal.get_spectrum()

    # Setting up frequency range
    ids = _find_nearest(f_range_hz, f)
    f = f[ids[0]:ids[1]]
    df = f[1]-f[0]

    cmif = _complex_mode_identification(sp[ids[0]:ids[1], :]).squeeze()
    sum_sp = _sum_magnitude_spectra(sp[ids[0]:ids[1], :])

    # Group delay
    _, group_ms = group_delay(signal)
    group_ms = group_ms[ids[0]:ids[1]]*1e3

    # Find peaks
    width = int(np.ceil(dist_hz / df))
    id_sum, _ = find_peaks(sum_sp, width=width)
    id_cmif, _ = find_peaks(cmif, width=width)
    id_group = []
    for n in range(signal.number_of_channels):
        id_, _ = find_peaks(group_ms[:, n], width=width)
        id_group.append(id_)

    if proximity_effect:
        f_modes = np.array([])
        for n in range(signal.number_of_channels):
            f_modes = \
                np.append(f_modes, f[id_group[n]][f[id_group[n]] < 199.9])
        ind_200 = np.where(f >= 199.9)
        if len(np.squeeze(ind_200)) < 1:
            ind_200 = len(f)
        else:
            ind_200 = ind_200[0][0]
        f_modes = f_modes.flatten()
        f_modes = list(f_modes)
        temp = []
        for f_m in f_modes:
            if f_modes.count(f_m) >= 2:
                temp.append(f_m)
        f_modes = set(temp)
    else:
        f_modes = set()
        ind_200 = 0

    f_modes = set(f_modes)

    # Same frequency appears in at least two of three peaks vectors
    for n in range(ind_200, len(f)):
        cond1 = f[n] in f[id_sum]
        cond2 = f[n] in f[id_cmif]
        cond3 = f[n] in f[id_group[0]]
        cond_1 = cond1 and cond2
        cond_2 = cond1 and cond3
        cond_3 = cond2 and cond3
        if cond_1 or cond_2 or cond_3:
            f_modes.add(f[n])
    f_modes = np.sort(list(f_modes))

    return f_modes.astype(int)


def convolve_rir_on_signal(signal: Signal, rir: Signal,
                           keep_peak_level: bool = True,
                           keep_length: bool = True):
    """Applies an RIR to a given signal. The RIR should also be a signal object
    with a single channel containing the RIR time data. Signal type should
    also be set to IR or RIR. By default, all channels are convolved with
    the RIR.

    Parameters
    ----------
    signal : Signal
        Signal to which the RIR is applied. All channels are affected.
    rir : Signal
        Single-channel Signal object containing the RIR.
    keep_peak_level : bool, optional
        When `True`, output signal is normalized to the peak level of
        the original signal. Default: `True`.
    keep_length : bool, optional
        When `True`, the original length is kept after convolution, otherwise
        the output signal is longer than the input one. Default: `True`.

    Returns
    -------
    new_sig : Signal
        Convolved signal with RIR.
    """
    assert rir.signal_type in ('rir', 'ir'), \
        f'{rir.signal_type} is not a valid signal type. Set it to rir or ir.'
    assert signal.time_data.shape[0] > rir.time_data.shape[0], \
        'The RIR is longer than the signal to convolve it with.'
    assert rir.number_of_channels == 1, \
        'RIR should not contain more than one channel.'
    assert rir.sampling_rate_hz == signal.sampling_rate_hz, \
        'The sampling rates do not match'

    if keep_length:
        total_length_samples = signal.time_data.shape[0]
    else:
        total_length_samples = \
            signal.time_data.shape[0] + rir.time_data.shape[0] - 1
    new_time_data = np.zeros((total_length_samples, signal.number_of_channels))

    for n in range(signal.number_of_channels):
        if keep_peak_level:
            old_peak = 20*np.log10(np.max(np.abs(signal.time_data[:, n])))
        new_time_data[:, n] = \
            convolve(
                signal.time_data[:, n],
                rir.time_data[:, 0],
                mode='full')[:total_length_samples]
        if keep_peak_level:
            new_time_data[:, n] = \
                _normalize(new_time_data[:, n], old_peak, mode='peak')

    new_sig = Signal(
        None,
        new_time_data,
        signal.sampling_rate_hz,
        signal_type=signal.signal_type,
        signal_id=signal.signal_id+' (convolved with RIR)')
    return new_sig
