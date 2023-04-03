"""
High-level methods for room acoustics functions
"""
import numpy as np
from scipy.signal import find_peaks, convolve
# from warnings import warn

from dsptoolbox.classes import Signal, MultiBandSignal, Filter
from dsptoolbox.filterbanks import (fractional_octave_bands,
                                    linkwitz_riley_crossovers)
from dsptoolbox.transfer_functions import group_delay
from ._room_acoustics import (_reverb,
                              _complex_mode_identification,
                              _sum_magnitude_spectra,
                              _find_ir_start,
                              _generate_rir,
                              ShoeboxRoom,
                              _add_reverberant_tail_noise,
                              _d50_from_rir,
                              _c80_from_rir,
                              _ts_from_rir)
from dsptoolbox._general_helpers import _find_nearest, _normalize, _pad_trim


def reverb_time(signal: Signal | MultiBandSignal, mode: str = 'T20',
                ir_start: int = None) -> np.ndarray:
    """Computes reverberation time. T20, T30, T60 and EDT.

    Parameters
    ----------
    signal : `Signal` or `MultiBandSignal`
        Signal for which to compute reverberation times. It must be type
        `'ir'` or `'rir'`.
    mode : str, optional
        Reverberation time mode. Options are `'T20'`, `'T30'`, `'T60'` or
        `'EDT'`. Default: `'T20'`.
    ir_start : int, optional
        When not `None`, it is used as the index of the start of the impulse
        response. Otherwise it is automatically computed as the first point
        where the normalized IR arrives at -20 dBFS. Default: `None`.

    Returns
    -------
    reverberation_times : `np.ndarray`
        Reverberation times for each channel. Shape is (band, channel)
        if MultiBandSignal object is passed.

    References
    ----------
    - DIN EN ISO 3382-1:2009-10, Acoustics - Measurement of the reverberation
      time of rooms with reference to other acoustical parameters.

    """
    if type(signal) == Signal:
        assert signal.signal_type in ('ir', 'rir'), \
            f'{signal.signal_type} is not a valid signal type for ' +\
            'reverb_time. It should be ir or rir'
        mode = mode.upper()
        valid_modes = ('T20', 'T30', 'T60', 'EDT')
        assert mode in valid_modes, \
            f'{mode} is not valid. Use either one of ' +\
            'these: T20, T30, T60 or EDT'
        reverberation_times = np.zeros((signal.number_of_channels))
        for n in range(signal.number_of_channels):
            reverberation_times[n] = _reverb(
                signal.time_data[:, n].copy(), signal.sampling_rate_hz,
                mode, ir_start=ir_start, return_ir_start=False)
    elif type(signal) == MultiBandSignal:
        reverberation_times = \
            np.zeros(
                (signal.number_of_bands, signal.bands[0].number_of_channels))
        for ind in range(signal.number_of_bands):
            reverberation_times[ind, :] = reverb_time(
                signal.bands[ind], mode, ir_start=ir_start)
    else:
        raise TypeError(
            'Passed signal should be of type Signal or MultiBandSignal')
    return reverberation_times
    # return reverberation_times.squeeze()


def find_modes(signal: Signal, f_range_hz=[50, 200],
               proximity_effect: bool = False, dist_hz: float = 5,
               prune_antimodes: bool = False) -> np.ndarray:
    """This metod is NOT validated. It might not be sufficient to find all
    modes in the given range.

    Computes the room modes of a set of RIR using different criteria:
    Complex mode indication function, sum of magnitude responses and group
    delay peaks of RIRs. If modes are identified in at least two of the three
    criteria, they are considered as such.

    The parameter prune antimodes is used to avoid getting modes that are
    dips (and not peaks) in the frequency responses. This is done after
    mode identification and is therefore only needed when proximity effect is
    set to `True` and mode identification is done using group delay criteria,
    since, for modes to be identified as such, they would need
    to exhibit peaks in at least the CMIF or sum of all magnitude spectra. If
    they were dips, they would be ignored anyway.

    Parameters
    ----------
    signal : `Signal`
        Signal containing the RIR'S from which to find the modes.
    f_range_hz : array-like, optional
        Vector setting range for mode search. Default: [50, 200].
    proximity_effect : bool, optional
        When `True`, only group delay criteria is used for finding modes
        up until 200 Hz. This is done since a gradient transducer will not
        easily see peaks in its magnitude response in low frequencies
        due to near-field effects. Default: `False`.
    dist_hz : float, optional
        Minimum distance (in Hz) between modes. Default: 5.
    prune_antimodes : bool, optional
        See if the detected modes are dips in the frequency response of the
        first RIR. This is only needed for the group delay method, which
        is essential when proximity_effect is set to `True`. Default: `False`.

    Returns
    -------
    f_modes : `np.ndarray`
        Vector containing frequencies where modes have been localized.

    References
    ----------
    - http://papers.vibetech.com/Paper17-CMIF.pdf

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

    # Compute CMIF and sum of all magnitude spectra
    cmif = _complex_mode_identification(sp[ids[0]:ids[1], :]).squeeze()
    sum_sp = _sum_magnitude_spectra(sp[ids[0]:ids[1], :])

    # Group delay
    _, group_ms = group_delay(signal)
    group_ms = group_ms[ids[0]:ids[1]]*1e3

    # Find peaks
    dist_samp = int(np.ceil(dist_hz / df))
    dist_samp = 1 if dist_samp < 1 else dist_samp

    id_sum, _ = find_peaks(sum_sp, distance=dist_samp, width=dist_samp)
    id_cmif, _ = find_peaks(cmif, distance=dist_samp, width=dist_samp)
    id_group = []
    for n in range(signal.number_of_channels):
        id_, _ = find_peaks(group_ms[:, n], distance=dist_samp,
                            width=dist_samp)
        id_group.append(id_)

    # When proximity effect is activated, only group delays will be used up
    # until 200 Hz
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

        # Assessment that lower modes are peaks (not dips)
        # of the magnitude response (first RIR)
        if prune_antimodes:
            antimodes, _ = \
                find_peaks(1/np.abs(sp[ids[0]:ids[1], 0]), distance=dist_samp,
                           width=dist_samp)
            f_antimodes = f[antimodes]
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

    # "Antimode" detection – only when proximity effect is True
    if proximity_effect and prune_antimodes:
        anti = np.intersect1d(f_antimodes, f_modes)
        f_modes = np.setdiff1d(f_modes, anti)
    return f_modes


def convolve_rir_on_signal(signal: Signal, rir: Signal,
                           keep_peak_level: bool = True,
                           keep_length: bool = True) -> Signal:
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
    new_sig : `Signal`
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
        new_time_data[:, n] = convolve(
            signal.time_data[:, n], rir.time_data[:, 0],
            mode='full')[:total_length_samples]
        if keep_peak_level:
            new_time_data[:, n] = _normalize(
                new_time_data[:, n], old_peak, mode='peak')

    new_sig = signal.copy()
    new_sig.time_data = new_time_data
    new_sig.signal_id += ' (convolved with RIR)'
    return new_sig


def find_ir_start(signal: Signal, threshold_dbfs: float = -20) -> np.ndarray:
    """This function finds the start of an IR defined as the first sample
    where a certain threshold is surpassed.

    Parameters
    ----------
    signal : `Signal`
        IR signal.
    threshold_dbfs : float, optional
        Threshold that should be passed (in dBFS). Default: -20.

    Returns
    -------
    start_index : `np.ndarray`
        Index of IR start for each channel. Returns an integer when signal
        only has one channel

    References
    ----------
    - ISO 3382-1:2009-10, Acoustics - Measurement of the reverberation time of
      rooms with reference to other acoustical parameters. pp. 22.

    """
    assert threshold_dbfs <= 0, \
        'Threshold must be negative'
    start_index = np.empty(signal.number_of_channels)
    for n in range(signal.number_of_channels):
        start_index[n] = \
            _find_ir_start(signal.time_data[:, n], threshold_dbfs)
    return start_index.squeeze()


def generate_synthetic_rir(room: ShoeboxRoom, source_position,
                           receiver_position,
                           sampling_rate_hz: int,
                           total_length_seconds: float = 0.5,
                           add_noise_reverberant_tail: bool = False,
                           apply_bandpass: bool = False,
                           use_detailed_absorption: bool = False,
                           max_order: int = None) \
        -> Signal:
    """This function returns a synthetized RIR in a shoebox-room using the
    image source model. The implementation is based on Brinkmann,
    et al. See References for limitations and advantages of this method.

    Parameters
    ----------
    room : `ShoeboxRoom`
        Room object with the information about the room properties.
    source_position : array-like
        Vector with length 3 corresponding to the source's position (x, y, z)
        in meters.
    receiver_position : array-like
        Vector with length 3 corresponding to the receiver's position (x, y, z)
        in meters.
    total_length_seconds : float, optional
        Total length of the output RIR in seconds. Default: 0.5.
    sampling_rate_hz : int
        Sampling rate of the generated impulse (in Hz). Default: `None`.
    add_noise_reverberant_tail : bool, optional
        When `True`, decaying noise is added to the IR in order to model
        the late reflections of the room. Default: `True`.
    apply_bandpass : bool, optional
        When `True`, a bandpass filter is applied to signal in order to obtain
        a realistic audio representation of the RIR. Default: `True`.
    use_detailed_absorption : bool, optional
        When `True`, The detailed absorption data of the room is used to
        generate the impulse response. This allows a more realistic RIR but at
        the expense of a much higher computational cost. Default: `False`.
    max_order : int, optional
        This gives the option to limit the order of reflections computed for
        the method. This is specially useful when detailed absorption is used
        and the room has a long reverberation time, since this kind of setting
        will take a specially long time to run. Pass `None` to use an automatic
        estimation for the maximum order. Default: `None`.

    Returns
    -------
    rir : `Signal`
        Newly generated RIR.

    References
    ----------
    - Brinkmann, Fabian & Erbes, Vera & Weinzierl, Stefan. (2018). Extending
      the closed form image source model for source directivity.
    - pyroomacoustics: https://github.com/LCAV/pyroomacoustics

    Notes
    -----
    Depending on the computer and the given reverberation time, this function
    can take a relatively long runtime. If a faster or more flexible
    implementation is needed, please refer to the pyroomacoustics package
    (see references).

    """
    assert sampling_rate_hz is not None, \
        'Sampling rate can not be None'
    assert type(room) == ShoeboxRoom, \
        'Room must be of type ShoeboxRoom'
    source_position = np.asarray(source_position)
    receiver_position = np.asarray(receiver_position)
    assert room.check_if_in_room(source_position), \
        'Source is not located inside the room'
    assert room.check_if_in_room(receiver_position), \
        'Receiver is not located inside the room'

    total_length_samples = int(total_length_seconds*sampling_rate_hz)

    if not use_detailed_absorption:
        # ====== Frequency independent
        rir = _generate_rir(
            room_dim=room.dimensions_m, alpha=room.absorption_coefficient,
            s_pos=source_position, r_pos=receiver_position, rt=room.t60_s,
            mo=max_order, sr=sampling_rate_hz)
        rir = _pad_trim(rir, total_length_samples)
        # Prune possible nan values
        np.nan_to_num(rir, copy=False, nan=0)
    else:
        # ====== Frequency dependent
        assert hasattr(room, 'detailed_absorption'), \
            'Given room has no detailed absorption dictionary'
        # Create filter bank
        freqs = room.detailed_absorption['center_frequencies'][:-1]*np.sqrt(2)
        fb = linkwitz_riley_crossovers(
            crossover_frequencies_hz=freqs, order=10,
            sampling_rate_hz=sampling_rate_hz)

        # Accumulator
        rir = np.zeros(total_length_samples)

        print('\nRIR Generator\n')
        for ind in range(fb.number_of_bands):
            print(f'Band {ind+1} of {fb.number_of_bands} is being computed...')
            alphas = room.detailed_absorption['absorption_matrix'][:, ind]
            rir_band = _generate_rir(
                room_dim=room.dimensions_m, alpha=alphas,
                s_pos=source_position, r_pos=receiver_position, rt=room.t60_s,
                mo=max_order, sr=sampling_rate_hz)
            rir_band = _pad_trim(rir_band, total_length_samples)
            # Prune possible nan values
            np.nan_to_num(rir_band, copy=False, nan=0)
            rir0 = Signal(None, rir_band, sampling_rate_hz)
            rir_multi = fb.filter_signal(rir0, zero_phase=True)
            rir += rir_multi.bands[ind].time_data[:, 0]

    # Add decaying noise as reverberant tail
    if add_noise_reverberant_tail:
        if not hasattr(room, 'mixing_time_s'):
            room.get_mixing_time('physical', n_reflections=1000)
        if room.mixing_time_s is None:
            room.get_mixing_time('physical', n_reflections=1000)
        rir = _add_reverberant_tail_noise(
            rir, room.mixing_time_s, room.t60_s, sr=sampling_rate_hz)

    rir = Signal(
        None, rir, sampling_rate_hz, signal_type='rir',
        signal_id='Synthetized RIR using the image source method')

    # Bandpass signal in order to have a realistic audio signal representation
    if apply_bandpass:
        f = Filter(
            'iir', dict(order=12, filter_design_method='butter',
                        type_of_pass='bandpass',
                        freqs=[30, (sampling_rate_hz//2)*0.9]),
            sampling_rate_hz=sampling_rate_hz)
        rir = f.filter_signal(rir)

    return rir


def descriptors(rir: Signal | MultiBandSignal, mode: str = 'd50'):
    """Returns a desired room acoustics descriptor from an RIR.

    Parameters
    ----------
    rir : `Signal` or `MultiBandSignal`
        Room impulse response. If it is a multi-channel signal, the descriptor
        given back has the shape (channel). If it is a `MultiBandSignal`,
        the descriptor has shape (band, channel).
    mode : {'d50', 'c80', 'br', 'ts'} str, optional
        This defines the descriptor to be computed. Options are:
        - `'d50'`: Definition. It takes values between [0, 1] and should
          correlate (positively) with speech inteligibility.
        - `'c80'`: Clarity. It is a value in dB. The higher, the more energy
          arrives in the early part of the RIR compared to the later part.
        - `'br'`: Bass-ratio. It exposes the ratio of reverberation times
          of the lower-frequency octave bands (125, 250) to the higher ones
          (500, 1000). T20 is always used.
        - `'ts'`: Center time. It is the central time computed of the RIR.

    Returns
    -------
    output_descriptor : `np.ndarray`
        Array containing the output descriptor. If RIR is a `Signal`,
        it has shape (channel). If RIR is a `MultiBandSignal`, the array has
        shape (band, channel).

    """
    mode = mode.lower()
    assert mode in ('d50', 'c80', 'br', 'ts'), \
        'Given mode is not in the available descriptors'
    if type(rir) == Signal:
        if mode == 'd50':
            func = _d50_from_rir
        elif mode == 'c80':
            func = _c80_from_rir
        elif mode == 'ts':
            func = _ts_from_rir
        else:
            # Bass ratio
            return _bass_ratio(rir)
        desc = np.zeros(rir.number_of_channels)
        for ch in range(rir.number_of_channels):
            desc[ch] = func(rir.time_data[:, ch], rir.sampling_rate_hz)
    elif type(rir) == MultiBandSignal:
        assert mode != 'br', \
            'Bass-ratio is not a valid descriptor to be used on a ' +\
            'MultiBandSignal. Pass a RIR as Signal to compute it'
        desc = np.zeros((rir.number_of_bands, rir.number_of_channels))
        for ind, b in enumerate(rir):
            desc[ind, :] = descriptors(b, mode=mode)
    else:
        raise TypeError('RIR must be of type Signal or MultiBandSignal')
    return desc


def _bass_ratio(rir: Signal) -> np.ndarray:
    """Core computation of bass ratio.

    Parameters
    ----------
    rir : `Signal`
        RIR.

    Returns
    -------
    br : `np.ndarray`
        Bass ratio per channel.

    """
    fb = fractional_octave_bands(
        [125, 1000], filter_order=10, sampling_rate_hz=rir.sampling_rate_hz)
    rir_multi = fb.filter_signal(rir, zero_phase=True)
    rt = reverb_time(rir_multi)
    br = np.zeros(rir.number_of_channels)
    for ch in range(rir.number_of_channels):
        br[ch] = (rt[0, ch]+rt[1, ch]) / (rt[2, ch]+rt[3, ch])
    return br
