"""
This contains signal generators that might be useful for measurements.
See measure.py for routines where the signals that are created here can be
used
"""
import numpy as np
from dsptoolbox.classes.signal_class import Signal
from dsptoolbox._general_helpers import (
    _normalize, _fade, _pad_trim, _frequency_weightning)
from dsptoolbox.classes._filter import _impulse


def noise(type_of_noise: str = 'white', length_seconds: float = 1,
          sampling_rate_hz: int = 48000, peak_level_dbfs: float = -10,
          number_of_channels: int = 1, fade: str = 'log',
          padding_end_seconds: float = None) -> Signal:
    """Creates a noise signal.

    Parameters
    ----------
    type_of_noise : str, optional
        Choose from `'white'`, `'pink'`, `'red'`, `'blue'`, `'violet'` or
        `'grey'`.
        Default: `'white'`.
    length_seconds : float, optional
        Length of the generated signal in seconds. Default: 1.
    sampling_rate_hz : int, optional
        Sampling rate in Hz. Default: 48000.
    peak_level_dbfs : float, optional
        Peak level of the signal in dBFS. Default: -10.
    number_of_channels : int, optional
        Number of channels (with different noise signals) to be created.
        Default: 1.
    fade : str, optional
        Type of fade done on the generated signal. By default, 10% of signal
        length (without the padding in the end) is faded at the beginning and
        end. Options are `'exp'`, `'lin'`, `'log'`.
        Pass `None` for no fading. Default: `'log'`.
    padding_end_seconds : float, optional
        Padding at the end of signal. Use `None` to avoid any padding.
        Default: `None`.

    Returns
    -------
    noise_sig : `Signal`
        Noise Signal object.

    References
    ----------
    https://en.wikipedia.org/wiki/Colors_of_noise

    """
    valid_noises = ('white', 'pink', 'red', 'blue', 'violet', 'grey')
    type_of_noise = type_of_noise.lower()
    assert type_of_noise in valid_noises, \
        f'{type_of_noise} is not valid'
    assert length_seconds > 0, 'Length has to be positive'
    assert peak_level_dbfs <= 0, 'Peak level cannot surpass 0 dBFS'
    assert number_of_channels >= 1, 'At least one channel should be generated'
    if padding_end_seconds is not None:
        assert padding_end_seconds > 0, 'Padding has to be a positive time'

    l_samples = int(length_seconds * sampling_rate_hz)
    f = np.fft.rfftfreq(l_samples, 1/sampling_rate_hz)

    if padding_end_seconds is not None:
        p_samples = int(padding_end_seconds * sampling_rate_hz)
    else:
        p_samples = 0
    time_data = np.zeros((l_samples+p_samples, number_of_channels))

    for n in range(number_of_channels):
        mag = np.ones(len(f)) + np.random.normal(0, 0.025, len(f))
        mag[0] = 0
        ph = np.random.uniform(-np.pi, np.pi, len(f))
        if type_of_noise == 'pink':
            mag[1:] /= f[1:]
        elif type_of_noise == 'red':
            mag[1:] /= (f[1:]**2)
        elif type_of_noise == 'blue':
            mag[1:] *= f[1:]
        elif type_of_noise == 'violet':
            mag[1:] *= (f[1:]**2)
        elif type_of_noise == 'grey':
            # Set to 15 Hz to cover whole audible spectrum but without
            # numerical instabilities because of large values in lower
            # frequencies
            id_low = np.argmin(np.abs(f - 15))
            w = _frequency_weightning(f, 'a', db_output=False)
            mag[id_low:] /= w[id_low:]
        vec = _normalize(np.fft.irfft(mag*np.exp(1j*ph)),
                         dbfs=peak_level_dbfs, mode='peak')
        if fade is not None:
            fade_length = 0.05 * length_seconds
            vec = _fade(s=vec, length_seconds=fade_length, mode=fade,
                        sampling_rate_hz=sampling_rate_hz, at_start=True)
            vec = _fade(s=vec, length_seconds=fade_length, mode=fade,
                        sampling_rate_hz=sampling_rate_hz, at_start=False)
        time_data[:l_samples, n] = vec

    id = type_of_noise.lower()+' noise'
    noise_sig = Signal(None, time_data, sampling_rate_hz, signal_type='noise',
                       signal_id=id)
    return noise_sig


def chirp(type_of_chirp: str = 'log', range_hz=None, length_seconds: float = 1,
          sampling_rate_hz: int = 48000, peak_level_dbfs: float = -10,
          number_of_channels: int = 1, fade: str = 'log',
          padding_end_seconds: float = None) -> Signal:
    """Creates a sweep signal.

    Parameters
    ----------
    type_of_chirp : str, optional
        Choose from `'lin'`, `'log'`.
        Default: `'log'`.
    range_hz : array-like with length 2
        Define range of chirp in Hz. When `None`, all frequencies between
        1 and nyquist are taken. Default: `None`.
    length_seconds : float, optional
        Length of the generated signal in seconds. Default: 1.
    sampling_rate_hz : int, optional
        Sampling rate in Hz. Default: 48000.
    peak_level_dbfs : float, optional
        Peak level of the signal in dBFS. Default: -10.
    number_of_channels : int, optional
        Number of channels (with the same chirp) to be created. Default: 1.
    fade : str, optional
        Type of fade done on the generated signal. By default, 10% of signal
        length (without the padding in the end) is faded at the beginning and
        end. Options are `'exp'`, `'lin'`, `'log'`.
        Pass `None` for no fading. Default: `'log'`.
    padding_end_seconds : float, optional
        Padding at the end of signal. Use `None` to avoid any padding.
        Default: `None`.

    Returns
    -------
    chirp_sig : `Signal`
        Chirp Signal object.

    References
    ----------
    https://de.wikipedia.org/wiki/Chirp

    """
    type_of_chirp = type_of_chirp.lower()
    assert type_of_chirp in ('lin', 'log'), \
        f'{type_of_chirp} is not a valid type. Select lin or np.log'
    if range_hz is not None:
        assert len(range_hz) == 2, \
            'range_hz has to contain exactly two frequencies'
        range_hz = sorted(range_hz)
        assert range_hz[0] > 0, \
            'Range has to start with positive frequencies excluding 0'
        assert range_hz[1] <= sampling_rate_hz//2, \
            'Upper limit for frequency range cannot be bigger than the ' +\
            'nyquist frequency'
    else:
        range_hz = [1, sampling_rate_hz//2]
    if padding_end_seconds is not None:
        p_samples = int(padding_end_seconds * sampling_rate_hz)
    else:
        p_samples = 0
    l_samples = int(sampling_rate_hz * length_seconds)
    t = np.linspace(0, length_seconds, l_samples)

    if type_of_chirp == 'lin':
        k = (range_hz[1]-range_hz[0])/length_seconds
        freqs = (range_hz[0] + k/2*t)*2*np.pi
        chirp_td = np.sin(freqs*t)
    elif type_of_chirp == 'log':
        k = np.exp((np.log(range_hz[1])-np.log(range_hz[0]))/length_seconds)
        chirp_td = \
            np.sin(2*np.pi*range_hz[0]/np.log(k)*(k**t-1))
    chirp_td = _normalize(chirp_td, peak_level_dbfs, mode='peak')

    if fade is not None:
        fade_length = 0.05 * length_seconds
        chirp_td = _fade(s=chirp_td, length_seconds=fade_length, mode=fade,
                         sampling_rate_hz=sampling_rate_hz, at_start=True)
        chirp_td = _fade(s=chirp_td, length_seconds=fade_length, mode=fade,
                         sampling_rate_hz=sampling_rate_hz, at_start=False)

    chirp_td = _pad_trim(chirp_td, l_samples+p_samples)

    chirp_n = chirp_td[..., None]
    if number_of_channels != 1:
        for n in range(1, number_of_channels):
            chirp_n = np.append(chirp_n, chirp_td[..., None], axis=1)
    # Signal
    chirp_sig = Signal(None, chirp_n, sampling_rate_hz,
                       signal_type='chirp', signal_id=type_of_chirp)
    return chirp_sig


def dirac(length_samples: int = 512, delay_samples: int = 0,
          number_of_channels: int = 1, sampling_rate_hz: int = 48000) \
        -> Signal:
    """Generates a dirac impulse Signal with the specified length and
    sampling rate.

    Parameters
    ----------
    length_samples : int, optional
        Length in samples. Default: 512.
    delay_samples : int, optional
        Delay of the impulse in samples. Default: 0.
    number_of_channels : int, optional
        Number of channels to be generated with the same impulse. Default: 1.
    sampling_rate_hz : int, optional
        Sampling rate to be used. Default: 480000.

    Returns
    -------
    imp : `Signal`
        Signal with dirac impulse.

    """
    assert type(length_samples) == int and length_samples > 0, \
        'Only positive lengths are valid'
    assert type(delay_samples) == int and delay_samples >= 0, \
        'Only positive delay is supported'
    assert delay_samples < length_samples, \
        'Delay is bigger than the samples of the signal'
    assert number_of_channels > 0, 'At least one channel has to be created'
    assert sampling_rate_hz > 0, 'Sampling rate can only be positive'
    td = np.zeros((length_samples, number_of_channels))
    for n in range(number_of_channels):
        td[:, n] = _impulse(
            length_samples=length_samples, delay_samples=delay_samples)
    imp = Signal(None, td, sampling_rate_hz, signal_type='dirac')
    return imp


def sinus(frequency_hz: float = 1000, length_seconds: float = 1,
          sampling_rate_hz: int = 48000, peak_level_dbfs: float = -10,
          number_of_channels: int = 1, uncorrelated: bool = False,
          fade: str = 'log', padding_end_seconds: float = None) -> Signal:
    """Creates a multi-channel sinus tone.

    Parameters
    ----------
    frequency_hz : float, optional
        Frequency (in Hz) for the sinus tone. Default: 1000.
    length_seconds : float, optional
        Length of the generated signal in seconds. Default: 1.
    sampling_rate_hz : int, optional
        Sampling rate in Hz. Default: 48000.
    peak_level_dbfs : float, optional
        Peak level of the signal in dBFS. Default: -10.
    number_of_channels : int, optional
        Number of channels (with the same chirp) to be created. Default: 1.
    uncorrelated : bool, optional
        When `True`, each tone gets a random phase shift so that the signals
        are not perfectly correlated. Default: `False`.
    fade : str, optional
        Type of fade done on the generated signal. By default, 10% of signal
        length (without the padding in the end) is faded at the beginning and
        end. Options are `'exp'`, `'lin'`, `'log'`.
        Pass `None` for no fading. Default: `'log'`.
    padding_end_seconds : float, optional
        Padding at the end of signal. Use `None` to avoid any padding.
        Default: `None`.

    Returns
    -------
    sinus_sig : `Signal`
        Sinus tone signal.

    """
    assert frequency_hz < sampling_rate_hz//2, \
        'Frequency must be beneath nyquist frequency'
    assert frequency_hz > 0, \
        'Frequency must be bigger than 0'

    if padding_end_seconds is not None:
        p_samples = int(padding_end_seconds * sampling_rate_hz)
    else:
        p_samples = 0
    l_samples = int(sampling_rate_hz * length_seconds)
    n_vec = np.arange(l_samples)

    td = np.empty((l_samples, number_of_channels))
    for n in range(number_of_channels):
        if uncorrelated:
            phase_shift = np.random.uniform(-np.pi, np.pi)
        else:
            phase_shift = 0
        td[:, n] = np.sin(
            frequency_hz / sampling_rate_hz * 2 * np.pi * n_vec + phase_shift)

    td = _normalize(td, peak_level_dbfs, mode='peak')

    if fade is not None:
        fade_length = 0.05 * length_seconds
        td = _fade(
            s=td, length_seconds=fade_length, mode=fade,
            sampling_rate_hz=sampling_rate_hz, at_start=True)
        td = _fade(
            s=td, length_seconds=fade_length, mode=fade,
            sampling_rate_hz=sampling_rate_hz, at_start=False)

    td = _pad_trim(td, l_samples+p_samples)

    # Signal
    sinus_sig = Signal(None, td, sampling_rate_hz,
                       signal_type='general')
    return sinus_sig
