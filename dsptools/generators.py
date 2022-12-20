"""
This contains signal generators that might be useful for measurements.
See measure.py for routines where the signals that are created here can be
used
"""
from numpy import (fft, zeros, ones, random, pi, linspace,
                   sin, log, append, exp)
from .classes.signal_class import Signal
from .backend._general_helpers import _normalize, _fade
from .other import pad_trim
from .backend._filter import _impulse

# __all__ = ['noise', 'chirp', 'dirac', ]


def noise(type_of_noise: str = 'white', length_seconds: float = 1,
          sampling_rate_hz: int = 48000, peak_level_dbfs: float = -10,
          number_of_channels: int = 1, fade: str = 'log',
          padding_end_seconds: float = None):
    """Creates a noise signal.

    Parameters
    ----------
    type_of_noise : str, optional
        Choose from `'white'`, `'pink'`, `'red'`, `'blue'`, `'violet'`.
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
        Type of fade done on the generated signal. Options are `'exp'`,
        `'lin'`, `'log'`. Pass `None` for no fading. Default: `'log'`.
    padding_end_seconds : float, optional
        Padding at the end of signal. Use `None` to avoid any padding.
        Default: `None`.

    Returns
    -------
    noise_sig : Signal
        Noise Signal object.

    References
    ----------
    https://en.wikipedia.org/wiki/Colors_of_noise
    """
    valid_noises = ('white', 'pink', 'red', 'blue', 'violet', 'grey')
    valid_noises = (n.casefold() for n in valid_noises)
    assert type_of_noise.casefold() in valid_noises, \
        f'{type_of_noise} is not valid'
    type_of_noise = type_of_noise.casefold()
    assert length_seconds > 0, 'Length has to be positive'
    assert peak_level_dbfs <= 0, 'Peak level cannot surpass 0 dBFS'
    assert number_of_channels >= 1, 'At least one channel should be generated'
    if padding_end_seconds is not None:
        assert padding_end_seconds > 0, 'Padding has to be a positive time'

    l_samples = int(length_seconds * sampling_rate_hz)
    f = fft.rfftfreq(l_samples, 1/sampling_rate_hz)

    time_data = zeros((l_samples, number_of_channels))

    for n in range(number_of_channels):
        mag = ones(len(f)) + random.normal(0, 0.025, len(f))
        mag[0] = 0
        ph = random.uniform(-pi, pi, len(f))
        if type_of_noise == 'pink'.casefold():
            mag[1:] /= f[1:]
        elif type_of_noise == 'red'.casefold():
            mag[1:] /= (f[1:]**2)
        elif type_of_noise == 'blue'.casefold():
            mag[1:] *= f[1:]
        elif type_of_noise == 'violet'.casefold():
            mag[1:] *= (f[1:]**2)
        vec = _normalize(fft.irfft(mag*exp(1j*ph)),
                         dbfs=peak_level_dbfs, mode='peak')
        if fade is not None:
            fade_length = 0.05 * length_seconds
            vec = _fade(s=vec, length_seconds=fade_length, mode=fade,
                        sampling_rate_hz=sampling_rate_hz, at_start=True)
            vec = _fade(s=vec, length_seconds=fade_length, mode=fade,
                        sampling_rate_hz=sampling_rate_hz, at_start=False)
        time_data[:, n] = vec

    id = type_of_noise.lower()+' noise'
    noise_sig = Signal(None, time_data, sampling_rate_hz, signal_type='noise',
                       signal_id=id)
    if padding_end_seconds is not None:
        p_samples = int(padding_end_seconds * sampling_rate_hz)
        noise_sig = pad_trim(noise_sig, l_samples+p_samples)
    return noise_sig


def chirp(type_of_chirp: str = 'log', range_hz=None, length_seconds: float = 1,
          sampling_rate_hz: int = 48000, peak_level_dbfs: float = -10,
          number_of_channels: int = 1, fade: str = 'log',
          padding_end_seconds: float = None):
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
        Number of channels (with different noise signals) to be created.
        Default: 1.
    fade : str, optional
        Type of fade done on the generated signal. Options are `'exp'`,
        `'lin'`, `'log'`. Pass `None` for no fading. Default: `'log'`.
    padding_end_seconds : float, optional
        Padding at the end of signal. Use `None` to avoid any padding.
        Default: `None`.

    Returns
    -------
    chirp_sig : Signal
        Chirp Signal object.

    Reference
    ---------
    https://de.wikipedia.org/wiki/Chirp
    """
    type_of_chirp = type_of_chirp.lower()
    assert type_of_chirp in ('lin', 'log'), \
        f'{type_of_chirp} is not a valid type. Select lin or log'
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
    l_samples = int(sampling_rate_hz * length_seconds)
    t = linspace(0, length_seconds, l_samples)

    if type_of_chirp == 'lin':
        k = (range_hz[1]-range_hz[0])/length_seconds
        freqs = (range_hz[0] + k/2*t)*2*pi
        chirp = sin(freqs*t)
    elif type_of_chirp == 'log':
        k = exp((log(range_hz[1])-log(range_hz[0]))/length_seconds)
        chirp = \
            sin(2*pi*range_hz[0]/log(k)*(k**t-1))
    chirp = _normalize(chirp, peak_level_dbfs, mode='peak')

    if fade is not None:
        fade_length = 0.05 * length_seconds
        chirp = _fade(s=chirp, length_seconds=fade_length, mode=fade,
                      sampling_rate_hz=sampling_rate_hz, at_start=True)
        chirp = _fade(s=chirp, length_seconds=fade_length, mode=fade,
                      sampling_rate_hz=sampling_rate_hz, at_start=False)

    chirp_n = chirp[..., None]
    if number_of_channels != 1:
        for n in range(number_of_channels):
            chirp_n = append(chirp_n, chirp[..., None], axis=1)
    # Signal
    chirp_sig = Signal(None, chirp_n, sampling_rate_hz,
                       signal_type='chirp', signal_id=type_of_chirp)
    if padding_end_seconds is not None:
        p_samples = int(padding_end_seconds * sampling_rate_hz)
        chirp_sig = pad_trim(chirp_sig, l_samples+p_samples)
    return chirp_sig


def dirac(length_samples: int = 512, number_of_channels: int = 1,
          sampling_rate_hz: int = 48000):
    """Generates a dirac impulse Signal with the specified length and
    sampling rate.

    Parameters
    ----------
    length_samples : int, optional
        Length in samples. Default: 512.
    number_of_channels : int, optional
        Number of channels to be generated with the same impulse. Default: 1.
    sampling_rate_hz : int, optional
        Sampling rate to be used. Default: 480000.

    Returns
    -------
    imp : Signal
        Signal with dirac impulse.
    """
    assert length_samples > 0, 'Only positive lengths are valid'
    assert number_of_channels > 0, 'At least one channel has to be created'
    assert sampling_rate_hz > 0, 'Sampling rate can only be positive'
    td = zeros((length_samples, number_of_channels))
    for n in range(number_of_channels):
        td[:, n] = _impulse(length_samples=length_samples)
    imp = Signal(None, td, sampling_rate_hz, signal_type='dirac')
    return imp
