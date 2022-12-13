'''
This contains signal generators that might be useful for measurements.
See measure.py for routines where the signals that are created here can be
used
'''
import numpy as np
from .signal_class import Signal
from .backend._general_helpers import _normalize, _fade


def noise(type_of_noise: str = 'white', length_seconds: float = 1,
          sampling_rate_hz: int = 48000, peak_level_dbfs: float = -10,
          number_of_channels: int = 1, faded: bool = True):
    '''
    Creates a noise signal.

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
    faded : bool, optional
        When `True`, start and end of the signal are faded (5% of length each).
        Default: `True`.

    Returns
    -------
    noise_sig : Signal
        Noise Signal object.
    '''
    valid_noises = ('white', 'pink', 'red', 'blue', 'violet', 'grey')
    valid_noises = (n.casefold() for n in valid_noises)
    assert type_of_noise.casefold() in valid_noises, \
        f'{type_of_noise} is not valid'
    type_of_noise = type_of_noise.casefold()
    assert length_seconds > 0, 'Length has to be positive'
    assert peak_level_dbfs <= 0, 'Peak level cannot surpass 0 dBFS'
    assert number_of_channels >= 1, 'At least one channel should be generated'

    fade_length = 0.05 * length_seconds

    l_samples = int(length_seconds * sampling_rate_hz)
    f = np.fft.rfftfreq(l_samples, 1/sampling_rate_hz)

    time_data = np.zeros((l_samples, number_of_channels))

    for n in range(number_of_channels):
        mag = np.ones(len(f)) + np.random.normal(0, 0.2, len(f))
        mag[0] = 0
        ph = np.random.uniform(-np.pi, np.pi, len(f))
        if type_of_noise == 'pink'.casefold():
            mag[1:] /= f[1:]
        elif type_of_noise == 'red'.casefold():
            mag[1:] /= (f[1:]**2)
        elif type_of_noise == 'blue'.casefold():
            mag[1:] *= f[1:]
        elif type_of_noise == 'violet'.casefold():
            mag[1:] *= (f[1:]**2)
        vec = _normalize(np.fft.irfft(mag*np.exp(1j*ph)),
                         dbfs=peak_level_dbfs, mode='peak')
        if faded:
            vec = _fade(vec, fade_length, sampling_rate_hz, True)
            vec = _fade(vec, fade_length, sampling_rate_hz, False)
        time_data[:, n] = vec

    id = type_of_noise.lower()+' noise'
    noise_sig = Signal(None, time_data, sampling_rate_hz, signal_id=id)
    noise_sig.set_spectrum_parameters(method='standard')
    return noise_sig


def chirp(type_of_noise: str = 'white', length_seconds: float = 1,
          sampling_rate_hz: int = 48000, peak_level_dbfs: float = -10,
          number_of_channels: int = 1):
    '''
    Creates a sweep signal.

    Parameters
    ----------
    type_of_noise : str, optional
        Choose from `'linear'`, `'log'`, `'loglog'`.
        Default: `'log'`.
    length_seconds : float, optional
        Length of the generated signal in seconds. Default: 1.
    sampling_rate_hz : int, optional
        Sampling rate in Hz. Default: 48000.
    peak_level_dbfs : float, optional
        Peak level of the signal in dBFS. Default: -10.
    number_of_channels : int, optional
        Number of channels (with different noise signals) to be created.
        Default: 1.

    Returns
    -------
    noise_sig : Signal
        Noise Signal object.
    '''
    print()
