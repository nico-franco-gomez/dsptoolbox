'''
Here are wrappers for streams with sounddevice. This is useful for
measurements.
'''
import sounddevice as sd
from .signal_class import Signal


def print_device_info(device_number: int = None):
    '''
    Prints available audio devices or information about a certain device
    when the device number is given.

    Parameters
    ----------
    device_number : int, optional
        Prints information about the specific device and returns it as
        a dictionary. Use `None` to ignore. Default: `None`.

    Returns
    -------
    d : dict
        Only when `device_number is not None`.
    '''
    if device_number is None:
        print(sd.query_devices())
    else:
        d = sd.query_devices(device_number)
        print(d)
        return d


def play_and_record(signal: Signal, device: str = None, play_channels=[1],
                    rec_channels=[1]):
    '''
    Play and record using some available device.

    Parameters
    ----------
    signal : Signal
        Signal object to be played. The number of channels has to match the
        total length of play_channels. The sampling rate of signal will
        define the sampling rate of the recorded signals.
    device : str, optional
        I/O device to be used. If `None`, the default device is used.
        Default: `None`.
    play_channels : int or array-like, optional
        Output channels that will play the signal. The number of channels
        should match the number of channels in signal. Default: [1].
    rec_channels : int or array-like, optional
        Number that will be recorded. Default: [1].

    Returns
    -------
    rec_sig : Signal
        Recorded signal.
    '''
    if type(play_channels) == int:
        play_channels = [play_channels]
    if type(rec_channels) == int:
        rec_channels = [rec_channels]
    assert signal.number_of_channels == len(play_channels), \
        'The number of channels in signal does not match the number of ' +\
        'channels in play_channels'
    assert not any([p < 1 for p in play_channels]), \
        'Play channel has to be 1 or more'
    assert not any([r < 1 for r in rec_channels]), \
        'Recording channel has to be 1 or more'
    if device is not None:
        sd.default.device = device
    rec_time_data = \
        sd.playrec(
            signal.time_data,
            samplerate=signal.sampling_rate_hz,
            input_mapping=rec_channels,
            output_mapping=play_channels)
    sd.wait()
    rec_sig = Signal(None, rec_time_data, signal.sampling_rate_hz)
    return rec_sig


def record(duration_seconds: float = 5, sampling_rate_hz: int = 48000,
           device: str = None, rec_channels=[1]):
    '''
    Play and record using some available device.

    Parameters
    ----------
    duration_seconds : float, optional
        Duration of recording in seconds. Default: 5.
    sampling_rate_hz : int, optional
        Sampling rate used for recording. Default: 48000.
    device : str, optional
        I/O device to be used. If `None`, the default device is used.
        Default: `None`.
    rec_channels : int or array-like, optional
        Number that will be recorded. Default: [0].

    Returns
    -------
    rec_sig : Signal
        Recorded signal.
    '''
    if type(rec_channels) == int:
        rec_channels = [rec_channels]
    assert not any([r < 1 for r in rec_channels]), \
        'Recording channel has to be 1 or more'
    sd.default.device = device
    rec_time_data = \
        sd.rec(
            frames=int(duration_seconds * sampling_rate_hz),
            samplerate=sampling_rate_hz,
            mapping=rec_channels)
    sd.wait()
    rec_sig = Signal(None, rec_time_data, sampling_rate_hz)
    return rec_sig
