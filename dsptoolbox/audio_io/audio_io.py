"""
Here are wrappers for streams with sounddevice. This is useful for
measurements and testing audio streams
"""
import sounddevice as sd
from dsptoolbox import Signal, normalize, pad_trim
from dsptoolbox._general_helpers import _normalize
from ._audio_io import standard_callback


def print_device_info(device_number: int = None):
    """Prints available audio devices or information about a certain device
    when the device number is given.

    Parameters
    ----------
    device_number : int, optional
        Prints information about the specific device and returns it as
        a dictionary. Use `None` to only print information about all devices
        without returning anything. Default: `None`.

    Returns
    -------
    d : dict
        Only when `device_number is not None`.

    """
    if device_number is None:
        print(sd.query_devices())
    else:
        d = sd.query_devices(device_number)
        print(d)
        return d


def set_device(device_number: int = None):
    """Takes in a device number to set it as the default. If `None` is passed,
    the available devices are first shown and then the user is asked for
    input to set the device.

    Parameters
    ----------
    device_number : int, optional
        Sets the device as default. Use `None` to ignore. Default: `None`.

    """
    if device_number is None:
        txt = 'List of available devices'
        print(txt+'\n'+'-'*len(txt))
        print(sd.query_devices())
        print('-'*len(txt))
        device_number = int(input(
            'Which device should be set as default? Between ' +
            f'0 and {len(sd.query_devices())-1}: '))
    d = sd.query_devices(device_number)['name']
    print(f"""{d} will be used!""")
    sd.default.device = d


def play_and_record(signal: Signal, duration_seconds: float = None,
                    normalized_dbfs: float = -6, device: str = None,
                    play_channels=None, rec_channels=[1]):
    """Play and record using some available device. Note that the channel
    numbers start here with 1.

    Parameters
    ----------
    signal : Signal
        Signal object to be played. The number of channels has to match the
        total length and order of play_channels. The sampling rate of signal
        will define the sampling rate of the recorded signals.
    duration_seconds : float, optional
        If `None`, the whole signal is played, otherwise it is trimmed to the
        given length. Default: `None`.
    normalized_dbfs: float, optional
        Normalizes the signal (dBFS peak level) before playing it.
        Set to `None` to ignore normalization. Default: -6.
    device : str, optional
        I/O device to be used. If `None`, the default device is used.
        Default: `None`.
    play_channels : int or array-like, optional
        Output channels that will play the signal. The number of channels
        should match the number of channels in signal. When `None`, the
        channels are automatically set. Default: `None`.
    rec_channels : int or array-like, optional
        Channel numbers that will be recorded. Default: [1].

    Returns
    -------
    rec_sig : Signal
        Recorded signal.

    """
    # Asserts
    if play_channels is None:
        play_channels = list(range(1, signal.number_of_channels+1))
    if type(play_channels) == int:
        play_channels = [play_channels]
    if type(rec_channels) == int:
        rec_channels = [rec_channels]
    play_channels = sorted(play_channels)
    rec_channels = sorted(rec_channels)
    assert signal.number_of_channels == len(play_channels), \
        'The number of channels in signal does not match the number of ' +\
        'channels in play_channels'
    assert not any([p < 1 for p in play_channels]), \
        'Play channel has to be 1 or more'
    assert not any([r < 1 for r in rec_channels]), \
        'Recording channel has to be 1 or more'
    #
    if duration_seconds is not None:
        assert duration_seconds > 0, 'Duration must be positive'
        duration_samples = duration_seconds * signal.sampling_rate_hz
    else:
        duration_seconds = signal.time_data.shape[0] / signal.sampling_rate_hz
        duration_samples = signal.time_data.shape[0]

    play_data = signal.time_data.copy()[:duration_samples, :]

    if normalized_dbfs is not None:
        assert normalized_dbfs <= 0, 'Only values beneath 0 dBFS are allowed'
        play_data = _normalize(play_data, dbfs=normalized_dbfs, mode='peak')

    if device is not None:
        sd.default.device = device

    print('\nReproduction and recording have started ' +
          f'({duration_seconds:.1f} s)...')
    rec_time_data = \
        sd.playrec(
            data=play_data,
            samplerate=signal.sampling_rate_hz,
            input_mapping=rec_channels,
            output_mapping=play_channels)
    sd.wait()
    print('Reproduction and recording have ended\n')

    rec_sig = Signal(None, rec_time_data, signal.sampling_rate_hz)
    return rec_sig


def record(duration_seconds: float = 5, sampling_rate_hz: int = 48000,
           device: str = None, rec_channels=[1]):
    """Record using some available device. Note that the channel numbers
    start here with 1.

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
        Number that will be recorded. Default: [1].

    Returns
    -------
    rec_sig : Signal
        Recorded signal.

    """
    # Asserts
    if type(rec_channels) == int:
        rec_channels = [rec_channels]
    rec_channels = sorted(rec_channels)
    assert not any([r < 1 for r in rec_channels]), \
        'Recording channel has to be 1 or more'
    #
    if device is not None:
        sd.default.device = device

    print(f'\nRecording started ({duration_seconds:.1f} s)...')
    rec_time_data = \
        sd.rec(
            frames=int(duration_seconds * sampling_rate_hz),
            samplerate=sampling_rate_hz,
            mapping=rec_channels)
    sd.wait()
    print('Recording has ended\n')

    rec_sig = Signal(None, rec_time_data, sampling_rate_hz)
    return rec_sig


def play(signal: Signal, duration_seconds: float = None,
         normalized_dbfs: float = -6, device: str = None, play_channels=None):
    """Play some available device. Note that the channel numbers
    start here with 1.

    Parameters
    ----------
    signal : Signal
        Signal to be reproduced. Its channel number must match the the length
        of the play_channels vector.
    duration_seconds : float, optional
        If `None`, the whole signal is played, otherwise it is trimmed to the
        given length. Default: `None`.
    normalized_dbfs: float, optional
        Normalizes the signal (dBFS peak level) before playing it.
        Set to `None` to ignore normalization. Default: -6.
    device : str, optional
        I/O device to be used. If `None`, the default device is used.
        Default: `None`.
    play_channels : int or array-like, optional
        Output channels that will play the signal. The number of channels
        should match the number of channels in signal. When `None`, the
        channels are automatically set. Default: `None`.

    """
    # Asserts and preprocessing
    if play_channels is None:
        play_channels = list(range(1, signal.number_of_channels+1))
    if type(play_channels) == int:
        play_channels = [play_channels]
    play_channels = sorted(play_channels)
    assert not any([r < 1 for r in play_channels]), \
        'Play channel has to be 1 or more'
    if duration_seconds is not None:
        assert duration_seconds > 0, 'Duration must be positive'
        duration_samples = duration_seconds * signal.sampling_rate_hz
    else:
        duration_seconds = signal.time_data.shape[0] / signal.sampling_rate_hz
        duration_samples = signal.time_data.shape[0]
    play_data = signal.time_data.copy()[:duration_samples, :]
    if normalized_dbfs is not None:
        assert normalized_dbfs <= 0, 'Only values beneath 0 dBFS are allowed'
        play_data = _normalize(play_data, dbfs=normalized_dbfs, mode='peak')
    #
    if device is not None:
        sd.default.device = device

    print(f'\nReproduction started ({duration_seconds:.1f} s)...')
    sd.play(
        data=play_data,
        samplerate=signal.sampling_rate_hz,
        mapping=play_channels)
    sd.wait()
    print('Reproduction has ended\n')


def play_stream(signal: Signal, duration_seconds: float = None,
                normalized_dbfs: float = -6, blocksize: int = 2048,
                audio_callback=standard_callback,
                device: str = None):
    """Plays a signal using a stream and a callback function.
    See `sounddevice.OutputStream` for extensive information about
    functionalities.

    NOTE: The `Signal`'s channel shape must match the device's
    output channels. The start of the signal can be set using `Signal`'s method
    `set_streaming_position`.

    Parameters
    ----------
    signal : `Signal`
        Signal to be reproduced.
    duration_seconds : float, optional
        Duration of the signal to be reproduced in seconds. If `None`, the
        whole signal is played. Default: `None`.
    normalized_dbfs : float, optional
        Normalizes the signal to a certain dBFS value. Pass `None` to avoid
        normalization. Default: -6.
    blocksize : int, optional
        Block size used for the stream. Powers of two are recommended.
        Default: 2048.
    audio_callback : callable, optional
        This is the core of the stream! Callback modifies the signal as it is
        passed to the audio device. The `standard_callback` just passes it
        through. `audio_callback` is a function that take in a signal object
        and passes a valid sounddevice callback. Their signatures are::

            audio_callback(signal: Signal) -> callback

            callback(outdata: numpy.ndarray, frames: int,
                     time: CData, status: CallbackFlags) -> None

        See `sounddevice`'s examples of callbacks for more general
        information. Default: `standard_callback`.
    device : str, optional
        Device to be used in the audio stream. Pass `None` to use the default
        device. Default: `None`.

    """
    if duration_seconds is not None:
        duration_ms = int(duration_seconds*1000)
        duration_samples = int(duration_seconds*signal.sampling_rate_hz)
        signal = pad_trim(signal, duration_samples)
    else:
        duration_seconds = signal.time_vector_s[-1]
        duration_ms = int(duration_seconds*1000)

    if normalized_dbfs is not None:
        signal = normalize(signal, peak_dbfs=normalized_dbfs)

    if not hasattr(signal, '_streaming_position'):
        signal.set_streaming_position()

    if device is not None:
        sd.default.device = device

    with sd.OutputStream(signal.sampling_rate_hz, blocksize=blocksize,
                         callback=audio_callback(signal),
                         channels=signal.number_of_channels):
        sd.sleep(duration_ms + 5)


def CallbackStop():
    """Wrapper around sounddevice's CallbackStop. Used for stopping audio
    streamings.

    """
    sd.CallbackStop()
