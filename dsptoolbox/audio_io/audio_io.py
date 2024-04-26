"""
Here are wrappers for streams with sounddevice. This is useful for
measurements and testing audio streams
"""

import sounddevice as sd
from .. import Signal
from .._general_helpers import _normalize
from ._audio_io import standard_callback

default_config = sd.default


def print_device_info(device_number: int | None = None):
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
    d : dict or `sounddevice.DeviceList`
        dict of a device when device number is passed or DeviceList when `None`
        is passed.

    """
    if device_number is None:
        d = sd.query_devices()
        print(d)
        return d
    else:
        d = sd.query_devices(device_number)
        print(d)
        return d


def set_latency(input_low: bool, output_low: bool):
    """
    Set the desired latency (Default is high). This can vary for each device
    and host. Sounddevice only allows for setting a low or a high latency.
    High latency is more robust, but it might be too large for some
    applications.

    Parameters
    ----------
    input_low : bool
        When `True`, low latency will be requested to the host for input
        streams.
    output_low : bool
        When `True`, low latency will be requested to the host for output
        streams.

    """
    sd.default.latency = (
        "low" if input_low else "high",
        "low" if output_low else "high",
    )


def set_blocksize(blocksize: int):
    """
    Set a default blocksize for any stream. This can lead to a stable latency
    for most interfaces. Not setting it will lead to a default value.

    Parameters
    ----------
    blocksize : int
        Desired block size.

    """
    sd.default.blocksize = blocksize


def set_device(
    device: list[int] | list[str] | str | int | None = None,
    sampling_rate_hz: int | None = None,
):
    """Takes in a device number to set it as the default for the input and the
    output. If `None` is passed, the available devices are first shown and
    then the user is asked for input to set the device two values separated by
    a comma "input_int, output_int".

    Parameters
    ----------
    device : list[int | str] with length 2, str, int, optional
        Sets the input and output devices from two integers, e.g. [1, 2].
        Alternatively, two strings contained in the interface's name (or the
        name itself) can be passed. The first interface to match will be taken.
        If passing only one string or integer, the interface will be taken for
        both input and output. Use `None` to be prompted with the options and
        pass only two values separated by a comma, e.g., `1, 2`.
        Default: `None`.
    sampling_rate_hz : int, optional
        Pass a default sampling rate to the devices. Pass `None` to ignore.
        Default: `None`.

    Returns
    -------
    device_list : `sounddevice.DeviceList`
        Device List with dictionaries containing information about each
        available device if `device=None`.

    """
    if device is None:
        txt = "List of available devices"
        print(txt + "\n" + "-" * len(txt))
        print(sd.query_devices())
        print("-" * len(txt))
        device = input(
            "Which device should be set as default? Between "
            + f"0 and {len(sd.query_devices()) - 1}: "
        )
        device = [int(d) for d in device.split(",")]
        if len(device) == 1:
            device = device[0]
    device_list = sd.query_devices()
    if type(device) is int:
        d = device_list[device]["name"]
        print(f"""{d} will be used for input and output!""")
        sd.default.device = device
    elif type(device) is str:
        d_id, d_name = get_interface_number_by_name(device, device_list)
        print(f"{d_name} will be used for input and output!")
        sd.default.device = d_id
    elif type(device) is list:
        assert len(device) == 2, "List with device numbers must be exactly 2"

        if type(device[0]) is int and type(device[1]) is int:
            d = device_list[device[0]]["name"]
            print(f"{d} will be used for input!")

            d = device_list[device[1]]["name"]
            print(f"{d} will be used for output!")
            sd.default.device = device
        elif type(device[0]) is str and type(device[1]) is str:
            d_id_in, d_name_in = get_interface_number_by_name(
                device[0], device_list
            )
            print(f"{d_name_in} will be used for input!")

            d_id_out, d_name_out = get_interface_number_by_name(
                device[1], device_list
            )
            print(f"{d_name_out} will be used for output!")
            sd.default.device = [d_id_in, d_id_out]
        else:
            raise TypeError(
                "device must be either a homogenouos list of int and "
                + "str, or an int or a str"
            )
    else:
        raise TypeError(
            "device must be either a homogenouos list of int and "
            + "str, or an int or a str"
        )

    # Sampling rate
    if sampling_rate_hz is not None:
        sd.default.samplerate = sampling_rate_hz
    return sd.query_devices()


def get_interface_number_by_name(
    name: str, device_list: sd.DeviceList
) -> tuple[int, str]:
    """
    Return the interface ID (number) by looking at its name.

    Parameters
    ----------
    name : str
        Name of the interface or string contained in the name (the first
        interface to match will be returned). The comparison is case-invariant.

    Returns
    -------
    ind : int
        Interface ID
    full_name : str
        Interface's full name

    """
    for ind, d in enumerate(device_list):
        full_name: str = d["name"]
        if name.lower() in full_name.lower():
            return ind, full_name
    raise ValueError(f"No device was found with name {name}")


def play_and_record(
    signal: Signal,
    duration_seconds: float | None = None,
    normalized_dbfs: float | None = -6,
    device: str | None = None,
    play_channels=None,
    rec_channels=[1],
) -> Signal:
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
    rec_sig : `Signal`
        Recorded signal.

    """
    # Asserts
    if play_channels is None:
        play_channels = list(range(1, signal.number_of_channels + 1))
    if type(play_channels) is int:
        play_channels = [play_channels]
    if type(rec_channels) is int:
        rec_channels = [rec_channels]
    play_channels = sorted(play_channels)
    rec_channels = sorted(rec_channels)
    assert signal.number_of_channels == len(play_channels), (
        "The number of channels in signal does not match the number of "
        + "channels in play_channels"
    )
    assert not any(
        [p < 1 for p in play_channels]
    ), "Play channel has to be 1 or more"
    assert not any(
        [r < 1 for r in rec_channels]
    ), "Recording channel has to be 1 or more"
    #
    if duration_seconds is not None:
        assert duration_seconds > 0, "Duration must be positive"
        duration_samples = duration_seconds * signal.sampling_rate_hz
    else:
        duration_seconds = signal.time_data.shape[0] / signal.sampling_rate_hz
        duration_samples = signal.time_data.shape[0]

    play_data = signal.time_data.copy()[:duration_samples, :]

    if normalized_dbfs is not None:
        assert normalized_dbfs <= 0, "Only values beneath 0 dBFS are allowed"
        play_data = _normalize(play_data, dbfs=normalized_dbfs, mode="peak")

    if device is not None:
        sd.default.device = device

    print(
        "Playback and recording have started "
        + f"({duration_seconds:.1f} s)..."
    )
    rec_time_data = sd.playrec(
        data=play_data,
        samplerate=signal.sampling_rate_hz,
        input_mapping=rec_channels,
        output_mapping=play_channels,
        blocking=True,
    )
    print("Playback and recording have ended\n")

    rec_sig = Signal(None, rec_time_data, signal.sampling_rate_hz)
    return rec_sig


def record(
    duration_seconds: float = 5,
    sampling_rate_hz: int = 48000,
    device: str | int | None = None,
    rec_channels=[1],
) -> Signal:
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
    rec_sig : `Signal`
        Recorded signal.

    """
    # Asserts
    if type(rec_channels) is int:
        rec_channels = [rec_channels]
    rec_channels = sorted(rec_channels)
    assert not any(
        [r < 1 for r in rec_channels]
    ), "Recording channel has to be 1 or more"
    #
    if device is not None:
        sd.default.device = device

    print(f"\nRecording started ({duration_seconds:.1f} s)...")
    rec_time_data = sd.rec(
        frames=int(duration_seconds * sampling_rate_hz),
        samplerate=sampling_rate_hz,
        mapping=rec_channels,
        blocking=True,
    )
    print("Recording has ended\n")

    rec_sig = Signal(None, rec_time_data, sampling_rate_hz)
    return rec_sig


def play(
    signal: Signal,
    duration_seconds: float | None = None,
    normalized_dbfs: float | None = -6,
    device: str | None = None,
    play_channels: int | list | tuple | None = None,
):
    """Playback of signal using some available device. Note that the channel
    numbers start here with 1.

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
        play_channels = list(range(1, signal.number_of_channels + 1))
    if type(play_channels) is int:
        play_channels = [play_channels]
    play_channels = sorted(play_channels)
    assert not any(
        [r < 1 for r in play_channels]
    ), "Play channel has to be 1 or more"
    if duration_seconds is not None:
        assert duration_seconds > 0, "Duration must be positive"
        duration_samples = int(duration_seconds * signal.sampling_rate_hz)
    else:
        duration_seconds = signal.time_data.shape[0] / signal.sampling_rate_hz
        duration_samples = signal.time_data.shape[0]
    play_data = signal.time_data.copy()[:duration_samples, :]
    if normalized_dbfs is not None:
        assert normalized_dbfs <= 0, "Only values beneath 0 dBFS are allowed"
        play_data = _normalize(play_data, dbfs=normalized_dbfs, mode="peak")
    #
    if device is not None:
        sd.default.device = device

    print(f"Playback started ({duration_seconds:.1f} s)...")
    sd.play(
        data=play_data,
        samplerate=signal.sampling_rate_hz,
        mapping=play_channels,
        blocking=True,
    )
    print("Playback has ended\n")


def play_through_stream(
    signal: Signal,
    blocksize: int = 2048,
    audio_callback=standard_callback,
    device: str | None = None,
):
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
    blocksize : int, optional
        Block size used for the stream. Powers of two are recommended.
        Default: 2048.
    audio_callback : callable, optional
        This is the core of the stream! Callback modifies the signal as it is
        passed to the audio device. `audio_callback` is a function that takes
        in a signal object and passes a valid sounddevice callback.
        There can be preprocessing but it shouldn't change the total length
        of the signal! Refer to `standard_callback` for an example.
        Their signatures are::

            audio_callback(signal: Signal) -> callable

            callback(outdata: np.ndarray, frames: int,
                     time: CData, status: CallbackFlags) -> None

        See `sounddevice`'s examples of callbacks for more general
        information. Default: `standard_callback`.
    device : str, optional
        Device to be used in the audio stream. Pass `None` to use the default
        device. Default: `None`.

    References
    ----------
    - https://python-sounddevice.readthedocs.io/en/0.4.5/

    """

    if not hasattr(signal, "streaming_position"):
        signal.set_streaming_position()

    duration_samples = signal.time_data.shape[0] - signal.streaming_position
    duration_ms = int(duration_samples / signal.sampling_rate_hz * 1000)

    if device is not None:
        sd.default.device = device

    try:
        with sd.OutputStream(
            signal.sampling_rate_hz,
            blocksize=blocksize,
            callback=audio_callback(signal),
            channels=signal.number_of_channels,
        ):
            sd.sleep(duration_ms + 5)
    except Exception as e:
        print(e)


def CallbackStop():
    """Wrapper around sounddevice's CallbackStop. Used for stopping audio
    streamings.

    """
    sd.CallbackStop()


def sleep(seconds: float):
    """Wrapper around sounddevice's sleep. Use for waiting while a stream
    happens.

    Parameters
    ----------
    seconds : float
        Seconds to wait.

    """
    sd.sleep(int(seconds * 1000))


def output_stream(
    signal: Signal,
    blocksize=2048,
    device=None,
    latency=None,
    extra_settings=None,
    callback=None,
    finished_callback=None,
    clip_off=None,
    dither_off=None,
    never_drop_input=None,
    prime_output_buffers_using_stream_callback=None,
):
    """Creates and return a sounddevice's OutputStream object. See
    sounddevice's documentation for more information.

    Parameters
    ----------
    signal : `Signal`
        Signal for which the output stream will be created.
    blocksize : int, optional
        Block size to be used during the stream. Default: 2048.
    device : str, optional
        Device to be used. Pass `None` to use default device. Default: `None`.
    callback : callable
        Function that defines the audio callback::

            callback(outdata: np.ndarray, frames: int,
                     time: CData, status: CallbackFlags) -> None

    finished_callback : callable
    clip_off : optional
    dither_off : optional
    never_drop_input : optional
    prime_output_buffers_using_stream_callback : optional

    Returns
    -------
    stream : `sd.OutputStream`
        Stream object.

    References
    ----------
    - https://python-sounddevice.readthedocs.io/en/0.4.5/

    """
    pobusc = prime_output_buffers_using_stream_callback
    stream = sd.OutputStream(
        samplerate=signal.sampling_rate_hz,
        blocksize=blocksize,
        device=device,
        channels=signal.number_of_channels,
        dtype=None,
        latency=latency,
        extra_settings=extra_settings,
        callback=callback,
        finished_callback=finished_callback,
        clip_off=clip_off,
        dither_off=dither_off,
        never_drop_input=never_drop_input,
        prime_output_buffers_using_stream_callback=pobusc,
    )
    return stream
