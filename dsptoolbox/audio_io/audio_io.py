"""
Here are wrappers for streams with sounddevice. This is useful for
measurements and testing audio streams
"""

import os
import sys
from typing import Any

from .. import Signal
from ..helpers.gain_and_level import _normalize

ASIO_ENVIRONMENT_VARIABLE = "SD_ENABLE_ASIO"


def _sd() -> Any:
    """Import sounddevice on first use.

    It is not imported with this module so that neither the library import
    nor `enable_asio()` depend on the audio backend being present or already
    initialized.

    """
    import sounddevice

    return sounddevice


def enable_asio() -> bool:
    """Ask sounddevice to load the ASIO-enabled PortAudio dll on Windows, by
    setting the `SD_ENABLE_ASIO` environment variable.

    This has to happen before sounddevice is imported anywhere in the process,
    which is why it is not done when importing this module.

    Returns
    -------
    bool
        True when the variable was set by this call. It is False on other
        platforms, when it was already set, or when sounddevice has already
        been imported, in which case the setting has no effect any more.

    """
    if sys.platform != "win32":
        return False
    if "sounddevice" in sys.modules:
        return False
    if ASIO_ENVIRONMENT_VARIABLE in os.environ:
        return False
    os.environ[ASIO_ENVIRONMENT_VARIABLE] = "1"
    return True


def get_default_config() -> Any:
    """Return sounddevice's default configuration object, on which the device,
    sampling rate, latency and block size can be inspected or set."""
    return _sd().default


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
        d = _sd().query_devices()
        print(d)
        return d
    else:
        d = _sd().query_devices(device_number)
        print(d)
        return d


def set_latency(input_low: bool, output_low: bool):
    """Set the desired latency (Default is high). This can vary for each device
    and host. Sounddevice only allows for setting a low or a high latency.
    High latency is more robust, but it might be too large for some
    applications.

    This method modifies the global default value.

    Parameters
    ----------
    input_low : bool
        When `True`, low latency will be requested to the host for input
        streams.
    output_low : bool
        When `True`, low latency will be requested to the host for output
        streams.

    """
    _sd().default.latency = (
        "low" if input_low else "high",
        "low" if output_low else "high",
    )


def set_blocksize(blocksize: int):
    """Set a default blocksize for any stream. This can lead to a stable latency
    for most interfaces. Not setting it will lead to a default value.

    This method modifies the global default value.

    Parameters
    ----------
    blocksize : int
        Desired block size.

    """
    _sd().default.blocksize = blocksize


def set_device(
    device: list[int] | list[str] | str | int,
    sampling_rate_hz: int | None = None,
):
    """Set the default input and output device.

    Parameters
    ----------
    device : list[int | str] with length 2, str, int
        Sets the input and output devices from two integers, e.g. [1, 2].
        Alternatively, two strings contained in the interface's name (or the
        name itself) can be passed. The first interface to match will be taken.
        If passing only one string or integer, the interface will be taken for
        both input and output.
    sampling_rate_hz : int, None, optional
        Pass a default sampling rate to the devices. Pass `None` to ignore.
        Default: `None`.

    Returns
    -------
    device_list : `sounddevice.DeviceList`
        Device List with dictionaries containing information about each
        available device.

    Notes
    -----
    - Use `list_devices()` to see which devices are available.

    """
    device_list = _sd().query_devices()
    if type(device) is int:
        d = device_list[device]["name"]
        print(f"""{d} will be used for input and output!""")
        _sd().default.device = device
    elif type(device) is str:
        d_id, d_name = get_interface_number_by_name(device, device_list)
        print(f"{d_name} will be used for input and output!")
        _sd().default.device = d_id
    elif type(device) is list:
        assert len(device) == 2, "List with device numbers must be exactly 2"

        if type(device[0]) is int and type(device[1]) is int:
            d = device_list[device[0]]["name"]
            print(f"{d} will be used for input!")

            d = device_list[device[1]]["name"]
            print(f"{d} will be used for output!")
            _sd().default.device = device
        elif type(device[0]) is str and type(device[1]) is str:
            d_id_in, d_name_in = get_interface_number_by_name(device[0], device_list)
            print(f"{d_name_in} will be used for input!")

            d_id_out, d_name_out = get_interface_number_by_name(device[1], device_list)
            print(f"{d_name_out} will be used for output!")
            _sd().default.device = [d_id_in, d_id_out]
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
        _sd().default.samplerate = sampling_rate_hz
    return _sd().query_devices()


def list_devices():
    """Return the available audio devices, and print them.

    Returns
    -------
    device_list : `sounddevice.DeviceList`
        Device list with dictionaries containing information about each
        available device. Its indices are the ones `set_device()` expects.

    """
    device_list = _sd().query_devices()
    title = "List of available devices"
    print(title + "\n" + "-" * len(title))
    print(device_list)
    print("-" * len(title))
    return device_list


def get_interface_number_by_name(name: str, device_list: "Any") -> tuple[int, str]:
    """Return the interface ID (number) by looking at its name.

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
    rec_channels: int | list[int] | None = None,
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
    if rec_channels is None:
        rec_channels = [1]
    if type(rec_channels) is int:
        rec_channels = [rec_channels]
    play_channels = sorted(play_channels)
    rec_channels = sorted(rec_channels)
    assert signal.number_of_channels == len(play_channels), (
        "The number of channels in signal does not match the number of "
        + "channels in play_channels"
    )
    assert not any([p < 1 for p in play_channels]), "Play channel has to be 1 or more"
    assert not any([r < 1 for r in rec_channels]), (
        "Recording channel has to be 1 or more"
    )
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
        play_data = _normalize(
            play_data,
            dbfs=normalized_dbfs,
            peak_normalization="peak",
            per_channel=False,
        )

    if device is not None:
        _sd().default.device = device

    print("Playback and recording have started " + f"({duration_seconds:.1f} s)...")
    rec_time_data = _sd().playrec(
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
    rec_channels: int | list[int] | None = None,
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
    if rec_channels is None:
        rec_channels = [1]
    if type(rec_channels) is int:
        rec_channels = [rec_channels]
    rec_channels = sorted(rec_channels)
    assert not any([r < 1 for r in rec_channels]), (
        "Recording channel has to be 1 or more"
    )
    #
    if device is not None:
        _sd().default.device = device

    print(f"\nRecording started ({duration_seconds:.1f} s)...")
    rec_time_data = _sd().rec(
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
        Signal to be reproduced. Its channel number must match the length
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
    assert not any([r < 1 for r in play_channels]), "Play channel has to be 1 or more"
    if duration_seconds is not None:
        assert duration_seconds > 0, "Duration must be positive"
        duration_samples = int(duration_seconds * signal.sampling_rate_hz)
    else:
        duration_seconds = signal.time_data.shape[0] / signal.sampling_rate_hz
        duration_samples = signal.time_data.shape[0]
    play_data = signal.time_data.copy()[:duration_samples, :]
    if normalized_dbfs is not None:
        assert normalized_dbfs <= 0, "Only values beneath 0 dBFS are allowed"
        play_data = _normalize(
            play_data,
            dbfs=normalized_dbfs,
            peak_normalization="peak",
            per_channel=False,
        )
    #
    if device is not None:
        _sd().default.device = device

    print(f"Playback started ({duration_seconds:.1f} s)...")
    _sd().play(
        data=play_data,
        samplerate=signal.sampling_rate_hz,
        mapping=play_channels,
        blocking=True,
    )
    print("Playback has ended\n")


def CallbackStop():
    """Wrapper around sounddevice's CallbackStop. Used for stopping audio
    streamings.

    """
    _sd().CallbackStop()


def sleep(seconds: float):
    """Wrapper around sounddevice's sleep. Use for waiting while a stream
    happens.

    Parameters
    ----------
    seconds : float
        Seconds to wait.

    """
    _sd().sleep(int(seconds * 1000))


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

            callback(outdata: NDArray[np.float64], frames: int,
                     time: CData, status: CallbackFlags) -> None

    finished_callback : callable
    clip_off : optional
    dither_off : optional
    never_drop_input : optional
    prime_output_buffers_using_stream_callback : optional

    Returns
    -------
    stream : `sounddevice.OutputStream`
        Stream object.

    References
    ----------
    - https://python-sounddevice.readthedocs.io/en/0.4.5/

    """
    pobusc = prime_output_buffers_using_stream_callback
    stream = _sd().OutputStream(
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
