"""
Audio IO
--------
This module handles audio playback and recording. It is based on sounddevice
(see link down below).

Setting audio device:

- `list_devices()`
- `set_device()`
- `print_device_info()`
- `get_default_config()`
- `set_latency()`
- `set_blocksize()`
- `enable_asio()` (Windows only, before sounddevice is imported)

Playing audio:

- `play()`
- `play_and_record()`
- `output_stream()`

Recording:

- `record()`

Others:

- `CallbackStop()` (used for stopping callbacks)
- `sleep()` (sleep while audio playback is finished)

References
----------
- https://pypi.org/project/sounddevice/

"""

from .audio_io import (
    CallbackStop,
    enable_asio,
    get_default_config,
    list_devices,
    output_stream,
    play,
    play_and_record,
    print_device_info,
    record,
    set_blocksize,
    set_device,
    set_latency,
    sleep,
)

__all__ = [
    "play",
    "play_and_record",
    "set_device",
    "list_devices",
    "record",
    "print_device_info",
    "CallbackStop",
    "sleep",
    "output_stream",
    "get_default_config",
    "enable_asio",
    "set_latency",
    "set_blocksize",
]
