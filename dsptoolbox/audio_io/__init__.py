"""
Audio IO
--------
This module handles audio playback and recording. It is based on sounddevice
(see link down below).

Setting audio device:

- `set_device()`
- `print_device_info()`
- `default_config`
- `set_latency()`
- `set_blocksize()`

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
    default_config,
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
    "record",
    "print_device_info",
    "CallbackStop",
    "sleep",
    "output_stream",
    "default_config",
    "set_latency",
    "set_blocksize",
]
