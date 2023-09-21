"""
Audio IO
--------
This module handles audio playback and recording. It is based on sounddevice
(see link down below). While some functions offer full functionality, some
are just wrappers around sounddevice's functions:

Setting audio device:

- `set_device()`
- `print_device_info()`
- `default_config`

Playing audio:

- `play()`
- `play_through_stream()`
- `play_and_record()`
- `output_stream()`

Recording:

- `record()`

Others:

- `CallbackStop()` (used for stopping callbacks)
- `standard_callback()` (an example of an audio callback)
- `sleep()` (sleep while audio playback is finished)

References
----------
- https://pypi.org/project/sounddevice/

"""
from .audio_io import (play, play_and_record, set_device, record,
                       print_device_info, play_through_stream, CallbackStop,
                       standard_callback, sleep, output_stream, default_config)

__all__ = [
    'play',
    'play_and_record',
    'set_device',
    'record',
    'print_device_info',
    'play_through_stream',
    'CallbackStop',
    'standard_callback',
    'sleep',
    'output_stream',
    'default_config',
]
