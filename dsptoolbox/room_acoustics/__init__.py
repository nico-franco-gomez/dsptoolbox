"""
Room Acoustics
--------------
Here are contained some functions that are related to room acoustics.

- `reverb_time()`
- `find_modes()`
- `convolve_rir_on_signal()`
- `find_ir_start()`
- `generate_synthetic_rir()`
- `ShoeboxRoom`: class for shoebox rooms. It can be passed to
  `generate_synthetic_rir`.

"""
from .room_acoustics import (
    reverb_time,
    find_modes,
    convolve_rir_on_signal,
    find_ir_start,
    generate_synthetic_rir,
    ShoeboxRoom,
    descriptors,
)

__all__ = [
    "reverb_time",
    "find_modes",
    "convolve_rir_on_signal",
    "find_ir_start",
    "generate_synthetic_rir",
    "ShoeboxRoom",
    "descriptors",
]
