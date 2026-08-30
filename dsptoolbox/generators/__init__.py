"""
Generators
----------
This module contains some utility signal generators. Choose from:

- `chirp()` (linear or logarithmic sweep)
- `sync_log_chirp()` (synchronized logarithmic sweep)
- `noise()` (white, pink, red, blue, violet, grey)
- `dirac()` (impulse)
- `oscillator()`

"""

from .enums import ChirpType, NoiseType, WaveForm
from .generators import chirp, dirac, noise, oscillator, sync_log_chirp

__all__ = [
    "chirp",
    "sync_log_chirp",
    "noise",
    "dirac",
    "oscillator",
    "NoiseType",
    "ChirpType",
    "WaveForm",
]
