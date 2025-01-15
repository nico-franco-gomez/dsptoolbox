"""
Generators
----------
This module contains some utility signal generators. Choose from:

- `chirp()` (sweep)
- `noise()` (white, pink, red, blue, violet, grey)
- `dirac()` (impulse)
- `oscillator()`

"""

from .generators import chirp, noise, dirac, oscillator
from .enums import NoiseType, ChirpType, WaveForm

__all__ = [
    "chirp",
    "noise",
    "dirac",
    "oscillator",
    "NoiseType",
    "ChirpType",
    "WaveForm",
]
