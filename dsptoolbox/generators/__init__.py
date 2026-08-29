"""
Generators
----------
This module contains some utility signal generators. Choose from:

- `chirp()` (sweep)
- `noise()` (white, pink, red, blue, violet, grey)
- `dirac()` (impulse)
- `oscillator()`

"""

from .enums import ChirpType, NoiseType, WaveForm
from .generators import chirp, dirac, noise, oscillator

__all__ = [
    "chirp",
    "noise",
    "dirac",
    "oscillator",
    "NoiseType",
    "ChirpType",
    "WaveForm",
]
