"""
Generators
----------
This module contains some utility signal generators. Choose from:

- `chirp()` (sweep)
- `noise()` (white, pink, red, blue, violet, grey)
- `dirac()` (impulse)
- `harmonic()`

"""
from .generators import chirp, noise, dirac, harmonic, oscillator

__all__ = [
    "chirp",
    "noise",
    "dirac",
    "harmonic",
    "oscillator",
]
