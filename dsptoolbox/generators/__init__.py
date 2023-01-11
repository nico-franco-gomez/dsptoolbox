"""
Generators
----------
This module contains some utility signal generators. Choose from:

- `chirp()` (sweep)
- `noise()` (white, pink, red, blue, violet, grey)
- `dirac()` (impulse)
- `sinus()` (harmonic)

"""
from .generators import chirp, noise, dirac, sinus

__all__ = [
    'chirp',
    'noise',
    'dirac',
    'sinus',
]
