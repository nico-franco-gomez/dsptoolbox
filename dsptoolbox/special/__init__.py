"""
Special
-------
This module contains functions that might be regarded as not very common use.

- `cepstrum()`
- `min_phase_from_mag()` (generate a signal with minimum phase from a
  magnitude spectrum)
- `lin_phase_from_mag()` (generate a signal with linear phase from a
  magnitude spectrum)

"""
from .special import cepstrum, min_phase_from_mag, lin_phase_from_mag

__all__ = [
    'cepstrum',
    'min_phase_from_mag',
    'lin_phase_from_mag',
]
