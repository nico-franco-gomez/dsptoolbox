"""
Transfer functions
------------------
In this module there are functions used to obtain or modify transfer functions
(TF).

Acquire TF from signals:

- `spectral_deconvolve()` (direct deconvolution)
- `compute_transfer_function()` (using welch's method for estimating auto- and
  cross correlation spectra from measurements)

Modify TF:

- `window_ir()` (Windows a TF in time domain)

Generate TF from magnitude spectrum:

- `min_phase_from_mag()` (generate a signal with minimum phase from a
  magnitude spectrum)
- `lin_phase_from_mag()` (generate a signal with linear phase from a
  magnitude spectrum)

"""
from .transfer_functions import (spectral_deconvolve, window_ir,
                                 compute_transfer_function,
                                 spectral_average,
                                 min_phase_from_mag,
                                 lin_phase_from_mag)

__all__ = [
    'spectral_deconvolve',
    'window_ir',
    'compute_transfer_function',
    'spectral_average',
    'min_phase_from_mag',
    'lin_phase_from_mag'
]
