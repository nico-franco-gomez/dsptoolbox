"""
Transfer functions
------------------
In this module there are functions used to obtain, modify or analyze
transfer functions (TF)/impulse responses (IR).

Acquire TF/IR from signals:

- `spectral_deconvolve()` (direct deconvolution)
- `compute_transfer_function()` (using welch's method for estimating auto- and
  cross correlation spectra from measurements of stochastic signals)

Modify TF/IR:

- `window_ir()` (Windows a TF in time domain)
- `min_phase_ir()` (returns a minimum-phase version of the IR)
- `combine_ir_with_dirac()` (combines an IR with a time-aligned dirac impulse)

Generate TF/IR from magnitude spectrum:

- `min_phase_from_mag()` (generate a signal with minimum phase from a
  magnitude spectrum using the distcrete hilbert transform)
- `lin_phase_from_mag()` (generate a signal with linear phase from a
  magnitude spectrum)

Analyze TF/IR:

- `group_delay()`
- `minimum_group_delay()`
- `excess_group_delay()`
- `minimum_phase()`
- `find_ir_delay()`
- `window_frequency_dependent()` (obtain a spectrum with a frequency-dependent
  window)

"""
from .transfer_functions import (
    spectral_deconvolve,
    window_ir,
    window_frequency_dependent,
    window_centered_ir,
    compute_transfer_function,
    spectral_average,
    min_phase_from_mag,
    lin_phase_from_mag,
    min_phase_ir,
    group_delay,
    minimum_group_delay,
    excess_group_delay,
    minimum_phase,
    ir_to_filter,
    filter_to_ir,
    combine_ir_with_dirac,
    warp_ir,
)

__all__ = [
    "spectral_deconvolve",
    "window_ir",
    "compute_transfer_function",
    "spectral_average",
    "min_phase_from_mag",
    "lin_phase_from_mag",
    "min_phase_ir",
    "group_delay",
    "minimum_group_delay",
    "excess_group_delay",
    "minimum_phase",
    "window_frequency_dependent",
    "ir_to_filter",
    "filter_to_ir",
    "combine_ir_with_dirac",
    "window_centered_ir",
    "warp_ir",
]
