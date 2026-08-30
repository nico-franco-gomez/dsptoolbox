"""
Filter Banks
------------
This is a collection of useful filters and filter banks.

Perfect magnitude reconstruction:

- `linkwitz_riley_crossovers()`

Perfect reconstruction:

- `reconstructing_fractional_octave_bands()`

Psychoacoustics:

- `auditory_filters_gammatone()`

Other:

- `qmf_crossover()`: Quadrature mirror filters crossover.
- `fractional_octave_bands()`: Butterworth bandpass filter bank with signal
  energy conservation.
- `weighting_filter()`: A- or C-Weighting filter.
- `complementary_fir_filter()`: Create a complementary FIR filter from a
  linear-phase FIR prototype.
- `pinking_filter()`: Get a -3 dB/octave filter.
- `matched_biquad()`: Analog-matched biquad filters.
- `gaussian_kernel()`: IIR first-order approximation of a gaussian window.
- `fractional_delay()`: IIR filter with tunable, fractional delay.
- `arma()`: IIR filter approximation of an impulse response.
- `ArmaMethod`: Method used by `arma()`.

FIR designers:

- `FirDesigner`: Base class of the two designers below.
- `PhaseLinearizer()`: Design an FIR filter that linearizes a phase spectrum.
- `GroupDelayDesigner()`: Design an FIR filter that matches a target group
  delay.

The filter structures meant for sample- or block-wise processing live in
``dsptoolbox.realtime``.

"""

from ..classes.group_delay_designer_phase_linearizer import (
    FirDesigner,
    GroupDelayDesigner,
    PhaseLinearizer,
)
from ._filterbank import ArmaMethod, arma
from .filterbanks import (
    auditory_filters_gammatone,
    complementary_fir_filter,
    fractional_delay,
    fractional_octave_bands,
    gaussian_kernel,
    linkwitz_riley_crossovers,
    matched_biquad,
    pinking_filter,
    qmf_crossover,
    reconstructing_fractional_octave_bands,
    weighting_filter,
)

__all__ = [
    "linkwitz_riley_crossovers",
    "reconstructing_fractional_octave_bands",
    "fractional_octave_bands",
    "auditory_filters_gammatone",
    "qmf_crossover",
    "weighting_filter",
    "complementary_fir_filter",
    "pinking_filter",
    "matched_biquad",
    "gaussian_kernel",
    "arma",
    "fractional_delay",
    "ArmaMethod",
    "FirDesigner",
    "PhaseLinearizer",
    "GroupDelayDesigner",
]
