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
- `weightning_filter()`: A- or C-Weightning filter.
- `complementary_fir_filter()`: Create a complementary FIR filter from a
  linear-phase FIR prototype.
- `LatticeLadderFilter()`: Filter with lattice-ladder topology.
- `PhaseLinearizer()`: Design an FIR filter that linearizes a phase spectrum
  or matches a target group delay.
- `StateVariableFilter()`: SV-Filter discretized with a topology-preserving
  transform.
- `convert_into_lattice_filter()`: Turns a conventional filter into its
  lattice/ladder representation.

"""

from .filterbanks import (
    linkwitz_riley_crossovers,
    reconstructing_fractional_octave_bands,
    auditory_filters_gammatone,
    fractional_octave_bands,
    qmf_crossover,
    weightning_filter,
    complementary_fir_filter,
    convert_into_lattice_filter,
)

from ..classes._lattice_ladder_filter import LatticeLadderFilter
from ..classes._phaseLinearizer import PhaseLinearizer
from ..classes._svfilter import StateVariableFilter

__all__ = [
    "linkwitz_riley_crossovers",
    "reconstructing_fractional_octave_bands",
    "fractional_octave_bands",
    "auditory_filters_gammatone",
    "qmf_crossover",
    "weightning_filter",
    "complementary_fir_filter",
    "convert_into_lattice_filter",
    "LatticeLadderFilter",
    "PhaseLinearizer",
    "StateVariableFilter",
]
