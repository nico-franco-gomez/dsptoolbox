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
- `PhaseLinearizer()`: Design an FIR filter that linearizes a phase spectrum.
- `GroupDelayDesigner()`: Design an FIR filter that matches a target group
  delay.
- `StateVariableFilter()`: SV-Filter discretized with a topology-preserving
  transform.
- `IIRFilter()`: General IIR filter implemented as transposed direct-form 2.
- `FIRFilter()`: FIR filter implemented in the time domain.
- `ParallelFilter()`: Find the (least-squares) optimal linear combination of
  parallel SOS to approximate an IR.
- `KautzFilter()`: Kautz filters with an orthonormal pole basis.
- `ExponentialAverageFilter()`.
- `StateSpaceFilter()`: Filter with state space representation.
- `FilterChain()`: Filter structure for applying all other filters
  sequentially.
- `convert_into_lattice_filter()`: Turns a conventional filter into its
  lattice/ladder representation.
- `pinking_filter()`: Get a -3 dB/octave filter.
- `matched_biquad()`: Analog-matched biquad filters.
- `gaussian_kernel()`: IIR first-order approximation of a gaussian window.
- `arma()`: IIR filter approximation of an impulse response.

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
    pinking_filter,
    matched_biquad,
    gaussian_kernel,
)

from ._filterbank import arma

from ..classes.lattice_ladder_filter import LatticeLadderFilter
from ..classes.parallel_filter import ParallelFilter
from ..classes.iir_filter_realtime import IIRFilter
from ..classes.fir_filter_realtime import FIRFilter
from ..classes.sv_filter import StateVariableFilter
from ..classes.kautz_filter import KautzFilter
from ..classes.exponential_average_filter import ExponentialAverageFilter
from ..classes.filter_chain import FilterChain
from ..classes.state_space_filter import StateSpaceFilter
from ..classes.group_delay_designer_phase_linearizer import (
    PhaseLinearizer,
    GroupDelayDesigner,
)

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
    "IIRFilter",
    "FIRFilter",
    "ParallelFilter",
    "StateSpaceFilter",
    "KautzFilter",
    "ExponentialAverageFilter",
    "FilterChain",
    "GroupDelayDesigner",
    "StateVariableFilter",
    "pinking_filter",
    "matched_biquad",
    "gaussian_kernel",
    "arma",
]
