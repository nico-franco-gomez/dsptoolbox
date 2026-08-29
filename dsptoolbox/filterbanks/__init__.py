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
- `LatticeLadderFilter()`: Filter with lattice-ladder topology.
- `PhaseLinearizer()`: Design an FIR filter that linearizes a phase spectrum.
- `GroupDelayDesigner()`: Design an FIR filter that matches a target group
  delay.
- `StateVariableFilter()`: SV-Filter discretized with a topology-preserving
  transform.
- `IIRFilter()`: General IIR filter implemented as transposed direct-form 2.
- `FIRFilter()`: FIR filter implemented in the time domain.
- `FIRFilterOverlapSave()`: FIR filter implemented for real-time processing
  using RFFT and overlap-save.
- `FIRUniformPartitioned()`: FIR filter implemented for real-time processing
  using RFFT and overlap-save and a partitioned filter.
- `FIRUniformPartitioned()`: FIR filter implemented for real-time processing
  using RFFT and overlap-save and a partitioned filter. Capable of vectorized
  multichannel processing.
- `WarpedFIR()`: FIR filter implemented in the time domain with efficient
  warping.
- `WarpedIIR()`: IIR filter implemented in the time domain with warping.
- `ParallelFilter()`: Find the (least-squares) optimal linear combination of
  parallel SOS to approximate an IR.
- `KautzFilter()`: Kautz filters with an orthonormal pole basis.
- `ExponentialAverageFilter()`.
- `StateSpaceFilter()`: Filter with state space representation.
- `FilterChain()`: Filter structure for applying all other filters
  sequentially.
- `pinking_filter()`: Get a -3 dB/octave filter.
- `matched_biquad()`: Analog-matched biquad filters.
- `gaussian_kernel()`: IIR first-order approximation of a gaussian window.
- `fractional_delay()`: IIR filter with tunable, fractional delay.
- `arma()`: IIR filter approximation of an impulse response.

"""

from ..classes.exponential_average_filter import ExponentialAverageFilter
from ..classes.filter_chain import FilterChain
from ..classes.fir_filter_realtime import (
    FIRFilter,
    FIRFilterOverlapSave,
    FIRUniformPartitioned,
    FIRUniformPartitionedMultichannel,
)
from ..classes.group_delay_designer_phase_linearizer import (
    FirDesigner,
    GroupDelayDesigner,
    PhaseLinearizer,
)
from ..classes.iir_filter_realtime import IIRFilter
from ..classes.kautz_filter import KautzFilter
from ..classes.lattice_ladder_filter import LatticeLadderFilter
from ..classes.parallel_filter import ParallelFilter
from ..classes.state_space_filter import StateSpaceFilter
from ..classes.sv_filter import StateVariableFilter
from ..classes.warped_filters import WarpedFIR, WarpedIIR
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
    "LatticeLadderFilter",
    "PhaseLinearizer",
    "FirDesigner",
    "IIRFilter",
    "FIRFilter",
    "FIRFilterOverlapSave",
    "FIRUniformPartitioned",
    "FIRUniformPartitionedMultichannel",
    "WarpedFIR",
    "WarpedIIR",
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
    "fractional_delay",
    "ArmaMethod",
]
