"""
Realtime
--------
Filter implementations meant for sample- or block-wise processing. They all
derive from `RealtimeFilter` and share its three-method interface
(`process_sample`, `process_block`, `reset_state`, `set_n_channels`), so they
can be combined freely through `FilterChain`.

For offline filtering of a whole `Signal`, prefer `Filter` and `FilterBank`
from the top-level namespace, which delegate to scipy.

Time-domain structures:

- `FIRFilter()`: FIR filter implemented in the time domain.
- `IIRFilter()`: General IIR filter implemented as transposed direct-form 2.
- `LatticeLadderFilter()`: Filter with lattice-ladder topology.
- `StateSpaceFilter()`: Filter with state space representation.
- `StateVariableFilter()`: SV-Filter discretized with a topology-preserving
  transform.
- `ExponentialAverageFilter()`: One-pole smoother with separate attack and
  release coefficients.

Frequency-domain convolution:

- `FIRFilterOverlapSave()`: FIR filter implemented for real-time processing
  using RFFT and overlap-save.
- `FIRUniformPartitioned()`: The same with a partitioned filter, for long
  impulse responses.
- `FIRUniformPartitionedMultichannel()`: Partitioned overlap-save capable of
  vectorized multichannel processing.

Warped and parallel structures:

- `WarpedFIR()`: FIR filter implemented in the time domain with efficient
  warping.
- `WarpedIIR()`: IIR filter implemented in the time domain with warping.
- `ParallelFilter()`: Find the (least-squares) optimal linear combination of
  parallel SOS to approximate an IR.
- `KautzFilter()`: Kautz filters with an orthonormal pole basis.

Composition:

- `FilterChain()`: Filter structure for applying all other filters
  sequentially.
- `RealtimeFilter`: Abstract base class of all of the above.

"""

from .exponential_average_filter import ExponentialAverageFilter
from .filter_chain import FilterChain
from .fir_filter_realtime import (
    FIRFilter,
    FIRFilterOverlapSave,
    FIRUniformPartitioned,
    FIRUniformPartitionedMultichannel,
)
from .iir_filter_realtime import IIRFilter
from .kautz_filter import KautzFilter
from .lattice_ladder_filter import LatticeLadderFilter
from .parallel_filter import ParallelFilter
from .realtime_filter import RealtimeFilter
from .state_space_filter import StateSpaceFilter
from .sv_filter import StateVariableFilter
from .warped_filters import WarpedFIR, WarpedIIR

__all__ = [
    "RealtimeFilter",
    "FIRFilter",
    "IIRFilter",
    "LatticeLadderFilter",
    "StateSpaceFilter",
    "StateVariableFilter",
    "ExponentialAverageFilter",
    "FIRFilterOverlapSave",
    "FIRUniformPartitioned",
    "FIRUniformPartitionedMultichannel",
    "WarpedFIR",
    "WarpedIIR",
    "ParallelFilter",
    "KautzFilter",
    "FilterChain",
]
