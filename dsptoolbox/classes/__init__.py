"""
Classes
-------
Here are the classes of the dsptoolbox:

- `Signal` (core class for all computations, it is constructed from time data
            and a sampling rate)
- `ImpulseResponse` (class containing a signal that characterizes a system's
  response)
- `MultiBandSignal` (signal with multiple bands and multirate capabilities)
- `Filter` (filter class with filtering methods)
- `FilterBank` (class containing a group of `Filters` and their metadata)
- `Spectrum` (class for containing a complex or magnitude spectrum)

"""

from .filter import Filter
from .filterbank import FilterBank
from .signal import Signal
from .impulse_response import ImpulseResponse
from .multibandsignal import MultiBandSignal
from .spectrum import Spectrum

__all__ = [
    "Filter",
    "FilterBank",
    "Signal",
    "ImpulseResponse",
    "MultiBandSignal",
    "Spectrum",
]
