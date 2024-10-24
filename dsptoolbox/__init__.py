"""
# dsptoolbox

Collection of dsp, audio and acoustics-related functions.

### Useful links:
- Github Repo: https://github.com/nico-franco-gomez/dsptoolbox
- Docs: https://dsptoolbox.readthedocs.io/en/latest/?badge=latest

"""

from .standard_functions import (
    latency,
    merge_signals,
    merge_filterbanks,
    pad_trim,
    fractional_delay,
    activity_detector,
    fade,
    normalize,
    true_peak_level,
    resample,
    load_pkl_object,
    detrend,
    rms,
    CalibrationData,
    envelope,
    dither,
    apply_gain,
    resample_filter,
)
from .classes import (
    Filter,
    FilterBank,
    Signal,
    ImpulseResponse,
    MultiBandSignal,
)
from . import transfer_functions
from . import distances
from . import room_acoustics
from . import plots
from . import generators
from . import filterbanks
from . import transforms
from . import audio_io
from . import beamforming
from . import effects
from . import tools

__all__ = [
    # Basic classes
    "Signal",
    "ImpulseResponse",
    "MultiBandSignal",
    "Filter",
    "FilterBank",
    # Functions in standard module
    "latency",
    "pad_trim",
    "fade",
    "merge_signals",
    "merge_filterbanks",
    "resample",
    "activity_detector",
    "normalize",
    "fractional_delay",
    "true_peak_level",
    "load_pkl_object",
    "detrend",
    "rms",
    "CalibrationData",
    "envelope",
    "dither",
    "apply_gain",
    "resample_filter",
    # Modules
    "transfer_functions",
    "distances",
    "room_acoustics",
    "plots",
    "generators",
    "filterbanks",
    "transforms",
    "audio_io",
    "beamforming",
    "effects",
    "tools",
]

__version__ = "0.4.7"
