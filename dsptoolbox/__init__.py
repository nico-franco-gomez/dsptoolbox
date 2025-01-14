"""
# dsptoolbox

Collection of dsp, audio and acoustics-related functions.

### Useful links:
- Github Repo: https://github.com/nico-franco-gomez/dsptoolbox
- Docs: https://dsptoolbox.readthedocs.io/en/latest/?badge=latest

"""

from .standard import (
    latency,
    append_signals,
    append_filterbanks,
    pad_trim,
    fractional_delay,
    delay,
    activity_detector,
    fade,
    normalize,
    true_peak_level,
    resample,
    load_pkl_object,
    detrend,
    rms,
    envelope,
    dither,
    apply_gain,
    resample_filter,
    modify_signal_length,
    merge_fir_filters,
    spectral_difference,
    append_spectra,
)
from .classes import (
    Filter,
    FilterBank,
    Signal,
    ImpulseResponse,
    MultiBandSignal,
    Spectrum,
)
from .classes.calibration_data import CalibrationData
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
    "Spectrum",
    "CalibrationData",
    # Functions in standard module
    "latency",
    "pad_trim",
    "fade",
    "append_signals",
    "append_filterbanks",
    "resample",
    "activity_detector",
    "normalize",
    "fractional_delay",
    "delay",
    "true_peak_level",
    "load_pkl_object",
    "detrend",
    "rms",
    "envelope",
    "dither",
    "apply_gain",
    "resample_filter",
    "modify_signal_length",
    "merge_fir_filters",
    "spectral_difference",
    "append_spectra",
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

__version__ = "0.5dev"
