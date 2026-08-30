"""
# dsptoolbox

Collection of dsp, audio and acoustics-related functions.

### Useful links:
- Github Repo: https://github.com/nico-franco-gomez/dsptoolbox
- Docs: https://dsptoolbox.readthedocs.io/en/latest/?badge=latest

"""

# Import order below is load-bearing; don't let an import sorter reshuffle it:
# .standard must be imported before .classes (several .standard submodules
# import from ..classes, which would otherwise see a partially-initialized
# .classes module if .classes started importing first), and both must be imported
# before the submodule block at the bottom, since several of those submodules
# (audio_io, distances, filterbanks, beamforming, effects) do
# `from .. import <name>`, which requires that name already bound here.
from .standard import (
    BiquadEqType,
    FadeType,
    FilterBankMode,
    FilterCoefficientsType,
    FilterPassType,
    FrequencySpacing,
    IirDesignMethod,
    InterpolationDomain,
    InterpolationEdgeHandling,
    InterpolationScheme,
    MagnitudeNormalization,
    SpectrumMethod,
    # Enums
    SpectrumScaling,
    SpectrumType,
    Window,
    crest_factor,
    envelope,
    latency,
    load_pkl_object,
    lufs_integrated,
    rms,
    true_peak_level,
)
from .classes import (
    Filter,
    FilterBank,
    ImpulseResponse,
    MultiBandSignal,
    Signal,
    Spectrum,
)
from .classes.calibration_data import CalibrationData
from . import (
    audio_io,
    beamforming,
    distances,
    effects,
    filterbanks,
    generators,
    plots,
    realtime,
    room_acoustics,
    tools,
    transfer_functions,
    transforms,
)

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
    "crest_factor",
    "lufs_integrated",
    "true_peak_level",
    "load_pkl_object",
    "rms",
    "envelope",
    # Modules
    "transfer_functions",
    "distances",
    "room_acoustics",
    "plots",
    "generators",
    "filterbanks",
    "realtime",
    "transforms",
    "audio_io",
    "beamforming",
    "effects",
    "tools",
    # Enums
    "SpectrumScaling",
    "SpectrumMethod",
    "FilterCoefficientsType",
    "BiquadEqType",
    "FilterBankMode",
    "FilterPassType",
    "MagnitudeNormalization",
    "SpectrumType",
    "InterpolationDomain",
    "InterpolationScheme",
    "InterpolationEdgeHandling",
    "FrequencySpacing",
    "IirDesignMethod",
    "Window",
    "FadeType",
]

__version__ = "0.9"
