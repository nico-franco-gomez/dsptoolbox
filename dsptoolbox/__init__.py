"""
# dsptoolbox

Collection of dsp, audio and acoustics-related functions.

### Useful links:
- Github Repo: https://github.com/nico-franco-gomez/dsptoolbox
- Docs: https://dsptoolbox.readthedocs.io/en/latest/?badge=latest

"""

from .standard import (
    BiquadEqType,
    FadeType,
    FilterBankMode,
    FilterCoefficientsType,
    FilterPassType,
    FrequencySpacing,
    IirDesignMethod,
    InterpolationConversion,
    InterpolationDomain,
    InterpolationEdgeHandling,
    InterpolationKind,
    InterpolationScheme,
    IrLatencyRemoval,
    MagnitudeNormalization,
    Power2Rounding,
    SampleFormat,
    SpectrumAverageMethod,
    SpectrogramParameters,
    SpectrumMethod,
    SpectrumParameters,
    # Enums
    SpectrumScaling,
    SpectrumType,
    WarpingFactor,
    Window,
    append_filterbanks,
    append_signals,
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
    "append_signals",
    "append_filterbanks",
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
    "InterpolationKind",
    "InterpolationConversion",
    "SampleFormat",
    "InterpolationEdgeHandling",
    "FrequencySpacing",
    "IirDesignMethod",
    "IrLatencyRemoval",
    "Power2Rounding",
    "SpectrumAverageMethod",
    "SpectrumParameters",
    "SpectrogramParameters",
    "Window",
    "WarpingFactor",
    "FadeType",
]

__version__ = "0.10.4"
