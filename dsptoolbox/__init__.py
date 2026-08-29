"""
# dsptoolbox

Collection of dsp, audio and acoustics-related functions.

### Useful links:
- Github Repo: https://github.com/nico-franco-gomez/dsptoolbox
- Docs: https://dsptoolbox.readthedocs.io/en/latest/?badge=latest

"""

# Import order below is load-bearing; don't let an import sorter reshuffle it:
# .standard must be imported before .classes (.standard.appending imports
# from ..classes, which would otherwise see a partially-initialized .classes
# module if .classes started importing first), and both must be imported
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
    activity_detector,
    append_filterbanks,
    append_signals,
    append_spectra,
    apply_gain,
    crest_factor,
    delay,
    detrend,
    dither,
    envelope,
    fade,
    fractional_delay,
    latency,
    load_pkl_object,
    lufs_integrated,
    merge_filters,
    modify_signal_length,
    normalize,
    pad_trim,
    resample,
    resample_filter,
    rms,
    spectral_difference,
    trim_with_level_threshold,
    trim_with_time_selection,
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
    "pad_trim",
    "trim_with_level_threshold",
    "fade",
    "append_signals",
    "append_filterbanks",
    "resample",
    "crest_factor",
    "lufs_integrated",
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
    "merge_filters",
    "spectral_difference",
    "append_spectra",
    "trim_with_time_selection",
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
