"""
Standard functions
------------------
This module contains a general collection of DSP functions that do not fall
under a same category. These functions act on the custom classes of
`dsptoolbox` and not on primitive data types such as arrays.

Access to these functions should be done via `dsptoolbox.*` (without reference
to `standard`).

"""

from .appending import append_filterbanks, append_signals, append_spectra
from .enums import (
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
    SpectrumScaling,
    SpectrumType,
    Window,
)
from .gain_and_level import (
    apply_gain,
    crest_factor,
    fade,
    lufs_integrated,
    normalize,
    rms,
    true_peak_level,
)
from .latency_delay import delay, fractional_delay, latency
from .other import (
    activity_detector,
    detrend,
    dither,
    envelope,
    load_pkl_object,
    merge_filters,
    spectral_difference,
)
from .pad_trim_methods import (
    modify_signal_length,
    pad_trim,
    trim_with_level_threshold,
    trim_with_time_selection,
)
from .resampling import resample, resample_filter

__all__ = [
    # Append
    "append_filterbanks",
    "append_signals",
    "append_spectra",
    # Latency+Delay
    "latency",
    "delay",
    "fractional_delay",
    # Padding and trimming
    "pad_trim",
    "modify_signal_length",
    "trim_with_level_threshold",
    "trim_with_time_selection",
    # Resampling
    "resample",
    "resample_filter",
    # Gain-related functions
    "apply_gain",
    "normalize",
    "fade",
    "true_peak_level",
    "rms",
    "crest_factor",
    "lufs_integrated",
    # Other
    "load_pkl_object",
    "activity_detector",
    "detrend",
    "envelope",
    "dither",
    "merge_filters",
    "spectral_difference",
    # Enums
    "SpectrumMethod",
    "SpectrumScaling",
    "FilterCoefficientsType",
    "BiquadEqType",
    "FilterBankMode",
    "FilterPassType",
    "IirDesignMethod",
    "MagnitudeNormalization",
    "SpectrumType",
    "InterpolationDomain",
    "InterpolationScheme",
    "InterpolationEdgeHandling",
    "FrequencySpacing",
    "Window",
    "FadeType",
]
