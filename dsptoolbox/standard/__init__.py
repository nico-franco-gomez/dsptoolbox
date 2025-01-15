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
from .latency_delay import latency, delay, fractional_delay
from .pad_trim_methods import pad_trim, modify_signal_length
from .resampling import resample, resample_filter
from .gain_and_level import apply_gain, normalize, fade, true_peak_level, rms
from .other import (
    load_pkl_object,
    activity_detector,
    detrend,
    envelope,
    dither,
    merge_filters,
    spectral_difference,
)
from .enums import (
    SpectrumMethod,
    SpectrumScaling,
    FilterCoefficientsType,
    BiquadEqType,
    FilterBankMode,
    FilterPassType,
    IirDesignMethod,
    MagnitudeNormalization,
    SpectrumType,
    InterpolationDomain,
    InterpolationScheme,
    InterpolationEdgeHandling,
    FrequencySpacing,
    Window,
    FadeType,
)

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
    # Resampling
    "resample",
    "resample_filter",
    # Gain-related functions
    "apply_gain",
    "normalize",
    "fade",
    "true_peak_level",
    "rms",
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
