"""
Standard functions
------------------
This module contains a general collection of DSP functions that do not fall
under a same category. These functions act on the custom classes of
`dsptoolbox` and not on primitive data types such as arrays.

Access to these functions should be done via `dsptoolbox.*` (without reference
to `standard`).

"""

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
    crest_factor,
    lufs_integrated,
    rms,
    true_peak_level,
)
from .latency_delay import latency
from .other import (
    envelope,
    load_pkl_object,
)

__all__ = [
    # Latency+Delay
    "latency",
    # Gain-related functions
    "true_peak_level",
    "rms",
    "crest_factor",
    "lufs_integrated",
    # Other
    "load_pkl_object",
    "envelope",
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
