"""
Standard functions
------------------
This module contains a general collection of DSP functions that do not fall
under a same category. These functions act on the custom classes of
`dsptoolbox` and not on primitive data types such as arrays.

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
    merge_fir_filters,
    spectral_difference,
)

__all__ = [
    "append_filterbanks",
    "append_signals",
    "append_spectra",
    "latency",
    "delay",
    "fractional_delay",
    "pad_trim",
    "modify_signal_length",
    "resample",
    "resample_filter",
    "apply_gain",
    "normalize",
    "fade",
    "true_peak_level",
    "rms",
    "load_pkl_object",
    "activity_detector",
    "detrend",
    "envelope",
    "dither",
    "merge_fir_filters",
    "spectral_difference",
]
