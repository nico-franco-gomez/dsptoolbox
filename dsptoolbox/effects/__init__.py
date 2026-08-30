"""
Effects
-------
This module is a collection of basic audio effects implemented for offline
processing. The effects can be applied to the signal as a whole (with possibly
anti-causal operations) or in a block-processing manner.

All effects derive from `AudioEffect` and return a new signal, leaving the
input untouched.

"""

from .effects import (
    LFO,
    AudioEffect,
    Chorus,
    Compressor,
    DigitalDelay,
    Distortion,
    SpectralSubtractor,
    Tremolo,
    get_frequency_from_musical_rhythm,
    get_time_period_from_musical_rhythm,
)
from .enums import DistortionType, SaturationType, Waveform

__all__ = [
    "AudioEffect",
    "SpectralSubtractor",
    "Distortion",
    "Compressor",
    "LFO",
    "Tremolo",
    "Chorus",
    "DigitalDelay",
    "get_frequency_from_musical_rhythm",
    "get_time_period_from_musical_rhythm",
    "DistortionType",
    "Waveform",
    "SaturationType",
]
