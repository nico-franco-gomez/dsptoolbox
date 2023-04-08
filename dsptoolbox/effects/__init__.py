"""
Effects
-------
This module is a collection of basic audio effects implemented for offline
processing. The effects can be applied to the signal as a whole (with possibly
anti-causal operations) or in a block-processing manner.

"""
from .effects import SpectralSubtractor, Distortion, Compressor

__all__ = [
    'SpectralSubtractor',
    'Distortion',
    'Compressor',
]
