"""
Filter Banks
------------
This is a collection of useful filter banks. They use primarily the
`FilterBank` class or some derivation from it.

Perfect magnitude reconstruction:

- `linkwitz_riley_crossovers()`

Perfect reconstruction:

- `reconstructing_fractional_octave_bands()`

Psychoacoustics:

- `auditory_filters_gammatone()`

"""
from .filterbanks import (linkwitz_riley_crossovers,
                          reconstructing_fractional_octave_bands,
                          auditory_filters_gammatone)

__all__ = [
    'linkwitz_riley_crossovers',
    'reconstructing_fractional_octave_bands',
    'auditory_filters_gammatone',
]
