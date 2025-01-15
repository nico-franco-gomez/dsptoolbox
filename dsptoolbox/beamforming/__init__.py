"""
Beamforming
-----------
This module contains basic beamforming algorithms and classes used for their
implementation.

Classes:

- `Grid()`
- `Regular2DGrid()`
- `Regular3DGrid()`
- `LineGrid()`
- `MicArray()`
- `SteeringVector()`
- `BeamformerDASFrequency()`
- `BeamformerDASTime()`
- `BeamformerCleanSC()`
- `BeamformerOrthogonal()`
- `BeamformerFunctional()`
- `BeamformerMVDR()`
- `Source()`

Functions:

- `mix_sources_on_array()`: produces a mixed multi-channel signal on a given
  microphone array by receiving multiple source objects.

References:

- For a more powerful and flexible beamforming library, please refer to the
  acoular package: http://acoular.org

"""

from .beamforming import (
    Grid,
    Regular2DGrid,
    Regular3DGrid,
    LineGrid,
    MicArray,
    SteeringVector,
    BeamformerDASFrequency,
    BeamformerCleanSC,
    BeamformerOrthogonal,
    BeamformerFunctional,
    BeamformerMVDR,
    BeamformerDASTime,
    MonopoleSource,
    mix_sources_on_array,
)
from .enums import SteeringVectorType

__all__ = [
    "Grid",
    "Regular2DGrid",
    "Regular3DGrid",
    "LineGrid",
    "MicArray",
    "SteeringVector",
    "BeamformerDASFrequency",
    "BeamformerCleanSC",
    "BeamformerOrthogonal",
    "BeamformerFunctional",
    "BeamformerMVDR",
    "BeamformerDASTime",
    "MonopoleSource",
    "mix_sources_on_array",
    "SteeringVectorType",
]
