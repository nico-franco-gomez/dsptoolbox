from enum import Enum, auto


class SteeringVectorType(Enum):
    """Types of steering vectors. See [1] for details.

    References
    ----------
    - [1]: Sarradj, Ennes. (2012). Three-Dimensional Acoustic Source Mapping
      with Different Beamforming Steering Vector Formulations. Advances in
      Acoustics and Vibration. 2012. 10.1155/2012/292695.

    """

    Classic = auto()
    Inverse = auto()
    TruePower = auto()
    TrueLocation = auto()


class PointsProjection(Enum):
    """Projection to use when plotting a set of points:

    - Automatic: 3D for points that extend in three dimensions, 2D otherwise.
    - TwoDimensional, ThreeDimensional: force the projection. Points that
      extend in three dimensions are always plotted in 3D.

    """

    Automatic = auto()
    TwoDimensional = auto()
    ThreeDimensional = auto()


class SpatialDimension(Enum):
    """Cartesian spatial dimensions."""

    X = auto()
    Y = auto()
    Z = auto()

    def to_str(self) -> str:
        return self.name.lower()
