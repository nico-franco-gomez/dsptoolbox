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
