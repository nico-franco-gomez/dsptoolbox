from enum import Enum, auto


class DistortionType(Enum):
    Arctan = auto()
    HardClip = auto()
    SoftClip = auto()
    NoDistortion = auto()
