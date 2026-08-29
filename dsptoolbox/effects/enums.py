from enum import Enum, auto


class DistortionType(Enum):
    Arctan = auto()
    HardClip = auto()
    SoftClip = auto()
    NoDistortion = auto()


class Waveform(Enum):
    """Waveforms available for the LFO."""

    Harmonic = auto()
    Sawtooth = auto()
    Square = auto()
    Triangle = auto()


class SaturationType(Enum):
    """Named saturation presets for the delay effect."""

    Digital = auto()
    Arctan = auto()
