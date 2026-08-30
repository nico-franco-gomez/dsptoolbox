from enum import Enum, auto


class NoiseType(Enum):
    White = auto()
    Pink = auto()
    Red = auto()
    Blue = auto()
    Violet = auto()
    Grey = auto()


class ChirpType(Enum):
    """Chirp types:

    - Logarithmic (or exponential).
    - Linear.

    The synchronized logarithmic chirp has its own generator,
    `generators.sync_log_chirp()`, since it also returns its effective
    duration.

    References
    ----------
    - https://de.wikipedia.org/wiki/Chirp

    """

    Linear = auto()
    Logarithmic = auto()


class WaveForm(Enum):
    Harmonic = auto()
    Square = auto()
    Triangle = auto()
    Sawtooth = auto()
