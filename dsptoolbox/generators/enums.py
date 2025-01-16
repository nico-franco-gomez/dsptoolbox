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
    - The `SyncLog` chirp is defined according to [2] and ensures that the
      harmonic responses have coherent phase with the linear response.

    References
    ----------
    - https://de.wikipedia.org/wiki/Chirp
    - [2]: Antonin Novak, Laurent Simon, Pierrick Lotton. Synchronized
      Swept-Sine: Theory, Application and Implementation.

    """

    Linear = auto()
    Logarithmic = auto()
    SyncLog = auto()


class WaveForm(Enum):
    Harmonic = auto()
    Square = auto()
    Triangle = auto()
    Sawtooth = auto()
