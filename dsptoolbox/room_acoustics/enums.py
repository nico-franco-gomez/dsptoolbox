from enum import Enum, auto


class ReverbTime(Enum):
    """Types of computation of the reverberation time:

    - Adaptive: adapts the computation to the best linear fit of energy decay.
    - EDT: Early decay time.

    """

    Adaptive = auto()
    T20 = auto()
    T30 = auto()
    T60 = auto()
    EDT = auto()


class RoomAcousticsDescriptor(Enum):
    """Descriptors:

    - D50: Definition. It takes values between [0, 1] and should
      correlate (positively) with speech inteligibility.
    - C80: Clarity. It is a value in dB. The higher, the more energy
      arrives in the early part of the RIR compared to the later part.
    - BassRatio: It exposes the ratio of reverberation times
      of the lower-frequency octave bands (125, 250) to the higher ones
      (500, 1000). T20 is always used.
    - CenterTime: It is the first-order moment RIR's energy.

    """

    D50 = auto()
    C80 = auto()
    BassRatio = auto()
    CenterTime = auto()
