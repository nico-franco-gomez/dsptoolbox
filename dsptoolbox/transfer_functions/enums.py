from enum import Enum, auto


class TransferFunctionType(Enum):
    """Types of transfer functions for stochastic signals:

    - H1: for noise in the output signal. `Gxy/Gxx`.
    - H2: for noise in the input signal. `Gyy/Gyx`.
    - H3: for noise in both signals. `G_xy / abs(G_xy) * (G_yy/G_xx)**0.5`.

    """

    H1 = auto()
    H2 = auto()
    H3 = auto()


class SmoothingDomain(Enum):
    """These are the different domains to realize smoothing:

    - RealImaginary: directly on spectrum (real and imaginary).
    - PowerPhase: Power and phase separately.
    - MagnitudePhase.
    - Power: smoothing on power response, phase response is maintained.
    - Magnitude: smoothing on magnitude response, phase response is maintained.
    - EquivalentComplex: smoothing on power response, phase is obtained from
      the smoothed `RealImaginary` variant. This is the scheme proposed as
      equivalent complex smoothing by [1].

    References
    ----------
    - [1]: GENERALIZED FRACTIONAL OCTAVE SMOOTHING OF  AUDIO / ACOUSTIC
      RESPONSES. PANAGIOTIS D. HATZIANTONIOU AND JOHN N. MOURJOPOULOS.

    """

    PowerPhase = auto()
    RealImaginary = auto()
    MagnitudePhase = auto()
    Power = auto()
    Magnitude = auto()
    EquivalentComplex = auto()
