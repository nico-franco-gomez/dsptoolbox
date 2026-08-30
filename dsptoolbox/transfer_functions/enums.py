from dataclasses import dataclass
from enum import Enum, auto


class FirPhaseMode(Enum):
    """Phase of an FIR filter obtained from an impulse response:

    - Direct: the phase of the impulse response is left untouched.
    - Minimum: minimum phase.
    - Linear: minimum linear phase.

    """

    Direct = auto()
    Minimum = auto()
    Linear = auto()


class DiracNormalization(Enum):
    """Normalization of the dirac impulse that is combined with an impulse
    response:

    - NoNormalization: the dirac is used as it is.
    - Energy: the dirac band is scaled to match the energy (RMS) contained in
      the band of the impulse response.
    - Peak: the peak values of both bands are matched.
    - Custom: an explicit gain in dB for the dirac part. Bind it with
      `DiracNormalization.Custom.with_gain_db()`.

    """

    NoNormalization = auto()
    Energy = auto()
    Peak = auto()
    Custom = auto()

    def with_gain_db(self, gain_db: float) -> "ParametrizedDiracNormalization":
        """Bind an explicit gain to the normalization, returning a
        `ParametrizedDiracNormalization`. Only `Custom` takes one.

        Parameters
        ----------
        gain_db : float
            Gain in dB with which to scale the dirac part.

        Returns
        -------
        ParametrizedDiracNormalization
            Normalization bound to its gain. It can be passed wherever a
            `DiracNormalization` is expected.

        """
        if not self.needs_gain():
            raise ValueError(f"{self.name} does not take a gain")
        return ParametrizedDiracNormalization(self, float(gain_db))

    def needs_gain(self) -> bool:
        """When True, the member requires an explicit gain."""
        return self == DiracNormalization.Custom


@dataclass(frozen=True)
class ParametrizedDiracNormalization:
    """A `DiracNormalization` bound to an explicit gain in dB.

    Instances are produced by `DiracNormalization.Custom.with_gain_db()` and
    are immutable.

    """

    normalization: DiracNormalization
    gain_db: float

    def needs_gain(self) -> bool:
        """The gain is already bound, so this is always False."""
        return False


DiracNormalizationType = DiracNormalization | ParametrizedDiracNormalization


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

    - RealImaginary: smoothing directly on spectrum (real and imaginary).
    - PowerPhase: Smoothing on power and phase separately.
    - MagnitudePhase: Smoothing on magnitude and phase separately.
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

    RealImaginary = auto()
    PowerPhase = auto()
    MagnitudePhase = auto()
    Power = auto()
    Magnitude = auto()
    EquivalentComplex = auto()
