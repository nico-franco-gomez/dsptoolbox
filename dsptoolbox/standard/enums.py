from numpy.typing import NDArray
import numpy as np
from enum import Enum, auto
from scipy.signal.windows import get_window as get_window_scipy


class SpectrumMethod(Enum):
    """Methods to compute a spectrum.
    - Welch: produces a spectrum that is averaged over time. If it is the
      autospectrum, it is always real-valued (magnitude or power). If it is
      a cross-spectrum, it is complex.
    - FFT: produces the spectrum of a deterministic signal or an impulse
      response using a DFT directly on the time signal.

    """

    WelchPeriodogram = auto()
    FFT = auto()


class SpectrumScaling(Enum):
    """Amplitude scalings are:
    - AmplitudeSpectrum
    - AmplitudeSpectralDensity
    - FFTBackward
    - FFTForward
    - FFTOrthogonal

    Power scalings are:
    - PowerSpectrum
    - PowerSpectralDensity

    Notes
    -----
    - FFT scalings just normalized by the length of the data but have no direct
      physical units, since they do not regard windows or sampling rates.
    - Power (magnitude-squared) scalings usually deliver real data, i.e., no
      complex spectra. This is not the case for cross-spectral matrices (CSM),
      where the power scalings do deliver complex cross-spectra.
    - Amplitude scalings can deliver complex or real (magnitude) spectra
      depending on the applied method to compute the spectrum.

    """

    AmplitudeSpectrum = auto()
    AmplitudeSpectralDensity = auto()
    PowerSpectrum = auto()
    PowerSpectralDensity = auto()
    FFTBackward = auto()
    FFTForward = auto()
    FFTOrthogonal = auto()

    def fft_norm(self) -> str:
        """Return the expected FFT normalization to use for the given
        scaling.

        Returns
        -------
        str
            FFT normalization as expected by numpy or scipy FFT.

        """
        if self in (
            SpectrumScaling.AmplitudeSpectrum,
            SpectrumScaling.AmplitudeSpectralDensity,
            SpectrumScaling.PowerSpectrum,
            SpectrumScaling.PowerSpectralDensity,
            SpectrumScaling.FFTBackward,
        ):
            return "backward"

        if self == SpectrumScaling.FFTForward:
            return "forward"

        return "ortho"

    def is_amplitude_scaling(self) -> bool:
        """When True, it is an amplitude scaling of spectrum. False should
        then be regarded as a power scaling.

        Returns
        -------
        bool

        """
        return self in (
            SpectrumScaling.AmplitudeSpectrum,
            SpectrumScaling.AmplitudeSpectralDensity,
            SpectrumScaling.FFTBackward,
            SpectrumScaling.FFTForward,
            SpectrumScaling.FFTOrthogonal,
        )

    def outputs_complex_spectrum(self, method: SpectrumMethod) -> bool:
        """True means that the output spectrum should be complex, otherwise it
        will real-valued.

        Parameters
        ----------
        method : SpectrumMethod
            Method for computing the spectrum.

        """
        if method == SpectrumMethod.WelchPeriodogram:
            return False

        return self.is_amplitude_scaling()

    def has_physical_units(self) -> bool:
        """When True, the spectrum scaling has a physical unit. Otherwise, it
        is just an FFT normalization scheme.

        Returns
        -------
        bool

        """
        return self in (
            SpectrumScaling.AmplitudeSpectrum,
            SpectrumScaling.AmplitudeSpectralDensity,
            SpectrumScaling.PowerSpectrum,
            SpectrumScaling.PowerSpectralDensity,
        )

    def is_spectral_density(self) -> bool:
        """When True, the scaling is a spectral density and its power
        representation can be integrated over frequency to get the signal's
        energy (Parserval's theorem applies). False means it is either a
        spectrum or it has no physical units.

        Returns
        -------
        bool

        """
        return self in (
            SpectrumScaling.AmplitudeSpectralDensity,
            SpectrumScaling.PowerSpectralDensity,
        )

    def conversion_factor(
        self,
        output: "SpectrumScaling",
        length_time_data_samples: int,
        sampling_rate_hz: int,
        window: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Obtain the conversion factor from the current scaling to another
        one. If the input and output do not match on whether scaling is
        linear or squared, the conversion factor is always computed to be
        multiplied with the squared data.

        Parameters
        ----------
        output : SpectrumScaling
            Scaling output.
        length_time_data_samples : int
        sampling_rate_hz : int
        window : NDArray[np.float64], None

        Returns
        -------
        NDArray[np.float64]

        """
        input_factor = self.get_scaling_factor(
            length_time_data_samples, sampling_rate_hz, window
        )
        output_factor = output.get_scaling_factor(
            length_time_data_samples, sampling_rate_hz, window
        )

        # Consistent amplitude or power representation
        if not (self.is_amplitude_scaling() ^ output.is_amplitude_scaling()):
            return output_factor / input_factor

        if self.is_amplitude_scaling():
            input_factor **= 2.0
        else:
            output_factor **= 2.0
        return output_factor / input_factor

    def get_scaling_factor(
        self,
        length_time_data_samples: int,
        sampling_rate_hz: int,
        window: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Obtain the scaling factor for the given spectrum scaling and
        parameters. This factor is always valid for applying on the linear or
        squared data directly. This factor applies for the forward transform
        and a one-sided spectrum. Correction for DC and Nyquist must be done
        manually.

        Parameters
        ----------
        output : SpectrumScaling
            Scaling output.
        length_time_data_samples : int
        sampling_rate_hz : int
        window : NDArray[np.float64], None

        """
        if self == SpectrumScaling.FFTBackward:
            return np.atleast_1d(1.0)

        if self == SpectrumScaling.FFTForward:
            return np.atleast_1d(1.0 / length_time_data_samples)

        if self == SpectrumScaling.FFTOrthogonal:
            return np.atleast_1d((1.0 / length_time_data_samples) ** 0.5)

        if self.is_spectral_density():
            if window is None:
                factor = (
                    2 / length_time_data_samples / sampling_rate_hz
                ) ** 0.5
            else:
                factor = (
                    2
                    / np.sum(window**2, axis=0, keepdims=True)
                    / sampling_rate_hz
                ) ** 0.5
        else:  # Spectrum
            if window is None:
                factor = 2**0.5 / length_time_data_samples
            else:
                factor = 2**0.5 / np.sum(window, axis=0, keepdims=True)

        if self.is_amplitude_scaling():
            return factor

        return factor**2.0


class FilterCoefficientsType(Enum):
    """Coefficients accepted by scipy.

    - Zpk: zero, poles, gain.
    - Sos: second-order sections.
    - Ba: feed-forward and feed-backward coefficients.

    """

    Zpk = auto()
    Sos = auto()
    Ba = auto()


class BiquadEqType(Enum):
    Lowpass = auto()
    Highpass = auto()
    Peaking = auto()
    Lowshelf = auto()
    Highshelf = auto()
    BandpassSkirt = auto()
    BandpassPeak = auto()
    LowpassFirstOrder = auto()
    HighpassFirstOrder = auto()
    AllpassFirstOrder = auto()
    Allpass = auto()
    Notch = auto()
    Inverter = auto()


class FilterBankMode(Enum):
    """Ways to apply a filter bank to a signal:
    - Parallel: returns a MultiBandSignal where each band is the output of
      each filter.
    - Sequential: applies each filter to the given Signal in a sequential
      manner and returns output with same dimension.
    - Summed: applies every filter as parallel and then sums the outputs
      returning same dimensional output as input.

    """

    Parallel = auto()
    Sequential = auto()
    Summed = auto()


class FilterPassType(Enum):
    Lowpass = auto()
    Highpass = auto()
    Bandpass = auto()
    Bandstop = auto()

    def __str__(self):
        return self.name.lower()

    def to_str(self):
        return str(self)


class IirDesignMethod(Enum):
    """Methods for IIR filter design:

    - Butterworth: maximally flat in the passband. Good for general use.
    - Bessel: mild rolloff but with approximately linear phase response in the
      passband.
    - Chebyshev1: ripples in the passband, monotonically decreasing in the
      stopband. Steep Rolloff.
    - Chebyshev2: flat in the passband, ripples in the stopband.
    - Elliptic: ripples in passband and stopband. Very steep rolloff.

    """

    Bessel = auto()
    Butterworth = auto()
    Chebyshev1 = auto()
    Chebyshev2 = auto()
    Elliptic = auto()

    def to_scipy_str(self) -> str:
        """Return the scipy string variant."""
        if self == IirDesignMethod.Bessel:
            return "bessel"
        if self == IirDesignMethod.Butterworth:
            return "butter"
        if self == IirDesignMethod.Chebyshev1:
            return "cheby1"
        if self == IirDesignMethod.Chebyshev2:
            return "cheby2"
        if self == IirDesignMethod.Elliptic:
            return "ellip"


class Window(Enum):
    """Different window types. They are computed via
    `scipy.signal.windows.get_window()`.

    """

    Boxcar = auto()
    Triang = auto()
    Blackman = auto()
    Hamming = auto()
    Hann = auto()
    Bartlett = auto()
    Flattop = auto()
    Parzen = auto()
    Bohman = auto()
    Blackmanharris = auto()
    Nuttall = auto()
    Barthann = auto()
    Cosine = auto()
    Exponential = auto()
    Tukey = auto()
    Taylor = auto()
    Lanczos = auto()
    Kaiser = auto()
    KaiserBesselDerived = auto()
    Gaussian = auto()
    GeneralCosine = auto()
    GeneralGaussian = auto()
    GeneralHamming = auto()
    Dpss = auto()
    Chebwin = auto()

    @property
    def extra_parameter(self):
        return self.__extra_parameter

    def with_extra_parameter(
        self, extra_parameter: float | tuple[float, float]
    ):
        """Add the extra parameter parameter to the window. Windows that
        require an extra parameter are:
        - Kaiser
        - KaiserBesselDerived
        - Gaussian
        - GeneralCosine
        - GeneralGaussian (two parameters)
        - GeneralHamming
        - Dpss
        - Chebwin

        Refer to `scipy.signal.windows` for more information.

        """
        self.__extra_parameter = extra_parameter
        return self

    def to_scipy_format(self):
        """Parse to format for passing to
        `scipy.signal.windows.get_window()`.

        """
        if self.needs_extra_parameter():
            if self == Window.GeneralGaussian:
                return (
                    self.__to_str(),
                    self.extra_parameter[0],
                    self.extra_parameter[1],
                )
            return (self.__to_str(), self.extra_parameter)
        return self.__to_str()

    def __to_str(self) -> str:
        if self == Window.KaiserBesselDerived:
            return "kaiser_bessel_derived"
        if self == Window.GeneralCosine:
            return "general_cosine"
        if self == Window.GeneralGaussian:
            return "general_gaussian"
        if self == Window.GeneralHamming:
            return "general_hamming"

        return self.name.lower()

    def needs_extra_parameter(self) -> bool:
        """When True, window type requires a new parameter."""
        return self in (
            Window.Kaiser,
            Window.KaiserBesselDerived,
            Window.Gaussian,
            Window.GeneralCosine,
            Window.GeneralGaussian,  # 2 parameters
            Window.GeneralHamming,
            Window.Dpss,
            Window.Chebwin,
        )

    def __call__(self, n_values: int, symmetric: bool):
        """Get window values from `scipy.signal.windows.get_window()`."""
        return get_window_scipy(
            self.to_scipy_format(), n_values, not symmetric
        )


class MagnitudeNormalization(Enum):
    """Normalization for magnitude responses:

    - NoNormalization.
    - OneKhz: @ 1 kHz for each channel.
    - Max: @ peak.
    - Energy: use average energy (per frequency) as normalization value.

    All variants exist for the first channel, thus taking the same
    normalization value for all channels, or for each channel independently.

    """

    NoNormalization = auto()
    OneKhz = auto()
    OneKhzFirstChannel = auto()
    Max = auto()
    MaxFirstChannel = auto()
    Energy = auto()
    EnergyFirstChannel = auto()


class SpectrumType(Enum):
    """Spectrum representations."""

    Power = auto()
    Magnitude = auto()
    Complex = auto()
    Db = auto()


class InterpolationDomain(Enum):
    """For Complex and MagnitudePhase domains, the underlying data must be
    complex.

    """

    Magnitude = auto()
    Power = auto()
    Complex = auto()
    MagnitudePhase = auto()

    def is_complex(self) -> bool:
        return self in (
            InterpolationDomain.Complex,
            InterpolationDomain.MagnitudePhase,
        )

    def is_linear(self) -> bool:
        return self != InterpolationDomain.Power


class InterpolationScheme(Enum):
    """The interpolation schemes are:

    - Linear: linear interpolation. It is the fastest and most stable.
    - Cubic: CubicSplines. It delivers smoother results, but can lead\
      to overshooting and other interpolation artifacts.
    - Pchip: PchipInterpolator. It is a polynomial interpolator that\
      avoids overshooting between interpolation points.

    """

    Linear = auto()
    Cubic = auto()
    Pchip = auto()


class InterpolationEdgeHandling(Enum):
    """Handling of edges during interpolation:

    - ZeroPad: fills with 0 values the frequency bins outside the range.
    - Extend: uses the values at the edges of the spectrum.
    - Error: raises an assertion error if frequency bins outside the saved
      range are requested.

    """

    ZeroPad = auto()
    Extend = auto()
    Error = auto()


class FrequencySpacing(Enum):
    Logarithmic = auto()
    Linear = auto()
    Other = auto()


# ====== Other
class FadeType(Enum):
    Linear = auto()
    Exponential = auto()
    Logarithmic = auto()
    NoFade = auto()
