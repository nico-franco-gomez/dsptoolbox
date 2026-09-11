from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
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

    def fft_norm(self) -> Literal["backward", "forward", "ortho"]:
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
        length_time_data_samples : int
            Length of the time data the spectrum was computed from.
        sampling_rate_hz : int
            Sampling rate in Hz.
        window : NDArray[np.float64], None
            Window that was applied to the time data. None means that no
            window (i.e. a boxcar) was used.

        Returns
        -------
        NDArray[np.float64]
            Scaling factor.

        """
        if self == SpectrumScaling.FFTBackward:
            return np.atleast_1d(1.0)

        if self == SpectrumScaling.FFTForward:
            return np.atleast_1d(1.0 / length_time_data_samples)

        if self == SpectrumScaling.FFTOrthogonal:
            return np.atleast_1d((1.0 / length_time_data_samples) ** 0.5)

        if self.is_spectral_density():
            if window is None:
                factor = (2 / length_time_data_samples / sampling_rate_hz) ** 0.5
            else:
                factor = (
                    2 / np.sum(window**2, axis=0, keepdims=True) / sampling_rate_hz
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
    """Available types:

    - Lowpass
    - Highpass
    - Peaking
    - Lowshelf
    - Highshelf
    - BandpassSkirt
    - BandpassPeak
    - LowpassFirstOrder
    - HighpassFirstOrder
    - AllpassFirstOrder
    - Allpass
    - Notch
    - Inverter
    """

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

    def __str__(self) -> str:
        return self.name.lower()

    def to_str(self) -> str:
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

    def with_extra_parameter(
        self, extra_parameter: float | tuple[float, float]
    ) -> "ParametrizedWindow":
        """Bind an extra parameter to the window, returning a
        `ParametrizedWindow`. Windows that require an extra parameter are:
        - Kaiser
        - KaiserBesselDerived
        - Gaussian
        - GeneralCosine
        - GeneralGaussian (two parameters)
        - GeneralHamming
        - Dpss
        - Chebwin

        Refer to `scipy.signal.windows` for more information.

        Returns
        -------
        ParametrizedWindow
            Window bound to its extra parameter. It exposes the same
            interface as `Window` and can be passed wherever one is expected.

        """
        if not self.needs_extra_parameter():
            raise ValueError(f"{self.name} does not take an extra parameter")
        if self == Window.GeneralGaussian:
            if len(np.atleast_1d(extra_parameter)) != 2:
                raise ValueError("GeneralGaussian requires exactly two parameters")
        return ParametrizedWindow(self, extra_parameter)

    def to_scipy_format(self) -> str:
        """Parse to format for passing to
        `scipy.signal.windows.get_window()`.

        """
        if self.needs_extra_parameter():
            raise ValueError(
                f"{self.name} requires an extra parameter. Pass it with "
                + f"Window.{self.name}.with_extra_parameter(...)"
            )
        return self._scipy_name()

    def _scipy_name(self) -> str:
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

    def __call__(self, n_values: int, symmetric: bool) -> NDArray[np.float64]:
        """Get window values from `scipy.signal.windows.get_window()`."""
        return get_window_scipy(self.to_scipy_format(), n_values, not symmetric)


@dataclass(frozen=True)
class ParametrizedWindow:
    """A `Window` bound to the extra parameter(s) that scipy requires for it.

    Instances are produced by `Window.with_extra_parameter()` and are
    immutable, so binding a parameter never affects other users of the same
    `Window` member. They can be passed anywhere a `Window` is accepted.

    """

    window: Window
    extra_parameter: float | tuple[float, float]

    def needs_extra_parameter(self) -> bool:
        """The parameter is already bound, so this is always True."""
        return True

    def to_scipy_format(self) -> tuple:
        """Parse to format for passing to
        `scipy.signal.windows.get_window()`.

        """
        if self.window == Window.GeneralGaussian:
            assert isinstance(self.extra_parameter, tuple)
            return (
                self.window._scipy_name(),
                self.extra_parameter[0],
                self.extra_parameter[1],
            )
        return (self.window._scipy_name(), self.extra_parameter)

    def __call__(self, n_values: int, symmetric: bool) -> NDArray[np.float64]:
        """Get window values from `scipy.signal.windows.get_window()`."""
        return get_window_scipy(self.to_scipy_format(), n_values, not symmetric)


WindowType = Window | ParametrizedWindow


class WarpingFactor(Enum):
    r"""Factor :math:`\lambda` of the allpass used for frequency warping. A
    negative factor increases the resolution of the lower frequencies at the
    expense of the higher ones, and a positive one does the opposite:

    - Bark, Erb: approximation to the psychoacoustically motivated Bark or
      ERB scale. The factor is obtained from the sampling rate according to
      [1], where the Bark approximation is the more accurate one.
    - BarkInverse, ErbInverse: the dewarping (backwards) stage of the above,
      i.e. the same factor with the opposite sign.
    - Custom: an explicit factor in ]-1; 1[. Bind it with
      `WarpingFactor.Custom.with_factor()`.

    References
    ----------
    - [1]: III, J.O. & Abel, Jonathan. (1999). Bark and ERB Bilinear
      Transforms. Speech and Audio Processing, IEEE Transactions on. 7.
      697 - 708. 10.1109/89.799695.

    """

    Bark = auto()
    BarkInverse = auto()
    Erb = auto()
    ErbInverse = auto()
    Custom = auto()

    def with_factor(self, factor: float) -> "ParametrizedWarpingFactor":
        """Bind an explicit warping factor, returning a
        `ParametrizedWarpingFactor`. Only `Custom` takes one, since the other
        members derive their factor from the sampling rate.

        Parameters
        ----------
        factor : float
            Warping factor. It has to be in the range ]-1; 1[.

        Returns
        -------
        ParametrizedWarpingFactor
            Factor bound to its value. It exposes the same interface as
            `WarpingFactor` and can be passed wherever one is expected.

        """
        if not self.needs_factor():
            raise ValueError(f"{self.name} does not take an explicit factor")
        return ParametrizedWarpingFactor(self, float(factor))

    def needs_factor(self) -> bool:
        """When True, the member requires an explicit factor."""
        return self == WarpingFactor.Custom

    def get_factor(self, sampling_rate_hz: int) -> float:
        """Return the warping factor for a given sampling rate.

        Parameters
        ----------
        sampling_rate_hz : int
            Sampling rate to assume while warping.

        Returns
        -------
        float
            Warping factor in ]-1; 1[.

        """
        if self.needs_factor():
            raise ValueError(
                f"{self.name} requires an explicit factor. Pass it with "
                + f"WarpingFactor.{self.name}.with_factor(...)"
            )
        if self in (WarpingFactor.Bark, WarpingFactor.BarkInverse):
            # Eq. (26)
            factor = -1.0 * (
                1.0674 * (2.0 / np.pi * np.arctan(0.06583 * sampling_rate_hz)) ** 0.5
                - 0.1916
            )
        else:
            # Eq. (30)
            factor = -1.0 * (
                0.7446 * (2.0 / np.pi * np.arctan(0.1418 * sampling_rate_hz)) ** 0.5
                + 0.03237
            )
        return (
            -factor
            if self in (WarpingFactor.BarkInverse, WarpingFactor.ErbInverse)
            else factor
        )


@dataclass(frozen=True)
class ParametrizedWarpingFactor:
    """A `WarpingFactor` bound to an explicit value.

    Instances are produced by `WarpingFactor.Custom.with_factor()` and are
    immutable. They can be passed anywhere a `WarpingFactor` is accepted.

    """

    warping_factor: WarpingFactor
    factor: float

    def __post_init__(self) -> None:
        if not abs(self.factor) < 1.0:
            raise ValueError("Warping factor has to be in ]-1; 1[")

    def needs_factor(self) -> bool:
        """The factor is already bound, so this is always False."""
        return False

    def get_factor(self, sampling_rate_hz: int) -> float:
        """Return the bound warping factor. The sampling rate is ignored."""
        return float(self.factor)


WarpingFactorType = WarpingFactor | ParametrizedWarpingFactor


class SampleFormat(Enum):
    """Representations for audio samples. `Int` is a signed integer, `Uint`
    an unsigned one, and the number is the bit depth. The 24-bit formats have
    no numpy equivalent, so they are only available as byte arrays with
    3-byte samples and the endianness of the current platform.

    """

    Float32 = auto()
    Float64 = auto()
    Int8 = auto()
    Int16 = auto()
    Int24 = auto()
    Int32 = auto()
    Uint8 = auto()
    Uint16 = auto()
    Uint24 = auto()
    Uint32 = auto()

    def is_float(self) -> bool:
        """When True, samples are floating-point values in [-1; 1]."""
        return self in (SampleFormat.Float32, SampleFormat.Float64)

    def is_signed(self) -> bool:
        """When True, the format is signed."""
        return not self.name.startswith("Uint")

    def bit_depth(self) -> int:
        """Number of bits per sample."""
        return int(
            self.name.removeprefix("Float").removeprefix("Uint").removeprefix("Int")
        )

    def to_numpy_dtype(self) -> type:
        """Equivalent numpy data type."""
        if self.is_float():
            return np.float32 if self == SampleFormat.Float32 else np.float64
        bits = self.bit_depth()
        if bits == 24:
            raise ValueError(f"{self.name} has no numpy data type equivalent")
        signed = {8: np.int8, 16: np.int16, 32: np.int32}
        unsigned = {8: np.uint8, 16: np.uint16, 32: np.uint32}
        return signed[bits] if self.is_signed() else unsigned[bits]


class MagnitudeNormalization(Enum):
    """Normalization for magnitude responses:

    - NoNormalization.
    - OneKhz: @ 1 kHz for each channel.
    - Max: @ peak.
    - Energy: use average energy (per frequency) as normalization value.

    All variants exist either for the first channel, thus taking the same
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


class InterpolationKind(Enum):
    """Kinds of interpolation for a frequency response, as passed to
    `scipy.interpolate.interp1d`. Quadratic and cubic are splines of the
    respective order. They deliver smoother results than the linear
    interpolation, but can overshoot.

    """

    Linear = auto()
    Quadratic = auto()
    Cubic = auto()

    def to_scipy_str(self) -> str:
        """Return the scipy string variant."""
        return self.name.lower()


class InterpolationConversion(Enum):
    """Representation to convert to for the interpolation of a frequency
    response, and back afterwards. `DbToPower` means input in dB,
    interpolation on the power spectrum and output in dB again.

    """

    DbToAmplitude = auto()
    DbToPower = auto()
    AmplitudeToDb = auto()
    AmplitudeToPower = auto()
    PowerToDb = auto()
    PowerToAmplitude = auto()

    def input_is_db(self) -> bool:
        """When True, the input is expected in dB."""
        return self in (
            InterpolationConversion.DbToAmplitude,
            InterpolationConversion.DbToPower,
        )


class InterpolationEdgeHandling(Enum):
    """Handling of edges during interpolation:

    - ZeroPad: fills with 0 values the frequency bins outside the range.
    - OnePad: fills with 1 values the frequency bins outside the range.
    - Extend: uses the values at the edges of the spectrum.
    - Error: raises an assertion error if frequency bins outside the saved
      range are requested.

    """

    ZeroPad = auto()
    OnePad = auto()
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


class SpectrumAverageMethod(Enum):
    """Statistic used to average the periodograms of Welch's method:

    - Mean.
    - Median: more robust against outlying frames, at the cost of a bias that
      this implementation corrects for.

    """

    Mean = auto()
    Median = auto()


class IrLatencyRemoval(Enum):
    """Way of estimating the latency of an impulse response so that it can be
    removed from its phase response:

    - NoRemoval: the phase response is left untouched.
    - Peak: the position of the peak in the time signal.
    - MinimumPhase: the delay in relation to the minimum-phase equivalent.
    - Custom: an explicit delay in samples. Bind it with
      `IrLatencyRemoval.Custom.with_delay_samples()`.

    """

    NoRemoval = auto()
    Peak = auto()
    MinimumPhase = auto()
    Custom = auto()

    def with_delay_samples(
        self, delay_samples: float | ArrayLike
    ) -> "ParametrizedIrLatencyRemoval":
        """Bind an explicit delay to the latency removal, returning a
        `ParametrizedIrLatencyRemoval`. Only `Custom` takes one, since the
        other members estimate the delay from the impulse response.

        Parameters
        ----------
        delay_samples : float, ArrayLike
            Delay in samples. A single value is applied to every channel, and
            an array-like is expected to hold one delay per channel. It can
            be fractional.

        Returns
        -------
        ParametrizedIrLatencyRemoval
            Latency removal bound to its delay. It can be passed wherever an
            `IrLatencyRemoval` is expected.

        """
        if not self.needs_delay():
            raise ValueError(f"{self.name} does not take an explicit delay")
        delays = np.atleast_1d(np.asarray(delay_samples, dtype=np.float64))
        if delays.ndim != 1:
            raise ValueError("Delays can only have one dimension")
        return ParametrizedIrLatencyRemoval(self, tuple(delays))

    def needs_delay(self) -> bool:
        """When True, the member requires an explicit delay."""
        return self == IrLatencyRemoval.Custom


@dataclass(frozen=True)
class ParametrizedIrLatencyRemoval:
    """An `IrLatencyRemoval` bound to an explicit delay in samples.

    Instances are produced by `IrLatencyRemoval.Custom.with_delay_samples()`
    and are immutable.

    """

    latency_removal: IrLatencyRemoval
    delay_samples: tuple[float, ...]

    def needs_delay(self) -> bool:
        """The delay is already bound, so this is always False."""
        return False

    def get_delay_samples(self, number_of_channels: int) -> NDArray[np.float64]:
        """Return the bound delay as one value per channel.

        Parameters
        ----------
        number_of_channels : int
            Number of channels that the delay is applied to.

        Returns
        -------
        NDArray[np.float64]
            Delay in samples with length `number_of_channels`.

        """
        delays = np.asarray(self.delay_samples, dtype=np.float64)
        if len(delays) == 1:
            return np.repeat(delays, number_of_channels)
        if len(delays) != number_of_channels:
            raise ValueError(
                f"{len(delays)} delays do not match {number_of_channels} channels"
            )
        return delays


IrLatencyRemovalType = IrLatencyRemoval | ParametrizedIrLatencyRemoval


class Power2Rounding(Enum):
    """Rounding towards a power of 2:

    - Closest.
    - Floor: the next smaller power of 2.
    - Ceil: the next larger power of 2.

    """

    Closest = auto()
    Floor = auto()
    Ceil = auto()
