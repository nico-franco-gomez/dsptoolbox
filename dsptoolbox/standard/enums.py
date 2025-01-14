from numpy.typing import NDArray
import numpy as np
from enum import Enum, auto


class SpectrumMethod(Enum):
    """Methods to compute a spectrum.
    - Welch: produces a spectrum that is averaged over time. If it is the
      autospectrum, it is always real-valued (magnitude or power). If it is
      a cross-spectrum, then it is complex.
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
    - Power (magnitude-squared) scalings always deliver real data, i.e., no
      complex spectra.
    - Amplitude scalings can deliver complex or real (magnitude) spectra.

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
    Zpk = auto()
    Sos = auto()
    Ba = auto()


class FilterType(Enum):
    Iir = auto()
    Fir = auto()
    Biquad = auto()
    Other = auto()


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
    Allpass = auto()
    Notch = auto()
    Inverter = auto()
