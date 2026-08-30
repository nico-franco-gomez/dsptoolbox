from copy import deepcopy
from pickle import HIGHEST_PROTOCOL, dump

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate as int_sci
from scipy.integrate import trapezoid

from .. import plots
from ..helpers.gain_and_level import from_db, to_db
from ..helpers.other import _check_path_format, _pad_trim
from ..helpers.spectrum_utilities import (
    _get_normalization_offset_db,
    _warp_frequency_vector,
)
from ..standard.enums import (
    FilterBankMode,
    FrequencySpacing,
    InterpolationDomain,
    InterpolationEdgeHandling,
    InterpolationScheme,
    MagnitudeNormalization,
    SpectrumType,
    Window,
    WindowType,
)
from ..tools import fractional_octave_smoothing
from ._multichannel_data import MultichannelData
from .filter import Filter
from .filterbank import FilterBank
from .signal import Signal


class Spectrum(MultichannelData):
    def __init__(
        self,
        frequency_vector_hz: NDArray[np.float64],
        spectral_data: ArrayLike,
    ):
        """Spectrum class. If the data is complex, it is regarded as the
        complex spectrum. Otherwise, it is assumed to be the magnitude (linear)
        spectrum. No rescaling is applied.

        Parameters
        ----------
        frequency_vector_hz : NDArray[np.float64]
            Frequency vector for the spectral data.
        spectral_data : ArrayLike
            Spectral data. It should be broadcastable to a 2D-Array. If its
            data is not complex, it is assumed to be the magnitude spectrum (no
            negative values are supported).

        """
        self.frequency_vector_hz = frequency_vector_hz
        self.spectral_data = spectral_data
        self._set_interpolator_parameters()

    @staticmethod
    def from_signal(sig: Signal, complex: bool = False) -> "Spectrum":
        """Instantiate from the spectrum of a signal. It can be complex or not.
        The spectrum is always generated with the `Signal.get_spectrum()`
        method without changing its parameters.

        Parameters
        ----------
        sig : Signal
            Input signal.
        complex : bool, optional
            When True, the spectral data will be complex. To this end,
            the spectrum of the signal must be complex. An assertion will be
            raised if this is not the case. Default: False.

        """
        if complex:
            assert sig.spectrum_scaling.outputs_complex_spectrum(sig.spectrum_method), (
                "Method or scaling do not deliver a complex spectrum"
            )

        f, sp = sig.get_spectrum()
        if complex:
            assert np.iscomplexobj(sp), "Spectrum of signal is not complex"
            return Spectrum(f, sp)

        return Spectrum(
            f,
            (
                np.abs(sp)
                if sig.spectrum_scaling.is_amplitude_scaling()
                else np.abs(sp) ** 0.5
            ),
        )

    @staticmethod
    def from_filter(
        frequency_vector_hz: NDArray[np.float64],
        filt: Filter,
        complex: bool = False,
    ) -> "Spectrum":
        """Obtain spectrum from computing the filter transfer function.

        Parameters
        ----------
        frequency_vector_hz : NDArray[np.float64]
            Frequency vector to compute.
        filt : Filter
            Filter to obtain the spectrum from.
        complex : bool, optional
            When True, the complex spectrum is saved. Otherwise, it is the
            magnitude spectrum. Default: False.

        """
        data = filt.get_transfer_function(frequency_vector_hz)
        return Spectrum(frequency_vector_hz, data if complex else np.abs(data))

    @staticmethod
    def from_filterbank(
        frequency_vector_hz: NDArray[np.float64],
        filter_bank: FilterBank,
        mode: FilterBankMode,
        complex: bool = False,
    ) -> "Spectrum":
        """Obtain spectrum from computing the transfer function of a filter
        bank.

        Parameters
        ----------
        frequency_vector_hz : NDArray[np.float64]
            Frequency vector to compute.
        filter_bank : FilterBank
            FilterBank to obtain the spectrum from.
        mode : FilterBankMode
            Mode for the filter bank.
        complex : bool, optional
            When True, the complex spectrum is saved. Otherwise, it is the
            magnitude spectrum. Default: False.

        """
        data = filter_bank.get_transfer_function(frequency_vector_hz, mode)
        return Spectrum(frequency_vector_hz, data if complex else np.abs(data))

    @property
    def frequency_vector_hz(self):
        """Get the frequency vector in Hz.

        Returns
        -------
        NDArray[np.float64]
            Frequency vector in Hz. Must be 1D, non-negative, and strictly
            ascending.

        """
        return self.__frequency_vector_hz

    @frequency_vector_hz.setter
    def frequency_vector_hz(self, new_freqs: NDArray[np.float64]):
        """Set the frequency vector.

        Parameters
        ----------
        new_freqs : NDArray[np.float64]
            Frequency vector in Hz. Must be 1D, contain only non-negative values,
            and be strictly ascending (no repeated values).

        Raises
        ------
        AssertionError
            If frequency vector is complex, not 1D, contains negative values,
            or is not strictly ascending.

        Notes
        -----
        The frequency vector type (e.g., linear, logarithmic) is automatically
        determined based on the spacing.

        """
        assert not np.iscomplexobj(new_freqs), "Complex frequencies are invalid"
        f = np.atleast_1d(new_freqs).astype(np.float64)
        assert f.ndim == 1, "Frequency vector can only have a single dimension"
        assert np.all(f >= 0.0), "Negative frequencies are not supported"
        assert np.all(np.ediff1d(f) > 0.0), "Frequency vector is not strictly ascending"
        self.__frequency_vector_type = self.__check_frequency_vector_type(f)
        self.__frequency_vector_hz = f

    @property
    def frequency_vector_type(self) -> FrequencySpacing:
        """Get the type of frequency spacing in the frequency vector.

        Returns
        -------
        FrequencySpacing
            Frequency spacing type (Linear, Logarithmic, or Other).

        """
        return self.__frequency_vector_type

    @property
    def number_frequency_bins(self) -> int:
        """Get the number of frequency bins in the spectrum.

        Returns
        -------
        int
            Number of frequency bins (length of the frequency vector).

        """
        return len(self.frequency_vector_hz)

    @property
    def length_frequency_bins(self) -> int:
        """Get the number of frequency bins in the spectrum.

        Returns
        -------
        int
            Alias for `number_frequency_bins`. Number of frequency bins
            (length of the frequency vector).

        """
        return self.number_frequency_bins

    @property
    def spectral_data(self) -> NDArray[np.float64 | np.complex128]:
        """Get the spectral data.

        Returns
        -------
        NDArray[np.float64 | np.complex128]
            Spectral data as a 2D array with shape (number_of_frequency_bins,
            number_of_channels). Real values represent magnitude spectrum,
            complex values represent complex spectrum.

        """
        return self.__spectral_data

    @spectral_data.setter
    def spectral_data(self, new_data: ArrayLike):
        """Set the spectral data.

        Parameters
        ----------
        new_data : ArrayLike
            Spectral data as a 2D array with shape (number_of_frequency_bins,
            number_of_channels). Data is automatically transposed if needed
            to match the expected shape.

        Raises
        ------
        AssertionError
            If spectral data is not 2D or the number of frequency bins does not
            match the current frequency vector.

        Notes
        -----
        Data must have the same number of frequency bins as the current
        frequency vector.

        """
        data = np.atleast_2d(new_data)
        assert data.ndim == 2, "Spectral data must have two dimensions"
        if data.shape[0] < data.shape[1]:
            data = data.T
        assert data.shape[0] == self.number_frequency_bins, (
            "Spectral data and frequency vector lengths do not match"
        )
        is_magnitude = np.isrealobj(data)
        self.__spectral_data = data.astype(
            np.float64 if is_magnitude else np.complex128
        )
        if self.is_magnitude:
            assert np.all(self.__spectral_data >= 0.0), (
                "No negative values are allowed for the magnitude spectrum"
            )

    @property
    def is_magnitude(self) -> bool:
        """Check if the spectral data is in magnitude form.

        Returns
        -------
        bool
            True if the spectral data contains only real (magnitude) values,
            False if it contains complex values.

        """
        return np.isrealobj(self.__spectral_data)

    @property
    def is_complex(self) -> bool:
        """When True, the spectral data is complex. Counterpart to is_magnitude
        for consistency with Signal class which has is_complex_signal."""
        return not self.is_magnitude

    @property
    def spectrum_type(self) -> SpectrumType:
        """Get the spectrum type (Magnitude or Complex).

        Returns
        -------
        SpectrumType
            Spectrum type indicating whether the data is magnitude or complex.

        """
        if self.is_magnitude:
            return SpectrumType.Magnitude
        return SpectrumType.Complex

    @property
    def has_coherence(self) -> bool:
        """Check if coherence data has been set for this spectrum.

        Returns
        -------
        bool
            True if coherence attribute exists, False otherwise.

        """
        return hasattr(self, "coherence")

    @staticmethod
    def __check_frequency_vector_type(
        f_vec_hz: NDArray[np.float64],
    ) -> FrequencySpacing:
        try:
            if np.all(np.isclose(np.ediff1d(f_vec_hz), f_vec_hz[-1] - f_vec_hz[-2])):
                return FrequencySpacing.Linear

            if np.all(
                np.isclose(f_vec_hz[2:] / f_vec_hz[1:-1], f_vec_hz[-1] / f_vec_hz[-2])
            ):
                return FrequencySpacing.Logarithmic
        except Exception as e:
            print(e)

        return FrequencySpacing.Other

    def to_signal(
        self,
        sampling_rate_hz: int,
        length_seconds: float | None = None,
    ) -> Signal:
        """Convert the current spectrum to a time signal using an inverse
        rFFT. Its data must be complex.

        Parameters
        ----------
        sampling_rate_hz : int
            Requested sampling rate of the time signal.
        length_seconds : float, None, optional
            Length of time signal in seconds. For linearly-spaced data and
            None, it will be inferred from the frequency resolution. For other
            frequency spacings, it is required. Default: None.

        Returns
        -------
        Signal

        Notes
        -----
        - For linearly-spaced frequency data, it will be checked if an
          interpolation is necessary considering the current frequency vector.
          If a length in seconds is requested, the time signal is first
          computed and then zero-padded or trimmed accordingly in case no
          interpolation was needed.
        - Non-linear frequency data will always trigger an interpolation in
          the magnitude and phase domains with zero-padding outside the known
          frequency domain.

        """
        assert not self.is_magnitude, "Spectrum must be complex"

        def __td_from_spec(spec, length_seconds, sampling_rate_hz) -> Signal:
            time_data = np.fft.irfft(spec, axis=0)
            if length_seconds is not None:
                length_samples = int(length_seconds * sampling_rate_hz + 0.5)
                time_data = _pad_trim(time_data, length_samples)
            return Signal.from_time_data(time_data, sampling_rate_hz)

        if self.frequency_vector_type == FrequencySpacing.Linear:
            delta_f = self.frequency_vector_hz[1] - self.frequency_vector_hz[0]
            condition_sampling_rate = (
                abs(sampling_rate_hz / 2 - self.frequency_vector_hz[-1]) > delta_f
            )
            condition_start = not np.isclose(self.frequency_vector_hz[0], 0.0)

            if not (condition_sampling_rate or condition_start):
                return __td_from_spec(
                    self.spectral_data, length_seconds, sampling_rate_hz
                )

            requested_freqs = np.arange(
                0.0, sampling_rate_hz / 2 + delta_f / 2.0, delta_f
            )
        else:
            assert length_seconds is not None, "A length must be provided"

            requested_freqs = np.fft.rfftfreq(
                int(length_seconds * sampling_rate_hz + 0.5),
                1 / sampling_rate_hz,
            )

        working = self.copy()
        working._set_interpolator_parameters(
            InterpolationDomain.MagnitudePhase,
            InterpolationScheme.Pchip,
            InterpolationEdgeHandling.ZeroPad,
        )
        spectrum = working.get_interpolated_spectrum(
            requested_freqs, SpectrumType.Complex
        )

        return __td_from_spec(spectrum, length_seconds, sampling_rate_hz)

    def trim(
        self,
        f_lower_hz: float | None,
        f_upper_hz: float | None,
        inclusive: bool = True,
    ) -> "Spectrum":
        """Return a copy of the spectrum trimmed to the new boundaries.

        Parameters
        ----------
        f_lower_hz : float, None
            Lowest frequency point in Hz.
        f_upper_hz : float, None
            Highest frequency point in Hz.
        inclusive : bool, optional
            When True, the given frequencies are included in the result\
            vector. Default: True.

        Returns
        -------
        Spectrum
            New, trimmed spectrum.

        """
        s = self.__freqs_to_slice(f_lower_hz, f_upper_hz, inclusive)
        new = self.copy()
        new.frequency_vector_hz = new.frequency_vector_hz[s]
        new.spectral_data = new.spectral_data[s, ...]
        return new

    def sum_channels(self, power_sum: bool = True) -> "Spectrum":
        """Sum all channels of the spectrum and return new spectrum with single channel.

        Parameters
        ----------
        power_sum : bool, optional
            When `True`, all channels are summed in their power representation.
            Otherwise, they are summed in either magnitude or complex representation,
            depending on the type of spectral data. Default: True.

        Returns
        -------
        self

        """
        if power_sum:
            return self._create_copy_with_new_data(
                np.sum(
                    np.abs(self.spectral_data) ** 2.0,
                    axis=1,
                    keepdims=True,
                )
                ** 0.5
            )
        return super().sum_channels()

    def resample(self, new_freqs_hz: NDArray[np.float64]) -> "Spectrum":
        """Return a copy of the spectrum resampled to a new frequency vector.
        The stored interpolation parameters will be used.

        Parameters
        ----------
        new_freqs_hz : NDArray[np.float64]
            New frequency vector.

        Returns
        -------
        Spectrum
            New, resampled spectrum.

        """
        new = self.copy()
        new._set_interpolator_parameters(
            (
                InterpolationDomain.Power
                if new.is_magnitude
                else InterpolationDomain.MagnitudePhase
            ),
            new.__int_scheme,
            new.__int_edges,
        )
        new_sp = new.get_interpolated_spectrum(
            new_freqs_hz,
            (SpectrumType.Magnitude if new.is_magnitude else SpectrumType.Complex),
        )
        new.frequency_vector_hz = new_freqs_hz
        new.spectral_data = new_sp
        return new

    def normalize(
        self,
        reference_frequency_hz: float,
        reference_channel: int | None = None,
    ) -> "Spectrum":
        """Return a copy of the spectrum normalized to be 0 dB at the
        specified frequency.

        Parameters
        ----------
        reference_frequency_hz : float
            Frequency at which to normalize.
        reference_channel : int, None, optional
            Reference channel to normalize to. If None, each channel is normalized
            by itself at the reference frequency. Default: None.

        Returns
        -------
        Spectrum
            New, normalized spectrum.

        """
        values = self.get_interpolated_spectrum(
            np.array([reference_frequency_hz]), SpectrumType.Magnitude
        )
        normalization_value = (
            values if reference_channel is None else values[0, reference_channel]
        )
        new = self.copy()
        new.spectral_data = new.spectral_data / normalization_value
        return new

    def apply_gain(self, gain_db: float | NDArray[np.float64]) -> "Spectrum":
        """Return a copy of the spectrum with gain applied.

        Parameters
        ----------
        gain_db : float, NDArray[np.float64]
            Gain to apply. It should be either a single value for all channels or
            a vector with a gain for each channel.

        Returns
        -------
        Spectrum
            New spectrum with gain applied.

        """
        gains = np.atleast_1d(gain_db)
        assert len(gains) == 1 or len(gains) == self.number_of_channels, (
            "Number of gains is not compatible"
        )
        new = self.copy()
        new.spectral_data = new.spectral_data * from_db(gains, True)
        return new

    def get_interpolated_spectrum(
        self,
        requested_frequency: NDArray[np.float64],
        output_type: SpectrumType,
    ) -> NDArray:
        """Obtain an interpolated spectrum. Refer to
        `set_interpolator_parameters()` to modify the interpolation.

        Parameters
        ----------
        requested_frequency : NDArray[np.float64], None
            Frequencies to which to interpolate.
        output_type :  SpectrumType
            Output for the data.

        Returns
        -------
        NDArray

        """
        if output_type == SpectrumType.Complex:
            assert not self.is_magnitude, "Complex output is not supported"

        inds_outside_left = requested_frequency < self.frequency_vector_hz[0]
        inds_outside_right = requested_frequency > self.frequency_vector_hz[-1]
        if self.__int_edges == InterpolationEdgeHandling.Error:
            assert 0 == np.sum(inds_outside_left | inds_outside_right), (
                "Frequencies are not in the given range and edge handling "
                + "does not support it"
            )

        # Input for interpolation depending on domain
        if self.__int_domain == InterpolationDomain.Power:
            if self.is_magnitude:
                interp_data = self.spectral_data**2.0
            else:
                interp_data = np.abs(self.spectral_data) ** 2.0
        elif self.__int_domain == InterpolationDomain.Magnitude:
            if self.is_magnitude:
                interp_data = self.spectral_data
            else:
                interp_data = np.abs(self.spectral_data)
        elif self.__int_domain == InterpolationDomain.Complex:
            interp_data = np.real(self.spectral_data)
            interp_data_imag = np.imag(self.spectral_data)
        elif self.__int_domain == InterpolationDomain.MagnitudePhase:
            interp_data = np.abs(self.spectral_data)
            interp_data_imag = np.unwrap(np.angle(self.spectral_data), axis=0)

        # Get edge values
        if self.__int_edges == InterpolationEdgeHandling.ZeroPad:
            left_val = right_val = 0.0
        elif self.__int_edges == InterpolationEdgeHandling.OnePad:
            left_val = right_val = 1.0
        else:  # "extend"
            left_val = interp_data[0, ...]
            right_val = interp_data[-1, ...]

        # Interpolation and filling edges
        if self.__int_scheme != InterpolationScheme.Linear:
            func = (
                int_sci.CubicSpline
                if self.__int_scheme == InterpolationScheme.Cubic
                else int_sci.PchipInterpolator
            )
            output = func(self.frequency_vector_hz, interp_data, axis=0)(
                requested_frequency
            )
            if self.__int_domain == InterpolationDomain.Complex:
                output = output + 1j * func(
                    self.frequency_vector_hz, interp_data_imag, axis=0
                )(requested_frequency)
            elif self.__int_domain == InterpolationDomain.MagnitudePhase:
                output = output * np.exp(
                    1j
                    * func(self.frequency_vector_hz, interp_data_imag, axis=0)(
                        requested_frequency
                    )
                )

            if len(inds_outside_left) > 0:
                output[inds_outside_left, :] = left_val
            if len(inds_outside_right) > 0:
                output[inds_outside_right, :] = right_val
        else:
            output = np.zeros(
                (len(requested_frequency), self.number_of_channels),
                dtype=(np.complex128 if self.__int_domain.is_complex() else np.float64),
            )
            for ch in range(output.shape[1]):
                output[:, ch] = np.interp(
                    requested_frequency,
                    self.frequency_vector_hz,
                    interp_data[:, ch],
                    left=left_val,
                    right=right_val,
                )
                # Complex handling
                if self.__int_domain == InterpolationDomain.Complex:
                    output[:, ch] += 1j * np.interp(
                        requested_frequency,
                        self.frequency_vector_hz,
                        interp_data_imag[:, ch],
                        left=left_val,
                        right=right_val,
                    )
                elif self.__int_domain == InterpolationDomain.MagnitudePhase:
                    output[:, ch] = output[:, ch] * np.exp(
                        1j
                        * np.interp(
                            requested_frequency,
                            self.frequency_vector_hz,
                            interp_data_imag[:, ch],
                            left=left_val,
                            right=right_val,
                        )
                    )

        # Output type
        if output_type == SpectrumType.Complex:
            return output
        elif output_type == SpectrumType.Db:
            if self.__int_domain.is_complex():
                return to_db(np.abs(output), True)
            return to_db(output, self.__int_domain.is_linear())
        elif output_type == SpectrumType.Power:
            if self.__int_domain.is_complex():
                return np.abs(output) ** 2.0
            elif self.__int_domain.is_linear():
                return output**2.0
            else:  # "power"
                return output
        elif output_type == SpectrumType.Magnitude:
            if self.__int_domain.is_complex():
                return np.abs(output)
            elif self.__int_domain.is_linear():
                return output
            else:  # "power"
                return output**0.5

        raise ValueError("Some unexpected case happened!")

    def _set_interpolator_parameters(
        self,
        domain: InterpolationDomain = InterpolationDomain.Power,
        scheme: InterpolationScheme = InterpolationScheme.Linear,
        edges_handling: InterpolationEdgeHandling = InterpolationEdgeHandling.ZeroPad,
    ) -> None:
        """Set the parameters of the interpolator in place. Private: used by
        `__init__` and internally by methods that already operate on a
        disposable working copy. Public API is `set_interpolator_parameters`.
        """
        if domain in (
            InterpolationDomain.Complex,
            InterpolationDomain.MagnitudePhase,
        ):
            assert not self.is_magnitude, (
                "No complex interpolation is possible with this data"
            )
        self.__int_domain = domain
        self.__int_scheme = scheme
        self.__int_edges = edges_handling

    def set_interpolator_parameters(
        self,
        domain: InterpolationDomain = InterpolationDomain.Power,
        scheme: InterpolationScheme = InterpolationScheme.Linear,
        edges_handling: InterpolationEdgeHandling = InterpolationEdgeHandling.ZeroPad,
    ) -> "Spectrum":
        """Return a copy of the spectrum with new interpolator parameters.

        Parameters
        ----------
        domain : InterpolationDomain, optional
            Domain to use during the interpolation. Default: Power.
        interpolation_scheme : InterpolationScheme, optional
            Type of interpolation to realize. See notes for details. Default:
            Linear.
        edges_handling : InterpolationEdgeHandling, optional
            Type of handling for interpolating outside the saved range. See
            notes for details. Default: ZeroPad.

        Returns
        -------
        Spectrum
            New spectrum with the new interpolator parameters.

        Notes
        -----
        - The domain for spectrum interpolation defaults to power. This renders
          the most accurate results, though spectrum interpolation always has
          some error if done directly on the frequency data. Zero-padding the
          time series or computing DFT directly is the correct way of obtaining
          the values of different frequency bins.

        """
        new = self.copy()
        new._set_interpolator_parameters(domain, scheme, edges_handling)
        return new

    def get_energy(
        self, f_lower_hz: float | None = None, f_upper_hz: float | None = None
    ) -> NDArray[np.float64]:
        """Integrate the spectrum in order to obtain the total energy in a
        frequency region. Depending on the original scaling, this can represent
        the RMS value of the underlying signal exactly.

        This is computed by using trapezoidal integration across the frequency
        axis. The passed frequency limits are always included in the
        computation.

        Parameters
        ----------
        f_lower_hz : float, None, optional
            Lower frequency bound. Pass None to integrate until the last
            available frequency bin. Default: None.
        f_upper_hz : float, None, optional
            Upper frequency bound. Pass None to integrate until the last
            available frequency bin. Default: None.

        Returns
        -------
        NDArray[np.float64]

        """
        region = self.__freqs_to_slice(f_lower_hz, f_upper_hz, True)
        return trapezoid(
            (
                self.spectral_data[region, ...] ** 2.0
                if self.is_magnitude
                else np.abs(self.spectral_data[region, ...]) ** 2.0
            ),
            self.frequency_vector_hz[region],
            axis=0,
        )

    def warp(self, warping_factor: float, sampling_rate_hz: int) -> "Spectrum":
        """Return a copy of the spectrum warped through interpolation with
        the stored interpolation parameters. This is done according to the
        formula shown in [1].

        Parameters
        ----------
        warping_factor : float
            Warping factor between ]-1;1[.
        sampling_rate_hz : int
            Assumed sampling rate while warping. It must be valid for the
            current frequency vector, i.e., no aliasing is to be expected.

        Returns
        -------
        Spectrum
            New, warped spectrum.

        References
        ----------
        - [1]: Germán Ramos, José J. López, Basilio Pueo. Cascaded warped-FIR
          and FIR filter structure for loudspeaker equalization with low
          computational cost requirements. Digital Signal Processing, Volume
          19, Issue 3, 2009, Pages 393-409, ISSN 1051-2004,
          https://doi.org/10.1016/j.dsp.2008.01.003.

        Notes
        -----
        - The formula presented in [1] has been modified with a negative sign for
          lambda in order to match the warping formulation used in this python package.
        - Negative lambda values increase the resolution for lower frequencies, while
          positive values expand higher frequencies.

        """
        if not np.isclose(sampling_rate_hz / 2, self.frequency_vector_hz[-1]):
            assert sampling_rate_hz / 2 >= self.frequency_vector_hz[-1], (
                "Invalid sampling rate for frequency vector"
            )

        new = self.copy()
        new.frequency_vector_hz = _warp_frequency_vector(
            new.frequency_vector_hz, sampling_rate_hz, warping_factor
        )
        return new

    def apply_octave_smoothing(
        self, octave_fraction: float, window_type: WindowType = Window.Hann
    ) -> "Spectrum":
        """Return a copy of the spectrum with octave smoothing applied to
        the spectral data. When complex, the smoothing happens on the
        magnitude and phase representation. Otherwise, the smoothing happens
        on the magnitude spectrum.

        Parameters
        ----------
        octave_fraction : float
            Octave fraction across which to apply smoothing.
        window_type : WindowType, optional
            Type of window to use. Default: Hann.

        Returns
        -------
        Spectrum
            New, smoothed spectrum.

        Notes
        -----
        - If the frequency vector was other than logarithmic or linear, the
          resulting data will have a linear frequency vector with around
          1 Hz resolution. Interpolation using the saved interpolator
          parameters will be used.

        """
        beta = (
            np.log2(self.frequency_vector_hz[-1] / self.frequency_vector_hz[-2])
            if self.frequency_vector_type == FrequencySpacing.Logarithmic
            else None
        )

        new = self.copy()

        if self.frequency_vector_type in (
            FrequencySpacing.Linear,
            FrequencySpacing.Logarithmic,
        ):
            data = self.spectral_data
        else:  # Other: map to linear and interpolate
            linear_freqs = np.linspace(
                self.frequency_vector_hz[0],
                self.frequency_vector_hz[-1],
                max(
                    int(self.frequency_vector_hz[-1] - self.frequency_vector_hz[0]),
                    2,
                ),
                endpoint=True,
            )
            data = self.get_interpolated_spectrum(
                linear_freqs,
                (SpectrumType.Magnitude if self.is_magnitude else SpectrumType.Complex),
            )
            new.frequency_vector_hz = linear_freqs

        if self.is_magnitude:
            new.spectral_data = fractional_octave_smoothing(
                data, beta, octave_fraction, window_type.to_scipy_format()
            )
            return new

        mag = fractional_octave_smoothing(
            np.abs(data), beta, octave_fraction, window_type.to_scipy_format()
        )
        ph = fractional_octave_smoothing(
            np.unwrap(np.angle(data), axis=0),
            beta,
            octave_fraction,
            window_type.to_scipy_format(),
        )
        new.spectral_data = mag * np.exp(1j * ph)
        return new

    def set_coherence(self, coherence: NDArray[np.float64]) -> "Spectrum":
        """Return a copy of the spectrum with the coherence matrix from the
        transfer function computation set.

        Parameters
        ----------
        coherence : NDArray[np.float64]
            Coherence matrix. It must match the shape of the saved spectral
            data.

        Returns
        -------
        Spectrum
            New spectrum with coherence set.

        """
        assert coherence.shape == self.spectral_data.shape, (
            "Length of signals and given coherence do not match"
        )
        assert not np.iscomplexobj(coherence), "Coherence cannot be complex"
        new = self.copy()
        new.coherence = coherence
        return new

    def spectral_difference(
        self,
        other: "Signal | Spectrum",
        octave_fraction_smoothing: float = 0.0,
        energy_normalization: bool = True,
        complex: bool = False,
        dynamic_range_db: float | None = 100.0,
    ) -> "Spectrum":
        """Compute the spectral difference between this and another signal
        or spectrum. Their number of channels must match. It is computed as
        `self / other`.

        Parameters
        ----------
        other : Signal, Spectrum
        octave_fraction_smoothing : float, optional
            Smoothing can be applied prior to computing the difference.
            Default: 0 (no smoothing).
        energy_normalization : bool, optional
            When True, each channel is energy normalized before computing
            the difference. Default: True.
        complex : bool, optional
            When True, the output will be complex. This is only supported if
            the inputs are complex (for signals, the saved spectrum
            parameters must deliver a complex spectrum). Default: False.
        dynamic_range_db : float, None, optional
            Dynamic range in dB to regard when building the difference. Pass
            None to avoid limiting the range. Default: 100.

        Returns
        -------
        Spectrum
            Difference spectrum.

        """
        assert self.number_of_channels == other.number_of_channels, (
            "Number of channels does not match"
        )

        inp1 = self.copy()

        if isinstance(other, Signal):
            inp2 = Spectrum.from_signal(other, complex)
        else:
            if complex:
                assert not other.is_magnitude, "Input data should be complex"
            inp2 = other.copy()

        if energy_normalization:
            inp1.spectral_data = inp1.spectral_data / inp1.get_energy() ** 0.5
            inp2.spectral_data = inp2.spectral_data / inp2.get_energy() ** 0.5

        if octave_fraction_smoothing != 0:
            inp1 = inp1.apply_octave_smoothing(octave_fraction_smoothing)
            inp2 = inp2.apply_octave_smoothing(octave_fraction_smoothing)

        inp2 = inp2.set_interpolator_parameters(
            InterpolationDomain.MagnitudePhase if complex else InterpolationDomain.Power
        )
        mag2 = inp2.get_interpolated_spectrum(
            inp1.frequency_vector_hz,
            SpectrumType.Complex if complex else SpectrumType.Magnitude,
        )

        if dynamic_range_db is not None:
            dynamic_range_factor = from_db(-abs(dynamic_range_db), True)
            mag2 = np.clip(mag2, np.max(mag2, axis=0) * dynamic_range_factor, None)

        inp1.spectral_data = inp1.spectral_data / mag2
        return inp1

    def append_spectra(
        self, others: list["Spectrum"], complex_if_available: bool = True
    ) -> "Spectrum":
        """Return a new spectrum joining this and other spectra by
        appending their channels.

        Parameters
        ----------
        others : list[Spectrum]
            Other spectra to be appended. All will be interpolated to this
            spectrum's frequency vector.
        complex_if_available : bool, optional
            If this spectrum is complex and this is set to True, all other
            spectra are expected to be complex as well and will be appended.
            In any other case, only magnitude spectra will be appended.
            Default: True.

        Returns
        -------
        Spectrum
            New spectrum with all channels.

        """
        spectra = [self] + list(others)
        assert len(spectra) > 1, "There must be at least one other spectrum to join"
        complex_append = complex_if_available and not spectra[0].is_magnitude
        if complex_append:
            assert all([not s.is_magnitude for s in spectra]), (
                "At least one spectrum is not complex"
            )

        total_channels = sum([s.number_of_channels for s in spectra])
        freqs = spectra[0].frequency_vector_hz
        spec = np.zeros(
            (len(freqs), total_channels),
            dtype=np.complex128 if complex_append else np.float64,
        )

        ch_ind = 0
        for s in spectra:
            spec[:, ch_ind : ch_ind + s.number_of_channels] = (
                s.get_interpolated_spectrum(
                    freqs,
                    (
                        SpectrumType.Complex
                        if complex_append
                        else SpectrumType.Magnitude
                    ),
                )
            )
            ch_ind += s.number_of_channels

        return Spectrum(freqs, spec)

    def plot_magnitude(
        self,
        in_db: bool = True,
        normalization: MagnitudeNormalization = MagnitudeNormalization.NoNormalization,
        dynamic_range_db: float | None = None,
    ):
        """Plot the magnitude spectrum.

        Parameters
        ----------
        in_db : bool, True
            When True, the data is converted to dB.
        normalization : MagnitudeNormalization, optional
            Type of normalization (per-channel) to apply. Default:
            NoNormalization.
        dynamic_range_db : float, None, optional
            Pass a dynamic range in order to constrain the plot. Use None
            to avoid it. Default: `None`.

        """
        magnitude = np.abs(self.spectral_data)
        offset_db = _get_normalization_offset_db(
            normalization,
            self.frequency_vector_hz,
            to_db(magnitude, True),
            to_db(np.mean(magnitude**2.0, axis=0), False),
        )
        data = magnitude / from_db(offset_db, True)[None, :]
        if in_db:
            data = to_db(data, True, dynamic_range_db=dynamic_range_db)
        return plots.general_plot(
            self.frequency_vector_hz,
            data,
            log_x=True,
            labels=[f"Channel {i}" for i in range(self.number_of_channels)],
            ylabel="Magnitude / " + "dB" if in_db else "1",
        )

    def plot_coherence(self) -> tuple[Figure, list[Axes]]:
        """Plots coherence. If not available, an attribute error will be
        triggered.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : list of `matplotlib.axes.Axes`
            Axes.

        """
        fig, ax = plots.general_subplots_line(
            x=self.frequency_vector_hz,
            matrix=self.coherence,
            column=True,
            sharey=True,
            log_x=True,
            ylabels=[
                rf"$\gamma^2$ Coherence {n}" for n in range(self.number_of_channels)
            ],
            range_x=None,
            xlabels="Frequency / Hz",
            range_y=[-0.1, 1.1],
        )
        return fig, ax

    def save_spectrum(self, path: str):
        """Saves the Spectrum object as a pickle.

        Parameters
        ----------
        path : str
            Path for the filter to be saved. Use only folder1/folder2/name
            (it can be passed with .pkl at the end or without it).

        """
        _check_path_format(path, "pkl")
        with open(path, "wb") as data_file:
            dump(self, data_file, HIGHEST_PROTOCOL)
        return self

    def copy(self) -> "Spectrum":
        """Copy the spectral data.

        Returns
        -------
        Spectrum

        """
        return deepcopy(self)

    # ======== Multichannel Data Base Class Implementation ====================
    def _get_data(self) -> NDArray[np.float64 | np.complex128]:
        """Get the spectral data for multichannel operations."""
        return self.spectral_data

    def _set_data(self, data: NDArray[np.float64 | np.complex128]) -> None:
        """Set the spectral data for multichannel operations."""
        self.spectral_data = data

    def _create_copy_with_new_data(
        self, data: NDArray[np.float64 | np.complex128]
    ) -> "Spectrum":
        """Create a copy with new spectral data."""
        new_spectrum = Spectrum(self.frequency_vector_hz, data)
        # Copy interpolator parameters
        new_spectrum._set_interpolator_parameters(
            self.__int_domain,
            self.__int_scheme,
            self.__int_edges,
        )
        # Copy coherence if it exists
        if self.has_coherence:
            new_spectrum.coherence = self.coherence
        return new_spectrum

    def _update_state(self) -> None:
        """No state tracking needed for Spectrum."""
        pass

    def __freqs_to_slice(
        self,
        f_lower_hz: float | None,
        f_upper_hz: float | None,
        inclusive: bool,
    ) -> slice:
        """Return a slice of the given frequency boundaries."""
        ind_low = (
            int(np.searchsorted(self.frequency_vector_hz, f_lower_hz))
            if f_lower_hz is not None
            else 0
        )
        ind_high = (
            int(np.searchsorted(self.frequency_vector_hz, f_upper_hz))
            if f_upper_hz is not None
            else self.number_frequency_bins
        )
        if inclusive:
            if f_upper_hz is not None:
                ind_high += 1
                ind_high = min(ind_high, self.number_frequency_bins)
            if f_lower_hz is not None:
                if self.frequency_vector_hz[ind_low] != f_lower_hz:
                    ind_low -= 1
                    ind_low = max(ind_low, 0)
        else:
            if f_lower_hz is not None:
                ind_low += 1
        assert ind_low < ind_high, "Slice is invalid"
        return slice(ind_low, ind_high)
