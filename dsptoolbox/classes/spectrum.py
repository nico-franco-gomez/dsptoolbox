import numpy as np
from numpy.typing import NDArray, ArrayLike
from copy import deepcopy
from scipy import interpolate as int_sci
from scipy.integrate import trapezoid
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..tools import to_db, fractional_octave_smoothing
from .. import plots
from .signal import Signal
from .filter import Filter
from .filterbank import FilterBank


class Spectrum:
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
        self.set_interpolator_parameters()

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
        scaling = sig._spectrum_parameters["scaling"]
        if complex:
            if scaling is not None:
                assert (
                    "power" not in scaling
                ), "Power scaling can not be used for a complex spectrum"
            assert (
                sig._spectrum_parameters["method"] == "standard"
            ), "Method for obtaining a complex spectrum must be standard"

        f, sp = sig.get_spectrum()
        if complex:
            assert np.iscomplexobj(sp), "Spectrum of signal is not complex"
            return Spectrum(f, sp)

        return Spectrum(f, sp if "amplitude" in scaling else sp**0.5)

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
        mode: str = "sequential",
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
        mode : {"sequential", "parallel", "summed"} str, optional
            Mode for the filter bank. Default: "sequential".
        complex : bool, optional
            When True, the complex spectrum is saved. Otherwise, it is the
            magnitude spectrum. Default: False.

        """
        data = filter_bank.get_transfer_function(frequency_vector_hz, mode)
        return Spectrum(frequency_vector_hz, data if complex else np.abs(data))

    @property
    def frequency_vector_hz(self):
        return self.__frequency_vector_hz

    @frequency_vector_hz.setter
    def frequency_vector_hz(self, new_freqs: NDArray[np.float64]):
        assert not np.iscomplexobj(
            new_freqs
        ), "Complex frequencies are invalid"
        f = np.atleast_1d(new_freqs).astype(np.float64)
        assert f.ndim == 1, "Frequency vector can only have a single dimension"
        assert np.all(f >= 0.0), "Negative frequencies are not supported"
        assert np.all(
            np.ediff1d(f) > 0.0
        ), "Frequency vector is not strictly ascending"
        self.__frequency_vector_type = self.__check_frequency_vector_type(f)
        self.__frequency_vector_hz = f

    @property
    def frequency_vector_type(self) -> str:
        return self.__frequency_vector_type

    @property
    def n_frequency_bins(self) -> int:
        return len(self.frequency_vector_hz)

    @property
    def spectral_data(self) -> NDArray[np.float64 | np.complex128]:
        return self.__spectral_data

    @spectral_data.setter
    def spectral_data(self, new_data: ArrayLike):
        data = np.atleast_2d(new_data)
        assert data.ndim == 2, "Spectral data must have two dimensions"
        if data.shape[0] < data.shape[1]:
            data = data.T
        assert (
            data.shape[0] == self.n_frequency_bins
        ), "Spectral data and frequency vector lengths do not match"
        self.__is_magnitude_spectrum = not np.iscomplexobj(data)
        self.__spectral_data = data.astype(
            np.float64 if self.is_magnitude else np.complex128
        )
        if self.is_magnitude:
            assert np.all(
                self.__spectral_data >= 0.0
            ), "No negative values are allowed for the magnitude spectrum"

    @property
    def number_of_channels(self) -> int:
        return self.__spectral_data.shape[1]

    @property
    def is_magnitude(self) -> bool:
        return self.__is_magnitude_spectrum

    @property
    def has_coherence(self) -> bool:
        return hasattr(self, "coherence")

    @staticmethod
    def __check_frequency_vector_type(f_vec_hz: NDArray[np.float64]) -> str:
        if np.all(
            np.isclose(np.ediff1d(f_vec_hz), f_vec_hz[-1] - f_vec_hz[-2])
        ):
            return "linear"

        if np.all(
            np.isclose(
                f_vec_hz[2:] / f_vec_hz[1:-1], f_vec_hz[-1] / f_vec_hz[-2]
            )
        ):
            return "logarithmic"

        return "other"

    def trim(
        self,
        f_lower_hz: float | None,
        f_upper_hz: float | None,
        inclusive: bool = True,
    ):
        """Trim the spectrum (inplace) to the new boundaries.

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
        self

        """
        s = self.__freqs_to_slice(f_lower_hz, f_upper_hz, inclusive)
        self.frequency_vector_hz = self.frequency_vector_hz[s]
        self.spectral_data = self.spectral_data[s, ...]
        return self

    def resample(self, new_freqs_hz: NDArray[np.float64]):
        """Resample current spectrum (inplace) to new frequency vector. The
        stored interpolation parameters will be used.

        Parameters
        ----------
        new_freqs_hz : NDArray[np.float64]
            New frequency vector.

        Returns
        -------
        self

        """
        new_sp = self.get_interpolated_spectrum(
            new_freqs_hz, "magnitude" if self.is_magnitude else "complex"
        )
        self.frequency_vector_hz = new_freqs_hz
        self.spectral_data = new_sp
        return self

    def get_interpolated_spectrum(
        self, requested_frequency: NDArray[np.float64], output_type: str
    ) -> NDArray:
        """Obtain an interpolated spectrum. Refer to
        `set_interpolator_parameters()` to modify the interpolation.

        Parameters
        ----------
        requested_frequency : NDArray[np.float64], None
            Frequencies to which to interpolate.
        output_type :  str {"power", "magnitude", "complex", "db"}
            Output for the data.

        Returns
        -------
        NDArray

        """
        output_type = output_type.lower()
        assert output_type in (
            "power",
            "magnitude",
            "complex",
            "db",
        ), "Output type is not supported"
        if output_type == "complex":
            assert not self.is_magnitude, "Complex output is not supported"
            assert self.__int_domain in (
                "complex",
                "magphase",
            ), "Interpolation domain must be complex for a complex output"

        inds_outside_left = requested_frequency < self.frequency_vector_hz[0]
        inds_outside_right = requested_frequency > self.frequency_vector_hz[-1]
        if self.__int_edges == "error":
            assert 0 == np.sum(inds_outside_left | inds_outside_right), (
                "Frequencies are not in the given range and edge handling "
                + "does not support it"
            )

        # Input for interpolation depending on domain
        if self.__int_domain == "power":
            if self.is_magnitude:
                interp_data = self.spectral_data**2.0
            else:
                interp_data = np.abs(self.spectral_data) ** 2.0
        elif self.__int_domain == "magnitude":
            if self.is_magnitude:
                interp_data = self.spectral_data
            else:
                interp_data = np.abs(self.spectral_data)
        elif self.__int_domain == "complex":
            interp_data = np.real(self.spectral_data)
            interp_data_imag = np.imag(self.spectral_data)
        elif self.__int_domain == "magphase":
            interp_data = np.abs(self.spectral_data)
            interp_data_imag = np.unwrap(np.angle(self.spectral_data), axis=0)

        # Get edge values
        if self.__int_edges == "zero-pad":
            left_val = right_val = 0.0
        else:  # "extend"
            left_val = interp_data[0, ...]
            right_val = interp_data[-1, ...]

        # Interpolation and filling edges
        if self.__int_scheme != "linear":
            func = (
                int_sci.CubicSpline
                if self.__int_scheme == "cubic"
                else int_sci.PchipInterpolator
            )
            output = func(self.frequency_vector_hz, interp_data, axis=0)(
                requested_frequency
            )
            if self.__int_domain == "complex":
                output = output + 1j * func(
                    self.frequency_vector_hz, interp_data_imag, axis=0
                )(requested_frequency)
            elif self.__int_domain == "magphase":
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
                dtype=(
                    np.float64 if output_type != "complex" else np.complex128
                ),
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
                if self.__int_domain == "complex":
                    output[:, ch] += 1j * np.interp(
                        requested_frequency,
                        self.frequency_vector_hz,
                        interp_data_imag[:, ch],
                        left=left_val,
                        right=right_val,
                    )
                elif self.__int_domain == "magphase":
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
        if output_type == "complex":
            return output
        elif output_type == "db":
            if self.__int_domain in ("complex", "magphase"):
                return to_db(np.abs(output), True)
            return to_db(output, self.__int_domain == "magnitude")
        elif output_type == "power":
            if self.__int_domain in ("complex", "magphase"):
                return np.abs(output) ** 2.0
            elif self.__int_domain == "magnitude":
                return output**2.0
            else:  # "power"
                return output
        elif output_type == "magnitude":
            if self.__int_domain in ("complex", "magphase"):
                return np.abs(output)
            elif self.__int_domain == "magnitude":
                return output
            else:  # "power"
                return output**0.5

        raise ValueError("Some unexpected case happened!")

    def set_interpolator_parameters(
        self,
        domain: str = "power",
        scheme: str = "linear",
        edges_handling: str = "zero-pad",
    ):
        """Set the parameters of the interpolator.

        Parameters
        ----------
        domain : {"power", "magnitude", "complex", "magphase"} str, optional
            Domain to use during the interpolation. Default: "power".
        interpolation_scheme : {"linear", "cubic", "pchip"} str, optional
            Type of interpolation to realize. See notes for details. Default:
            "linear".
        edges_handling : {"zero-pad", "extend", "error"} str, optional
            Type of handling for interpolating outside the saved range. See
            notes for details. Default: "zero-pad".

        Notes
        -----
        - The domain for spectrum interpolation defaults to power. This renders
          the most accurate results, though spectrum interpolation always has
          some error if done directly on the frequency data. Zero-padding or
          computing DFT directly is the correct way of obtaining the values of
          different frequency bins.
        - For complex and magphase domains, the underlying data must be
          complex.
        - The interpolation schemes are:

            - linear: linear interpolation. It is the fastest and most stable.
            - cubic: CubicSplines. It delivers smoother results, but can lead\
              to overshooting and other interpolation artifacts.
            - pchip: PchipInterpolator. It is a polynomial interpolator that\
              avoids overshooting between interpolation points.

        - Handling of edges:
            - zero-pad: fills with 0 values the frequency bins outside the
              range.
            - extend: uses the values at the edges of the spectrum.
            - error: raises an assertion error if frequency bins outside the
              saved range are requested.

        """
        domain = domain.lower()
        assert domain in (
            "power",
            "magnitude",
            "complex",
            "magphase",
        ), "No supported interpolation domain"
        if domain in ("complex", "magphase"):
            assert (
                not self.is_magnitude
            ), "No complex interpolation is possible with this data"
        self.__int_domain = domain

        scheme = scheme.lower()
        assert scheme in (
            "linear",
            "cubic",
            "pchip",
        ), "Invalid interpolation scheme"
        self.__int_scheme = scheme

        edges_handling = edges_handling.lower()
        assert edges_handling in (
            "zero-pad",
            "extend",
            "error",
        ), "Handling of edges is not supported"
        self.__int_edges = edges_handling
        return self

    def get_energy(
        self, f_lower_hz: float | None = None, f_upper_hz: float | None = None
    ) -> NDArray[np.float64]:
        """Integrate the spectrum in order to obtain the total energy in a
        frequency region. Depending on the original scaling, this can represent
        the RMS value of the underlying signal exactly.

        This is computed by using trapezoidal integration across the frequency
        axis.

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

    def apply_octave_smoothing(
        self, octave_fraction: float, window_type="hann"
    ):
        """Apply octave smoothing (inplace) on the spectral data. When complex,
        the smoothing happens on the magnitude and phase representation.
        Otherwise, the smoothing happens on the magnitude spectrum.

        Parameters
        ----------
        octave_fraction : float
            Octave fraction across which to apply smoothing.
        window_type : str, tuple(str, float)
            Type of window to use. Refer to
            `tools.fractional_octave_smoothing()` for details on the available
            windows. Default: "hann".

        Returns
        -------
        self

        Notes
        -----
        - If the frequency vector was other than logarithmic or linear, the
          resulting data will have a linear frequency vector with around
          1 Hz resolution. Interpolation using the saved interpolator
          parameters will be used.

        """
        beta = (
            np.log2(
                self.frequency_vector_hz[-1] / self.frequency_vector_hz[-2]
            )
            if self.frequency_vector_type == "logarithmic"
            else None
        )

        if self.frequency_vector_type in ("linear", "logarithmic"):
            data = self.spectral_data
        else:  # Other: map to linear and interpolate
            data = self.get_interpolated_spectrum(
                np.linspace(
                    self.frequency_vector_hz[0],
                    self.frequency_vector_hz[-1],
                    self.frequency_vector_hz[-1] - self.frequency_vector_hz[0],
                    endpoint=True,
                ),
                "magnitude" if self.is_magnitude else "complex",
            )

        if self.is_magnitude:
            self.spectral_data = fractional_octave_smoothing(
                data, beta, octave_fraction, window_type
            )
            return self

        mag = fractional_octave_smoothing(
            np.abs(data), beta, octave_fraction, window_type
        )
        ph = fractional_octave_smoothing(
            np.unwrap(np.angle(data), axis=0),
            beta,
            octave_fraction,
            window_type,
        )
        self.spectral_data = mag * np.exp(1j * ph)
        return self

    def construct_time_signal(
        self,
        sampling_rate_hz: int,
        min_length_samples: int | None = None,
        magphase_interpolation: bool = True,
    ) -> Signal:
        """Create a time signal from the complex spectrum. Complex
        interpolation will be triggered.

        Parameters
        ----------
        sampling_rate_hz : int
            Sampling rate for the output signal.
        min_length_samples : int, None, optional
            Minimum length for the time signal. If None, it is inferred from
            the average frequency resolution.
        magphase_interpolation : bool, optional
            When True, the interpolation is set to be done on magnitude-phase
            domain. Otherwise, it is on the complex plane. Default: True.

        Returns
        -------
        Signal
            Reconstructed time signal.

        """
        assert not self.is_magnitude, "Not valid for magnitude spectrum"

        if min_length_samples is None:
            if self.frequency_vector_type == "linear":
                df = self.frequency_vector_hz[1] - self.frequency_vector_hz[0]
            else:
                df = np.mean(np.ediff1d(self.frequency_vector_hz))
            f_vec_length = int((sampling_rate_hz / 2) / df + 0.5)
            if f_vec_length % 2 == 0:
                f_vec_length += 1
        else:
            f_vec_length = min_length_samples // 2 + 1
            df = sampling_rate_hz / f_vec_length

        lin_freqs = np.linspace(
            0, sampling_rate_hz / 2, f_vec_length, endpoint=True
        )

        self.set_interpolator_parameters(
            "magphase" if magphase_interpolation else "complex"
        )
        sp = self.get_interpolated_spectrum(lin_freqs, "complex")
        return Signal.from_time_data(
            np.fft.irfft(
                sp,
                axis=0,
                n=max(min_length_samples, (f_vec_length - 1) * 2),
            ),
            sampling_rate_hz,
        )

    def set_coherence(self, coherence: NDArray[np.float64]):
        """Sets the coherence matrix from the transfer function computation.

        Parameters
        ----------
        coherence : NDArray[np.float64]
            Coherence matrix. It must match the shape of the saved spectral
            data.

        """
        assert (
            coherence.shape == self.spectral_data.shape
        ), "Length of signals and given coherence do not match"
        assert not np.iscomplexobj(coherence), "Coherence cannot be complex"
        self.coherence = coherence

    def plot_magnitude(
        self,
        in_db: bool = True,
        normalization: str | None = None,
        dynamic_range_db: float | None = 100.0,
    ):
        """Plot the magnitude spectrum.

        Parameters
        ----------
        in_db : bool, True
            When True, the data is converted to dB.
        normalization : str {"1khz", "max", "energy"}, None, optional
            Type of normalization (per-channel) to apply. Default: None.
        dynamic_range_db : float, None, optional
            Pass a dynamic range in order to constrain the plot. Use None
            to avoid it. Default: 100.

        """
        if normalization is not None:
            normalization = normalization.lower()
            assert normalization in (
                "1khz",
                "max",
                "energy",
            ), "Normalization is invalid"
            if normalization == "1khz":
                norm = self.get_interpolated_spectrum(
                    [1000.0], output_type="magnitude"
                )
            elif normalization == "max":
                norm = (
                    np.max(np.abs(self.spectral_data), axis=0)
                    if not self.is_magnitude
                    else np.max(self.spectral_data, axis=0)
                )
            elif normalization == "energy":
                norm = (self.get_energy() / self.n_frequency_bins) ** 0.5
        else:
            norm = np.ones(self.number_of_channels)

        data = np.abs(self.spectral_data) / norm
        if in_db:
            data = to_db(data, True, dynamic_range_db=dynamic_range_db)
        return plots.general_plot(
            self.frequency_vector_hz,
            data,
            log=True,
            labels=[f"Channel {i}" for i in range(self.number_of_channels)],
            ylabel="Magnitude / " + "dB" if in_db else "1",
            returns=True,
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
            log=True,
            ylabels=[
                rf"$\gamma^2$ Coherence {n}"
                for n in range(self.number_of_channels)
            ],
            range_x=None,
            xlabels="Frequency / Hz",
            range_y=[-0.1, 1.1],
            returns=True,
        )
        return fig, ax

    def copy(self) -> "Spectrum":
        """Copy the spectral data.

        Returns
        -------
        Spectrum

        """
        return deepcopy(self)

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
            else self.n_frequency_bins
        )
        if inclusive:
            ind_high += 1
            ind_high = min(ind_high, self.n_frequency_bins)
        else:
            ind_low -= 1
            ind_low = max(0, ind_low)
        return slice(ind_low, ind_high)