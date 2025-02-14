"""
Contains Filter class
"""

from pickle import dump, HIGHEST_PROTOCOL
from warnings import warn
from copy import deepcopy
import numpy as np
from fractions import Fraction
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scipy.signal as sig
from numpy.typing import NDArray, ArrayLike

from .signal import Signal
from .impulse_response import ImpulseResponse
from .filter_helpers import (
    _biquad_coefficients,
    _impulse,
    _group_delay_filter,
    _filter_on_signal,
    _filter_on_signal_ba,
    _filter_and_downsample,
    _filter_and_upsample,
)
from .plots import _zp_plot
from ..plots import general_plot
from ..helpers.other import _pad_trim, _check_format_in_path
from ..helpers.gain_and_level import to_db
from ..standard.enums import (
    FilterCoefficientsType,
    BiquadEqType,
    FilterPassType,
    IirDesignMethod,
    Window,
    MagnitudeNormalization,
)


class Filter:
    """Class for creating and storing linear digital filters with all their
    metadata.

    """

    # ======== Constructor and initializers ===================================
    def __init__(
        self,
        filter_coefficients: dict,
        sampling_rate_hz: int,
    ):
        """The Filter class contains all parameters and metadata needed for
        using a digital filter.

        Parameters
        ----------
        filter_coefficients : dict
            Dictionary containing configuration for the filter. The dictionary
            must exclusively contain one of the following keys:
            - FilterCoefficientsType.Zpk
            - FilterCoefficientsType.Sos
            - FilterCoefficientsType.Ba

        sampling_rate_hz : int
            Sampling rate in Hz for the digital filter.

        """
        self.warning_if_complex = True
        self.sampling_rate_hz = sampling_rate_hz
        assert (
            (FilterCoefficientsType.Ba in filter_coefficients)
            ^ (FilterCoefficientsType.Sos in filter_coefficients)
            ^ (FilterCoefficientsType.Zpk in filter_coefficients)
        ), (
            "Only (and at least) one type of filter coefficients "
            + "should be passed to create a filter"
        )
        if FilterCoefficientsType.Zpk in filter_coefficients:
            self.zpk = filter_coefficients[FilterCoefficientsType.Zpk]
            self.sos = sig.zpk2sos(*self.zpk, analog=False)
        elif FilterCoefficientsType.Sos in filter_coefficients:
            self.sos = filter_coefficients[FilterCoefficientsType.Sos]
        elif FilterCoefficientsType.Ba in filter_coefficients:
            b, a = filter_coefficients[FilterCoefficientsType.Ba]
            self.ba = [np.atleast_1d(b), np.atleast_1d(a)]

    @staticmethod
    def iir_filter(
        order: int,
        frequency_hz: float | ArrayLike,
        type_of_pass: FilterPassType,
        sampling_rate_hz: int,
        filter_design_method: IirDesignMethod = IirDesignMethod.Butterworth,
        passband_ripple_db: float | None = None,
        stopband_attenuation_db: float | None = None,
    ) -> "Filter":
        """Return an IIR filter using `scipy.signal.iirfilter`. IIR filters are
        always implemented as SOS by default.

        Parameters
        ----------
        order : int
            Filter order.
        frequency_hz : float | ArrayLike
            Frequency or frequencies of the filter in Hz.
        type_of_pass : FilterPassType
            Type of pass.
        sampling_rate_hz : int
            Sampling rate in Hz.
        filter_design_method : IirDesignMethod, optional
            Design method for the IIR filter. Default: Butterworth.
        passband_ripple_db : float, None, optional
            Passband ripple in dB for "cheby1" and "ellip". Default: None.
        stopband_attenuation_db : float, None, optional
            Minimum stopband attenutation in dB for "cheby2" and "ellip".
            Default: None.

        Returns
        -------
        Filter

        """
        zpk = sig.iirfilter(
            N=order,
            Wn=frequency_hz,
            btype=type_of_pass.to_str(),
            analog=False,
            fs=sampling_rate_hz,
            ftype=filter_design_method.to_scipy_str(),
            rp=passband_ripple_db,
            rs=stopband_attenuation_db,
            output="zpk",
        )
        return Filter(
            {FilterCoefficientsType.Zpk: zpk},
            sampling_rate_hz,
        )

    @staticmethod
    def biquad(
        eq_type: BiquadEqType,
        frequency_hz: float | ArrayLike,
        gain_db: float,
        q: float,
        sampling_rate_hz: int,
    ) -> "Filter":
        """Return a biquad filter according to [1].

        Parameters
        ----------
        eq_type : BiquadEqType
            EQ type.
        frequency_hz : float
            Frequency of the biquad in Hz.
        gain_db : float
            Gain of biquad in dB.
        q : float
            Quality factor.
        sampling_rate_hz : int
            Sampling rate in Hz.

        Returns
        -------
        Filter

        Reference
        ---------
        - [1]: https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-
            cookbook.html.

        """
        return Filter(
            {
                FilterCoefficientsType.Ba: _biquad_coefficients(
                    eq_type=eq_type,
                    frequency_hz=frequency_hz,
                    gain_db=gain_db,
                    q=q,
                    fs_hz=sampling_rate_hz,
                )
            },
            sampling_rate_hz,
        )

    @staticmethod
    def fir_filter(
        order: int,
        frequency_hz: float | ArrayLike,
        type_of_pass: FilterPassType,
        sampling_rate_hz: int,
        window: Window = Window.Hamming,
    ) -> "Filter":
        """Design an FIR filter using `scipy.signal.firwin`.

        Parameters
        ----------
        order : int
            Filter order. It corresponds to the number of taps - 1.
        frequency_hz : float | ArrayLike
            Frequency or frequencies of the filter in Hz.
        type_of_pass : FilterPassType
            Type of filter pass.
        sampling_rate_hz : int
            Sampling rate in Hz.
        window : Window, optional
            Window to apply to the FIR filter. Default: Hamming.

        Returns
        -------
        Filter

        """
        return Filter(
            {
                FilterCoefficientsType.Ba: [
                    sig.firwin(
                        numtaps=order + 1,
                        cutoff=frequency_hz,
                        window=(
                            window.to_scipy_format()
                            if window is not None
                            else Window.Hamming.to_scipy_format()
                        ),
                        pass_zero=type_of_pass.to_str(),
                        fs=sampling_rate_hz,
                    ),
                    np.asarray([1.0]),
                ]
            },
            sampling_rate_hz,
        )

    @staticmethod
    def from_ba(
        b: ArrayLike,
        a: ArrayLike,
        sampling_rate_hz: int,
    ) -> "Filter":
        """Create a filter from some b (numerator) and a (denominator)
        coefficients.

        Parameters
        ----------
        b : ArrayLike
            Numerator coefficients.
        a : ArrayLike
            Denominator coefficients.
        sampling_rate_hz : int
            Sampling rate in Hz.

        Returns
        -------
        Filter

        """
        return Filter({FilterCoefficientsType.Ba: [b, a]}, sampling_rate_hz)

    @staticmethod
    def from_sos(
        sos: NDArray[np.float64],
        sampling_rate_hz: int,
    ) -> "Filter":
        """Create a filter from second-order sections.

        Parameters
        ----------
        sos : NDArray[np.float64]
            Second-order sections.
        sampling_rate_hz : int
            Sampling rate in Hz.

        Returns
        -------
        Filter

        """
        return Filter({FilterCoefficientsType.Sos: sos}, sampling_rate_hz)

    @staticmethod
    def from_zpk(
        z: NDArray[np.float64],
        p: NDArray[np.float64],
        k: float,
        sampling_rate_hz: int,
    ) -> "Filter":
        """Create a filter from zero-pole representation.

        Parameters
        ----------
        z : NDArray[np.float64]
            Zeros
        p : NDArray[np.float64]
            Poles
        k : float
            Gain
        sampling_rate_hz : int
            Sampling rate in Hz.

        Returns
        -------
        Filter

        """
        return Filter(
            {FilterCoefficientsType.Zpk: [z, p, k]}, sampling_rate_hz
        )

    @staticmethod
    def fir_from_file(path: str, channel: int = 0) -> "Filter":
        """Read an FIR filter from an audio file.

        Parameters
        ----------
        path : str
            Path to audio file. It will be read using Signal.from_file().
        channel : int, optional
            Channel to take from the audio file for the FIR filter. Default: 0.

        Returns
        -------
        Filter
            FIR filter.

        """
        ir = ImpulseResponse.from_file(path)
        return Filter.from_ba(
            ir.time_data[:, channel], [1.0], ir.sampling_rate_hz
        )

    # ================
    def initialize_zi(self, number_of_channels: int = 1):
        """Initializes zi for steady-state filtering. The number of parallel
        zi's can be defined externally.

        Parameters
        ----------
        number_of_channels : int, optional
            Number of channels is needed for the number of filter's zi's.
            Default: 1.

        """
        assert (
            number_of_channels > 0
        ), """Zi's have to be initialized for at least one channel"""
        self.zi = []
        if hasattr(self, "sos"):
            for _ in range(number_of_channels):
                self.zi.append(sig.sosfilt_zi(self.sos))
        else:
            for _ in range(number_of_channels):
                self.zi.append(sig.lfilter_zi(self.ba[0], self.ba[1]))

        return self

    @property
    def metadata(self) -> dict:
        """Get a dictionary with metadata about the filter properties."""
        info: dict = {}
        info["order"] = self.order
        info["sampling_rate_hz"] = self.sampling_rate_hz
        info["filter_type"] = "iir" if self.is_iir else "fir"
        info["has_sos"] = self.has_sos
        info["has_zpk"] = self.has_zpk
        return info

    @property
    def metadata_str(self) -> str:
        """Get a string with metadata about the filter properties."""
        txt = """Filter:\n"""
        temp = ""
        for _ in range(len(txt)):
            temp += "-"
        txt += temp + "\n"
        metadata = self.metadata
        for k in metadata:
            if k == "ba":
                continue
            txt += f"""{str(k).replace("_", " ").
                        capitalize()}: {metadata[k]}\n"""
        return txt

    @property
    def sampling_rate_hz(self):
        return self.__sampling_rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, new_sampling_rate_hz):
        assert (
            new_sampling_rate_hz is not None
        ), "Sampling rate can not be None"
        assert (
            type(new_sampling_rate_hz) is int
        ), "Sampling rate can only be an integer"
        self.__sampling_rate_hz = new_sampling_rate_hz

    @property
    def warning_if_complex(self):
        return self.__warning_if_complex

    @warning_if_complex.setter
    def warning_if_complex(self, new_warning):
        assert (
            type(new_warning) is bool
        ), "This attribute must be of boolean type"
        self.__warning_if_complex = new_warning

    @property
    def is_iir(self) -> bool:
        if self.has_sos:
            return True

        a = self.ba[1]
        return not (len(a) == 1 and a[0] == 1.0)

    @property
    def is_fir(self) -> bool:
        return not self.is_iir

    @property
    def ba(self) -> list[NDArray[np.float64 | np.complex128]]:
        return self.__ba

    @ba.setter
    def ba(self, new_ba: tuple | list):
        ba: list[NDArray] = list(new_ba)
        assert len(ba) == 2, "ba coefficients must be a list of length two"
        for ind in range(len(ba)):
            coeff = np.atleast_1d(ba[ind])
            assert coeff.ndim == 1
            if np.issubdtype(coeff.dtype, np.complexfloating):
                coeff = coeff.astype(np.complex128)
            else:
                coeff = coeff.astype(np.float64)
            ba[ind] = coeff

        # Check lengths while trimming
        b, a = ba
        # Trim zeros for a
        a = np.atleast_1d(np.trim_zeros(a.copy(), "b"))
        # Change to FIR and normalize if only one a coefficient
        if len(a) == 1:
            b /= a[0]
            a = a / a[0]
            self.__ba = [b, a]
        else:
            self.__ba = ba

    @property
    def sos(self) -> NDArray[np.float64 | np.complex128]:
        return self.__sos

    @sos.setter
    def sos(self, sos):
        assert isinstance(sos, np.ndarray)
        assert sos.ndim == 2
        assert sos.shape[1] == 6
        self.__sos = sos

    @property
    def has_sos(self) -> bool:
        return hasattr(self, "sos")

    @property
    def has_zpk(self) -> bool:
        return hasattr(self, "zpk")

    @property
    def zpk(self) -> list:
        return self.__zpk

    @zpk.setter
    def zpk(self, new_zpk):
        self.__zpk = list(new_zpk)

    @property
    def order(self):
        if hasattr(self, "zpk"):
            return max(len(self.zpk[0]), len(self.zpk[1]))
        if hasattr(self, "sos"):
            n_first_order_sos = np.sum(
                (self.sos[:, 2] == 0.0) & (self.sos[:, 5] == 0.0)
            )
            return self.sos.shape[0] * 2 - n_first_order_sos
        if hasattr(self, "ba"):
            return max(len(self.ba[0]), len(self.ba[1])) - 1
        raise ValueError("No order found")

    def __len__(self):
        return self.order + 1

    def __str__(self):
        return self.metadata_str

    # ======== Filtering ======================================================
    def filter_signal(
        self,
        signal: Signal,
        channels=None,
        activate_zi: bool = False,
        zero_phase: bool = False,
    ) -> Signal:
        """Takes in a `Signal` object and filters selected channels. Exports a
        new `Signal` object.

        Parameters
        ----------
        signal : `Signal`
            Signal to be filtered.
        channels : int or array-like, optional
            Channel or array of channels to be filtered. When `None`, all
            channels are filtered. If only some channels are selected, these
            will be filtered and the others will be bypassed (and returned).
            Default: `None`.
        activate_zi : int, optional
            Gives the zi to update the filter values. Default: `False`.
        zero_phase : bool, optional
            Uses zero-phase filtering on signal. Be aware that the filter
            is applied twice in this case. Default: `False`.

        Returns
        -------
        new_signal : `Signal`
            New Signal object.

        """
        # Check sampling rates
        assert (
            self.sampling_rate_hz == signal.sampling_rate_hz
        ), "Sampling rates do not match"
        # Zero phase and zi
        assert not (activate_zi and zero_phase), (
            "Filter initial and final values cannot be updated when "
            + "filtering with zero-phase"
        )
        # Channels
        if channels is None:
            channels = np.arange(signal.number_of_channels)
        else:
            channels = np.squeeze(channels)
            channels = np.atleast_1d(channels)
            assert (
                channels.ndim == 1
            ), "channels can be only a 1D-array or an int"
            assert all(channels < signal.number_of_channels), (
                f"Selected channels ({channels}) are not valid for the "
                + f"signal with {signal.number_of_channels} channels"
            )

        # Zi – create always for all channels and selected channels will get
        # updated while filtering
        if activate_zi:
            if not hasattr(self, "zi"):
                self.initialize_zi(signal.number_of_channels)
            if len(self.zi) != signal.number_of_channels:
                warn(
                    "zi values of the filter have not been correctly "
                    + "intialized for the number of channels. They have now"
                    + " been corrected"
                )
                self.initialize_zi(signal.number_of_channels)
            zi_old = self.zi
        else:
            zi_old = None

        # Check filter length compared to signal
        if self.order > signal.time_data.shape[0]:
            warn(
                "Filter is longer than signal, results might be "
                + "meaningless!"
            )

        # Filter with SOS when possible
        if hasattr(self, "sos"):
            new_signal, zi_new = _filter_on_signal(
                signal=signal,
                sos=self.sos,
                channels=channels,
                zi=zi_old,
                zero_phase=zero_phase,
                warning_on_complex_output=self.warning_if_complex,
            )
        else:
            # Filter with ba
            new_signal, zi_new = _filter_on_signal_ba(
                signal=signal,
                ba=self.ba,
                channels=channels,
                zi=zi_old,
                zero_phase=zero_phase,
                is_fir=self.is_fir,
                warning_on_complex_output=self.warning_if_complex,
            )
        if activate_zi:
            self.zi = zi_new
        return new_signal

    def filter_and_resample_signal(
        self, signal: Signal, new_sampling_rate_hz: int
    ) -> Signal:
        """Filters and resamples signal. This is only available for all
        channels and sampling rates that are achievable by (only) down- or
        upsampling. This method is for allowing specific filters to be
        decimators/interpolators. If you just want to resample a signal,
        use the function in the standard module.

        If this filter is iir, standard resampling is applied. If it is
        fir, an efficient polyphase representation will be used.

        NOTE: Beware that no additional lowpass filter is used in the
        resampling step which can lead to aliases or other effects if this
        Filter is not adequate!

        Parameters
        ----------
        signal : `Signal`
            Signal to be filtered and resampled.
        new_sampling_rate_hz : int
            New sampling rate to resample to.

        Returns
        -------
        new_sig : `Signal`
            New down- or upsampled signal.

        """
        fraction = Fraction(
            new_sampling_rate_hz, signal.sampling_rate_hz
        ).as_integer_ratio()
        assert fraction[0] == 1 or fraction[1] == 1, (
            f"{new_sampling_rate_hz} is not valid because it needs down- "
            + f"AND upsampling (Up/Down: {fraction[0]}/{fraction[1]})"
        )

        # Check if standard or polyphase representation is to be used
        if self.is_fir:
            polyphase = True
        else:
            if not hasattr(self, "ba"):
                self.ba: list = list(sig.sos2tf(self.sos))
            polyphase = False

        # Check if down- or upsampling is required
        if fraction[0] == 1:
            assert (
                signal.sampling_rate_hz == self.sampling_rate_hz
            ), "Sampling rates do not match"
            new_time_data = _filter_and_downsample(
                time_data=signal.time_data,
                down_factor=fraction[1],
                ba_coefficients=self.ba,
                polyphase=polyphase,
            )
        elif fraction[1] == 1:
            assert (
                signal.sampling_rate_hz * fraction[0] == self.sampling_rate_hz
            ), (
                "Sampling rates do not match. For the upsampler, the "
                + """sampling rate of the filter should match the output's"""
            )
            new_time_data = _filter_and_upsample(
                time_data=signal.time_data,
                up_factor=fraction[0],
                ba_coefficients=self.ba,
                polyphase=polyphase,
            )

        new_sig = signal.copy_with_new_time_data(new_time_data)
        new_sig.sampling_rate_hz = new_sampling_rate_hz
        return new_sig

    # ======== Getters ========================================================
    def get_ir(
        self, length_samples: int = 512, zero_phase: bool = False
    ) -> ImpulseResponse:
        """Gets an impulse response of the filter with given length.

        Parameters
        ----------
        length_samples : int, optional
            Length for the impulse response in samples. Default: 512.

        Returns
        -------
        ir_filt : `ImpulseResponse`
            Impulse response of the filter.

        """
        # FIR with no zero phase filtering
        if self.is_fir and not zero_phase:
            b = self.ba[0].copy()
            if length_samples < len(b):
                warn(
                    f"{length_samples} is not enough for filter with "
                    + f"length {len(b)}. IR will have the latter length."
                )
                length_samples = len(b)
            b = _pad_trim(b, length_samples)
            return ImpulseResponse(
                None, b, self.sampling_rate_hz, constrain_amplitude=False
            )

        # IIR or zero phase IR
        ir_filt = _impulse(length_samples)
        ir_filt = ImpulseResponse(
            None,
            ir_filt,
            self.sampling_rate_hz,
            constrain_amplitude=False,
        )
        return self.filter_signal(ir_filt, zero_phase=zero_phase)

    def get_transfer_function(
        self, frequency_vector_hz: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Obtain the complex transfer function of the filter analytically
        evaluated for a given frequency vector.

        Parameters
        ----------
        frequency_vector_hz : NDArray[np.float64]
            Frequency vector for which to compute the transfer function

        Returns
        -------
        NDArray[np.complex128]
            Complex transfer function

        Notes
        -----
        - This method uses scipy's freqz to compute the transfer function. In
          the case of FIR filters, it might be significantly faster and more
          precise to use a direct FFT approach.

        """
        assert (
            frequency_vector_hz.ndim == 1
        ), "Frequency vector can only have one dimension"
        assert (
            frequency_vector_hz.max() <= self.sampling_rate_hz / 2
        ), "Queried frequency vector has values larger than nyquist"

        if self.is_iir and hasattr(self, "sos"):
            return sig.sosfreqz(
                self.sos, frequency_vector_hz, fs=self.sampling_rate_hz
            )[1]

        # IIR ba and FIR
        return sig.freqz(
            self.ba[0],
            self.ba[1],
            frequency_vector_hz,
            fs=self.sampling_rate_hz,
        )[1]

    def get_group_delay(
        self, frequency_vector_hz: NDArray[np.float64], in_seconds: bool = True
    ) -> NDArray[np.float64]:
        """Obtain the group delay of the filter using
        `scipy.signal.group_delay`. To this end, filter coefficients in ba-form
        are always used. This could lead to numerical imprecisions in case the
        filter has a large order.

        Parameters
        ----------
        frequency_vector_hz : NDArray[np.float64]
            Frequency vector for which to compute the group delay.
        in_seconds : bool, optional
            When True, the output is given in seconds. Otherwise it is in
            samples. Default: True.

        Returns
        -------
        group_delay : NDArray[np.float64]
            Group delay with shape (frequency).

        """
        ba = self.get_coefficients(FilterCoefficientsType.Ba)
        gd = sig.group_delay(
            ba, w=frequency_vector_hz, fs=self.sampling_rate_hz
        )[1]
        return gd / self.sampling_rate_hz if in_seconds else gd

    def get_coefficients(self, coefficients_mode: FilterCoefficientsType):
        """Return a copy of the filter coefficients.

        Parameters
        ----------
        coefficients_mode : FilterCoefficients
            Type of filter coefficients to be returned.

        Returns
        -------
        coefficients : array-like
            Array with filter coefficients with shape depending on mode:
            - ba: list(b, a) with b and a of type NDArray[np.float64].
            - sos: NDArray[np.float64] with shape (n_sections, 6).
            - zpk: tuple(z, p, k) with z, p, k of type
              NDArray[np.complex128] and float

        """
        if coefficients_mode == FilterCoefficientsType.Sos:
            if self.has_sos:
                return self.sos.copy()
            if self.order > 500:
                warn(
                    "Order is above 500. Computing SOS might take a "
                    + "long time"
                )
            return sig.tf2sos(self.ba[0], self.ba[1])
        elif coefficients_mode == FilterCoefficientsType.Ba:
            if self.has_sos:
                return sig.sos2tf(self.sos)
            return deepcopy(self.ba)
        elif coefficients_mode == FilterCoefficientsType.Zpk:
            if self.has_zpk:
                return tuple(deepcopy(self.zpk))
            elif self.has_sos:
                return sig.sos2zpk(self.sos)

            # Check if filter is too long
            if self.order > 500:
                warn(
                    "Order is above 500. Computing zpk might take a "
                    + "long time"
                )
            return sig.tf2zpk(self.ba[0], self.ba[1])
        else:
            raise ValueError(
                f"{coefficients_mode} is not valid. Use sos, ba or zpk"
            )

    # ======== Plots and prints ===============================================
    def show_info(self):
        """Prints all the filter parameters to the console."""
        print(self.metadata_str)

    def plot_magnitude(
        self,
        length_samples: int = 512,
        range_hz=[20, 20e3],
        normalize: MagnitudeNormalization = MagnitudeNormalization.NoNormalization,
        show_info_box: bool = True,
        zero_phase: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plots magnitude spectrum.
        Change parameters of spectrum with set_spectrum_parameters.

        Parameters
        ----------
        length_samples : int, optional
            Length of IR for magnitude plot. See notes for details.
            Default: 512.
        range_hz : array-like with length 2, optional
            Range for which to plot the magnitude response.
            Default: [20, 20000].
        normalize : MagnitudeNormalization, optional
            Mode for normalization. Default: NoNormalization.
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `True`.
        zero_phase : bool, optional
            Plots magnitude for zero phase filtering. Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        Notes
        -----
        - An IR of the filter is obtained by filtering a dirac impulse in the
          case of IIR filters. For FIR filters, the taps are used and,
          if necessary, zero-padded. The IR length determines the frequency
          resolution.

        """
        if self.order > length_samples:
            length_samples = self.order + 100
            warn(
                f"length_samples ({length_samples}) is shorter than the "
                + f"""filter order {self.order}. Length will be """
                + "automatically extended."
            )
        ir = self.get_ir(length_samples=length_samples, zero_phase=zero_phase)
        fig, ax = ir.plot_magnitude(range_hz, normalize, show_info_box=False)
        if show_info_box:
            txt = self.metadata_str
            ax.text(
                0.1,
                0.5,
                txt,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="grey", alpha=0.75),
            )
        return fig, ax

    def plot_group_delay(
        self,
        length_samples: int = 512,
        range_hz=[20, 20e3],
        show_info_box: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plots group delay of the filter. Different methods are used for
        FIR or IIR filters.

        Parameters
        ----------
        length_samples : int, optional
            Length of ir for magnitude plot. Default: 512.
        range_hz : array-like with length 2, optional
            Range for which to plot the magnitude response.
            Default: [20, 20000].
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        if self.order > length_samples:
            length_samples = self.order + 100
            warn(
                f"length_samples ({length_samples}) is shorter than the "
                + f"""filter order {self.order}. Length will be """
                + "automatically extended."
            )
        if hasattr(self, "sos"):
            ba = sig.sos2tf(self.sos)
        else:
            ba = self.ba
        f, gd = _group_delay_filter(ba, length_samples, self.sampling_rate_hz)
        gd *= 1e3
        ymax = None
        ymin = None
        if any(abs(gd) > 50):
            ymin = -2
            ymax = 50
        fig, ax = general_plot(
            x=f,
            matrix=gd[..., None],
            range_x=range_hz,
            range_y=[ymin, ymax],
            ylabel="Group delay / ms",
        )
        if show_info_box:
            txt = self.metadata_str
            ax.text(
                0.1,
                0.5,
                txt,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="grey", alpha=0.75),
            )
        return fig, ax

    def plot_phase(
        self,
        length_samples: int = 512,
        range_hz=[20, 20e3],
        unwrap: bool = False,
        show_info_box: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plots phase spectrum.

        Parameters
        ----------
        length_samples : int, optional
            Length of IR for phase plot. See notes for details. Default: 512.
        range_hz : array-like with length 2, optional
            Range for which to plot the magnitude response.
            Default: [20, 20000].
        unwrap : bool, optional
            Unwraps the phase to show. Default: `False`.
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        Notes
        -----
        - An IR of the filter is obtained by filtering a dirac impulse in the
          case of IIR filters. For FIR filters, the taps are used and,
          if necessary, zero-padded. The IR length determines the frequency
          resolution.

        """
        if self.order > length_samples:
            length_samples = self.order + 1
            warn(
                f"length_samples ({length_samples}) is shorter than the "
                + f"""filter order {self.order}. Length will be """
                + "automatically extended."
            )
        ir = self.get_ir(length_samples=length_samples)
        fig, ax = ir.plot_phase(range_hz, unwrap)
        if show_info_box:
            txt = self.metadata_str
            ax.text(
                0.1,
                0.5,
                txt,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="grey", alpha=0.75),
            )
        return fig, ax

    def plot_zp(self, show_info_box: bool = False) -> tuple[Figure, Axes]:
        """Plots zeros and poles with the unit circle. This returns `None` and
        produces no plot if user decides that conversion ba->sos is too costly.

        Parameters
        ----------
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        if self.has_zpk:
            z, p, k = self.zpk
        elif self.has_sos:
            z, p, k = sig.sos2zpk(self.sos)
        else:
            # Ask explicitely if filter is very long
            if self.order > 500:
                warn("Filter order is over 500. Computing zpk might take long")
            z, p, k = sig.tf2zpk(self.ba[0], self.ba[1])
        fig, ax = _zp_plot(z, p)
        ax.text(
            0.75,
            0.91,
            rf"$k={k:.1e}$",
            transform=ax.transAxes,
            verticalalignment="top",
        )
        if show_info_box:
            txt = self.metadata_str
            ax.text(
                0.1,
                0.5,
                txt,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="grey", alpha=0.75),
            )
        return fig, ax

    def plot_taps(
        self, show_info_box: bool = False, in_db: bool = False
    ) -> tuple[Figure, Axes]:
        """Plots filter taps for an FIR filter. IIR filters will raise an
        assertion error.

        Parameters
        ----------
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `False`.
        in_db : bool, optional
            When True, the FIR coefficients are shown in dB. Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        assert self.is_fir, "Plotting taps is only valid for FIR filters"
        t = np.arange(0, len(self)) / self.sampling_rate_hz
        txt = self.metadata_str if show_info_box else None
        return general_plot(
            t,
            to_db(self.ba[0], True) if in_db else self.ba[0],
            log_x=False,
            xlabel="Time / s",
            ylabel="Taps / 1",
            info_box=txt,
            tight_layout=True,
        )

    # ======== Saving and export ==============================================
    def save_filter(self, path: str):
        """Saves the Filter object as a pickle.

        Parameters
        ----------
        path : str
            Path for the filter to be saved with format `.pkl`.

        """
        path = _check_format_in_path(path, "pkl")
        with open(path, "wb") as data_file:
            dump(self, data_file, HIGHEST_PROTOCOL)
        return self

    def copy(self):
        """Returns a copy of the object.

        Returns
        -------
        new_sig : `Filter`
            Copy of filter.

        """
        return deepcopy(self)
