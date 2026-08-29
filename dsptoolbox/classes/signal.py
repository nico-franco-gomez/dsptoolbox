"""
Signal class
"""

from copy import deepcopy
from fractions import Fraction
from os.path import splitext
from pickle import HIGHEST_PROTOCOL, dump
from typing import TYPE_CHECKING, Self
from warnings import warn

import numpy as np
import soundfile as sf
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.fft import next_fast_len, rfft
from scipy.signal import oaconvolve, resample_poly

if TYPE_CHECKING:
    from .filter import Filter
    from .filterbank import FilterBank
    from .spectrum import Spectrum

from ..helpers.gain_and_level import _fade, _normalize, from_db, to_db
from ..helpers.latency import (
    _remove_ir_latency_from_phase,
    _remove_ir_latency_from_phase_peak,
)
from ..helpers.minimum_phase import _remove_ir_latency_from_phase_min_phase
from ..helpers.other import (
    _pad_trim,
    find_nearest_points_index_in_vector,
)
from ..helpers.smoothing import _fractional_octave_smoothing, _get_smoothing_factor_ema
from ..helpers.spectrum_utilities import (
    _get_normalized_spectrum,
    _scale_spectrum,
    _wrap_phase,
)
from ..plots import general_matrix_plot, general_plot, general_subplots_line
from ..standard._spectral_methods import _csm_fft, _csm_welch, _stft, _welch
from ..standard._standard_backend import (
    _detrend,
    _fractional_delay_filter,
    _group_delay_direct,
    _indices_above_threshold_dbfs,
)
from ..standard.enums import (
    FadeType,
    FilterBankMode,
    MagnitudeNormalization,
    SpectrumMethod,
    SpectrumScaling,
    Window,
)
from ._multichannel_data import MultichannelData
from .plots import _csm_plot


class Signal(MultichannelData):
    """Class for general signals (time series). Most of the methods and
    supported computations are focused on audio signals, but some features
    might be generalizable to all kinds of time series. It is assumed that
    audio is always represented in floating point type.

    """

    # ======== Constructor and State handler ==================================
    def __init__(
        self,
        path: str | None = None,
        time_data=None,
        sampling_rate_hz: int | None = None,
        constrain_amplitude: bool = False,
        activate_cache: bool = False,
    ):
        """Signal class that saves time data, channel and sampling rate
        information as well as spectrum, cross-spectral matrix and more.

        Parameters
        ----------
        path : str, optional
            A path to audio files. Reading is done with the soundfile library.
            Wave and Flac audio files are accepted. Default: `None`.
        time_data : array-like, NDArray[np.float64], optional
            Time data of the signal. It is saved as a matrix with the form
            (time samples, channel number). Default: `None`.
        sampling_rate_hz : int, optional
            Sampling rate of the signal in Hz. Default: `None`.
        constrain_amplitude : bool, optional
            When `True`, audio is normalized to 0 dBFS peak level in case that
            there are amplitude values greater than 1. Otherwise, there is no
            normalization and the audio data is not constrained to [-1, 1].
            A warning is always shown when audio gets normalized and the used
            normalization factor is saved as `amplitude_scale_factor`.
            Default: `False`.
        activate_cache : bool, optional
            When True, spectra, CSM and STFT will be cached. They will not
            be computed again if no parameters have changed. Set to False to
            avoid caching altogether. Default: False.

        """
        # Handling amplitude
        self.constrain_amplitude = constrain_amplitude
        self.calibrated_signal = False
        self.activate_cache = activate_cache
        # State tracker
        self.__update_state()
        # Import data
        if path is not None:
            assert time_data is None, (
                "Constructor cannot take a path and " + "a vector at the same time"
            )
            assert sampling_rate_hz is None, (
                "Constructor cannot take a path and a sampling rate at the"
                + " same time"
            )
            time_data, sampling_rate_hz = sf.read(path)
        else:
            assert time_data is not None, (
                "Either a path to an audio file or a time vector has to be " + "passed"
            )
            assert sampling_rate_hz is not None, "A sampling rate should be passed!"
        self.sampling_rate_hz = sampling_rate_hz
        self.time_data = time_data
        self._set_spectrum_parameters()
        self._set_spectrogram_parameters()

    @staticmethod
    def from_file(path: str):
        """Create a signal from a path to a wav or flac audio file.

        Parameters
        ----------
        path : str
            Path to file.

        Returns
        -------
        Signal

        """
        return Signal(path)

    @staticmethod
    def from_time_data(
        time_data: NDArray[np.float64],
        sampling_rate_hz: int,
        constrain_amplitude: bool = True,
    ):
        """Create a signal from an array of PCM samples.

        Parameters
        ----------
        time_data : array-like, NDArray[np.float64]
            Time data of the signal. It is saved as a matrix with the form
            (time samples, channel number).
        sampling_rate_hz : int
            Sampling rate of the signal in Hz.
        constrain_amplitude : bool, optional
            When `True`, audio is normalized to 0 dBFS peak level in case that
            there are amplitude values greater than 1. Otherwise, there is no
            normalization and the audio data is not constrained to [-1, 1].
            A warning is always shown when audio gets normalized and the used
            normalization factor is saved as `amplitude_scale_factor`.
            Default: `True`.

        Returns
        -------
        Signal

        """
        return Signal(None, time_data, sampling_rate_hz, constrain_amplitude)

    def __update_state(self):
        """Internal update of object state. If for instance time data gets
        added, new spectrum, csm or stft has to be computed.

        """
        self.__spectrum_state_update = True
        self.__csm_state_update = True
        self.__spectrogram_state_update = True
        self.__time_vector_update = True

    @property
    def metadata(self) -> dict:
        """Return dictionary with metadata about the signal."""
        info = {}
        info["sampling_rate_hz"] = self.sampling_rate_hz
        info["number_of_channels"] = self.number_of_channels
        info["signal_length_samples"] = self.length_samples
        info["signal_length_seconds"] = self.length_seconds
        info["constrain_amplitude"] = self.constrain_amplitude
        info["amplitude_scale_factor"] = self.amplitude_scale_factor
        info["is_complex_signal"] = self.is_complex_signal
        return info

    @property
    def metadata_str(self) -> str:
        """Generate string with metadata about the signal."""
        metadata = self.metadata
        txt = ""
        temp = ""
        for _ in range(len(txt)):
            temp += "-"
        txt += temp + "\n"
        for k in metadata.keys():
            txt += f"""{str(k).replace("_", " ").capitalize()}: {metadata[k]}\n"""
        return txt

    def _generate_time_vector(self):
        """Internal method to generate a time vector on demand."""
        self.__time_vector_update = False
        self.__time_vector_s = np.linspace(
            0, len(self.time_data) / self.sampling_rate_hz, len(self.time_data)
        )

    # ======== Properties and setters =========================================
    @property
    def time_data(self) -> NDArray[np.float64]:
        """Get the time domain signal data.

        Returns
        -------
        NDArray[np.float64]
            Time data as a 2D array with shape (time_samples, number_of_channels).
            Real part of the signal. Use `time_data_imaginary` for the imaginary
            part if it exists.

        """
        return self.__time_data

    @time_data.setter
    def time_data(self, new_time_data: ArrayLike):
        """Set the time data for the signal.

        Parameters
        ----------
        new_time_data : ArrayLike
            Time data as a 1D or 2D array with shape (time_samples, channels).
            If 1D, it is treated as a single-channel signal. If 2D, the array
            is automatically transposed if needed to ensure the dimension with
            more samples is interpreted as time. Complex-valued data is supported
            and will be stored with real and imaginary parts separated.

        Raises
        ------
        AssertionError
            If the array has more than 2 dimensions.

        Notes
        -----
        - Complex-valued input is automatically separated into real and imaginary
          parts and stored internally.
        - If `constrain_amplitude` is True, the signal is automatically normalized
          to 0 dBFS peak level if any amplitude exceeds 1.0. A warning is issued
          when this occurs.
        - When complex data is present, amplitude constraining uses the maximum
          of the peaks from both real and imaginary parts as the normalization
          factor.
        - Setting new time data triggers internal state updates for spectrum,
          cross-spectral matrix, and spectrogram computations.
        - Any existing time window is cleared when new time data is set.

        """
        # Shape of Time Data array
        new_time_data = np.atleast_2d(new_time_data).squeeze()
        assert new_time_data.ndim <= 2, (
            f"{new_time_data.ndim} are "
            + "too many dimensions for time data. Dimensions should"
            + " be [time samples, channels]"
        )
        if new_time_data.ndim < 2:
            new_time_data = new_time_data[..., None]

        # Assume always that there are more time samples than channels
        if new_time_data.shape[1] > new_time_data.shape[0]:
            new_time_data = new_time_data.T

        # Handle complex data
        if np.iscomplexobj(new_time_data):
            new_time_data_imag = np.imag(new_time_data)
            new_time_data = np.real(new_time_data)
        else:
            new_time_data_imag = None

        # Normalization
        if self.constrain_amplitude:
            time_data_max = np.max(np.abs(new_time_data))
            if new_time_data_imag is not None:
                time_data_max = max(time_data_max, np.max(np.abs(new_time_data_imag)))
            if time_data_max > 1.0:
                new_time_data /= time_data_max
                warn(
                    "Signal was over 0 dBFS, normalizing to 0 dBFS "
                    + "peak level was triggered",
                    stacklevel=2,
                )
                # Imaginary part is also scaled by same factor as real part
                if new_time_data_imag is not None:
                    new_time_data_imag /= time_data_max
                self.__amplitude_scale_factor = 1.0 / time_data_max
            else:
                self.__amplitude_scale_factor = 1.0
        else:
            self.__amplitude_scale_factor = 1.0

        # Set time data (real and imaginary)
        self.__time_data = new_time_data
        self.time_data_imaginary = new_time_data_imag
        self.__update_state()

        if hasattr(self, "window"):
            del self.window

    @property
    def amplitude_scale_factor(self) -> float:
        """This is the scaling factor (multiplied) when the amplitude is
        automatically constrained to the range [-1., 1.].

        This factor is computed by checking peak amplitude of the real and
        imaginary parts of the time signal independently. The largest peak
        value is then used as the normalization factor for both.

        """
        return self.__amplitude_scale_factor

    @property
    def sampling_rate_hz(self) -> int:
        """Get the sampling rate in Hz.

        Returns
        -------
        int
            Sampling rate in Hz.

        """
        return self.__sampling_rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, new_sampling_rate_hz):
        """Set the sampling rate in Hz.

        Parameters
        ----------
        new_sampling_rate_hz : int
            New sampling rate in Hz. Must be a positive integer.

        Raises
        ------
        AssertionError
            If new_sampling_rate_hz is not an integer.

        Notes
        -----
        Setting a new sampling rate triggers an internal state update.

        """
        assert type(new_sampling_rate_hz) is int, "Sampling rate can only be an integer"
        self.__sampling_rate_hz = new_sampling_rate_hz
        self.__update_state()

    @property
    def length_seconds(self) -> float:
        """Get the duration of the signal in seconds.

        Returns
        -------
        float
            Signal duration in seconds.

        """
        return len(self) / self.sampling_rate_hz

    @property
    def length_samples(self) -> int:
        """Get the number of samples in the signal.

        Returns
        -------
        int
            Number of time samples.

        """
        return len(self)

    @property
    def time_vector_s(self) -> NDArray[np.float64]:
        """Corresponding time vector for the signal."""
        if self.__time_vector_update:
            self._generate_time_vector()
        return self.__time_vector_s

    @property
    def time_data_imaginary(self) -> NDArray[np.float64] | None:
        """Imaginary part of the time data saved as np.float64. It can be None
        meaning that the signal is purely real."""
        if self.__time_data_imaginary is None:
            return None
        return self.__time_data_imaginary

    @time_data_imaginary.setter
    def time_data_imaginary(self, new_imag: NDArray[np.float64]):
        """Set the imaginary part of the time data.

        Parameters
        ----------
        new_imag : NDArray[np.float64] or None
            Imaginary part of the time data. If provided, must have the same
            shape as the real part. Can be None to remove imaginary data.

        Raises
        ------
        AssertionError
            If the shape of new_imag does not match the shape of the real part.

        """
        if new_imag is not None:
            assert new_imag.shape == self.__time_data.shape, (
                "Shape of imaginary part time data does not match"
            )
        self.__time_data_imaginary: NDArray[np.float64] | None = new_imag

    @property
    def is_complex_signal(self) -> bool:
        """When True, this signal contains an imaginary part."""
        return self.time_data_imaginary is not None

    @property
    def constrain_amplitude(self) -> bool:
        """When True, the amplitude of the signal is always constrained to the
        [-1., 1.] range. It will be automatically scaled if it surpasses these
        values so that peak values are either -1. or 1. Use False to avoid
        any amplitude scaling.

        If this is triggered, the scaling factor is also saved in the signal.

        """
        return self.__constrain_amplitude

    @constrain_amplitude.setter
    def constrain_amplitude(self, nca):
        """Set whether to constrain the signal amplitude to [-1., 1.].

        Parameters
        ----------
        nca : bool
            When True, the signal's amplitude is constrained to the [-1., 1.]
            range with automatic scaling if needed. When False, no amplitude
            scaling is applied.

        Raises
        ------
        AssertionError
            If nca is not a boolean.

        Notes
        -----
        When enabling amplitude constraining, the signal is automatically
        rescaled to fit within the [-1., 1.] range if needed, and the
        scaling factor is saved in the signal.

        """
        assert type(nca) is bool, "constrain_amplitude must be of type boolean"
        self.__constrain_amplitude = nca
        # Restart time data setter for triggering normalization if needed
        if nca and hasattr(self, "time_data"):
            ntd = self.time_data
            self.time_data = ntd

    @property
    def calibrated_signal(self) -> bool:
        """When True, this signal has been (amplitude) calibrated, so that it
        represents sound pressure in Pa."""
        return self.__calibrated_signal

    @calibrated_signal.setter
    def calibrated_signal(self, ncs):
        """Set whether the signal is (amplitude) calibrated.

        Parameters
        ----------
        ncs : bool
            When True, indicates that this signal has been amplitude calibrated
            and represents sound pressure in Pa. When False, the signal is
            not calibrated.

        Raises
        ------
        AssertionError
            If ncs is not a boolean.

        """
        assert type(ncs) is bool, "calibrated_signal must be of type boolean"
        self.__calibrated_signal = ncs

    def __len__(self):
        """Length of time signal in samples."""
        return self.time_data.shape[0]

    def __str__(self):
        """Metadata of the signal."""
        return self.metadata_str

    def __iter__(self):
        """Iterate over the channels of the signal. Modifications to the
        samples can be done through these slices."""
        return iter([self.time_data[:, x] for x in range(self.number_of_channels)])

    def _set_spectrum_parameters(
        self,
        method: SpectrumMethod = SpectrumMethod.WelchPeriodogram,
        smoothing: int = 0,
        pad_to_fast_length: bool = True,
        window_length_samples: int = 1024,
        window_type: Window = Window.Hann,
        overlap_percent: float = 50,
        detrend: bool = True,
        average: str = "mean",
        scaling: SpectrumScaling = SpectrumScaling.FFTBackward,
    ) -> None:
        """Set spectrum parameters in place. Private: used by `__init__` and
        internally where a disposable/owned object is already being mutated.
        Public API is `set_spectrum_parameters`.
        """
        _new_spectrum_parameters = dict(
            method=method,
            smoothing=smoothing,
            pad_to_fast_length=pad_to_fast_length,
            window_length_samples=window_length_samples,
            window_type=window_type,
            overlap_percent=overlap_percent,
            detrend=detrend,
            average=average,
            scaling=scaling,
        )
        if not hasattr(self, "_spectrum_parameters"):
            self._spectrum_parameters = _new_spectrum_parameters
            self.__spectrum_state_update = True
        else:
            if not all(
                [
                    self._spectrum_parameters[k] == _new_spectrum_parameters[k]
                    for k in self._spectrum_parameters
                ]
            ):
                self._spectrum_parameters = _new_spectrum_parameters
                self.__spectrum_state_update = True

                # Also CSM
                self.__csm_state_update = True

    def set_spectrum_parameters(
        self,
        method: SpectrumMethod = SpectrumMethod.WelchPeriodogram,
        smoothing: int = 0,
        pad_to_fast_length: bool = True,
        window_length_samples: int = 1024,
        window_type: Window = Window.Hann,
        overlap_percent: float = 50,
        detrend: bool = True,
        average: str = "mean",
        scaling: SpectrumScaling = SpectrumScaling.FFTBackward,
    ) -> Self:
        """Return a copy of the signal with new parameters set for the
        computation of the spectrum.

        Parameters
        ----------
        method : SpectrumMethod, optional
            Method to use in order to acquire the spectrum. See notes for
            details. Default: WelchPeriodogram.
        smoothing : int, optional
            Smoothing across (`1/smoothing`) octave bands. It will only be
            applied on the spectrum. Smoothes
            magnitude AND phase. For accesing the smoothing algorithm, refer to
            `dsptoolbox.tools.fractional_octave_smoothing()`.
            If smoothing is applied here, `Signal.get_spectrum()` returns
            the smoothed spectrum, but plotting ignores this parameter.
            Default: 0 (no smoothing).
        pad_to_fast_length : bool, optional
            When True and `method=FFT`, the spectrum will be zero-padded to
            have a length that is fast for computing the FFT. Default: True.
        window_length_samples : int, optional
            Window size. Default: 1024.
        window_type : Window, optional
            Choose type of window. Default: Hann.
        overlap_percent : float, optional
            Overlap in percent. Default: 50.
        detrend : bool, optional
            Detrending (subtracting mean). Default: True.
        average : str, optional
            Averaging method. Choose from `'mean'` or `'median'`.
            Default: `'mean'`.
        scaling : SpectrumScaling, optional
            Scaling of spectrum. See references for details about scaling.
            Default: FFTBackward.

        Returns
        -------
        Signal
            New signal with the new spectrum parameters.

        References
        ----------
        - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and
          spectral density estimation by the Discrete Fourier transform (DFT),
          including a comprehensive list of window functions and some new
          at-top windows.

        Notes
        -----
        - On the SpectrumComputation:
            - FFT should be done for deterministic signals and impulse
              responses.
            - WelchPeriodogram can be applied to stochastic signals and as an
              averaged spectrum for non-stationary signals.

        """
        new = self.copy()
        new._set_spectrum_parameters(
            method=method,
            smoothing=smoothing,
            pad_to_fast_length=pad_to_fast_length,
            window_length_samples=window_length_samples,
            window_type=window_type,
            overlap_percent=overlap_percent,
            detrend=detrend,
            average=average,
            scaling=scaling,
        )
        return new

    @property
    def spectrum_scaling(self) -> SpectrumScaling:
        """Get the spectrum scaling method.

        Returns
        -------
        SpectrumScaling
            The scaling method used for spectrum computation.

        """
        """Selected scaling for the spectrum."""
        return self._spectrum_parameters["scaling"]

    @spectrum_scaling.setter
    def spectrum_scaling(self, new_scaling: SpectrumScaling):
        """Set the spectrum scaling method.

        Parameters
        ----------
        new_scaling : SpectrumScaling
            The scaling method to use for spectrum computation.
            See `SpectrumScaling` enum for available options.

        Raises
        ------
        AssertionError
            If new_scaling is not a SpectrumScaling instance.

        Notes
        -----
        Changing the spectrum scaling triggers a state update for both
        the spectrum and the cross-spectral matrix.

        """
        assert isinstance(new_scaling, SpectrumScaling)
        self._spectrum_parameters["scaling"] = new_scaling
        self.__spectrum_state_update = True
        self.__csm_state_update = True

    @property
    def spectrum_method(self) -> SpectrumMethod:
        """Get the spectrum computation method.

        Returns
        -------
        SpectrumMethod
            The method used for spectrum computation (e.g., FFT, WelchPeriodogram).

        """
        return self._spectrum_parameters["method"]

    @spectrum_method.setter
    def spectrum_method(self, new_method: SpectrumMethod):
        """Set the spectrum computation method.

        Parameters
        ----------
        new_method : SpectrumMethod
            The method to use for spectrum computation.
            See `SpectrumMethod` enum for available options
            (e.g., FFT, WelchPeriodogram).

        Raises
        ------
        AssertionError
            If new_method is not a SpectrumMethod instance.

        Notes
        -----
        Changing the spectrum method triggers a state update for both
        the spectrum and the cross-spectral matrix.

        """
        assert isinstance(new_method, SpectrumMethod)
        self._spectrum_parameters["method"] = new_method
        self.__spectrum_state_update = True
        self.__csm_state_update = True

    @property
    def spectrum_smoothing(self) -> float:
        """Get the spectrum smoothing parameter in fraction of octaves.

        Returns
        -------
        float
            Smoothing parameter. 0 means no smoothing. Positive values indicate
            smoothing in fractions of octaves.

        """
        return self._spectrum_parameters["smoothing"]

    @spectrum_smoothing.setter
    def spectrum_smoothing(self, new_smoothing):
        """Set the spectrum smoothing parameter.

        Parameters
        ----------
        new_smoothing : float or int
            Smoothing parameter in fraction of octaves. Determines the width
            of the smoothing window applied to the spectrum. Must be zero or
            positive. Zero (default) means no smoothing.

        Raises
        ------
        AssertionError
            If new_smoothing is negative.

        Notes
        -----
        The smoothing is applied across 1/smoothing octave bands.
        This smoothes both magnitude and phase of the spectrum.

        """
        assert new_smoothing >= 0.0, "Smoothing must be positive or zero"
        self._spectrum_parameters["smoothing"] = float(new_smoothing)

    def _set_spectrogram_parameters(
        self,
        window_length_samples: int = 1024,
        window_type: Window = Window.Hann,
        overlap_percent: float = 50.0,
        fft_length_samples: int | None = None,
        detrend: bool = False,
        padding: bool = True,
        scaling: SpectrumScaling = SpectrumScaling.FFTBackward,
    ) -> None:
        """Set spectrogram parameters in place. Private: used by `__init__`
        and internally where a disposable/owned object is already being
        mutated. Public API is `set_spectrogram_parameters`.
        """
        _new_spectrogram_parameters = dict(
            window_length_samples=window_length_samples,
            window_type=window_type,
            overlap_percent=overlap_percent,
            fft_length_samples=fft_length_samples,
            detrend=detrend,
            padding=padding,
            scaling=scaling,
        )
        if not hasattr(self, "_spectrogram_parameters"):
            self._spectrogram_parameters = _new_spectrogram_parameters
            self.__spectrogram_state_update = True
        else:
            if not all(
                [
                    self._spectrogram_parameters[k] == _new_spectrogram_parameters[k]
                    for k in self._spectrogram_parameters
                ]
            ):
                self._spectrogram_parameters = _new_spectrogram_parameters
                self.__spectrogram_state_update = True

    def set_spectrogram_parameters(
        self,
        window_length_samples: int = 1024,
        window_type: Window = Window.Hann,
        overlap_percent: float = 50.0,
        fft_length_samples: int | None = None,
        detrend: bool = False,
        padding: bool = True,
        scaling: SpectrumScaling = SpectrumScaling.FFTBackward,
    ) -> Self:
        """Return a copy of the signal with new parameters set for the
        computation of the spectrogram.

        Parameters
        ----------
        window_length_samples : int, optional
            Window size. Default: 1024.
        window_type : Window, optional
            Type of window to use. Default: Hann.
        overlap_percent : float, optional
            Overlap in percent. Default: 50.
        fft_length_samples : int, optional
            Length of the FFT window for each time window. This affects
            the frequency resolution and can also crop the time window. Pass
            `None` to use the window length. Default: `None`.
        detrend : bool, optional
            Detrending (subtracting mean) for each time frame.
            Default: `False`.
        padding : bool, optional
            Padding signal in the beginning and end to center it in order
            to avoid losing energy because of windowing. Default: `True`.
        scaling : SpectrumScaling, optional
            Scaling of spectrum. Default: `FFTBackwards`.

        Returns
        -------
        Signal
            New signal with the new spectrogram parameters.

        References
        ----------
        - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and
          spectral density estimation by the Discrete Fourier transform (DFT),
          including a comprehensive list of window functions and some new
          at-top windows.

        """
        new = self.copy()
        new._set_spectrogram_parameters(
            window_length_samples=window_length_samples,
            window_type=window_type,
            overlap_percent=overlap_percent,
            fft_length_samples=fft_length_samples,
            detrend=detrend,
            padding=padding,
            scaling=scaling,
        )
        return new

    # ======== Add, remove and reorder channels ===============================
    def add_channel(
        self,
        path: str | None = None,
        new_time_data: NDArray[np.float64] | None = None,
        sampling_rate_hz: int | None = None,
        allow_padding_trimming: bool = True,
    ) -> Self:
        """Return a copy of the signal with new channels added.

        Parameters
        ----------
        path : str, optional
            Path to the file containing new channel information.
        new_time_data : NDArray[np.float64], optional
            np.array with new channel data.
        sampling_rate_hz : int, optional
            Sampling rate for the new data
        allow_padding_trimming : bool, optional
            Activates padding or trimming at the end of signal in case the
            new data does not match previous data. Default: `True`.

        Returns
        -------
        Signal
            New signal with the added channels.

        """
        if path is not None:
            assert new_time_data is None, (
                "Only path or new time data is " + "accepted, not both."
            )
            new_time_data, sampling_rate_hz = sf.read(path)
        else:
            if new_time_data is not None:
                assert path is None, (
                    "Only path or new time data is " + "accepted, not both."
                )
        assert sampling_rate_hz == self.sampling_rate_hz, (
            f"{sampling_rate_hz} does not match {self.sampling_rate_hz} "
            + "as the sampling rate"
        )
        if type(new_time_data) is not NDArray[np.float64]:
            new_time_data = np.array(new_time_data)
        if new_time_data.ndim > 2:
            new_time_data = new_time_data.squeeze()
        assert new_time_data.ndim <= 2, (
            f"{new_time_data.ndim} are "
            + "too many dimensions for time data. Dimensions should"
            + " be (time samples, channels)"
        )
        if new_time_data.ndim < 2:
            new_time_data = new_time_data[..., None]
        if new_time_data.shape[1] > new_time_data.shape[0]:
            new_time_data = new_time_data.T

        diff = new_time_data.shape[0] - self.time_data.shape[0]
        if diff != 0:
            txt = "Padding" if diff < 0 else "Trimming"
            if allow_padding_trimming:
                new_time_data = _pad_trim(
                    new_time_data,
                    self.time_data.shape[0],
                    axis=0,
                    in_the_end=True,
                )
                warn(
                    f"{txt} has been performed "
                    + "on the end of the new signal to match original one.",
                    stacklevel=2,
                )
            else:
                raise AttributeError(
                    f"{new_time_data.shape[0]} does not match "
                    + f"{self.time_data.shape[0]}. Activate allow_padding_trimming "
                    + "for allowing this channel to be added"
                )
        return self.copy_with_new_time_data(
            np.concatenate([self.time_data, new_time_data], axis=1)
        )

    def clear_time_window(self) -> Self:
        """Return a copy of the signal with the time window removed, if any."""
        new = self.copy()
        if hasattr(new, "window"):
            del new.window
        return new

    # ======== Getters ========================================================
    def get_spectrum(
        self, force_computation=False
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128 | np.float64]]:
        """Returns spectrum according to the stored parameters.

        Parameters
        ----------
        force_computation : bool, optional
            Forces spectrum computation.

        Returns
        -------
        spectrum_freqs : NDArray[np.float64]
            Frequency vector.
        spectrum : NDArray[np.complex128 | np.float64]
            Spectrum matrix for each channel.

        """
        condition = (
            not hasattr(self, "spectrum")
            or self.__spectrum_state_update
            or force_computation
        )

        if condition:
            if self.spectrum_method == SpectrumMethod.WelchPeriodogram:
                spectrum = _welch(
                    self.time_data,
                    None,
                    self.sampling_rate_hz,
                    self._spectrum_parameters["window_type"],
                    self._spectrum_parameters["window_length_samples"],
                    self._spectrum_parameters["overlap_percent"],
                    self._spectrum_parameters["detrend"],
                    self._spectrum_parameters["average"],
                    self._spectrum_parameters["scaling"],
                )
                fft_length = self._spectrum_parameters["window_length_samples"]
            else:  # FFT
                fft_length = (
                    next_fast_len(self.length_samples, True)
                    if self._spectrum_parameters["pad_to_fast_length"]
                    else self.length_samples
                )
                # Get spectrum
                spectrum = rfft(
                    self.time_data,
                    axis=0,
                    norm=self.spectrum_scaling.fft_norm(),
                    n=fft_length,
                )

                # Smoothing
                if self._spectrum_parameters["smoothing"] != 0:
                    # Smoothing the magnitude
                    temp_abs = _fractional_octave_smoothing(
                        np.abs(spectrum),
                        None,
                        self._spectrum_parameters["smoothing"],
                        clip_values=True,
                    )
                    # Smoothing the phase is not shift-invariant...
                    temp_phase = _fractional_octave_smoothing(
                        np.unwrap(np.angle(spectrum), axis=0),
                        None,
                        self._spectrum_parameters["smoothing"],
                    )
                    spectrum = temp_abs * np.exp(1j * temp_phase)

                # Length of signal for frequency vector and scaling
                if self.spectrum_scaling.has_physical_units():
                    spectrum = _scale_spectrum(
                        spectrum,
                        self.spectrum_scaling,
                        fft_length,
                        self.sampling_rate_hz,
                        None if not hasattr(self, "window") else self.window,
                    )

            freqs = np.fft.rfftfreq(fft_length, 1 / self.sampling_rate_hz)
            if self.activate_cache:
                self.spectrum = [freqs.copy(), spectrum.copy()]
                self.__spectrum_state_update = False
            return freqs, spectrum

        return self.spectrum[0].copy(), self.spectrum[1].copy()

    def get_csm(
        self, force_computation=False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get Cross spectral matrix for all channels with the shape
        (frequencies, channels, channels). It uses the parameters stored in
        `set_spectrum_parameters`.

        Parameters
        ----------
        force_computation : bool, optional
            When `True`, computation is forced even if there is cached data.
            Default: `False`.

        Returns
        -------
        f_csm : NDArray[np.float64]
            Frequency vector.
        csm : NDArray[np.float64]
            Cross spectral matrix with shape (frequency, channels, channels).

        """
        assert self.number_of_channels > 1, (
            "Cross spectral matrix can only be computed when at least two "
            + "channels are available"
        )
        condition = (
            not hasattr(self, "csm") or force_computation or self.__csm_state_update
        )

        if condition:
            if self.spectrum_method == SpectrumMethod.WelchPeriodogram:
                f, csm = _csm_welch(
                    self.time_data,
                    self.sampling_rate_hz,
                    self._spectrum_parameters["window_length_samples"],
                    self._spectrum_parameters["window_type"],
                    self._spectrum_parameters["overlap_percent"],
                    self._spectrum_parameters["detrend"],
                    self._spectrum_parameters["average"],
                    self._spectrum_parameters["scaling"],
                )
            else:
                # Ensure a complex type of scaling during computation of
                # spectrum
                old_scaling = self.spectrum_scaling
                self.spectrum_scaling = SpectrumScaling.FFTBackward

                f, sp = self.get_spectrum()
                csm = _csm_fft(
                    sp,
                    old_scaling,
                    self.window if hasattr(self, "window") else None,
                    self.sampling_rate_hz,
                )
                self.spectrum_scaling = old_scaling
            if self.activate_cache:
                self.csm = [f.copy(), csm.copy()]
                self.__csm_state_update = False
            return f, csm
        return self.csm[0].copy(), self.csm[1].copy()

    def get_spectrogram(
        self, force_computation: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
        """Returns a matrix containing the STFT of a specific channel.

        Parameters
        ----------
        force_computation : bool, optional
            Forces new computation of the STFT. Default: False.

        Returns
        -------
        t_s : NDArray[np.float64]
            Time vector.
        f_hz : NDArray[np.float64]
            Frequency vector.
        spectrogram : NDArray[np.complex128]
            Complex spectrogram with shape (frequency, time, channel).

        """
        condition = (
            not hasattr(self, "spectrogram")
            or force_computation
            or self.__spectrogram_state_update
        )

        if condition:
            spectrogram = _stft(
                self.time_data,
                self.sampling_rate_hz,
                self._spectrogram_parameters["window_length_samples"],
                self._spectrogram_parameters["window_type"],
                self._spectrogram_parameters["overlap_percent"],
                self._spectrogram_parameters["fft_length_samples"],
                self._spectrogram_parameters["detrend"],
                self._spectrogram_parameters["padding"],
                self._spectrogram_parameters["scaling"],
            )
            self.__spectrogram_state_update = False
            if self.activate_cache:
                self.spectrogram = deepcopy(spectrogram)
            return spectrogram[0], spectrogram[1], spectrogram[2]

        return (
            self.spectrogram[0].copy(),
            self.spectrogram[1].copy(),
            self.spectrogram[2].copy(),
        )

    # ======== Plots ==========================================================
    def plot_magnitude(
        self,
        range_hz: list[float] | None = (20.0, 20e3),
        normalize: MagnitudeNormalization = MagnitudeNormalization.NoNormalization,
        range_db=None,
        smoothing: int = 0,
        show_info_box: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plots magnitude spectrum.
        Change parameters of spectrum with set_spectrum_parameters.

        Parameters
        ----------
        range_hz : array-like with length 2, None, optional
            Range for which to plot the magnitude response. Use None to avoid
            setting any specific range. Default: [20, 20000].
        normalize : MagnitudeNormalization, optional
            Mode for normalization. Default: NoNormalization.
        range_db : array-like with length 2, optional
            Range in dB for which to plot the magnitude response.
            Default: `None`.
        smoothing : int, optional
            Smoothing across the (1/smoothing) octave band. It only applies to
            the plot data and not to `get_spectrum()`. Default: 0 (no
            smoothing).
        show_info_box : bool, optional
            Plots a info box regarding spectrum parameters and plot parameters.
            If it is str, it overwrites the standard message.
            Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        Notes
        -----
        - Smoothing is only applied on the plot data.
        - In case the signal has been calibrated and the time data is given in
          Pascal, the plotted values in dB will be scaled by p0=(20e-6 Pa)**2
          when no normalization is active.

        """
        # Handle smoothing
        prior_smoothing = self._spectrum_parameters["smoothing"]
        self._spectrum_parameters["smoothing"] = 0

        # Get spectrum
        f, sp = self.get_spectrum()

        self._spectrum_parameters["smoothing"] = prior_smoothing

        f, mag_db = _get_normalized_spectrum(
            f=f,
            spectra=sp,
            is_amplitude_scaling=self.spectrum_scaling.is_amplitude_scaling(),
            f_range_hz=range_hz,
            normalize=normalize,
            smoothing=smoothing,
            phase=False,
            calibrated_data=self.calibrated_signal,
        )

        if show_info_box:
            txt = "Info"
            txt += f"""\nMode: {self._spectrum_parameters["method"]}"""
            if range_hz is not None:
                txt += f"\nRange: [{range_hz[0]}, {range_hz[1]}]"
            txt += f"\nNormalized: {normalize}"
            txt += f"""\nSmoothing: {smoothing}"""
        else:
            txt = None

        match normalize:
            case MagnitudeNormalization.NoNormalization:
                y_extra = "" if self.calibrated_signal else "FS"
            case MagnitudeNormalization.OneKhz:
                y_extra = " (normalized @ 1 kHz)"
            case MagnitudeNormalization.OneKhzFirstChannel:
                y_extra = " (normalized @ 1 kHz for first channel)"
            case MagnitudeNormalization.Max:
                y_extra = " (normalized @ peak)"
            case MagnitudeNormalization.MaxFirstChannel:
                y_extra = " (normalized @ peak for first channel)"
            case MagnitudeNormalization.Energy:
                y_extra = " (normalized with average energy)"
            case MagnitudeNormalization.EnergyFirstChannel:
                y_extra = " (normalized with average energy of first channel)"
            case _:
                raise ValueError("No valid normalization")

        fig, ax = general_plot(
            f,
            mag_db,
            range_hz,
            ylabel="Magnitude / dB" + y_extra,
            info_box=txt,
            labels=[f"Channel {n}" for n in range(self.number_of_channels)],
            range_y=range_db,
        )
        return fig, ax

    def plot_time(self) -> tuple[Figure, list[Axes]]:
        """Plots time signals.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : list of `matplotlib.axes.Axes`
            Axes.

        """
        fig, ax = general_subplots_line(
            self.time_vector_s,
            self.time_data,
            sharex=True,
            ylabels=[f"Channel {n}" for n in range(self.number_of_channels)],
            xlabels="Time / s",
        )

        for n in range(self.number_of_channels):
            mx = np.max(np.abs(self.time_data[:, n])) * 1.1
            if self.is_complex_signal:
                ax[n].plot(
                    self.time_vector_s,
                    self.time_data_imaginary[:, n],
                    alpha=0.9,
                    linestyle="dotted",
                )
            ax[n].set_ylim([-mx, mx])
        return fig, ax

    def plot_spl(
        self,
        normalize_at_peak: bool = False,
        dynamic_range_db: float | None = 100.0,
        window_length_s: float = 0.0,
    ) -> tuple[Figure, list[Axes]]:
        """Plots the momentary sound pressure level (dB or dBFS) of each
        channel. If the signal is calibrated and not normalized at peak, the
        values correspond to dB, otherwise they are dBFS.

        Parameters
        ----------
        normalize_at_peak : bool, optional
            When `True`, each channel gets normalize by its peak value.
            Default: `False`.
        dynamic_range_db : float, optional
            This is the range in dB used for plotting. Each plot will be in the
            range [peak + 1 - dynamic_range_db, peak + 1]. Pass `None` to avoid
            setting any range. Default: 100.
        window_length_s : float, optional
            When different than 0, a moving average along the time axis is done
            with the given length. Default: 0.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : list of `matplotlib.axes.Axes`
            Axes.

        Notes
        -----
        - All values are clipped to be at least -800 dBFS.
        - If it is an analytic signal and normalization is applied, the peak
          value of the real part is used as the normalization factor.
        - If the time window is not 0, effects at the edges of the signal might
          be present due to zero-padding.

        """
        td_squared = self.time_data**2

        if window_length_s > 0:
            window = np.ones((int(window_length_s * self.sampling_rate_hz + 0.5), 1))
            window /= len(window)
            td_squared = oaconvolve(td_squared, window, mode="same", axes=0)

        if self.is_complex_signal:
            td_squared_imaginary = self.time_data_imaginary**2.0
            if window_length_s > 0:
                td_squared_imaginary = oaconvolve(
                    td_squared_imaginary, window, mode="same", axes=0
                )
            complex_etc = to_db(
                td_squared_imaginary,
                False,
                500 if dynamic_range_db is None else dynamic_range_db,
            )

        etc = to_db(td_squared, False, 500)
        peak_values = np.max(etc, axis=0)

        if normalize_at_peak:
            etc -= peak_values
            if self.is_complex_signal:
                complex_etc -= peak_values

        db_type = "dBFS"
        if self.calibrated_signal and not normalize_at_peak:
            # Convert to dB
            factor = 20 * np.log10(2e-5)
            etc -= factor
            peak_values -= factor
            db_type = "dB"
            if self.is_complex_signal:
                complex_etc -= factor

        fig, ax = general_subplots_line(
            self.time_vector_s,
            etc,
            sharex=True,
            ylabels=[
                f"Channel {n} / {db_type}" for n in range(self.number_of_channels)
            ],
            xlabels="Time / s",
        )

        add_to_peak = 1  # Add 1 dB for better plotting
        max_values = (
            peak_values + add_to_peak
            if not normalize_at_peak
            else np.ones(self.number_of_channels)
        )

        for n in range(self.number_of_channels):
            if self.is_complex_signal:
                ax[n].plot(self.time_vector_s, complex_etc[:, n], alpha=0.75)
            if dynamic_range_db is not None:
                ax[n].set_ylim(
                    [max_values[n] - np.abs(dynamic_range_db), max_values[n]]
                )
        return fig, ax

    def plot_group_delay(
        self,
        range_hz: list[float] | None = (20.0, 20e3),
        smoothing: int = 0,
        remove_ir_latency: str | ArrayLike | None = None,
    ) -> tuple[Figure, Axes]:
        """Plots group delay of each channel.

        Parameters
        ----------
        range_hz : array-like with length 2, None, optional
            Range of frequencies for which to show group delay. Pass None to
            avoid any specific range. Default: [20, 20e3].
        smoothing : int, optional
            When different than 0, smoothing is applied to the group delay
            along the (1/smoothing) octave band. This only affects the values
            in the plot. Default: 0.
        remove_ir_latency : str {"peak", "min_phase"}, ArrayLike, None,\
                optional
            If the signal is an impulse response, the delay of the impulse can
            be removed. IR delay removal options are:

            - str {"peak" or "min_phase"}: By regarding its delay in relation
              to the minimum-phase equivalent or its peak in the time signal.
            - ArrayLike: Delay in samples to remove from each channel.
            - None: no latency removal.

            Default: None.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        # Handle spectrum parameters
        prior_spectrum_parameters = self._spectrum_parameters
        self._set_spectrum_parameters(
            SpectrumMethod.FFT,
            scaling=SpectrumScaling.FFTBackward,
            smoothing=0,
        )
        f, sp = self.get_spectrum()
        self._spectrum_parameters = prior_spectrum_parameters

        ph = np.angle(sp)

        if remove_ir_latency is None:
            pass
        elif type(remove_ir_latency) is str:
            match remove_ir_latency.lower():
                case "peak":
                    ph = _remove_ir_latency_from_phase_peak(
                        f, ph, self.time_data, self.sampling_rate_hz
                    )
                case "min_phase":
                    ph = _remove_ir_latency_from_phase_min_phase(
                        f, ph, self.time_data, self.sampling_rate_hz, 8
                    )
                case _:
                    raise ValueError("No valid latency removal")
        else:
            delays_samples = np.atleast_1d(remove_ir_latency)
            ph = _remove_ir_latency_from_phase(
                f, ph, delays_samples, self.sampling_rate_hz
            )

        gd = _group_delay_direct(ph, f[1] - f[0])

        if smoothing != 0:
            gd = _fractional_octave_smoothing(gd, None, smoothing)

        if range_hz is not None:
            inds = find_nearest_points_index_in_vector(range_hz, f)
            f = f[inds[0] : inds[1]]
            gd = gd[inds[0] : inds[1], ...]

        fig, ax = general_plot(
            f,
            gd * 1e3,
            range_hz,
            labels=[f"Channel {n}" for n in range(self.number_of_channels)],
            ylabel="Group delay / ms",
        )
        return fig, ax

    def plot_spectrogram(
        self,
        channel_number: int = 0,
        log_freqs: bool = True,
        dynamic_range_db: float = 50,
    ) -> tuple[Figure, Axes]:
        """Plots STFT matrix of the given channel.

        Parameters
        ----------
        channel_number : int, optional
            Selected channel to plot spectrogram. Default: 0 (first).
        logfreqs : bool, optional
            When `True`, frequency axis is plotted logarithmically.
            Default: `True`.
        dynamic_range_db : float, optional
            This sets the dynamic range to show for the spectrogram. The
            plotted colormap goes from the maximum down to maximum minus
            dynamic range. For example, dynamic_range_db=50 plots for a peak
            value of 30 dB the colormap of the spectrogram between
            [30, -20] dB. Default: 50.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        # Get whole spectrogram
        t, f, stft = self.get_spectrogram()

        # Select channel
        stft = stft[:, :, channel_number]

        ids = find_nearest_points_index_in_vector(20.0, f)
        if ids == 0:
            ids += 1
        f = f[ids[0] :]
        stft = stft[ids[0] :, :]

        zlabel = "dBFS"
        stft_db = to_db(
            stft,
            self._spectrogram_parameters["scaling"].is_amplitude_scaling(),
        )

        if self.calibrated_signal:
            stft_db -= 20 * np.log10(2e-5)
            zlabel = "dB(SPL)"

        stft_db = np.nan_to_num(stft_db, nan=np.min(stft_db))
        fig, ax = general_matrix_plot(
            matrix=stft_db,
            range_x=(t[0], t[-1]),
            range_y=(f[0], f[-1]),
            range_z=np.abs(dynamic_range_db),
            xlabel="Time / s",
            ylabel="Frequency / Hz",
            zlabel=zlabel,
            xlog=False,
            ylog=log_freqs,
            colorbar=True,
        )
        return fig, ax

    def plot_phase(
        self,
        range_hz: list[float] | None = (20.0, 20e3),
        unwrap: bool = False,
        smoothing: int = 0,
        remove_ir_latency: str | None | ArrayLike = None,
    ) -> tuple[Figure, Axes]:
        """Plots phase of the frequency response, only available if the method
        for the spectrum is FFT.

        Parameters
        ----------
        range_hz : array-like with length 2, None, optional
            Range of frequencies for which to show group delay.
            Default: [20, 20e3].
        unwrap : bool, optional
            When `True`, the unwrapped phase is plotted. Default: `False`.
        smoothing : int, optional
            When different than 0, the phase response is smoothed across the
            1/smoothing-octave band. This only applies smoothing to the plot
            data. Default: 0.
        remove_ir_latency : str {"peak", "min_phase"}, ArrayLike,\
                None, optional
            If the signal is an impulse response, the delay of the impulse can
            be removed. IR delay removal options are:

            - str {"peak" or "min_phase"}: By regarding its delay in relation
              to the minimum-phase equivalent or its peak in the time signal.
            - ArrayLike: Delay in samples to remove from each channel.
            - None: no latency removal.

            Default: None.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        assert self.spectrum_method == SpectrumMethod.FFT, (
            "Phase cannot be plotted since the spectrum is welch."
        )

        prior_smoothing = self._spectrum_parameters["smoothing"]
        self._spectrum_parameters["smoothing"] = 0

        # Get spectrum
        f, sp = self.get_spectrum()
        ph = np.angle(sp)

        self._spectrum_parameters["smoothing"] = prior_smoothing

        if remove_ir_latency is None:
            pass
        elif type(remove_ir_latency) is str:
            match remove_ir_latency.lower():
                case "peak":
                    ph = _remove_ir_latency_from_phase_peak(
                        f, ph, self.time_data, self.sampling_rate_hz
                    )
                case "min_phase":
                    ph = _remove_ir_latency_from_phase_min_phase(
                        f, ph, self.time_data, self.sampling_rate_hz, 8
                    )
                case _:
                    raise ValueError("No valid latency removal")
        else:
            delays_samples = np.atleast_1d(remove_ir_latency)
            ph = _remove_ir_latency_from_phase(
                f, ph, delays_samples, self.sampling_rate_hz
            )

        if smoothing != 0:
            ph = _wrap_phase(
                _fractional_octave_smoothing(np.unwrap(ph, axis=0), None, smoothing)
            )

        if unwrap:
            ph = np.unwrap(ph, axis=0)

        if range_hz is not None:
            inds = find_nearest_points_index_in_vector(range_hz, f)
            f = f[inds[0] : inds[1]]
            ph = ph[inds[0] : inds[1], ...]

        fig, ax = general_plot(
            x=f,
            matrix=ph,
            range_x=range_hz,
            labels=[f"Channel {n}" for n in range(self.number_of_channels)],
            ylabel="Phase / rad",
        )
        return fig, ax

    def plot_csm(
        self, range_hz=(20, 20e3), with_phase: bool = True
    ) -> tuple[Figure, Axes]:
        """Plots the cross spectral matrix of the multichannel signal.

        Parameters
        ----------
        range_hz : array-like with length 2, optional
            Range of Hz to be showed. Default: [20, 20e3].
        with_phase : bool, optional
            When `True`, the unwrapped phase is also plotted. Default: `True`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        f, csm = self.get_csm()
        fig, ax = _csm_plot(f, csm, range_hz, True, with_phase)
        return fig, ax

    # ======== Saving and copy ================================================
    def save_signal(self, path: str, bit_depth: int = 32):
        """Saves the Signal object as wav, flac or pickle. The saving format
        is inferred from the file extension in `path`.

        Parameters
        ----------
        path : str
            Path for the signal to be saved. Its extension (`'.wav'`,
            `'.flac'` or `'.pkl'`) determines the saving format.
        bit_depth : int, optional
            Bit depth when saving a signal in `'wav'` or `'flac'` format.
            Only 16, 24, 32 and 64 are valid. 32 and 64 are only valid for
            `'wav'`. Default: 32.

        """
        extension = splitext(path)[1].lower().lstrip(".")
        if extension in ("wav", "flac"):
            if bit_depth == 32:
                subtype = "FLOAT"
            elif bit_depth == 64:
                subtype = "DOUBLE"
            elif bit_depth == 24:
                subtype = "PCM_24"
            elif bit_depth == 16:
                subtype = "PCM_16"
            else:
                raise ValueError(
                    "Selected bit depth is not valid. " + "Use either 16, 24, 32 or 64"
                )
            sf.write(path, self.time_data, self.sampling_rate_hz, subtype=subtype)
        elif extension == "pkl":
            with open(path, "wb") as data_file:
                dump(self, data_file, HIGHEST_PROTOCOL)
        else:
            raise ValueError(
                f"'{extension}' is not a supported saving format. Use a path "
                "ending in .wav, .flac or .pkl"
            )
        return self

    def copy(self) -> "Signal":
        """Returns a copy of the object.

        Returns
        -------
        new_sig : `Signal`
            Copy of Signal.

        """
        return deepcopy(self)

    # ======== Multichannel Data Base Class Implementation ====================
    def _get_data(self) -> NDArray[np.float64 | np.complex128]:
        """Get the time data for multichannel operations."""
        return (
            self.time_data + 1j * self.time_data_imaginary
            if self.is_complex_signal
            else self.time_data
        )

    def _set_data(self, data: NDArray[np.float64 | np.complex128]) -> None:
        """Set the time data for multichannel operations."""
        self.time_data = data

    def _create_copy_with_new_data(
        self, data: NDArray[np.float64 | np.complex128]
    ) -> "Signal":
        """Create a copy with new time data."""
        return self.copy_with_new_time_data(data)

    def _update_state(self) -> None:
        """Update internal state after data modification."""
        self.__update_state()

    def copy_with_new_time_data(self, new_time_data: ArrayLike) -> "Signal":
        """Copy all attributes of the signal but with new time data.

        Parameters
        ----------
        new_time_data : ArrayLike
            New valid time data.

        Returns
        -------
        Signal
            Signal with new time data

        Notes
        -----
        - This signal object will own the time data alone, so when passing an
          array, it is checked whether its memory belongs to another array or
          not. If so, a copy is made.

        """
        # Copy if the underlying memory belongs to another array
        if isinstance(new_time_data, np.ndarray):
            new_time_data = (
                new_time_data if new_time_data.base is None else new_time_data.copy()
            )
        #
        new_signal = Signal.from_time_data(
            new_time_data, self.sampling_rate_hz, self.constrain_amplitude
        )
        new_signal.calibrated_signal = self.calibrated_signal
        new_signal.activate_cache = self.activate_cache
        new_signal._spectrum_parameters = deepcopy(self._spectrum_parameters)
        new_signal._spectrogram_parameters = deepcopy(self._spectrogram_parameters)
        return new_signal

    # ======== Transforms (returning a new Signal) ============================
    def pad_trim(
        self, desired_length_samples: int, in_the_end: bool = True
    ) -> "Signal":
        """Return a copy of the signal with padded or trimmed time data.

        Parameters
        ----------
        desired_length_samples : int
            Length of resulting signal.
        in_the_end : bool, optional
            Defines if padding or trimming should be done in the beginning or
            in the end of the signal. Default: `True`.

        Returns
        -------
        Signal
            New padded or trimmed signal.

        """
        new_time_data = np.zeros((desired_length_samples, self.number_of_channels))
        for n in range(self.number_of_channels):
            new_time_data[:, n] = _pad_trim(
                self.time_data[:, n],
                desired_length_samples,
                in_the_end=in_the_end,
            )
        return self.copy_with_new_time_data(new_time_data)

    def modify_signal_length(
        self, start_seconds: float | None, end_seconds: float | None
    ) -> "Signal":
        """Return a copy of the signal with added silence at the beginning
        or the end. Time samples can also be trimmed when using negative
        time values.

        Parameters
        ----------
        start_seconds : float, None
            Seconds to add or remove from the start. Positive values append
            samples while negative ones remove them. Pass None to avoid any
            modification.
        end_seconds : float, None
            Seconds to add or remove from the end. Positive values append
            samples while negative ones remove them. Pass None to avoid any
            modification.

        Returns
        -------
        Signal
            Copy of the signal with new length.

        """
        assert start_seconds is not None or end_seconds is not None, (
            "At least the start or the end should be modified"
        )
        fs = self.sampling_rate_hz
        start_samples = (
            0
            if start_seconds is None
            else int(start_seconds * fs + 0.5 * np.sign(start_seconds))
        )
        end_samples = (
            0
            if end_seconds is None
            else int(end_seconds * fs + 0.5 * np.sign(end_seconds))
        )

        # Avoid cutting too many samples
        if start_samples < 0:
            assert len(self) > -start_samples, "Trimming is too much"
        if end_samples < 0:
            assert len(self) > -end_samples, "Trimming is too much"
        if start_samples < 0 and end_samples < 0:
            assert len(self) > -(start_samples + end_samples), "Trimming is too much"

        td = self.time_data
        if start_samples >= 0:
            td = np.pad(td, ((start_samples, 0), (0, 0)))
        else:
            td = td[-start_samples:, ...]

        if end_samples >= 0:
            td = np.pad(td, ((0, end_samples), (0, 0)))
        else:
            td = td[:end_samples, ...]
        return self.copy_with_new_time_data(td)

    def trim_with_level_threshold(
        self, threshold_db: float, at_start: bool = True, at_end: bool = True
    ) -> tuple["Signal", int, int]:
        """Return a copy of the signal trimmed by discarding the edge
        samples below a certain threshold.

        Parameters
        ----------
        threshold_db : float
            (Inclusive) Threshold for trimming. Generally in dBFS, but it can
            be in dBSPL if the signal has been calibrated.
        at_start : bool, optional
            Activate trimming in the beginning. Default: True.
        at_end : bool, optional
            Activate trimming in the end. Default: True.

        Returns
        -------
        Signal
            Copy of the signal with trimmed time series.
        int
            Start index in the original array.
        int
            Stop index in the original array.

        """
        assert at_start or at_end, "Either start or end should be trimmed"

        threshold_linear = from_db(threshold_db, True)
        above_threshold = np.where(np.abs(self.time_data) >= threshold_linear)
        if at_start:
            indices_along_first_axis = above_threshold[0][: self.number_of_channels]
            start = int(np.min(indices_along_first_axis))
        else:
            start = 0

        if at_end:
            indices_along_first_axis = above_threshold[0][-self.number_of_channels :]
            stop = min(self.length_samples, int(np.max(indices_along_first_axis)) + 1)
        else:
            stop = self.length_samples

        return (
            self.copy_with_new_time_data(self.time_data[start:stop]),
            start,
            stop,
        )

    def trim_with_time_selection(
        self,
        start_time_s: float | None,
        end_time_s: float | None,
        inclusive: bool = True,
    ) -> "Signal":
        """Return a copy of the signal trimmed to a selected time window.

        Parameters
        ----------
        start_time_s : float, None
            Start time for the window. Pass None to start the time window
            at the beginning of the signal.
        end_time_s : float, None
            End time for the window. Pass None to place the end of the time
            window at the end of the signal.
        inclusive : bool, optional
            When True, the bounds are inclusive. Default: True.

        Returns
        -------
        Signal
            Trimmed copy.

        """
        assert start_time_s is not None or end_time_s is not None, (
            "At least one bound must be other than None"
        )
        if start_time_s:
            assert start_time_s >= 0.0, "Start time must be at least zero"
            assert start_time_s < self.length_seconds, (
                "Start time must be less than signal's length"
            )
            start_sample = int(start_time_s * self.sampling_rate_hz)
            if not inclusive:
                start_sample += 1
        else:
            start_sample = 0

        if end_time_s:
            assert end_time_s > 0.0, "End time must be greater than 0"
            assert end_time_s <= self.length_seconds, (
                "End time must be less than signal length"
            )
            end_sample = int(end_time_s * self.sampling_rate_hz)
            if inclusive:
                end_sample += 1
        else:
            end_sample = self.length_samples

        assert end_sample > start_sample, "Invalid time window"
        selection = slice(start_sample, end_sample)
        return self.copy_with_new_time_data(self.time_data[selection, ...])

    def normalize(
        self,
        norm_dbfs: float,
        peak_normalization: bool = True,
        each_channel: bool = False,
    ) -> "Signal":
        """Return a copy of the signal normalized to a given dBFS value. It
        either normalizes each channel or the signal as a whole.

        Parameters
        ----------
        norm_dbfs : float
            Value in dBFS to reach after normalization.
        peak_normalization : bool, optional
            When True, signal is normalized at peak. False uses RMS value.
            See notes. Default: True.
        each_channel : bool, optional
            When `True`, each channel on its own is normalized. When `False`,
            the peak or rms value across all channels is regarded.
            Default: `False`.

        Returns
        -------
        Signal
            Normalized signal.

        Notes
        -----
        - Normalization can be done for peak or RMS. The latter might
          generate a signal with samples above 0 dBFS if
          `signal.constrain_amplitude=False`.

        """
        return self.copy_with_new_time_data(
            _normalize(self.time_data, norm_dbfs, peak_normalization, each_channel)
        )

    def fade(
        self,
        fade_type: FadeType,
        length_fade_seconds: float | None = None,
        at_start: bool = True,
        at_end: bool = True,
    ) -> "Signal":
        """Return a copy of the signal with fading applied.

        Parameters
        ----------
        fade_type : FadeType
            Type of fading to be applied.
        length_fade_seconds : float, optional
            Fade length in seconds. If `None`, 2.5% of the signal's length is
            used for the fade. Default: `None`.
        at_start : bool, optional
            When `True`, the start of signal is faded. Default: `True`.
        at_end : bool, optional
            When `True`, the ending of signal is faded. Default: `True`.

        Returns
        -------
        Signal
            New signal.

        """
        assert at_start or at_end, "At least start or end of signal should be faded"
        if length_fade_seconds is None:
            length_fade_seconds = self.time_vector_s[-1] * 0.025
        assert length_fade_seconds < self.time_vector_s[-1], (
            "Fade length should not be longer than the signal itself"
        )

        new_time_data = np.empty_like(self.time_data)
        for n in range(self.number_of_channels):
            vec = self.time_data[:, n].copy()
            if at_start:
                vec = _fade(
                    vec,
                    length_fade_seconds,
                    mode=fade_type,
                    sampling_rate_hz=self.sampling_rate_hz,
                    at_start=True,
                )
            if at_end:
                vec = _fade(
                    vec,
                    length_fade_seconds,
                    mode=fade_type,
                    sampling_rate_hz=self.sampling_rate_hz,
                    at_start=False,
                )
            new_time_data[:, n] = vec
        return self.copy_with_new_time_data(new_time_data)

    def apply_gain(self, gain_db: float | NDArray[np.float64]) -> "Signal":
        """Return a copy of the signal with gain applied, either to the
        signal as a whole or per channel.

        Parameters
        ----------
        gain_db : float, NDArray[np.float64]
            Gain in dB to be applied. If it is an array, it should have as
            many elements as there are channels in the signal.

        Returns
        -------
        Signal
            Signal with new gain.

        Notes
        -----
        - If `constrain_amplitude=True` in the signal, the resulting time
          data might get rescaled after applying the gain.

        """
        gain_linear = from_db(np.atleast_1d(gain_db), True)
        if len(gain_linear) == 1:
            gain_linear = gain_linear[0]
        new_sig = self.copy_with_new_time_data(self.time_data * gain_linear)
        if new_sig.is_complex_signal:
            new_sig.time_data_imaginary *= gain_linear
        return new_sig

    def detrend(self, polynomial_order: int = 0) -> "Signal":
        """Return the detrended signal.

        Parameters
        ----------
        polynomial_order : int, optional
            Polynomial order of the fitted polynomial that will be removed
            from time data. 0 is equal to mean removal. Default: 0.

        Returns
        -------
        Signal
            Detrended signal.

        """
        assert polynomial_order >= 0, "Polynomial order should be positive"
        return self.copy_with_new_time_data(
            _detrend(self.time_data.copy(), polynomial_order)
        )

    def dither(
        self,
        triangular_distribution: bool = True,
        epsilon: float = float(np.finfo(np.float16).smallest_subnormal),
        noise_shaping_filterbank: "FilterBank | None" = None,
        truncate: bool = False,
    ) -> "Signal":
        """Return a copy of the signal with dither applied and, optionally,
        truncated to 16-bit floating point representation.

        Parameters
        ----------
        triangular_distribution : bool, optional
            Type of probability distribution to acquire noise from. When
            True, a rectangular distribution is used, otherwise it is
            uniform. Default: True.
        epsilon : float, optional
            Value that represents the quantization step. The default value
            supposes quantization to 16-bit floating point. It is obtained
            through numpy's smallest subnormal for np.float16. See notes for
            the value concerning the 24-bit case. Default: 6e-08.
        noise_shaping_filterbank : `FilterBank`, `None`, optional
            Noise can be arbitrarily shaped using a filter bank (in
            sequential mode). Pass `None` to avoid any noise-shaping.
            Default: `None`.
        truncate : bool, optional
            When `True`, the time samples are truncated to np.float16
            resolution. `False` only applies dither noise to the signal
            without truncating. Default: `False`.

        Returns
        -------
        Signal
            Signal with dither.

        Notes
        -----
        - The output signal has time samples with 16-bit precision, but the
          data type of the array is `np.float64` for consistency.
        - Rectangular distribution applies noise with samples coming from a
          uniform distribution [-epsilon/2, epsilon/2]. Triangular has a
          triangle shape for the noise distribution with values between
          [-epsilon, epsilon]. See [1] for more details.
        - Dither might be only necessary when lowering the bit-depth down to
          16 bits, though the 24-bit case might be relevant if there are
          signal components with very low volumes.
        - 24-bit signed integers range from -8388608 to 8388607. The
          quantization step is therefore `1/8388608=1.1920928955078125e-07`.

        References
        ----------
        - [1]: Lerch, Weinzierl. Handbuch der Audiotechnik: Chapter 14.

        """
        shape = self.time_data.shape

        if not triangular_distribution:
            noise = np.random.uniform(-epsilon / 2, epsilon / 2, size=shape)
        else:
            noise = np.random.uniform(
                -epsilon / 2, epsilon / 2, size=shape
            ) + np.random.uniform(-epsilon / 2, epsilon / 2, size=shape)

        if noise_shaping_filterbank is not None:
            noise_s = Signal(None, noise, self.sampling_rate_hz)
            noise_s = noise_shaping_filterbank.filter_signal(
                noise_s, mode=FilterBankMode.Sequential
            )
            noise = noise_s.time_data

        if truncate:
            return self.copy_with_new_time_data(
                (self.time_data + noise).astype(np.float16).astype(np.float64)
            )

        return self.copy_with_new_time_data(self.time_data + noise)

    def activity_detector(
        self,
        threshold_dbfs: float = -20,
        channel: int = 0,
        relative_to_peak: bool = True,
        pre_filter: "Filter | None" = None,
        attack_time_ms: float = 1,
        release_time_ms: float = 25,
    ) -> tuple["Signal", dict]:
        """This is a simple signal activity detector that uses a power
        threshold. It can be used relative to the signal's peak value or
        absolute. It is only applicable to one channel of the signal. This
        method returns the signal and a dictionary containing noise (as a
        signal) and the time indices corresponding to the bins that were
        found to surpass the threshold according to attack and release
        times.

        Prefiltering (for example with a bandpass filter) is possible when a
        `pre_filter` is passed.

        See Returns to gain insight into the returned dictionary and its
        keys.

        Parameters
        ----------
        threshold_dbfs : float
            Threshold in dBFS to separate noise from activity.
        channel : int, optional
            Channel in which to perform the detection. Default: 0.
        relative_to_peak : bool, optional
            When `True`, the threshold value is relative to the signal's peak
            value. Otherwise, it is regarded as an absolute threshold.
            Default: `True`.
        pre_filter : `Filter`, optional
            Filter used for prefiltering the signal. It can be for instance a
            bandpass filter selecting the relevant frequencies in which the
            activity might be. Pass `None` to avoid any pre filtering. The
            filter is applied using zero-phase filtering. Default: `None`.
        attack_time_ms : float, optional
            Attack time (in ms). It corresponds to a lag time for detecting
            activity after surpassing the threshold. Default: 1.
        release_time_ms : float, optional
            Release time (in ms) for activity detector after signal has
            fallen below power threshold. Pass 0 to release immediately.
            Default: 25.

        Returns
        -------
        detected_sig : `Signal`
            Detected signal.
        others : dict
            Dictionary containing following keys:
            - `'noise'`: left-out noise in original signal (below threshold)
              as `Signal` object.
            - `'signal_indices'`: array of boolean that describes which
              indices of the original time series belong to signal and which
              to noise. `True` at index n means index n was passed to
              signal.
            - `'noise_indices'`: the inverse array to `'signal_indices'`.

        """
        assert isinstance(channel, int), (
            "Channel must be type integer. Function is not implemented for "
            + "multiple channels."
        )
        assert threshold_dbfs < 0, "Threshold must be below zero"
        assert release_time_ms >= 0, "Release time must be positive"
        assert attack_time_ms >= 0, "Attack time must be positive"

        # Get channel
        signal = self.get_channels(channel)

        # Pre-filtering
        if pre_filter is not None:
            from .filter import Filter

            assert isinstance(pre_filter, Filter), "pre_filter must be of type Filter"
            signal_filtered = pre_filter.filter_signal(signal, zero_phase=True)
        else:
            signal_filtered = signal

        # Release samples
        attack_coeff = _get_smoothing_factor_ema(
            attack_time_ms / 1e3, signal.sampling_rate_hz
        )
        release_coeff = _get_smoothing_factor_ema(
            release_time_ms / 1e3, signal.sampling_rate_hz
        )

        # Get indices
        signal_indices = _indices_above_threshold_dbfs(
            signal_filtered.time_data.copy(),
            threshold_dbfs=threshold_dbfs,
            attack_smoothing_coeff=attack_coeff,
            release_smoothing_coeff=release_coeff,
            normalize=relative_to_peak,
        )
        noise_indices = ~signal_indices

        # Separate signals
        detected_sig = signal.copy()
        noise = signal.copy()

        try:
            detected_sig.time_data = signal.time_data[signal_indices, 0]
        except ValueError as e:
            warn(
                "No detected activity, threshold might be too high. Detected "
                + "signal will be a vector filled with zeroes",
                stacklevel=2,
            )
            print("Numpy error: ", e)
            detected_sig.time_data = np.zeros(500)

        try:
            noise.time_data = signal.time_data[noise_indices, 0]
        except ValueError as e:
            warn(
                "No detected noise, threshold might be too low. Noise will be "
                + "a vector filled with zeroes",
                stacklevel=2,
            )
            print("Numpy error: ", e)
            noise.time_data = np.zeros(500)

        others = dict(
            noise=noise, signal_indices=signal_indices, noise_indices=noise_indices
        )
        return detected_sig, others

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
        from .spectrum import Spectrum

        return Spectrum.from_signal(self, complex).spectral_difference(
            other,
            octave_fraction_smoothing,
            energy_normalization,
            complex,
            dynamic_range_db,
        )

    def fractional_delay(
        self,
        delay_seconds: float,
        channels=None,
        keep_length: bool = False,
        order: int = 30,
        side_lobe_suppression_db: float = 60,
    ) -> "Signal":
        """Return a copy of the signal with fractional time delay applied.

        Parameters
        ----------
        delay_seconds : float
            Delay in seconds.
        channels : int or array-like, optional
            Channels to be delayed. Pass `None` to delay all channels.
            Default: `None`.
        keep_length : bool, optional
            When `True`, the signal retains its original length and loses
            information for the latest samples. If only specific channels
            are to be delayed, and keep_length is set to `False`, the
            remaining channels are zero-padded in the end. Default: `False`.
        order : int, optional
            Order of the sinc filter, higher order yields better results at
            the expense of computation time. Default: 30.
        side_lobe_suppression_db : float, optional
            Side lobe suppression in dB for the Kaiser window. Default: 60.

        Returns
        -------
        Signal
            Delayed signal.

        """
        assert delay_seconds >= 0, "Delay must be positive"
        if delay_seconds == 0:
            return self.copy()
        if self.time_data_imaginary is not None:
            warn(
                "Imaginary time data will be ignored in this function. "
                + "Delay it manually by creating another signal object, if "
                + "needed.",
                stacklevel=2,
            )
        delay_samples = delay_seconds * self.sampling_rate_hz
        if keep_length:
            assert delay_samples < self.time_data.shape[0], (
                "Delay too large for the given signal"
            )
        if channels is None:
            channels = np.arange(self.number_of_channels)
        channels = np.atleast_1d(np.asarray(channels).squeeze())
        assert np.all(channels < self.number_of_channels) and len(
            np.unique(channels)
        ) == len(channels), "There is at least an invalid channel number"

        # Get filter and integer delay
        delay_int, frac_delay_filter = _fractional_delay_filter(
            delay_samples, order, side_lobe_suppression_db
        )

        # Copy data
        new_time_data = self.time_data

        # Create space for the filter in the end of signal
        new_time_data = _pad_trim(
            new_time_data, self.time_data.shape[0] + len(frac_delay_filter) - 1
        )

        # Delay channels
        new_time_data[:, channels] = oaconvolve(
            self.time_data[:, channels],
            frac_delay_filter[..., None],
            mode="full",
            axes=0,
        )

        # Handle delayed and undelayed channels
        channels_not = np.setdiff1d(np.arange(new_time_data.shape[1]), channels)
        not_delayed = new_time_data[:, channels_not]
        delayed = new_time_data[:, channels]

        # Delay respective channels in the beginning and add zeros in the end
        # to the others
        delayed = _pad_trim(
            delayed, delay_int + new_time_data.shape[0], in_the_end=False
        )
        not_delayed = _pad_trim(
            not_delayed, delay_int + new_time_data.shape[0], in_the_end=True
        )

        new_time_data = _pad_trim(
            new_time_data, delay_int + new_time_data.shape[0], in_the_end=True
        )
        new_time_data[:, channels_not] = not_delayed
        new_time_data[:, channels] = delayed

        # =========== handle length ===========================================
        if keep_length:
            new_time_data = new_time_data[: self.time_data.shape[0], :]

        return self.copy_with_new_time_data(new_time_data)

    def delay(
        self,
        delay_samples: int,
        channels=None,
        keep_length: bool = False,
    ) -> "Signal":
        """Return a copy of the signal with a time delay applied. This
        method is faster than `fractional_delay` because it only applies
        integer delay by zero-padding.

        Parameters
        ----------
        delay_samples : int
            Delay in samples.
        channels : int or array-like, optional
            Channels to be delayed. Pass `None` to delay all channels.
            Default: `None`.
        keep_length : bool, optional
            When `True`, the signal retains its original length and loses
            information for the latest samples. If only specific channels
            are to be delayed, and keep_length is set to `False`, the
            remaining channels are zero-padded in the end. Default: `False`.

        Returns
        -------
        Signal
            Delayed signal.

        """
        if delay_samples == 0:
            return self.copy()
        if keep_length:
            assert delay_samples < self.time_data.shape[0], (
                "Delay too large for the given signal"
            )
        if channels is None:
            channels = np.arange(self.number_of_channels)
        channels = np.atleast_1d(np.asarray(channels).squeeze())
        assert np.all(channels < self.number_of_channels) and len(
            np.unique(channels)
        ) == len(channels), "There is at least an invalid channel number"

        # Copy data
        new_time_data = self.time_data

        # Handle delayed and undelayed channels
        channels_not = np.setdiff1d(np.arange(new_time_data.shape[1]), channels)
        not_delayed = new_time_data[:, channels_not]
        delayed = new_time_data[:, channels]

        delayed = _pad_trim(
            delayed, delay_samples + new_time_data.shape[0], in_the_end=False
        )
        not_delayed = _pad_trim(
            not_delayed,
            delay_samples + new_time_data.shape[0],
            in_the_end=True,
        )

        new_time_data = _pad_trim(
            new_time_data,
            delay_samples + new_time_data.shape[0],
            in_the_end=True,
        )
        new_time_data[:, channels_not] = not_delayed
        new_time_data[:, channels] = delayed
        if keep_length:
            new_time_data = new_time_data[: self.time_data.shape[0], :]

        return self.copy_with_new_time_data(new_time_data)

    def resample(
        self, desired_sampling_rate_hz: int, rescaling: bool = False
    ) -> "Signal":
        """Return a copy of the signal resampled to the desired sampling
        rate using `scipy.signal.resample_poly` with an efficient polyphase
        representation.

        Parameters
        ----------
        desired_sampling_rate_hz : int
            Sampling rate to convert the signal to.
        rescaling : bool, optional
            When True, the data is rescaled by dividing by the resampling
            factor. This retains the magnitude scaling when regarding the
            unscaled spectrum. Default: False.

        Returns
        -------
        Signal
            Resampled signal.

        """
        if self.sampling_rate_hz == desired_sampling_rate_hz:
            return self.copy()
        ratio = Fraction(
            numerator=desired_sampling_rate_hz, denominator=self.sampling_rate_hz
        )
        u, d = ratio.as_integer_ratio()
        new_time_data = resample_poly(self.time_data, up=u, down=d, axis=0)
        new_sig = self.copy_with_new_time_data(
            new_time_data * (d / u) if rescaling else new_time_data
        )
        new_sig.sampling_rate_hz = desired_sampling_rate_hz
        return new_sig

    def append_signals(
        self,
        others: list["Signal"],
        allow_padding_trimming: bool = True,
        at_end: bool = True,
    ) -> "Signal":
        """Return a copy of the signal with the channels of other signals
        appended. If their lengths are not the same, trimming or padding can
        be applied to match this signal's length.

        Parameters
        ----------
        others : list[Signal]
            Other signals whose channels should be appended.
        allow_padding_trimming : bool, optional
            If the signals do not have the same length, all are trimmed or
            zero-padded to match this signal's length, when this is True.
            Otherwise, an error will be raised if the lengths do not match.
            Default: `True`.
        at_end : bool, optional
            When `True` and `allow_padding_trimming=True`, padding or
            trimming is done at the end of the signals. Otherwise, it is
            done in the beginning. Default: `True`.

        Returns
        -------
        Signal
            Signal with all channels.

        """
        signals = [self] + list(others)
        assert len(signals) > 1, "At least one other signal should be passed"

        complex_data = False
        for s in signals:
            assert isinstance(s, Signal), (
                "All signals must be of type Signal or ImpulseResponse"
            )
            assert s.sampling_rate_hz == signals[0].sampling_rate_hz, (
                "Sampling rates do not match"
            )
            if not allow_padding_trimming:
                assert len(s) == len(signals[0]), (
                    "Lengths do not match and padding or trimming " + "is not activated"
                )
            complex_data |= s.is_complex_signal

        total_n_channels = sum([s.number_of_channels for s in signals])
        total_length = len(signals[0])
        td = np.zeros(
            (len(signals[0]), total_n_channels),
            dtype=np.complex128 if complex_data else np.float64,
        )

        current_channel = 0
        for s in signals:
            if complex_data:
                if s.is_complex_signal:
                    td[
                        :,
                        current_channel : current_channel + s.number_of_channels,
                    ] = _pad_trim(
                        s.time_data + 1j * s.time_data_imaginary,
                        total_length,
                        in_the_end=at_end,
                    )
                else:
                    td[
                        :,
                        current_channel : current_channel + s.number_of_channels,
                    ] = _pad_trim(
                        s.time_data.astype(np.complex128),
                        total_length,
                        in_the_end=at_end,
                    )
            else:
                td[:, current_channel : current_channel + s.number_of_channels] = (
                    _pad_trim(s.time_data, total_length, in_the_end=at_end)
                )
            current_channel += s.number_of_channels
        new_sig = self.copy()
        new_sig.time_data = td
        return new_sig

    def show_info(self):
        """Prints all the signal information to the console."""
        print(self.metadata_str)
        return self
