import numpy as np
from numpy.typing import NDArray, ArrayLike
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from copy import deepcopy

from .signal import Signal
from ..helpers.gain_and_level import to_db
from ..standard.enums import SpectrumMethod, MagnitudeNormalization
from ..plots import general_plot_two_axes

from ..helpers.spectrum_utilities import _get_exact_gain_1khz
from ..helpers.latency import (
    _remove_ir_latency_from_phase,
    _remove_ir_latency_from_phase_peak,
)
from ..helpers.minimum_phase import _remove_ir_latency_from_phase_min_phase
from ..standard._standard_backend import _group_delay_direct


class ImpulseResponse(Signal):
    def __init__(
        self,
        path: str | None = None,
        time_data: NDArray[np.float64] | None = None,
        sampling_rate_hz: int | None = None,
        constrain_amplitude: bool = True,
        activate_cache: bool = False,
    ):
        """Instantiate impulse response.

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
            Default: `True`.
        activate_cache : bool, optional
            When True, spectra, CSM and STFT will be cached. They will not
            be computed again if no parameters have changed. Set to False to
            avoid caching altogether. Default: False.

        Returns
        -------
        ImpulseResponse

        """
        super().__init__(
            path,
            time_data,
            sampling_rate_hz,
            constrain_amplitude=constrain_amplitude,
            activate_cache=activate_cache,
        )
        self.spectrum_method = SpectrumMethod.FFT

    @staticmethod
    def from_signal(signal: Signal):
        """Create an impulse response from a signal.

        Parameters
        ----------
        signal : `Signal`

        Returns
        -------
        ImpulseResponse

        """
        ir = ImpulseResponse(
            None,
            signal.time_data,
            signal.sampling_rate_hz,
            signal.constrain_amplitude,
        )
        ir.calibrated_signal = signal.calibrated_signal
        ir.time_data_imaginary = signal.time_data_imaginary
        return ir

    @staticmethod
    def from_file(path: str):
        """Create an impulse response from a path to a wav or flac audio file.

        Parameters
        ----------
        path : str
            Path to file.

        Returns
        -------
        ImpulseResponse

        """
        s = Signal.from_file(path)
        return ImpulseResponse.from_signal(s)

    @staticmethod
    def from_time_data(
        time_data: NDArray[np.float64],
        sampling_rate_hz: int,
        constrain_amplitude: bool = True,
    ):
        """Create an impulse response from an array of PCM samples.

        Parameters
        ----------
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
            Default: `True`.

        Returns
        -------
        ImpulseResponse

        """
        s = Signal.from_time_data(
            time_data, sampling_rate_hz, constrain_amplitude
        )
        return ImpulseResponse.from_signal(s)

    def set_window(self, window: NDArray[np.float64]):
        """Sets the window used for the IR.

        Parameters
        ----------
        window : NDArray[np.float64]
            Window used for the IR.

        """
        assert (
            window.shape == self.time_data.shape
        ), f"{window.shape} does not match shape {self.time_data.shape}"
        self.window = window
        return self

    def plot_time(self) -> tuple[Figure, list[Axes]]:
        """Plots time signals.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : list of `matplotlib.axes.Axes`
            Axes.

        """
        fig, ax = super().plot_time()
        if hasattr(self, "window"):
            mx = np.max(np.abs(self.time_data), axis=0)

            for n in range(self.number_of_channels):
                ax[n].plot(
                    self.time_vector_s,
                    self.window[:, n] * mx[n],
                    alpha=0.75,
                )
        return fig, ax

    def plot_spl(
        self,
        normalize_at_peak: bool = False,
        range_db: float | None = 100.0,
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
        range_db : float, optional
            This is the range in dB used for plotting. Each plot will be in the
            range [peak + 1 - range_db, peak + 1]. Pass `None` to avoid setting
            any range. Default: 100.
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
        fig, ax = super().plot_spl(
            normalize_at_peak, range_db, window_length_s
        )

        peak_values = to_db(np.max(np.abs(self.time_data), axis=0), True)

        max_values = (
            peak_values + 1  # Add 1 dB for better plotting
            if not normalize_at_peak
            else np.ones(self.number_of_channels)
        )

        for n in range(self.number_of_channels):
            if hasattr(self, "window"):
                ax[n].plot(
                    self.time_vector_s,
                    to_db(self.window[:, n] / 1.1, True, dynamic_range_db=500)
                    + max_values[n],
                    alpha=0.75,
                )
        return fig, ax

    def plot_bode(
        self,
        range_hz=[20, 20e3],
        normalize: MagnitudeNormalization = MagnitudeNormalization.NoNormalization,
        range_db=None,
        show_group_delay: bool = False,
        range_rad_s=None,
        smoothing: int = 0,
        remove_ir_latency: str | None | ArrayLike = None,
    ) -> tuple[Figure, list[Axes]]:
        """Create a bode plot where magnitude and phase response are plotted
        together.

        Parameters
        ----------
        range_hz : array-like with length 2, optional
            Range for which to plot the magnitude response.
            Default: [20, 20000].
        normalize : MagnitudeNormalization, optional
            Mode for normalization. Default: NoNormalization.
        range_db : array-like with length 2, optional
            Range in dB for which to plot the magnitude response.
            Default: None.
        show_group_delay : bool, optional
            When True, the group delay is shown instead of the phase response.
            It is computed with the numerical derivative of the phase response.
            Default: False.
        range_s : array-like with length 2, optional
            Range for plotting the group delay or phase response. Default:
            None.
        smoothing : int, optional
            Smoothing across the (1/smoothing) octave band. It only applies to
            the plot data and not to `get_spectrum()`. It is applied to both
            magnitude and phase/group delay response. Default: 0
            (no smoothing).
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
        ax : list of `matplotlib.axes.Axes`
            List containing the two axes of the plot.

        """
        prior_smoothing = self.spectrum_smoothing
        self.spectrum_smoothing = smoothing
        f, sp = self.get_spectrum()
        self.spectrum_smoothing = prior_smoothing
        sp_abs = np.abs(sp)

        if normalize == MagnitudeNormalization.OneKhz:
            sp_abs /= _get_exact_gain_1khz(f, sp_abs)[None, ...]
        elif normalize == MagnitudeNormalization.Max:
            sp_abs /= np.max(sp_abs, axis=0, keepdims=True)
        elif normalize == MagnitudeNormalization.Energy:
            sp_abs /= np.mean(sp_abs**2.0, axis=0, keepdims=True) ** 0.5

        phase = np.angle(sp)
        if remove_ir_latency is None:
            pass
        elif type(remove_ir_latency) is str:
            match remove_ir_latency.lower():
                case "peak":
                    phase = _remove_ir_latency_from_phase_peak(
                        f, phase, self.time_data, self.sampling_rate_hz
                    )
                case "min_phase":
                    phase = _remove_ir_latency_from_phase_min_phase(
                        f, phase, self.time_data, self.sampling_rate_hz, 8
                    )
                case _:
                    raise ValueError("No valid latency removal")
        else:
            delays_samples = np.atleast_1d(remove_ir_latency)
            phase = _remove_ir_latency_from_phase(
                f, phase, delays_samples, self.sampling_rate_hz
            )

        fig, ax = general_plot_two_axes(
            f,
            to_db(sp_abs, True),
            f,
            (
                _group_delay_direct(phase, f[1] - f[0])
                if show_group_delay
                else phase
            ),
            range_x=range_hz,
            range_y1=range_db,
            range_y2=range_rad_s,
            log_x=True,
            labels1=[f"Channel {n}" for n in range(self.number_of_channels)],
            y1label="Magnitude / dB",
            y2label=("Group Delay / s" if show_group_delay else "Phase / rad"),
            y2_linestyle="dashed",
            y2_alpha=0.6,
        )
        ax[-1].grid(linestyle="dashed")

        return fig, ax

    def copy_with_new_time_data(
        self, new_time_data: ArrayLike
    ) -> "ImpulseResponse":
        # Copy if the underlying memory belongs to another array
        if isinstance(new_time_data, np.ndarray):
            new_time_data = (
                new_time_data
                if new_time_data.base is None
                else new_time_data.copy()
            )
        #
        new_signal = ImpulseResponse.from_time_data(
            new_time_data, self.sampling_rate_hz, self.constrain_amplitude
        )
        new_signal.calibrated_signal = self.calibrated_signal
        new_signal.activate_cache = self.activate_cache
        new_signal._spectrum_parameters = deepcopy(self._spectrum_parameters)
        new_signal._spectrogram_parameters = deepcopy(
            self._spectrogram_parameters
        )
        if self.spectrum_method != SpectrumMethod.FFT:
            new_signal.spectrum_method = SpectrumMethod.FFT
        return new_signal
