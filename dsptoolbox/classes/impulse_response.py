from typing import Self

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from ..helpers.gain_and_level import to_db
from ..helpers.latency import _apply_ir_latency_removal_to_phase
from ..helpers.spectrum_utilities import _get_normalization_offset_db
from ..plots import general_plot_two_axes
from ..standard._standard_backend import _group_delay_direct
from ..standard.enums import (
    IrLatencyRemoval,
    IrLatencyRemovalType,
    MagnitudeNormalization,
    SpectrumMethod,
)
from .signal import Signal


class ImpulseResponse(Signal):
    def __init__(
        self,
        path: str | None = None,
        time_data: NDArray[np.float64] | None = None,
        sampling_rate_hz: int | None = None,
        constrain_amplitude: bool = False,
        activate_cache: bool = False,
    ) -> None:
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
            Default: `False`.
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
    def from_signal(signal: Signal) -> "ImpulseResponse":
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
    def from_file(
        path: str,
        constrain_amplitude: bool = False,
        activate_cache: bool = False,
    ) -> "ImpulseResponse":
        """Create an impulse response from a path to a wav or flac audio file.

        Parameters
        ----------
        path : str
            Path to file.
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

        Returns
        -------
        ImpulseResponse

        """
        return ImpulseResponse.from_signal(
            Signal.from_file(path, constrain_amplitude, activate_cache)
        )

    @staticmethod
    def from_time_data(
        time_data: NDArray[np.float64],
        sampling_rate_hz: int,
        constrain_amplitude: bool = False,
        activate_cache: bool = False,
    ) -> "ImpulseResponse":
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
            Default: `False`.
        activate_cache : bool, optional
            When True, spectra, CSM and STFT will be cached. They will not
            be computed again if no parameters have changed. Set to False to
            avoid caching altogether. Default: False.

        Returns
        -------
        ImpulseResponse

        """
        return ImpulseResponse.from_signal(
            Signal.from_time_data(
                time_data, sampling_rate_hz, constrain_amplitude, activate_cache
            )
        )

    def set_window(self, window: NDArray[np.float64]) -> Self:
        """Return a copy of the IR with the window set.

        Parameters
        ----------
        window : NDArray[np.float64]
            Window used for the IR.

        Returns
        -------
        ImpulseResponse
            New impulse response with the window set.

        """
        assert window.shape == self.time_data.shape, (
            f"{window.shape} does not match shape {self.time_data.shape}"
        )
        new = self.copy()
        new.window = window
        return new

    def plot_time(
        self,
        ax: list[Axes] | None = None,
    ) -> tuple[Figure, list[Axes]]:
        """Plots time signals.

        ax : list of `matplotlib.axes.Axes`, None, optional
            Axes to draw on, one per channel. New ones are created when None.
            Default: None.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : list of `matplotlib.axes.Axes`
            Axes.

        """
        fig, ax = super().plot_time(ax=ax)
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
        dynamic_range_db: float | None = 100.0,
        window_length_s: float = 0.0,
        ax: list[Axes] | None = None,
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

        ax : list of `matplotlib.axes.Axes`, None, optional
            Axes to draw on, one per channel. New ones are created when None.
            Default: None.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : list of `matplotlib.axes.Axes`
            Axes.

        Notes
        -----
        - All values are clipped to at most 500 dB below the peak.
        - If it is an analytic signal and normalization is applied, the peak
          value of the real part is used as the normalization factor.
        - If the time window is not 0, effects at the edges of the signal might
          be present due to zero-padding.

        """
        fig, ax = super().plot_spl(
            normalize_at_peak, dynamic_range_db, window_length_s, ax=ax
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
        range_hz: tuple[float, float] | None = (20.0, 20e3),
        normalize: MagnitudeNormalization = MagnitudeNormalization.NoNormalization,
        range_db: tuple[float, float] | None = None,
        show_group_delay: bool = False,
        range_rad_s: tuple[float, float] | None = None,
        smoothing: int = 0,
        remove_ir_latency: IrLatencyRemovalType = IrLatencyRemoval.NoRemoval,
        ax: Axes | None = None,
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
        range_rad_s : array-like with length 2, optional
            Range for plotting the group delay or phase response. Default:
            None.
        smoothing : int, optional
            Smoothing across the (1/smoothing) octave band. It only applies to
            the plot data and not to `get_spectrum()`. It is applied to both
            magnitude and phase/group delay response. Default: 0
            (no smoothing).
        remove_ir_latency : IrLatencyRemoval, optional
            If the signal is an impulse response, the delay of the impulse can
            be removed. The delay is either estimated from the minimum-phase
            equivalent or from the peak in the time signal, or passed
            explicitly with `IrLatencyRemoval.Custom.with_delay_samples()`.
            Default: NoRemoval.

        ax : `matplotlib.axes.Axes`, None, optional
            Axes to draw on, so that several plots can share one axis. A new
            figure is created when None. Default: None.

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
        sp_abs_db = to_db(sp_abs, True)
        sp_abs_db -= _get_normalization_offset_db(
            normalize,
            f,
            sp_abs_db,
            to_db(np.mean(sp_abs**2.0, axis=0), False),
        )[None, :]

        phase = np.angle(sp)
        phase = _apply_ir_latency_removal_to_phase(
            remove_ir_latency, f, phase, self.time_data, self.sampling_rate_hz
        )

        fig, ax = general_plot_two_axes(
            f,
            sp_abs_db,
            f,
            (
                _group_delay_direct(phase, f[1] - f[0]) * 1e3
                if show_group_delay
                else phase
            ),
            range_x=range_hz,
            range_y1=range_db,
            range_y2=range_rad_s,
            log_x=True,
            labels1=[f"Channel {n}" for n in range(self.number_of_channels)],
            y1label="Magnitude / dB",
            y2label=("Group delay / ms" if show_group_delay else "Phase / rad"),
            y2_linestyle="dashed",
            y2_alpha=0.6,
            ax=ax,
        )
        ax[-1].grid(linestyle="dashed")

        return fig, ax

    def copy_with_new_time_data(self, new_time_data: ArrayLike) -> Self:
        new_signal = super().copy_with_new_time_data(new_time_data)
        if self.spectrum_method != SpectrumMethod.FFT:
            new_signal.spectrum_method = SpectrumMethod.FFT
        return new_signal
