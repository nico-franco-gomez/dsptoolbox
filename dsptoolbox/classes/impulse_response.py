import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .signal import Signal
from ..plots import general_subplots_line
from ..tools import to_db


class ImpulseResponse(Signal):
    def __init__(
        self,
        path: str | None = None,
        time_data: NDArray[np.float64] | None = None,
        sampling_rate_hz: int | None = None,
        constrain_amplitude: bool = True,
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

        Returns
        -------
        ImpulseResponse

        """
        super().__init__(
            path,
            time_data,
            sampling_rate_hz,
            constrain_amplitude=constrain_amplitude,
        )
        self.set_spectrum_parameters(method="standard")

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
        ir.amplitude_scale_factor = signal.amplitude_scale_factor
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

    def set_coherence(self, coherence: NDArray[np.float64]):
        """Sets the coherence measurements of the transfer function.
        It only works for `signal_type = ('ir', 'h1', 'h2', 'h3', 'rir')`.

        Parameters
        ----------
        coherence : NDArray[np.float64]
            Coherence matrix.

        """
        assert coherence.shape[0] == (
            self.time_data.shape[0] // 2 + 1
        ), "Length of signals and given coherence do not match"
        assert coherence.shape[1] == self.number_of_channels, (
            "Number of channels between given coherence and signal "
            + "does not match"
        )
        self.coherence = coherence

    def get_coherence(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns the coherence matrix.

        Returns
        -------
        f : NDArray[np.float64]
            Frequency vector.
        coherence : NDArray[np.float64]
            Coherence matrix.

        """
        assert hasattr(
            self, "coherence"
        ), "There is no coherence data saved in the Signal object"
        f, _ = self.get_spectrum()
        return f, self.coherence

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

    def plot_coherence(self) -> tuple[Figure, list[Axes]]:
        """Plots coherence measurements if there are any.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : list of `matplotlib.axes.Axes`
            Axes.

        """
        f, coh = self.get_coherence()
        fig, ax = general_subplots_line(
            x=f,
            matrix=coh,
            column=True,
            sharey=True,
            log=True,
            ylabels=[
                rf"$\gamma^2$ Coherence {n}"
                for n in range(self.number_of_channels)
            ],
            range_x=[20, 20e3],
            xlabels="Frequency / Hz",
            range_y=[-0.1, 1.1],
            returns=True,
        )
        return fig, ax
