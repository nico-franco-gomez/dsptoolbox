"""
State variable filter topology-Preserving (trapezoidal integrators)
2-Pole multimode filter
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from .signal_class import Signal
from .multibandsignal import MultiBandSignal
from ..generators import dirac


class StateVariableFilter:
    """This is a state variable filter discretized using the
    topology-preserving transform (trapezoidal integrator)."""

    def __init__(
        self, frequency_hz: float, resonance: float, sampling_rate_hz: int
    ):
        """Construct a state variable, 2-pole multimode filter. The
        implementation is based on [1], but the resonance parameter is here
        equal to 2R.

        Parameters
        ----------
        frequency_hz : float
            Cutoff frequency for the filter.
        resonance : float
            Resonance parameter. The filter becomes unstable for R = 0.
        sampling_rate_hz : int
            Sampling rate in Hz.

        Notes
        -----
        - Resonance can be linked to Quality factor as Q = 1/2R, i.e.,
          Q = 1/resonance.

        References
        ----------
        - [1]: Zavalishin, V.: The Art of VA Filter Design. Page 81.

        """
        self.sampling_rate_hz = sampling_rate_hz
        self.set_parameters(frequency_hz, resonance, 1)

    def set_parameters(
        self, frequency_hz: float, resonance: float, n_channels: int
    ):
        """Set filter parameters.

        Parameters
        ----------
        frequency_hz : float
            Cutoff frequency.
        resonance : float
            Resonance parameter.
        n_channels : int
            Number of channels to be filtered (Necessary for the filter's
            states).

        """
        assert frequency_hz > 0 and frequency_hz < self.sampling_rate_hz // 2
        self.g = np.tan(np.pi * frequency_hz / self.sampling_rate_hz)
        self.resonance = resonance
        self.intermediate_value = 1 / (1 + self.resonance * self.g + self.g**2)
        self.set_n_channels(n_channels)

    def set_n_channels(self, n_channels: int):
        """Set the number of channels for which the filter will be filtering.

        Parameters
        ----------
        n_channels : int
            Number of channels to be filtered.

        """
        assert n_channels > 0
        self.n_channels = n_channels
        self.state = np.zeros((2, self.n_channels))

    def _reset_state(self):
        """Reset filter states."""
        self.state.fill(0)

    def _process_sample(
        self, sample: float, channel: int = 0
    ) -> tuple[float, float, float, float]:
        """Process a single sample using a specific channel."""
        yh = (
            sample
            - (self.resonance + self.g) * self.state[0, channel]
            - self.state[1, channel]
        ) * self.intermediate_value

        yb = self.g * yh + self.state[0, channel]
        self.state[0, channel] = self.g * yh + yb

        yl = self.g * yb + self.state[1, channel]
        self.state[1, channel] = self.g * yb + yl

        return yl, yh, yb, yl - self.resonance * yb + yh

    def _process_vector(self, input: np.ndarray) -> np.ndarray:
        """Process a whole multichannel array. The outputs are a 3d-array with
        shape (time sample, band, channel). There are 4 bands: lowpass,
        highpass, bandpass and allpass. They are returned in this order.

        """
        input = np.atleast_2d(input)
        assert input.ndim < 3, "Too many dimensions for input array"
        outputs = np.zeros((max(input.shape), 4, min(input.shape)))
        if self.n_channels != input.shape[1]:
            self.set_n_channels(input.shape[1])

        for ch in range(input.shape[1]):
            for i in np.arange(len(input)):
                outputs[i, :, ch] = self._process_sample(
                    input[i, ch], channel=ch
                )
        return outputs

    def filter_signal(self, signal: Signal) -> MultiBandSignal:
        """Filter a signal.

        Parameters
        ----------
        signal : `Signal`
            Input signal to be filtered.

        Returns
        -------
        `MultiBandSignal`
            Multiband signal containing the four outputs of the filter:
            lowpass, highpass, bandpass and allpass (saved in this order in
            the multiband signal).

        """
        assert (
            self.sampling_rate_hz == signal.sampling_rate_hz
        ), "Sampling rates do not match"
        td = self._process_vector(signal.time_data)
        return MultiBandSignal(
            [
                Signal(
                    None,
                    td[:, i, :],
                    sampling_rate_hz=self.sampling_rate_hz,
                    signal_type=signal.signal_type,
                )
                for i in range(4)
            ]
        )

    def get_ir(self, length_samples: int = 1024) -> MultiBandSignal:
        """Get an IR from the VS-Filter.

        Parameters
        ----------
        length_samples : int, optional
            Length of the IR in samples. Default: 1024.

        Returns
        -------
        `MultiBandSignal`
            Impulse responses of the filter in following order: lowpass,
            highpass, bandpass and allpass.

        """
        d = dirac(length_samples, sampling_rate_hz=self.sampling_rate_hz)
        d.signal_type = "ir"
        self._reset_state()
        return self.filter_signal(d)

    def plot_magnitude(
        self,
        length_samples: int = 1024,
        range_hz: list | None = [20, 20e3],
        range_db: list | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the magnitude response of each band output of the filter.

        Parameters
        ----------
        length_samples : int, optional
            Length of the IR. Default: 1024.
        range_hz : list, None, optional
            Range of Hz to plot. Default: [20, 20e3].
        range_db : list, None, optional
            Range of dB to plot. Pass `None` to plot automatically.
            Default: `None`.

        Returns
        -------
        Figure, Axes
            Matplotlib's figure and axes.

        """
        d = self.get_ir(length_samples).get_all_bands()
        d.signal_type = "ir"
        d.set_spectrum_parameters(method="standard")
        fig, ax = d.plot_magnitude(
            range_hz=range_hz,
            normalize=None,
            range_db=range_db,
            smoothe=0,
            scale=False,
        )
        ax.legend(["Lowpass", "Highpass", "Bandpass", "Allpass"])
        return fig, ax

    def plot_group_delay(
        self,
        length_samples: int = 1024,
        range_hz: list | None = [20, 20e3],
    ) -> tuple[Figure, Axes]:
        """Plot the group delay of each band output of the filter.

        Parameters
        ----------
        length_samples : int, optional
            Length of the IR. Default: 1024.
        range_hz : list, None, optional
            Range of Hz to plot. Default: [20, 20e3].

        Returns
        -------
        Figure, Axes
            Matplotlib's figure and axes.

        """
        d = self.get_ir(length_samples).get_all_bands()
        d.signal_type = "ir"
        d.set_spectrum_parameters(method="standard")
        fig, ax = d.plot_group_delay(range_hz=range_hz)
        ax.legend(["Lowpass", "Highpass", "Bandpass", "Allpass"])
        return fig, ax

    def plot_phase(
        self,
        length_samples: int = 1024,
        range_hz: list | None = [20, 20e3],
        unwrap: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plot the phase of each band output of the filter.

        Parameters
        ----------
        length_samples : int, optional
            Length of the IR. Default: 1024.
        range_hz : list, None, optional
            Range of Hz to plot. Default: [20, 20e3].
        unwrap : bool, optional
            When `True`, the phase response is unwrapped. Default: `False`.

        Returns
        -------
        Figure, Axes
            Matplotlib's figure and axes.

        """
        d = self.get_ir(length_samples).get_all_bands()
        d.signal_type = "ir"
        d.set_spectrum_parameters(method="standard")
        fig, ax = d.plot_phase(range_hz=range_hz, unwrap=unwrap)
        ax.legend(["Lowpass", "Highpass", "Bandpass", "Allpass"])
        return fig, ax
