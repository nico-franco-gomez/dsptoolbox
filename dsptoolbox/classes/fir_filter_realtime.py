from numpy.typing import NDArray
import numpy as np

from .realtime_filter import RealtimeFilter


class FIRFilter(RealtimeFilter):
    """FIR filter implemented in the time domain. This class is
    written for experimentation purposes, but using `scipy.signal.lfilter` or
    some convolution function should be preferred for usual filtering tasks."""

    def __init__(self, b: NDArray[np.float64]):
        """Instantiate an FIR filter from b (numerator) coefficients.

        Parameters
        ----------
        b : NDArray[np.float64]
            Numerator coefficients.

        Notes
        -----
        - The state is stored as a circular buffer.

        """
        self.order = len(b) - 1
        self.b = b
        self.set_n_channels(1)

    def set_n_channels(self, n_channels: int):
        self.state = np.zeros((self.order, n_channels))
        self.current_state_ind = np.zeros(n_channels, dtype=np.int_)

    def reset_state(self):
        self.state.fill(0.0)

    def process_sample(self, x: float, channel: int):
        """Process a sample."""
        y = self.b[0] * x

        write_index = self.current_state_ind[channel]
        for i in range(self.order):
            read_index = (write_index - i) % self.order
            y += self.state[read_index, channel] * self.b[i + 1]
        write_index = (write_index + 1) % self.order
        self.state[write_index, channel] = x
        self.current_state_ind[channel] = write_index
        return y
