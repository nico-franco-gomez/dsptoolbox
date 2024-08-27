from numpy.typing import NDArray
import numpy as np

from .realtime_filter import RealtimeFilter


class IIRFilter(RealtimeFilter):
    """IIR filter implemented as a transposed direct form 2."""

    def __init__(self, b: NDArray[np.float64], a: NDArray[np.float64]):
        """Instantiate an IIR filter from b (numerator) and a (denominator)
        coefficients.

        Parameters
        ----------
        b : NDArray[np.float64]
            Numerator coefficients
        a : NDArray[np.float64]
            Denominator coefficients

        """
        b /= a[0]
        a /= a[0]
        self.order = max(len(b), len(a)) - 1
        self.b = np.pad(b, ((0, self.order + 1 - len(b))))
        self.a = np.pad(a, ((0, self.order + 1 - len(a))))

        self.set_n_channels(1)

    def set_n_channels(self, n_channels: int):
        self.state = np.zeros((self.order, n_channels))

    def reset_state(self):
        self.state.fill(0.0)

    def process_sample(self, x: float, channel: int):
        """Process a sample."""
        y = self.b[0] * x + self.state[0, channel]
        for i in range(self.order - 1):
            self.state[i, channel] = (
                x * self.b[i + 1]
                - y * self.a[i + 1]
                + self.state[i + 1, channel]
            )
        self.state[-1, channel] = x * self.b[-1] - y * self.a[-1]
        return y
