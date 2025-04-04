from numpy.typing import NDArray
import numpy as np

from .signal import Signal
from .realtime_filter import RealtimeFilter


class WarpedFIR(RealtimeFilter):
    """The Warped FIR filter has a structure like a common FIR filter but with
    allpasses instead of unit delays between each coefficient. This warps
    the input during the filtering stage.

    This implementation is done efficiently according to [1].

    References
    ----------
    - [1]: Karjalainen, M. & Härmä, Aki & Laine, Unto & Huopaniemi, J.. (1997).
      Warped filters and their audio applications. 4 pp..
      10.1109/ASPAA.1997.625615.

    """

    def __init__(
        self, b_coefficients: NDArray[np.float64], warping_factor: float
    ):
        """Instantiate a warped FIR filter with its coefficients and a warping
        factor. See [1] for details on use and implementation.

        Parameters
        ----------
        b_coefficients : NDArray[np.float64]
            Feedforward filter coefficients.
        warping_factor : float
            Factor to use for warping.

        References
        ----------
        - [1]: Karjalainen, M. & Härmä, Aki & Laine, Unto & Huopaniemi, J..
          (1997). Warped filters and their audio applications. 4 pp..
          10.1109/ASPAA.1997.625615.

        """
        assert (
            abs(warping_factor) < 1.0
        ), "Warping factor must be in range ]-1;1["
        self.b = b_coefficients
        self.warp = warping_factor
        self.N = len(self.b)
        self.order = len(self.b) - 1
        self.set_n_channels(1)

    def set_n_channels(self, n_channels: int):
        assert n_channels > 0
        self.buffer = np.zeros((self.N, n_channels))

    def reset_state(self):
        self.buffer.fill(0.0)

    def process_sample(self, x: float, channel: int) -> float:
        # Start delay-free output
        output = x * self.b[0]
        residue = x

        # Update states and accumulate in output
        for nn in range(self.order):
            # New value
            new_residue = (
                self.buffer[nn + 1, channel] - residue
            ) * self.warp + self.buffer[nn, channel]
            # Accumulate old in buffer
            self.buffer[nn, channel] = residue
            # Swap
            residue = new_residue
            # Accumulate output
            output += new_residue * self.b[nn + 1]

        return output

    def filter_signal(self, signal: Signal) -> Signal:
        """Filter a whole signal with the warped FIR filter. No checking of the
        sampling rates is done here and the existing buffers are left
        unmodified in this operation.

        Parameters
        ----------
        signal : Signal
            Signal to be filtered.

        """
        buffer_prior = self.buffer.copy()
        self.set_n_channels(signal.number_of_channels)
        new_signal = signal.copy_with_new_time_data(
            self.__process_time_data_vector(signal.time_data)
        )
        self.buffer = buffer_prior
        return new_signal

    def __process_time_data_vector(self, time_data: NDArray[np.float64]):
        output = np.zeros_like(time_data)
        n_channels = time_data.shape[1]
        for channel in range(n_channels):
            for n in range(len(time_data)):
                output[n, channel] = self.process_sample(
                    time_data[n, channel], channel
                )
        return output
