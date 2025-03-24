from numpy.typing import NDArray
import numpy as np
import scipy.fft as fft

from .realtime_filter import RealtimeFilter
from ..classes.filter import Filter


class FIRFilter(RealtimeFilter):
    """FIR filter implemented in the time domain. This class is
    written for experimentation purposes and realtime applications, but using
    `scipy.signal.lfilter` or some convolution function should be preferred for
    usual offline filtering tasks."""

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


class FIRFilterOverlapSave(RealtimeFilter):
    """Execute a convolution of an FIR filter with the overlap-save scheme.
    This can be used in realtime with block-processing."""

    def __init__(self, fir: Filter):
        """Instantiate FIR filter.

        Parameters
        ----------
        fir : NDArray[np.float64]
            FIR filter. It can only have a single dimension.

        """
        assert fir.is_fir, "Only valid for FIR filters"
        self.fir = fir.ba[0].copy()
        self.set_n_channels(1)

    def prepare(self, blocksize_samples: int, n_channels: int):
        """Prepare the filter for block processing.

        Parameters
        ----------
        blocksize_samples : int
            Size of blocks in samples.
        n_channels : int
            Number of channels to prepare the buffers.

        """
        self.set_n_channels(n_channels)
        self.blocksize = blocksize_samples
        self.total_length = fft.next_fast_len(
            len(self.fir) + blocksize_samples, True
        )
        self.fir_spectrum = fft.rfft(self.fir, n=self.total_length, axis=0)
        self.buffer = np.zeros((self.total_length, self.n_channels))

    def process_block(
        self, block: NDArray[np.float64], channel: int
    ) -> NDArray[np.float64]:
        """Apply FIR filter to a block. It is expected to have a single
        dimension.

        Parameters
        ----------
        block : NDArray[np.float64]
            Block of new samples. It is expected that it only has a single
            dimension corresponding to the defined block size.
        channel : int
            Channel to which the passed block corresponds. It is not checked
            for performance.

        """
        self.buffer[-self.blocksize :, channel] = block
        output_data = fft.irfft(
            fft.rfft(self.buffer[:, channel]) * self.fir_spectrum
        )[-self.blocksize :]

        # Roll buffer
        self.buffer[:, channel] = np.roll(
            self.buffer[:, channel], shift=-self.blocksize
        )
        return output_data

    def process_sample(self, x: float, channel: int):
        raise NotImplementedError(
            "The convolution can only done via block-processing"
        )

    def reset_state(self):
        """Reset all filter states to 0."""
        self.buffer.fill(0.0)

    def set_n_channels(self, n_channels: int):
        """Set the number of channels to be filtered."""
        self.n_channels = n_channels
