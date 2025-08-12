from numpy.typing import NDArray
import numpy as np
import scipy.fft as fft

from ..standard.enums import FilterCoefficientsType
from .realtime_filter import RealtimeFilter
from ..classes.filter import Filter
from ..classes.signal import Signal


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

    @staticmethod
    def from_filter(fir: Filter):
        """Instantiate FIR filter.

        Parameters
        ----------
        fir : Filter
            FIR filter.

        Returns
        -------
        FIRFilter

        """
        assert fir.is_fir, "Only valid for FIR filters"
        b, _ = fir.get_coefficients(FilterCoefficientsType.Ba)
        return FIRFilter(b)

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

    def __init__(self, b: NDArray[np.float64]):
        """Create a new FIR Filter to be used with the overlap-save scheme.
        It can only process data in blocks and the `prepare` method has to be
        called before the processing can start.

        Parameters
        ----------
        b : NDArray[np.float64]
            Feedforward coefficients of the FIR filter.

        """
        assert b.ndim == 1, "A single dimension should be provided"
        self.fir = b

    @staticmethod
    def from_filter(fir: Filter):
        """Instantiate FIR filter.

        Parameters
        ----------
        fir : Filter
            FIR filter.

        """
        assert fir.is_fir, "Only valid for FIR filters"
        b, _ = fir.get_coefficients(FilterCoefficientsType.Ba)
        return FIRFilterOverlapSave(b)

    def prepare(self, blocksize_samples: int, n_channels: int):
        """Prepare the filter for block processing.

        Parameters
        ----------
        blocksize_samples : int
            Size of blocks in samples.
        n_channels : int
            Number of channels to prepare the buffers.

        """
        self.blocksize = blocksize_samples
        self.total_length = fft.next_fast_len(
            len(self.fir) + blocksize_samples, True
        )
        self.fir_spectrum = fft.rfft(self.fir, n=self.total_length, axis=0)
        self.buffer = np.zeros((self.total_length, n_channels))

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
        self.buffer[: -self.blocksize, channel] = self.buffer[
            self.blocksize :, channel
        ]
        return output_data

    def process_sample(self, x: float, channel: int):
        raise NotImplementedError(
            "The convolution can only done via block-processing"
        )

    def reset_state(self):
        """Reset all filter states to 0."""
        self.buffer.fill(0.0)

    def set_n_channels(self, n_channels: int):
        raise NotImplementedError("Use prepare method for setting the filter")


class FIRUniformPartitioned(FIRFilterOverlapSave):
    """FIR filter implemented with overlap-save scheme and uniform filter
    partitions. This type of filter can be used when the FIR filter is
    considerably long.

    """

    def __init__(self, fir: NDArray[np.float64]):
        """Instantiate a new FIR filter.

        Parameters
        ----------
        fir : NDArray[np.float64]
            Filter coefficients.

        """
        assert fir.ndim == 1
        self.fir = fir

    @staticmethod
    def from_filter(fir: Filter):
        assert fir.is_fir, "Only valid for FIR filters"
        b, _ = fir.get_coefficients(FilterCoefficientsType.Ba)
        return FIRUniformPartitioned(b)

    def prepare(self, blocksize_samples: int, n_channels: int):
        self.blocksize = blocksize_samples
        self.fft_size = blocksize_samples * 2
        self.__prepare_partitions(n_channels)

    def reset_state(self):
        self.buffer_spectra.fill(0.0 * 1j)
        self.input_buffer.fill(0.0)

    def __prepare_partitions(self, n_channels: int):
        self.n_partitions = len(self.fir) // self.blocksize + 1

        # Partitions
        partitioned = np.zeros((self.blocksize, self.n_partitions))
        for n in range(self.n_partitions):
            partition = self.fir[n * self.blocksize : (n + 1) * self.blocksize]
            partitioned[: len(partition), n] = partition
        self.partitioned_spectrum = fft.rfft(
            partitioned, axis=0, n=self.fft_size
        )

        # Buffer index for filter
        self.buffer_ind = 0

        # Helper for avoiding allocations in process
        self.buffer_index_helper = np.arange(self.n_partitions)

        # Channel buffers
        self.buffer_spectra = np.zeros(
            (self.fft_size // 2 + 1, self.n_partitions, n_channels),
            dtype=np.complex128,
        )
        self.input_buffer = np.zeros((self.fft_size, n_channels))

    def process_block(self, block: NDArray[np.float64], channel: int):
        # Store new block in input buffer
        self.input_buffer[: self.blocksize, channel] = self.input_buffer[
            -self.blocksize :, channel
        ]
        self.input_buffer[-self.blocksize :, channel] = block

        # Transform input
        self.buffer_spectra[:, self.buffer_ind, channel] = fft.rfft(
            self.input_buffer[:, channel]
        )

        # Accumulate output of all filters with buffers
        output = np.sum(
            self.partitioned_spectrum
            * self.buffer_spectra[
                :, self.buffer_ind - self.buffer_index_helper, channel
            ],
            axis=1,
        )

        # Advance filter buffer
        self.buffer_ind += 1
        self.buffer_ind %= self.n_partitions

        # Get output
        return fft.irfft(output)[-self.blocksize :]


class FIRUniformPartitionedMultichannel(FIRUniformPartitioned):
    """FIR filter implemented with overlap-save scheme and uniform filter
    partitions. This type of filter can be used when the FIR filter is
    considerably long. This version always processes multiple channels at once
    with different inputs and different outputs. This might be more efficient
    that doing each channel individually due to vectorization.

    """

    def __init__(self, fir: NDArray[np.float64]):
        """Instantiate a new FIR filter.

        Parameters
        ----------
        fir : NDArray[np.float64]
            Multi-channel Filter coefficients.

        """
        # Bring into standard form
        self.fir = Signal.from_time_data(fir, 10000).time_data

    def prepare(self, blocksize_samples: int):  # type: ignore
        """Prepares the processing.

        Parameters
        ----------
        blocksize_samples : int
            Block size to use

        """
        self.blocksize = blocksize_samples
        self.fft_size = blocksize_samples * 2
        self.__prepare_partitions()

    def __prepare_partitions(self):
        self.n_partitions = self.fir.shape[0] // self.blocksize + 1
        self.n_channels = self.fir.shape[1]

        # Partitions
        partitioned = np.zeros(
            (self.blocksize, self.n_partitions, self.n_channels)
        )
        for n in range(self.n_partitions):
            partition = self.fir[
                n * self.blocksize : (n + 1) * self.blocksize, ...
            ]
            partitioned[: len(partition), n, :] = partition
        self.partitioned_spectrum = fft.rfft(
            partitioned, axis=0, n=self.fft_size
        )

        # Buffer index for filter
        self.buffer_ind = 0

        # Helper for avoiding allocations in process
        self.buffer_index_helper = np.arange(self.n_partitions)

        # Channel buffers
        self.buffer_spectra = np.zeros(
            (self.fft_size // 2 + 1, self.n_partitions, self.n_channels),
            dtype=np.complex128,
        )
        self.input_buffer = np.zeros((self.fft_size, self.n_channels))

    def process_block(self, block: NDArray[np.float64]):  # type: ignore
        """Process an input block.

        Parameters
        ----------
        block : NDArray[np.float64]
            Block with input data. It is expected to have shape (time samples,
            channels) and always contain all channels to process.

        Returns
        -------
        NDArray[np.float64]
            Output of convolution with shape (time samples, channels)

        """
        # Store new block in input buffer
        self.input_buffer[: self.blocksize] = self.input_buffer[
            -self.blocksize :
        ]
        self.input_buffer[-self.blocksize :] = block

        # Transform input
        self.buffer_spectra[:, self.buffer_ind] = fft.rfft(
            self.input_buffer, axis=0
        )

        # Accumulate output of all filters with buffers
        output = np.sum(
            self.partitioned_spectrum
            * self.buffer_spectra[
                :, self.buffer_ind - self.buffer_index_helper, ...
            ],
            axis=1,
        )

        # Advance filter buffer
        self.buffer_ind += 1
        self.buffer_ind %= self.n_partitions

        # Get output
        return fft.irfft(output, axis=0)[-self.blocksize :]
