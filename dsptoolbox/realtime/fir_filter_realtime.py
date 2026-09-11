from collections.abc import Callable

import numpy as np
import scipy.fft as fft
from numpy.typing import NDArray

from ..classes.filter import Filter
from ..classes.signal import Signal
from ..standard.enums import FilterCoefficientsType
from .realtime_filter import RealtimeFilter

_fir_filtering_sample_rust: Callable | None
try:
    from .._rust import fir_filtering_sample as _fir_filtering_sample_rust
except ImportError:
    _fir_filtering_sample_rust = None


class FIRFilter(RealtimeFilter[float]):
    """FIR filter implemented in the time domain. This class is
    written for experimentation purposes and realtime applications, but using
    `scipy.signal.lfilter` or some convolution function should be preferred for
    usual offline filtering tasks."""

    def __init__(self, b: NDArray[np.float64]) -> None:
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
    def from_filter(fir: Filter) -> "FIRFilter":
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

    def set_n_channels(self, n_channels: int) -> None:
        self.state = np.zeros((self.order, n_channels))
        self.current_state_ind = np.zeros(n_channels, dtype=np.int_)

    def reset_state(self) -> None:
        self.state.fill(0.0)
        self.current_state_ind.fill(0)

    def process_sample(self, x: float, channel: int) -> float:
        """Process a sample."""
        if (
            _fir_filtering_sample_rust is not None
            and getattr(self.b, "dtype", None) == np.dtype(np.float64)
            and self.state.dtype == np.dtype(np.float64)
            and self.current_state_ind.dtype == np.dtype(np.int64)
            and 0 <= channel < self.state.shape[1]
        ):
            return _fir_filtering_sample_rust(
                self.b, x, self.state, self.current_state_ind, channel
            )

        y = self.b[0] * x

        write_index = self.current_state_ind[channel]
        for i in range(self.order):
            read_index = (write_index - i) % self.order
            y += self.state[read_index, channel] * self.b[i + 1]
        write_index = (write_index + 1) % self.order
        self.state[write_index, channel] = x
        self.current_state_ind[channel] = write_index
        return y

    def process_block(
        self, block: NDArray[np.float64], channel: int
    ) -> NDArray[np.float64]:
        if self.order == 0:
            return self.b[0] * block

        # Unroll the circular buffer into the oldest-first delay line that the
        # convolution expects, and rewrite it from the block's tail afterwards
        delay_line = self.state[
            (self.current_state_ind[channel] - np.arange(self.order)) % self.order,
            channel,
        ][::-1]
        extended = np.concatenate((delay_line, block))
        output = np.convolve(extended, self.b)[self.order : self.order + len(block)]

        self.state[:, channel] = extended[-self.order :]
        self.current_state_ind[channel] = self.order - 1
        return output


class FIRFilterOverlapSave(RealtimeFilter[float]):
    """Execute a convolution of an FIR filter with the overlap-save scheme.
    This can be used in realtime with block-processing."""

    def __init__(self, b: NDArray[np.float64]) -> None:
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
    def from_filter(fir: Filter) -> "FIRFilterOverlapSave":
        """Instantiate FIR filter.

        Parameters
        ----------
        fir : Filter
            FIR filter.

        Returns
        -------
        FIRFilterOverlapSave

        """
        assert fir.is_fir, "Only valid for FIR filters"
        b, _ = fir.get_coefficients(FilterCoefficientsType.Ba)
        return FIRFilterOverlapSave(b)

    def prepare(self, blocksize_samples: int, n_channels: int) -> None:
        """Prepare the filter for block processing.

        Parameters
        ----------
        blocksize_samples : int
            Size of blocks in samples.
        n_channels : int
            Number of channels to prepare the buffers.

        """
        self.blocksize = blocksize_samples
        self.total_length = fft.next_fast_len(len(self.fir) + blocksize_samples, True)
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
        output_data = fft.irfft(fft.rfft(self.buffer[:, channel]) * self.fir_spectrum)[
            -self.blocksize :
        ]

        # Roll buffer
        self.buffer[: -self.blocksize, channel] = self.buffer[self.blocksize :, channel]
        return output_data

    def process_sample(self, x: float, channel: int) -> float:
        raise NotImplementedError("The convolution can only done via block-processing")

    def reset_state(self) -> None:
        """Reset all filter states to 0."""
        self.buffer.fill(0.0)

    def set_n_channels(self, n_channels: int) -> None:
        raise NotImplementedError("Use prepare method for setting the filter")


class FIRUniformPartitioned(FIRFilterOverlapSave):
    """FIR filter implemented with overlap-save scheme and uniform filter
    partitions. This type of filter can be used when the FIR filter is
    considerably long.

    """

    def __init__(self, fir: NDArray[np.float64]) -> None:
        """Instantiate a new FIR filter.

        Parameters
        ----------
        fir : NDArray[np.float64]
            Filter coefficients.

        """
        assert fir.ndim == 1
        self.fir = fir

    @staticmethod
    def from_filter(fir: Filter) -> "FIRUniformPartitioned":
        assert fir.is_fir, "Only valid for FIR filters"
        b, _ = fir.get_coefficients(FilterCoefficientsType.Ba)
        return FIRUniformPartitioned(b)

    def prepare(self, blocksize_samples: int, n_channels: int) -> None:
        self.blocksize = blocksize_samples
        self.fft_size = blocksize_samples * 2
        self.__prepare_partitions(n_channels)

    def reset_state(self) -> None:
        self.buffer_spectra.fill(0.0 * 1j)
        self.input_buffer.fill(0.0)

    def __prepare_partitions(self, n_channels: int) -> None:
        self.n_partitions = len(self.fir) // self.blocksize + 1

        # Partitions
        partitioned = np.zeros((self.blocksize, self.n_partitions))
        for n in range(self.n_partitions):
            partition = self.fir[n * self.blocksize : (n + 1) * self.blocksize]
            partitioned[: len(partition), n] = partition
        self.partitioned_spectrum = fft.rfft(partitioned, axis=0, n=self.fft_size)

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

    def process_block(
        self, block: NDArray[np.float64], channel: int
    ) -> NDArray[np.float64]:
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

    def __init__(self, fir: NDArray[np.float64]) -> None:
        """Instantiate a new FIR filter.

        Parameters
        ----------
        fir : NDArray[np.float64]
            Multi-channel Filter coefficients.

        """
        # Bring into standard form
        self.fir = Signal.from_time_data(fir, 10000).time_data

    def prepare(self, blocksize_samples: int, n_channels: int | None = None) -> None:
        """Prepares the processing.

        Parameters
        ----------
        blocksize_samples : int
            Block size to use.
        n_channels : int, None, optional
            The number of channels is defined by the filter itself. When it is
            passed, it is only checked against it. Default: None.

        """
        if n_channels is not None:
            assert n_channels == self.fir.shape[1], (
                f"This filter processes {self.fir.shape[1]} channels, "
                + f"not {n_channels}"
            )
        self.blocksize = blocksize_samples
        self.fft_size = blocksize_samples * 2
        self.__prepare_partitions()

    def __prepare_partitions(self) -> None:
        self.n_partitions = self.fir.shape[0] // self.blocksize + 1
        self.n_channels = self.fir.shape[1]

        # Partitions
        partitioned = np.zeros((self.blocksize, self.n_partitions, self.n_channels))
        for n in range(self.n_partitions):
            partition = self.fir[n * self.blocksize : (n + 1) * self.blocksize, ...]
            partitioned[: len(partition), n, :] = partition
        self.partitioned_spectrum = fft.rfft(partitioned, axis=0, n=self.fft_size)

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

    def process_block(
        self, block: NDArray[np.float64], channel: int = -1
    ) -> NDArray[np.float64]:
        """Process an input block.

        Parameters
        ----------
        block : NDArray[np.float64]
            Block with input data. It is expected to have shape (time samples,
            channels) and always contain all channels to process.
        channel : int, optional
            Ignored, every channel is always processed at once. It is only
            accepted so that this class can be used through the
            `RealtimeFilter` interface. Default: -1.

        Returns
        -------
        NDArray[np.float64]
            Output of convolution with shape (time samples, channels)

        """
        # Store new block in input buffer
        self.input_buffer[: self.blocksize] = self.input_buffer[-self.blocksize :]
        self.input_buffer[-self.blocksize :] = block

        # Transform input
        self.buffer_spectra[:, self.buffer_ind] = fft.rfft(self.input_buffer, axis=0)

        # Accumulate output of all filters with buffers
        output = np.sum(
            self.partitioned_spectrum
            * self.buffer_spectra[:, self.buffer_ind - self.buffer_index_helper, ...],
            axis=1,
        )

        # Advance filter buffer
        self.buffer_ind += 1
        self.buffer_ind %= self.n_partitions

        # Get output
        return fft.irfft(output, axis=0)[-self.blocksize :]
