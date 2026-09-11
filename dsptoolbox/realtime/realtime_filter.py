import abc
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

# Most structures produce one output sample per input sample, but a multimode
# filter produces one per mode, so the sample type is a parameter of the class
SampleT = TypeVar("SampleT", bound=float | tuple[float, ...])


class RealtimeFilter(abc.ABC, Generic[SampleT]):
    @abc.abstractmethod
    def process_sample(self, x: float, channel: int) -> SampleT:
        """Process a sample with the filter for a given channel. Channel index
        is not checked for speed."""

    @abc.abstractmethod
    def reset_state(self) -> None:
        """Reset all filter states to 0."""

    @abc.abstractmethod
    def set_n_channels(self, n_channels: int) -> None:
        """Set the number of channels to be filtered."""

    def process_block(
        self, block: NDArray[np.float64], channel: int
    ) -> NDArray[np.float64]:
        """Process a block of samples with the filter for a given channel. The
        filter state is carried over between calls, so filtering a signal in
        blocks delivers the same result as filtering it at once.

        Parameters
        ----------
        block : NDArray[np.float64]
            Block of new samples with a single dimension.
        channel : int
            Channel to which the block belongs. It is not checked for speed.

        Returns
        -------
        NDArray[np.float64]
            Filtered block with the length of the input.

        Notes
        -----
        - This generic implementation calls `process_sample` for every sample
          and is therefore no faster than the per-sample loop it replaces.
          Structures that can filter a whole block at once override it.
        - Only valid for filters that produce a single sample per input
          sample. Multimode structures override it with their own layout.

        """
        output: NDArray[np.float64] = np.empty(len(block), dtype=np.float64)
        for index in range(len(block)):
            output[index] = self.process_sample(block[index], channel)
        return output
