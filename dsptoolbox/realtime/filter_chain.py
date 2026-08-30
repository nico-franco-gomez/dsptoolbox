import numpy as np
from numpy.typing import NDArray

from .realtime_filter import RealtimeFilter


class FilterChain(RealtimeFilter):
    """This is a utility class for applying a series of real-time filters
    sequentially."""

    def __init__(self, filters: list[RealtimeFilter]):
        """Instantiate a filter chain from a list of real-time filters. Their
        types are not checked and will be applied in the provided order.

        Parameters
        ----------
        filters : list[RealtimeFilter]
            List containing the filters.

        """
        self.filters = filters

    @property
    def n_filters(self):
        return len(self.filters)

    def set_n_channels(self, n_channels: int):
        for f in self.filters:
            f.set_n_channels(n_channels)

    def reset_state(self):
        for f in self.filters:
            f.reset_state()

    def process_sample(self, x: float, channel: int):
        for f in self.filters:
            x = f.process_sample(x, channel)
        return x

    def process_block(
        self, block: NDArray[np.float64], channel: int
    ) -> NDArray[np.float64]:
        for f in self.filters:
            block = f.process_block(block, channel)
        return block
