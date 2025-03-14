import abc


class RealtimeFilter(abc.ABC):
    @abc.abstractmethod
    def process_sample(self, x: float, channel: int):
        """Process a sample with the filter for a given channel. Channel index
        is not checked for speed."""
        pass

    @abc.abstractmethod
    def reset_state(self):
        """Reset all filter states to 0."""
        pass

    @abc.abstractmethod
    def set_n_channels(self, n_channels: int):
        """Set the number of channels to be filtered."""
        pass
