import abc


class RealtimeFilter(abc.ABC):
    @abc.abstractmethod
    def process_sample(x: float, channel: int):
        pass

    @abc.abstractmethod
    def reset_state():
        """Reset all filter states to 0."""
        pass
