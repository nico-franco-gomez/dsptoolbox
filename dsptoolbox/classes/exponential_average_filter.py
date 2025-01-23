import numpy as np

from ..helpers.smoothing import _get_smoothing_factor_ema
from .realtime_filter import RealtimeFilter


class ExponentialAverageFilter(RealtimeFilter):
    def __init__(
        self,
        increase_time_s: float,
        decrease_time_s: float,
        sampling_rate_hz: int,
        accuracy_step_response: float = 0.95,
    ):
        """The exponential average filter is a one-pole IIR filter which
        smoothes a the input (lowpass filter). It can have a different
        coefficients for increasing and decreasing values.

        Parameters
        ----------
        increase_time_s : float
            Time it would take the filter to obtain the given `accuracy` in
            the step response. This is applied for increasing values.
        decrease_time_s : float
            Time it would take the filter to obtain the given `accuracy` in
            the step response. This is applied for decreasing values.
        sampling_rate_hz : int
            Sampling rate of the filter.
        accuracy_step_response : float, optional
            This represents the value that the step response should reach after
            the increase or decrease time. It has to in ]0; 1[. Default: 0.95.

        """
        self.sampling_rate_hz = sampling_rate_hz
        self.increase_coefficient = _get_smoothing_factor_ema(
            increase_time_s, self.sampling_rate_hz, accuracy_step_response
        )
        self.decrease_coefficient = _get_smoothing_factor_ema(
            decrease_time_s, self.sampling_rate_hz, accuracy_step_response
        )
        self.set_n_channels(1)

    def set_n_channels(self, n_channels: int):
        self.state = np.zeros((1, n_channels))

    def reset_state(self):
        self.state.fill(0.0)

    def process_sample(self, x: float, channel: int):
        if x > self.state:  # Ascending
            y = (
                x * self.increase_coefficient
                + (1 - self.increase_coefficient) * self.state[0, channel]
            )
        else:  # Descending
            y = (
                x * self.decrease_coefficient
                + (1 - self.decrease_coefficient) * self.state[0, channel]
            )
        self.state[0, channel] = y
        return y
