from .filter_class import Filter
from .signal_class import Signal
import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import interp1d
from .._general_helpers import (
    _correct_for_real_phase_spectrum,
    _pad_trim,
    _wrap_phase,
)
from warnings import warn


class PhaseLinearizer:
    """This class designs an FIR filter that linearizes a known phase
    response.

    """

    def __init__(
        self,
        frequency_vector: np.ndarray,
        phase_response: np.ndarray,
        sampling_rate_hz: int,
    ):
        """PhaseLinearizer creates an FIR filter that can linearize a phase
        response. Use the method `set_parameters` to define specific design
        parameters.

        Parameters
        ----------
        frequency_vector : `np.ndarray`
            Frequency vector that corresponds to the phase response.
        phase_response : `np.ndarray`
            Wrapped phase response that should be linearized. It is expected
            to contain only the positive frequencies (including dc and
            eventually nyquist). It should have the same shape as
            `frequency_vector`.
        sampling_rate_hz : int
            Sampling rate corresponding to the passed phase response. It is
            also used for the designed FIR filter.

        """
        assert (
            frequency_vector.shape == phase_response.shape
        ), "Frequency vector and phase response vector do not match"
        self.nyquist_included = np.isclose(
            frequency_vector[-1], sampling_rate_hz // 2
        )
        self.phase_response = phase_response
        self.frequency_vector = frequency_vector
        self.sampling_rate_hz = sampling_rate_hz
        self.set_parameters()

    def set_parameters(
        self,
        delay_increase_percent: float = 100.0,
        total_length_factor: float = 0.5,
    ):
        """Set parameters for the FIR filter.

        Parameters
        ----------
        delay_increase_percent : float, optional
            This is the increase (in percentage) in delay to the current
            maximum group delay of the phase response. Increasing this improves
            the quality of the designed filter but also makes it longer.
            Passing a value of 100 means that the total group delay will be
            2 times larger than the longest group delay in the phase response.
            Default: 100.
        total_length_factor : float, optional
            The total length of the filter is by default two times the longest
            group delay of the designed filter. This can be reduced or
            augmented by this factor. A factor of 0.5 or less returns the
            minimum length. Default: 0.5.

        """
        assert (
            delay_increase_percent >= 0
        ), "Delay increase must be larger than zero"
        assert (
            total_length_factor > 0
        ), "Total length factor must be larger than zero"
        self.group_delay_increase_factor = 1 + delay_increase_percent / 100
        self.total_length_factor = total_length_factor

    def get_filter(self) -> Filter:
        """Get FIR filter."""
        return Filter(
            "other",
            {"ba": [self._design(), [1]]},
            sampling_rate_hz=self.sampling_rate_hz,
        )

    def get_filter_as_ir(self) -> Signal:
        return Signal(
            None, self._design(), self.sampling_rate_hz, signal_type="ir"
        )

    def _design(self) -> np.ndarray:
        """Compute filter."""
        # Get initial parameters
        if not hasattr(self, "group_delay"):
            self.gd = self._get_group_delay()
            self.gd_time_length = (len(self.phase_response) - 1) * 2
            if not self.nyquist_included:
                self.gd_time_length += 1

        max_delay_samples_synthesized = int(
            np.max(self.gd)
            * self._get_group_delay_factor_in_samples()
            * self.group_delay_increase_factor
            # Ceil
            + 0.9999999
        )

        # Interpolate if phase response is not much longer than maximum
        # expected delay
        if max_delay_samples_synthesized * 20 > self.gd_time_length:
            # Define new time length for the group delay
            self.gd_time_length = (
                int(max_delay_samples_synthesized * 20)
                + max_delay_samples_synthesized % 2
            )
            new_freqs = np.fft.rfftfreq(
                self.gd_time_length, 1 / self.sampling_rate_hz
            )
            self.gd = interp1d(
                self.frequency_vector,
                self.gd,
                "cubic",
                assume_sorted=True,
            )(new_freqs) * (len(self.gd) / len(new_freqs))
            warn(
                "Phase response is not much longer than maximum expected "
                + "group delay (less than 20 times longer). Spectrum "
                + "interpolation was done, but it is recommended to pass "
                + "a phase spectrum with finer resolution!"
            )

        # Get new phase using group target group delay
        target_gd = (
            np.max(self.gd) * self.group_delay_increase_factor - self.gd
        )
        new_phase = -cumulative_simpson(target_gd, initial=0)
        new_phase = _correct_for_real_phase_spectrum(_wrap_phase(new_phase))

        # Convert to time domain and trim
        ir = np.fft.irfft(np.exp(1j * new_phase))
        trim_length = (
            int(max_delay_samples_synthesized * 2 * self.total_length_factor)
            if self.total_length_factor > 0.5
            else int(max_delay_samples_synthesized + 1)
        )
        ir = _pad_trim(ir, trim_length)
        return ir

    def _get_group_delay(self) -> np.ndarray:
        """Return the unscaled group delay."""
        return -np.gradient(np.unwrap(self.phase_response))

    def _get_group_delay_factor_in_samples(self) -> float:
        """This is the conversion factor from unscaled to delay in samples."""
        length = (len(self.phase_response) - 1) * 2
        if not self.nyquist_included:
            length += 1
        return length / 2 / np.pi

    def _get_group_delay_factor_in_seconds(self) -> float:
        """This is the conversion factor from unscaled to delay in seconds."""
        length = (len(self.phase_response) - 1) * 2
        if not self.nyquist_included:
            length += 1
        return length / 2 / np.pi / self.sampling_rate_hz
