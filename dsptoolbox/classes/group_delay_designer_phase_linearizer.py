from .filter import Filter
from .impulse_response import ImpulseResponse
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from numpy.typing import NDArray
from .._general_helpers import _correct_for_real_phase_spectrum, _pad_trim
from warnings import warn


class GroupDelayDesigner:
    """This class designs an FIR filter with a desired group delay response."""

    def __init__(
        self,
        target_group_delay_s: NDArray[np.float64],
        time_data_length_samples: int,
        sampling_rate_hz: int,
    ):
        """GroupDelayDesigner creates an FIR filter with a desired group delay
        response. Use the method `set_parameters` to define specific design
        parameters.

        Parameters
        ----------
        target_group_delay_s : NDArray[np.float64]
            Target group delay in seconds. It is expected to contain the whole
            positive frequency spectrum (including dc and eventually nyquist)
            and be linearly spaced.
        time_data_length_samples : NDArray[np.float64]
            Length of the time signal that corresponds to the frequency vector
            of `target_group_delay_s`.
        sampling_rate_hz : int
            Sampling rate to design the filter. It corresponds to the implicit
            frequency vector of the group delay.

        """
        self.time_data_length_samples = time_data_length_samples
        self.sampling_rate_hz = sampling_rate_hz
        self._set_target_group_delay_s(target_group_delay_s)
        self.set_parameters()

    def set_parameters(
        self,
        delay_increase_ms: float = 0.0,
    ):
        """Set parameters for the FIR filter.

        Parameters
        ----------
        delay_increase_ms : float, optional
            This is an overall increase in delay to the current group delay (in
            milliseconds). Increasing this improves the quality of the
            designed filter but also makes it longer. Default: 0.

        """
        assert (
            delay_increase_ms >= 0
        ), "Delay increase must be larger than zero"
        self.group_delay_increase_ms = delay_increase_ms

    def _set_target_group_delay_s(
        self, target_group_delay_s: NDArray[np.float64]
    ):
        """Set target group delay to use instead of phase response.

        Parameters
        ----------
        target_group_delay : NDArray[np.float64]
            Target group delay (in samples) to use.

        """
        assert (
            target_group_delay_s.ndim == 1
        ), "Target group delay can only have 1 dimension"
        assert self.time_data_length_samples // 2 + 1 == len(
            target_group_delay_s
        ), (
            f"Target group delay with length {len(target_group_delay_s)} and "
            + f"length {self.time_data_length_samples} do not match."
        )
        self.target_group_delay_s = target_group_delay_s

    def _get_unscaled_preprocessed_group_delay(self) -> NDArray[np.float64]:
        """Obtain reprocessed group delay for designing the FIR filter."""
        return (
            self.target_group_delay_s + self.group_delay_increase_ms / 1e3
        ) / self._get_group_delay_factor_in_seconds()

    def _get_group_delay_factor_in_samples(self) -> float:
        """This is the conversion factor from unscaled to delay in samples."""
        return self.time_data_length_samples / 2 / np.pi

    def _get_group_delay_factor_in_seconds(self) -> float:
        """This is the conversion factor from unscaled to delay in seconds."""
        return (
            self.time_data_length_samples / 2 / np.pi / self.sampling_rate_hz
        )

    def get_filter(self) -> Filter:
        """Get FIR filter."""
        return Filter.from_ba(self.__design(), [1], self.sampling_rate_hz)

    def get_filter_as_ir(self) -> ImpulseResponse:
        """Get the phase filter as an ImpulseResponse."""
        return ImpulseResponse(None, self.__design(), self.sampling_rate_hz)

    def __design(self) -> NDArray[np.float64]:
        """Compute filter."""
        target_gd = self._get_unscaled_preprocessed_group_delay()
        max_delay_samples_synthesized = int(
            np.max(target_gd) * self._get_group_delay_factor_in_samples() + 1
        )
        gd_time_length_samples = self.time_data_length_samples

        # Interpolate if phase response is not much longer than maximum
        # expected delay to increase frequency resolution
        if max_delay_samples_synthesized * 20 > gd_time_length_samples:
            warn(
                f"Phase response (length {gd_time_length_samples}) "
                + "is not much longer than maximum expected "
                + f"group delay {max_delay_samples_synthesized} (less "
                + "than 20 times longer). Spectrum interpolation "
                + "is triggered, but it is recommended to pass a phase "
                + "spectrum with finer resolution!"
            )
            # Define new time length for the group delay
            new_gd_time_length_samples = (
                int(max_delay_samples_synthesized * 20) + 1
            )
            # Ensure even length
            new_gd_time_length_samples += new_gd_time_length_samples % 2
            # Interpolate
            new_freqs = np.fft.rfftfreq(
                new_gd_time_length_samples, 1 / self.sampling_rate_hz
            )
            frequency_vector_hz = np.fft.rfftfreq(
                self.time_data_length_samples, 1 / self.sampling_rate_hz
            )
            target_gd = PchipInterpolator(
                frequency_vector_hz, target_gd, extrapolate=True
            )(new_freqs) * (gd_time_length_samples / len(new_freqs))
            gd_time_length_samples = new_gd_time_length_samples

        # Get new phase using group target group delay
        new_phase = -cumulative_trapezoid(target_gd, initial=0)

        # Correct if nyquist is given
        add_extra_sample = False
        if gd_time_length_samples % 2 == 0:
            add_extra_sample = new_phase[-1] % np.pi > np.pi / 2.0
            new_phase = _correct_for_real_phase_spectrum(new_phase)

        # Convert to time domain and trim
        ir = np.fft.irfft(np.exp(1j * new_phase), gd_time_length_samples)
        trim_length = int(max_delay_samples_synthesized + 1 + add_extra_sample)
        ir = _pad_trim(ir, trim_length)
        return ir


class PhaseLinearizer(GroupDelayDesigner):
    """This class designs an FIR filter that linearizes a known phase
    response.

    """

    def __init__(
        self,
        phase_response: NDArray[np.float64],
        time_data_length_samples: int,
        sampling_rate_hz: int,
    ):
        """PhaseLinearizer creates an FIR filter that can linearize a phase
        response. Use the method `set_parameters` to define specific design
        parameters.

        Parameters
        ----------
        phase_response : NDArray[np.float64]
            Wrapped phase response that should be linearized. It is expected
            to contain only the positive frequencies (including dc and
            eventually nyquist).
        time_data_length_samples : NDArray[np.float64]
            Length of the time signal that gave the phase response.
        sampling_rate_hz : int
            Sampling rate corresponding to the passed phase response. It is
            also used for the designed FIR filter.

        """
        self.phase_response = phase_response
        self.set_parameters()
        self.time_data_length_samples = time_data_length_samples
        self.sampling_rate_hz = sampling_rate_hz
        target_group_delay_s = (
            self._get_target_group_delay_in_seconds_from_phase()
        )
        self._set_target_group_delay_s(target_group_delay_s)

    def set_parameters(
        self,
        delay_increase_percent: float = 100.0,
    ):
        """Set parameters for the FIR filter.

        Parameters
        ----------
        delay_increase_percent : float, optional
            This is the increase (in percentage) in delay to the current
            maximum group delay. Increasing this improves the quality of the
            designed filter but also makes it longer. Passing a value of 100
            means that the total group delay will be 2 times larger than the
            longest group delay. Default: 100.

        """
        assert (
            delay_increase_percent >= 0
        ), "Delay increase must be larger than zero"
        self.group_delay_increase_factor = 1 + delay_increase_percent / 100

    def __get_group_delay(self, phase_response) -> NDArray[np.float64]:
        """Return the unscaled group delay from the phase response."""
        return -np.gradient(np.unwrap(phase_response))

    def _get_target_group_delay_in_seconds_from_phase(
        self,
    ) -> NDArray[np.float64]:
        """Return the target group delay in seconds. It is computed from the
        phase response and has already the increase factor."""
        gd = self.__get_group_delay(self.phase_response)
        target_gd = np.max(gd) * self.group_delay_increase_factor - gd
        return target_gd * self._get_group_delay_factor_in_seconds()

    def _get_unscaled_preprocessed_group_delay(self) -> NDArray[np.float64]:
        """Get the target group delay already corrected by the increase factor
        and in no physical units (neither samples nor seconds)."""
        return (
            self._get_target_group_delay_in_seconds_from_phase()
            / self._get_group_delay_factor_in_seconds()
        )
