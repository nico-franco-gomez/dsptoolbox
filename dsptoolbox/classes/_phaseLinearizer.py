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
        phase_response: np.ndarray,
        time_data_length_samples: int,
        sampling_rate_hz: int,
        target_group_delay_samples: np.ndarray | None = None,
    ):
        """PhaseLinearizer creates an FIR filter that can linearize a phase
        response. Use the method `set_parameters` to define specific design
        parameters.

        Parameters
        ----------
        phase_response : `np.ndarray`
            Wrapped phase response that should be linearized. It is expected
            to contain only the positive frequencies (including dc and
            eventually nyquist).
        time_data_length_samples : `np.ndarray`
            Length of the time signal that gave the phase response.
        sampling_rate_hz : int
            Sampling rate corresponding to the passed phase response. It is
            also used for the designed FIR filter.
        target_group_delay_samples : `np.ndarray` or `None`, optional
            If passed, this overwrites the phase response and becomes the
            target for the FIR filter. It must be given in samples for the
            whole spectrum (only positive frequencies). For producing
            satisfactory amplitude responses of the filter, it is recommended
            that the target group delay be smooth and not very close to 0.
            Default: `None`.

        """
        self.frequency_vector = np.fft.rfftfreq(
            time_data_length_samples, 1 / sampling_rate_hz
        )
        self.time_data_length_samples = time_data_length_samples
        assert len(self.frequency_vector) == len(phase_response), (
            f"Phase response with length {len(phase_response)} and "
            + f"length {time_data_length_samples} do not match."
        )
        self.phase_response = phase_response
        self.sampling_rate_hz = sampling_rate_hz
        self.set_parameters()
        if target_group_delay_samples is not None:
            self._set_target_group_delay(target_group_delay_samples)

    def _set_target_group_delay(self, target_group_delay: np.ndarray):
        """Set target group delay to use instead of phase response.

        Parameters
        ----------
        target_group_delay : `np.ndarray`
            Target group delay (in samples) to use.

        """
        assert (
            target_group_delay.ndim == 1
        ), "Target group delay can only have 1 dimension"
        assert len(self.frequency_vector) == len(
            target_group_delay
        ), f"Target group delay shape {target_group_delay.shape} is invalid"
        self.target_group_delay = target_group_delay

    def set_parameters(
        self,
        delay_increase_percent: float = 100.0,
        total_length_factor: float = 1.5,
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
            The total length of the filter is based on the longest group delay.
            This factor can augment it. Default: 1.5.

        Notes
        -----
        - If there is a target group delay, no increase is applied by
          `delay_increase_percent`, but `total_length_factor` is still used
          for the output filter.

        """
        assert (
            delay_increase_percent >= 0
        ), "Delay increase must be larger than zero"
        if total_length_factor < 1.0:
            warn(
                "Total length factor should not be less than 1. It "
                + "will be clipped."
            )
        self.group_delay_increase_factor = 1 + delay_increase_percent / 100
        self.total_length_factor = np.clip(
            total_length_factor, a_min=1.0, a_max=None
        )

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
        if not hasattr(self, "target_group_delay"):
            gd = self._get_group_delay()
            gd_time_length_samples = self.time_data_length_samples

            max_delay_samples_synthesized = int(
                np.max(gd)
                * self._get_group_delay_factor_in_samples()
                * self.group_delay_increase_factor
                # Ceil
                + 0.99999999999
            )
            target_gd = np.max(gd) * self.group_delay_increase_factor - gd
        else:
            target_gd = (
                self.target_group_delay
                / self._get_group_delay_factor_in_samples()
            )
            max_delay_samples_synthesized = int(
                np.max(self.target_group_delay) + 1
            )
            gd_time_length_samples = self.time_data_length_samples

        # Interpolate if phase response is not much longer than maximum
        # expected delay
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
            target_gd = interp1d(
                self.frequency_vector,
                target_gd,
                "cubic",
                assume_sorted=True,
                # Extrapolate if nyquist goes above last frequency bin
                bounds_error=False,
                fill_value=(target_gd[0], target_gd[-1]),
            )(new_freqs) * (gd_time_length_samples / len(new_freqs))
            gd_time_length_samples = new_gd_time_length_samples

        # Get new phase using group target group delay
        new_phase = -cumulative_simpson(target_gd, initial=0)
        # Correct if nyquist is given
        if gd_time_length_samples % 2 == 0:
            new_phase = _correct_for_real_phase_spectrum(
                _wrap_phase(new_phase)
            )

        # Convert to time domain and trim
        ir = np.fft.irfft(np.exp(1j * new_phase), gd_time_length_samples)
        trim_length = (
            int(max_delay_samples_synthesized * self.total_length_factor)
            if self.total_length_factor
            > 1.0 + (1 / max_delay_samples_synthesized)
            else int(max_delay_samples_synthesized + 1)
        )
        ir = _pad_trim(ir, trim_length)
        return ir

    def _get_group_delay(self) -> np.ndarray:
        """Return the unscaled group delay."""
        return -np.gradient(np.unwrap(self.phase_response))

    def _get_group_delay_factor_in_samples(self) -> float:
        """This is the conversion factor from unscaled to delay in samples."""
        return self.time_data_length_samples / 2 / np.pi

    def _get_group_delay_factor_in_seconds(self) -> float:
        """This is the conversion factor from unscaled to delay in seconds."""
        return (
            self.time_data_length_samples / 2 / np.pi / self.sampling_rate_hz
        )
