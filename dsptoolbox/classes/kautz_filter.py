import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from .realtime_filter import RealtimeFilter
from .impulse_response import ImpulseResponse
from .signal import Signal
from .iir_filter_realtime import IIRFilter
from ..generators import dirac


class KautzFilter(RealtimeFilter):
    """Class for a Kautz filter that can process real-valued signals. See
    references for details on Kautz Filters and their uses.

    References
    ----------
    - [1]: Bank, B. (2022). Warped, Kautz, and Fixed-Pole Parallel Filters: A
      Review. Journal of the Audio Engineering Society.
    - This function is a port from the matlab toolbox:
      http://legacy.spa.aalto.fi/software/kautz/kautz.htm

    """

    def __init__(
        self,
        poles: NDArray[np.complex128],
        sampling_rate_hz: int,
    ):
        """Get a Kautz filter with a set of poles that defines the orthonormal
        basis. This filter only supports processing of real-valued signals.

        Parameters
        ----------
        poles : NDArray[np.complex128]
            Poles that define the orthonormal basis for the filter. These can
            be complex and real. If complex, they should only have a positive
            imaginary part. The complex-conjugated pole will be automatically
            generated.
        sampling_rate_hz : int
            Sampling rate of the filter.

        Notes
        -----
        - All filter coefficients are initialized with 1.0.

        References
        ----------
        - [1]: Bank, B. (2022). Warped, Kautz, and Fixed-Pole Parallel Filters:
          A Review. Journal of the Audio Engineering Society.
        - This function is a port from the matlab toolbox:
          http://legacy.spa.aalto.fi/software/kautz/kautz.htm

        """
        assert not np.any(
            poles.imag < 0.0
        ), "No poles with negative imaginary part should be passed"
        assert not np.any(
            np.abs(poles) >= 1.0
        ), "No poles should lie outside the unit circle"
        self.sampling_rate_hz = sampling_rate_hz

        self.__set_poles(poles)
        self.__compute_filters()
        self.set_filter_coefficients(
            np.ones(self.n_real_poles), np.ones(self.n_complex_poles)
        )
        self.set_n_channels(1)

    def __set_poles(self, poles: NDArray[np.complex128]):
        # Separate into real and complex
        real_indices = poles.imag == 0.0
        self.poles_real = np.real(poles[real_indices])
        self.poles_complex = poles[~real_indices]

        self.n_complex_poles = len(self.poles_complex) * 2
        self.n_real_poles = len(self.poles_real)
        self.total_n_poles = self.n_complex_poles + self.n_real_poles

    def set_filter_coefficients(
        self, c_real: NDArray[np.float64], c_complex: NDArray[np.float64]
    ):
        """Set the filter coefficients for each section of the Kautz filter.
        Optimal filter coefficients (in a least-squares sense) can be found
        by analyzing an IR with the desired magnitude and phase response in
        `fit_coefficients_to_ir()`.

        Parameters
        ----------
        c_real : NDArray[np.float64]
            Coefficients corresponding to the real poles.
        c_complex : NDArray[np.float64]
            Coefficients for the complex poles. The coefficients will be used
            with adjacent complex-conjugated pairs, i.e., pole `a+1j*b` and
            then pole `a-1j*b`.

        """
        assert self.n_complex_poles == len(c_complex)
        assert self.n_real_poles == len(c_real)
        self.coefficients_real_poles = c_real
        self.coefficients_complex_poles = c_complex

    def __compute_filters(self):
        self.__filters_real: list[IIRFilter] = []
        self.__filters_real_advance_sample: list[IIRFilter] = []
        self.__filters_complex: list[IIRFilter] = []
        self.__filters_complex_advance_sample: list[IIRFilter] = []

        # Real poles
        for ii, preal in enumerate(self.poles_real):
            self.__filters_real.append(
                IIRFilter(
                    b=np.array([(1.0 - preal**2.0) ** 0.5]),
                    a=np.array([1.0, -preal]),
                )
            )
            self.__filters_real_advance_sample.append(
                IIRFilter(
                    b=np.array([-preal, 1.0]),
                    a=np.array([1.0, -preal]),
                )
            )

        # Complex poles
        q = -2.0 * np.real(self.poles_complex)
        r = np.abs(self.poles_complex) ** 2.0
        for ii in range(len(self.poles_complex)):
            a = np.array([1.0, q[ii], r[ii]])
            self.__filters_complex.append(
                IIRFilter(
                    b=np.array([1.0, -1.0])
                    * ((1.0 - r[ii]) * (1.0 + r[ii] - q[ii]) / 2.0) ** 0.5,
                    a=a,
                )
            )
            self.__filters_complex.append(
                IIRFilter(
                    b=np.array([1.0, 1.0])
                    * ((1.0 - r[ii]) * (1.0 + r[ii] + q[ii]) / 2.0) ** 0.5,
                    a=a,
                )
            )
            self.__filters_complex_advance_sample.append(
                IIRFilter(
                    b=np.array([r[ii], q[ii], 1.0]),
                    a=a,
                )
            )

    def set_n_channels(self, n_channels: int):
        for f in self.__filters_complex:
            f.set_n_channels(n_channels)
        for f in self.__filters_real:
            f.set_n_channels(n_channels)
        for f in self.__filters_complex_advance_sample:
            f.set_n_channels(n_channels)
        for f in self.__filters_real_advance_sample:
            f.set_n_channels(n_channels)

    def reset_state(self):
        for f in self.__filters_real:
            f.reset_state()
        for f in self.__filters_complex:
            f.reset_state()
        for f in self.__filters_real_advance_sample:
            f.reset_state()
        for f in self.__filters_complex_advance_sample:
            f.reset_state()

    def process_sample(self, x: float, channel: int):
        y = 0.0
        for ind, f in enumerate(self.__filters_real):
            y += (
                f.process_sample(x, channel)
                * self.coefficients_real_poles[ind]
            )
            x = self.__filters_real_advance_sample[ind].process_sample(
                x, channel
            )

        for ind in range(0, len(self.__filters_complex), 2):
            x1 = self.__filters_complex[ind].process_sample(x, channel)
            x2 = self.__filters_complex[ind + 1].process_sample(x, channel)
            y += (
                x1 * self.coefficients_complex_poles[ind]
                + x2 * self.coefficients_complex_poles[ind + 1]
            )
            x = self.__filters_complex_advance_sample[ind // 2].process_sample(
                x, channel
            )
        return y

    def fit_coefficients_to_ir(self, ir: ImpulseResponse):
        """Fit Kautz filter coefficients to an impulse response.

        Parameters
        ----------
        ir : ImpulseResponse
            Single-channel impulse response to which to set the filter
            coefficients. This works best for minimum-phase IRs and IRs without
            initial delay.

        References
        ----------
        - [1]: Bank, B. (2022). Warped, Kautz, and Fixed-Pole Parallel Filters:
          A Review. Journal of the Audio Engineering Society.
        - This function is a port from the matlab toolbox:
          http://legacy.spa.aalto.fi/software/kautz/kautz.htm

        """
        assert (
            ir.number_of_channels == 1
        ), "Only a single-channel IR is supported"
        self.set_filter_coefficients(
            np.ones(self.n_real_poles), np.ones(self.n_complex_poles)
        )
        coefficients = self.__process_time_data_vector(
            ir.time_data[::-1], True
        )
        # Last samples of the single channel for each pole are the optimal LS
        # filter coefficients
        coefficients = coefficients[-1, :, 0]
        self.set_filter_coefficients(
            coefficients[: self.n_real_poles],
            coefficients[self.n_real_poles :],
        )

        self.sampling_rate_hz = ir.sampling_rate_hz

    def filter_signal(self, signal: Signal) -> Signal:
        """Filter a whole signal with the Kautz filter."""
        assert (
            signal.sampling_rate_hz == self.sampling_rate_hz
        ), "Sampling rates do not match"

        new_td = self.__process_time_data_vector(signal.time_data, False)
        new_sig = signal.copy()
        new_sig.time_data = new_td
        return new_sig

    def get_ir(self, length_samples: int) -> ImpulseResponse:
        """Return an impulse response from the Kautz filter."""
        d = dirac(length_samples, 0, sampling_rate_hz=self.sampling_rate_hz)
        return self.filter_signal(d)

    def __process_time_data_vector(
        self,
        time_data: NDArray[np.float64],
        compute_tap_out_matrix: bool = False,
    ) -> NDArray[np.float64]:
        """Process a whole time series with the filter. Output can be either
        the filtered signal with shape (sample, channel), or the tap-out
        matrix (sample, pole, channel)."""
        output_length = time_data.shape[0]
        n_channels = time_data.shape[1]

        if compute_tap_out_matrix:
            output = np.zeros((output_length, self.total_n_poles, n_channels))
        else:
            output = np.zeros((output_length, n_channels))

        # Real poles
        for ii, preal in enumerate(self.poles_real):
            output_tapout = (
                (1.0 - preal**2.0) ** 0.5
                * lfilter([1], [1, -preal], time_data, axis=0)
                * self.coefficients_real_poles[ii]
            )
            if compute_tap_out_matrix:
                output[:, ii, :] = output_tapout
            else:
                output += output_tapout

            time_data = lfilter([-preal, 1], [1, -preal], time_data, axis=0)

        # Complex poles
        q = -2.0 * np.real(self.poles_complex)
        r = np.abs(self.poles_complex) ** 2.0

        ind_tapout = 0
        for ii in range(len(self.poles_complex)):
            output_tapout = (
                ((1 - r[ii]) * (1 + r[ii] - q[ii]) / 2) ** 0.5
                * lfilter([1, -1], [1, q[ii], r[ii]], time_data, axis=0)
                * self.coefficients_complex_poles[ind_tapout]
            )
            if compute_tap_out_matrix:
                output[:, len(self.poles_real) + ind_tapout, :] = output_tapout
            else:
                output += output_tapout
            ind_tapout += 1

            output_tapout = (
                ((1 - r[ii]) * (1 + r[ii] + q[ii]) / 2) ** 0.5
                * lfilter([1, 1], [1, q[ii], r[ii]], time_data, axis=0)
                * self.coefficients_complex_poles[ind_tapout]
            )
            if compute_tap_out_matrix:
                output[:, len(self.poles_real) + ind_tapout, :] = output_tapout
            else:
                output += output_tapout

            ind_tapout += 1
            time_data = lfilter(
                [r[ii], q[ii], 1], [1, q[ii], r[ii]], time_data, axis=0
            )
        return output
