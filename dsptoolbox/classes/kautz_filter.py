import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter
from scipy.linalg import lstsq

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
        self.set_filter_coefficients(
            np.ones(self.n_real_poles), np.ones(self.n_complex_poles)
        )
        self.set_n_channels(1)

    @staticmethod
    def from_ir(ir: ImpulseResponse, order: int, iterations: int):
        """Approximate the IR with an optimal pole basis and coefficients. The
        algorithm is a port from [1] and based on [2].

        Parameters
        ----------
        ir : ImpulseResponse
            IR to approximate
        order : int
            Order of the Kautz filter.
        iterations : int
            Number of iterations for finding optimal poles.

        Returns
        -------
        KautzFilter

        References
        ----------
        - [1]: This function is a port from the matlab toolbox:
          http://legacy.spa.aalto.fi/software/kautz/kautz.htm
        - [2]: H.Brandenstein and R. Unbehauen, "Least-Square Approximation of
          FIR by IIR Filters", IEEE Transactions on Signal Processing, vol. 46,
          no. 1, pp. 21 - 30, 1998.

        """
        f = KautzFilter(np.ones(2) * 0.5, ir.sampling_rate_hz)
        f.fit_poles_and_coefficients_to_ir(ir, order, iterations)
        return f

    def __set_poles(self, poles: NDArray[np.complex128]):
        """Set poles and compute real time filters."""
        # Separate into real and complex
        real_indices = poles.imag == 0.0
        self.poles_real = np.real(poles[real_indices])
        self.poles_complex = poles[~real_indices]

        self.n_complex_poles = len(self.poles_complex) * 2
        self.n_real_poles = len(self.poles_real)
        self.total_n_poles = self.n_complex_poles + self.n_real_poles

        self.__compute_filters()

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
        return self

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
        """Fit Kautz filter coefficients to an impulse response. See references
        for the details on how this is accomplished.

        Parameters
        ----------
        ir : ImpulseResponse
            Single-channel impulse response to which to fit the filter
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
        return self

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
        d = dirac(
            length_samples,
            delay_samples=0,
            sampling_rate_hz=self.sampling_rate_hz,
        )
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

    def fit_poles_and_coefficients_to_ir(
        self, ir: ImpulseResponse, order: int, iterations: int
    ):
        """Find optimal poles for fitting an IR using the algorithm of [1] and
        based on [2].

        References
        ----------
        - [1]: This function is a port from the matlab toolbox:
          http://legacy.spa.aalto.fi/software/kautz/kautz.htm
        - [2]: H.Brandenstein and R. Unbehauen, "Least-Square Approximation of
          FIR by IIR Filters", IEEE Transactions on Signal Processing, vol. 46,
          no. 1, pp. 21 - 30, 1998.

        """
        assert (
            ir.number_of_channels == 1
        ), "Only a single-channel IR is supported"
        poles = KautzFilter.__find_optimal_poles_for_ir(
            order, iterations, ir.time_data.squeeze().copy()
        )
        self.__set_poles(poles)
        self.fit_coefficients_to_ir(ir)
        return self

    @staticmethod
    def __find_optimal_poles_for_ir(
        order: int, iterations: int, target_response: NDArray[np.float64]
    ):
        """Port from [1]. Based on [2].

        References
        ----------
        - [1]: This function is a port from the matlab toolbox:
          http://legacy.spa.aalto.fi/software/kautz/kautz.htm
        - [2]: H.Brandenstein and R. Unbehauen, "Least-Square Approximation of
          FIR by IIR Filters", IEEE Transactions on Signal Processing, vol. 46,
          no. 1, pp. 21 - 30, 1998.

        """
        assert (
            target_response.ndim == 1
        ), "This is only valid for 1D time series"

        response_length = len(target_response)

        # time inverse of target_response
        target_response = target_response[::-1]

        # Accumulators
        matrix_a = np.zeros((response_length, order))
        polynomial_coefficients = np.array([1.0] + [0.0] * order)
        coefficients_matrix = np.zeros((iterations, order + 1))
        error_array = np.zeros(iterations)

        for i in range(iterations):
            filtered_response = lfilter(
                [1.0], polynomial_coefficients, target_response
            )
            vector_b = np.hstack(
                [np.zeros(order), -filtered_response[:-order]]
            )

            # Make A matrix
            matrix_a.fill(0.0)
            matrix_a[:, 0] = filtered_response
            for k in range(1, order):
                matrix_a[k:, k] = filtered_response[:-k]

            least_squares_solution = lstsq(matrix_a, vector_b)[0]

            # Form from LS solution: p[0] * x**n + p[1] * x**(n-1) + ...
            polynomial_coefficients = np.hstack(
                [[1.0], least_squares_solution[::-1]]
            )

            # Numerator z^-order * polynomial_coefficients(z^-1) coefficients
            inverse_polynomial = polynomial_coefficients[::-1]

            # AP filtering  -> Unstable outputs
            allpass_filtered = lfilter(
                inverse_polynomial, polynomial_coefficients, target_response
            )

            # Store coefficients in rows of coefficients_matrix
            coefficients_matrix[i, :] = polynomial_coefficients

            # normalized RMSE
            error_array[i] = np.sum(allpass_filtered**2)

        # Extract poles with the smallest error
        inds = ~np.isnan(error_array)
        min_error_index = np.argmin(error_array[inds])
        poles = np.roots(coefficients_matrix[inds, :][min_error_index, :])
        return poles[poles.imag >= 0.0]
