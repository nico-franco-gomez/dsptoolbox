import numpy as np
from numpy.typing import NDArray
import scipy.signal as sig
from scipy.linalg import lstsq

from .realtime_filter import RealtimeFilter
from .iir_filter_realtime import IIRFilter
from .fir_filter_realtime import FIRFilter
from .impulse_response import ImpulseResponse
from .filterbank import FilterBank
from .filter import Filter
from .signal import Signal
from ..generators import dirac


class ParallelFilter(RealtimeFilter):
    """Filter bank that processes filtering using SOS in parallel and an FIR
    part. See [1] for details.

    References
    ----------
    - [1]: Bank, B. (2022). Warped, Kautz, and Fixed-Pole Parallel Filters: A
    Review. Journal of the Audio Engineering Society.

    """

    def __init__(
        self, poles: NDArray[np.complex128], n_fir: int, sampling_rate_hz: int
    ):
        """Instantiate a parallel filter bank from a pole basis and a number of
        FIR coefficients. Details are given in [1]. Use `set_parameters()` and
        `set_coefficients()` for configuring the filter bank.

        Parameters
        ----------
        poles : NDArray[np.complex128]
            Poles to use as basis for the approximation. Only real poles or
            complex poles with positive imaginary parts should be passed
            (they will be automatically conjugated). See [1] for detailed
            information on how to find a suitable basis.
        n_fir : int, optional
            Number of FIR coefficients to use during the approximation. At
            least 1 is recommended, though an all-pole approximation could also
            deliver satisfactory results in some cases. Default: 1.
        sampling_rate_hz : int
            Sampling rate for the filter bank.

        Notes
        -----
        - The frequency resolution of the filter bank is given by the set of
          poles.
        - Methods for finding optimal poles are reviewed in depth in [1].
        - It was noticed that delaying the IIR coefficients can be advantageous
          in terms of quality of the approximation.

        References
        ----------
        - [1]: Bank, B. (2022). Warped, Kautz, and Fixed-Pole Parallel Filters:
        A Review. Journal of the Audio Engineering Society.

        """
        assert n_fir >= 0, "n_fir must be at least 0"
        assert np.all(
            np.abs(poles) < 1.0
        ), "At least one pole lies outside the unit circle"
        assert np.all(
            poles.imag >= 0.0
        ), "Only poles with positive imaginary part are accepted"
        assert np.all(
            np.abs(poles) > 0.0
        ), "No poles at the origin should be used"
        assert all(
            [np.sum(np.isclose(poles, p)) == 1 for p in poles]
        ), "Pole multiplicity cannot be more than 1"
        assert sampling_rate_hz > 0, "Sampling rate must be greater than 0"

        self.poles = poles
        self.n_fir = n_fir
        self.sampling_rate_hz = sampling_rate_hz
        self.set_parameters()

    def set_parameters(
        self,
        delay_iir_samples: int = 0,
        fir_offset_ms: float = 0.0,
    ):
        """Parameters for the parallel filter bank.

        Parameters
        ----------
        delay_iir_samples : int, optional
            This delay is applied to the IIR part of the filter bank. It can be
            used in order to separate the influence of the zeros from the poles
            in the transfer function when setting `delay_iir_samples = n_fir`.
            Default: 0.
        fir_offset_ms : float, optional
            A delay between the FIR coefficients can be added with this offset.
            Set to 0 if all FIR coefficients should be contiguous. Default: 0.

        """
        assert delay_iir_samples >= 0, "Delay should not be negative"
        self.fir_offset_samples = max(
            1, int(self.sampling_rate_hz * fir_offset_ms / 1e3 + 0.5)
        )
        self.delay_iir_samples = (
            self.n_fir + 1 + self.fir_offset_samples * (self.n_fir - 1)
            if delay_iir_samples is None
            else delay_iir_samples
        )
        return self

    def set_coefficients(
        self,
        iir_coefficients: NDArray[np.float64],
        fir: NDArray[np.float64] | None = None,
    ):
        """Set the parallel filter coefficients.

        Parameters
        ----------
        iir_coefficients : NDArray[np.float64]
            Coefficients for each SOS. It must have shape (n_sos, 2).
        fir : NDArray[np.float64]
            FIR coefficients.

        """
        assert iir_coefficients.ndim == 2
        assert iir_coefficients.shape[0] == self.__sos.shape[0]

        for ss in range(self.__sos.shape[0]):
            self.__sos[ss, :2] = iir_coefficients[ss, :]

        if fir is not None:
            assert fir.ndim == 1
            self.__fir_coefficients = fir
        else:
            self.__fir_coefficients = np.array([])
        self.n_fir = len(self.__fir_coefficients)
        return self

    def fit_to_ir(self, ir: ImpulseResponse):
        """Fit the filter coefficients of this filter bank to an IR using the
        frequency-domain least-squares approximation as outlined in [1].

        Parameters
        ----------
        ir : ImpulseResponse
            IR to approximate. The IR should contain a single channel. For the
            approximation to deliver satisfactory results, it is recommended
            that the IR be minimum phase.

        References
        ----------
        - [1]: Bank, B. (2022). Warped, Kautz, and Fixed-Pole Parallel Filters:
        A Review. Journal of the Audio Engineering Society.

        """
        assert (
            ir.number_of_channels == 1
        ), "This is only valid for a single-channel IR"
        freqs, spectrum_channels = ir.get_spectrum()
        freqs = freqs[1:]
        spectrum_channels = spectrum_channels[1:]
        fs_hz = ir.sampling_rate_hz

        # Get SOS
        comp_inds = self.poles.imag != 0
        poles = np.hstack([self.poles, self.poles[comp_inds].conjugate()])
        self.__sos = sig.zpk2sos([], poles, 1.0)
        n_sos = self.__sos.shape[0]

        # ========== Create model matrix
        n_parameters = n_sos * 3 + self.n_fir
        L = len(freqs)
        M = np.zeros((L, n_parameters), dtype=np.complex128)

        for ind in range(0, n_sos * 3, 3):
            M[:, ind] = sig.sosfreqz(
                self.__sos[ind // 3, :][None, :], freqs, fs=fs_hz
            )[1]

            # Delayed by one sample
            sos_delayed = self.__sos[ind // 3, :].copy()
            sos_delayed[0] = 0.0
            sos_delayed[1] = 1.0
            M[:, ind + 1] = sig.sosfreqz(
                sos_delayed[None, :], freqs, fs=fs_hz
            )[1]

            # Delayed by two samples
            sos_delayed = self.__sos[ind // 3, :].copy()
            sos_delayed[0] = 0.0
            sos_delayed[1] = 0.0
            sos_delayed[2] = 1.0
            M[:, ind + 2] = sig.sosfreqz(
                sos_delayed[None, :], freqs, fs=fs_hz
            )[1]

        # Apply IIR delay to SOS sections
        if self.delay_iir_samples > 0:
            M[:, : n_sos * 3] *= sig.freqz(
                [0.0] * self.delay_iir_samples + [1.0], [1.0], freqs, fs=fs_hz
            )[1][:, None]

        for n in range(self.n_fir):
            # Put impulse at index n for the FIR part
            M[:, n_sos * 3 + n] = sig.freqz(
                np.hstack([[0.0] * (n * self.fir_offset_samples), [1.0]]),
                [1.0],
                freqs,
                fs=fs_hz,
            )[1]
        # ==========================

        # Make "real" model matrix
        M = np.vstack([np.real(M), np.imag(M)])

        # Solve "real" optimization problem
        spectrum = spectrum_channels[:, 0]
        spectrum = np.hstack([np.real(spectrum), np.imag(spectrum)])
        solution = lstsq(M, spectrum, overwrite_a=True, overwrite_b=True)[0]

        # Extract IIR and FIR solution and put into SOS
        for ind in range(0, n_sos * 3, 3):
            self.__sos[ind // 3, 0] = solution[ind]
            self.__sos[ind // 3, 1] = solution[ind + 1]
            self.__sos[ind // 3, 2] = solution[ind + 2]

        self.__fir_coefficients = solution[n_sos * 3 :]

        # Put delays in between the fir coefficients
        if self.fir_offset_samples > 1 and self.n_fir > 1:
            ff = np.zeros(
                (
                    (self.fir_offset_samples)
                    * (len(self.__fir_coefficients) - 1)
                    + 1
                )
            )
            ff[:: self.fir_offset_samples + 1] = self.__fir_coefficients[:-1]
            ff[-1] = self.__fir_coefficients[-1]
            self.__fir_coefficients = ff

        self.__compute_filter_bank()
        return self

    def __compute_filter_bank(self):
        fb = FilterBank(
            [
                Filter.from_sos(
                    self.__sos[n, :][None, ...], self.sampling_rate_hz
                )
                for n in range(self.__sos.shape[0])
            ]
        )
        if len(self.__fir_coefficients) > 0:
            fb.add_filter(
                Filter.from_ba(
                    self.__fir_coefficients, [1.0], self.sampling_rate_hz
                )
            )
        self.filter_bank = fb
        self.__compute_real_time_filters()

    def __compute_real_time_filters(self):
        assert hasattr(self, "filter_bank"), "Filter bank needed"
        self.iir: list[IIRFilter] = []
        for f in self.filter_bank:
            if f.filter_type == "fir":
                self.fir = FIRFilter(f.get_coefficients("ba")[0])
            else:
                self.iir.append(IIRFilter(*f.get_coefficients("ba")))
        if self.delay_iir_samples > 0:
            self.iir_delay = FIRFilter(
                np.array(self.delay_iir_samples * [0.0] + [1.0])
            )

    def filter_signal(self, signal: Signal) -> Signal:
        """Filter a signal using the parallel filter bank.

        Parameters
        ----------
        signal : Signal
            Signal to be filtered.

        Returns
        -------
        Signal
            Filtered signal. It has the same length as the input.

        """
        assert (
            self.sampling_rate_hz == signal.sampling_rate_hz
        ), "Sampling rates do not match"
        td = signal.time_data

        if self.n_fir > 0:
            output = sig.oaconvolve(
                td, self.__fir_coefficients[:, None], axes=0
            )[: td.shape[0], ...]
        else:
            output = np.zeros_like(td)

        if self.delay_iir_samples > 0:
            td = np.pad(td, ((self.delay_iir_samples, 0), (0, 0)))[
                : td.shape[0]
            ]

        for n_sos in range(self.__sos.shape[0]):
            output += sig.sosfilt(self.__sos[n_sos, :][None, :], td, axis=0)
        new_sig = signal.copy()
        new_sig.time_data = output
        return new_sig

    def get_ir(self, length_samples: int):
        """Get an impulse response from the filter bank."""
        d = dirac(length_samples, sampling_rate_hz=self.sampling_rate_hz)
        return self.filter_signal(d)

    def set_n_channels(self, n_channels: int):
        for f in self.iir:
            f.set_n_channels(n_channels)
        if self.n_fir > 0:
            self.fir.set_n_channels(n_channels)
        if self.delay_iir_samples > 0:
            self.iir_delay.set_n_channels(n_channels)

    def reset_state(self):
        for f in self.iir:
            f.reset_state()
        if self.n_fir > 1:
            self.fir.reset_state()
        if self.delay_iir_samples > 0:
            self.iir_delay.reset_state()

    def process_sample(self, x: float, channel: int):
        y = 0.0

        # FIR
        if self.n_fir > 1:
            y += self.fir.process_sample(x, channel)
        elif self.n_fir == 1:
            y += self.__fir_coefficients[0] * x

        # Delay
        if self.delay_iir_samples > 0:
            x = self.iir_delay.process_sample(x, channel)

        # IIR
        for f in self.iir:
            y += f.process_sample(x, channel)
        return y
