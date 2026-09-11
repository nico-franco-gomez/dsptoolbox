from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ..classes.filter import Filter
from ..classes.signal import Signal
from ..standard.enums import FilterCoefficientsType, WarpingFactorType
from .realtime_filter import RealtimeFilter

_warped_fir_filtering_rust: Callable | None
_warped_fir_filtering_block_rust: Callable | None
_warped_fir_filtering_sample_rust: Callable | None
_warped_iir_filtering_rust: Callable | None
_warped_iir_filtering_block_rust: Callable | None
_warped_iir_filtering_sample_rust: Callable | None
try:
    from .._rust import (
        warped_fir_filtering as _warped_fir_filtering_rust,
    )
    from .._rust import (
        warped_fir_filtering_block as _warped_fir_filtering_block_rust,
    )
    from .._rust import (
        warped_fir_filtering_sample as _warped_fir_filtering_sample_rust,
    )
    from .._rust import (
        warped_iir_filtering as _warped_iir_filtering_rust,
    )
    from .._rust import (
        warped_iir_filtering_block as _warped_iir_filtering_block_rust,
    )
    from .._rust import (
        warped_iir_filtering_sample as _warped_iir_filtering_sample_rust,
    )
except ImportError:
    _warped_fir_filtering_rust = None
    _warped_fir_filtering_block_rust = None
    _warped_fir_filtering_sample_rust = None
    _warped_iir_filtering_rust = None
    _warped_iir_filtering_block_rust = None
    _warped_iir_filtering_sample_rust = None


def _warped_fir_filtering_python(
    b: NDArray[np.float64],
    warp: float,
    time_data: NDArray[np.float64],
    state: NDArray[np.float64],
) -> None:
    for channel in range(time_data.shape[1]):
        for sample in range(time_data.shape[0]):
            residue = time_data[sample, channel]
            output = residue * b[0]
            for coefficient in range(len(b) - 1):
                new_residue = (
                    state[coefficient + 1, channel] - residue
                ) * warp + state[coefficient, channel]
                state[coefficient, channel] = residue
                residue = new_residue
                output += new_residue * b[coefficient + 1]
            state[-1, channel] = residue
            time_data[sample, channel] = output


def _warped_iir_filtering_python(
    b: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    warp: float,
    time_data: NDArray[np.float64],
    state: NDArray[np.float64],
) -> None:
    for channel in range(time_data.shape[1]):
        for sample in range(time_data.shape[0]):
            value = time_data[sample, channel]
            value += sigmas[1:] @ state[: len(sigmas) - 1, channel]
            value *= sigmas[0]
            residue = value
            output = residue * b[0]
            for coefficient in range(state.shape[0] - 1):
                new_residue = (
                    state[coefficient + 1, channel] - residue
                ) * warp + state[coefficient, channel]
                state[coefficient, channel] = residue
                residue = new_residue
                if coefficient + 1 < len(b):
                    output += new_residue * b[coefficient + 1]
            state[-1, channel] = residue
            time_data[sample, channel] = output


class WarpedFIR(RealtimeFilter[float]):
    """The Warped FIR filter has a structure like a common FIR filter but with
    allpasses instead of unit delays between each coefficient. This warps
    the input during the filtering stage.

    This implementation is done efficiently according to [1].

    References
    ----------
    - [1]: Karjalainen, M. & Härmä, Aki & Laine, Unto & Huopaniemi, J.. (1997).
      Warped filters and their audio applications. 4 pp..
      10.1109/ASPAA.1997.625615.

    """

    def __init__(
        self,
        b: NDArray[np.float64],
        warping_factor: WarpingFactorType,
        sampling_rate_hz: int,
    ) -> None:
        """Instantiate a warped FIR filter with its coefficients and a warping
        factor. See [1] for details on use and implementation.

        Parameters
        ----------
        b : NDArray[np.float64]
            Feedforward filter coefficients.
        warping_factor : WarpingFactor
            Factor to use for warping. Use
            `WarpingFactor.Custom.with_factor()` to pass an explicit value in
            ]-1; 1[.
        sampling_rate_hz : int
            Sampling rate of the filter. It is only relevant when filtering a
            whole signal and not in the sample-by-sample processing.

        References
        ----------
        - [1]: Karjalainen, M. & Härmä, Aki & Laine, Unto & Huopaniemi, J..
          (1997). Warped filters and their audio applications. 4 pp..
          10.1109/ASPAA.1997.625615.

        """
        self.sampling_rate_hz = sampling_rate_hz
        self.b = b
        self.warp = warping_factor.get_factor(sampling_rate_hz)
        self.N = len(self.b)
        self.order = len(self.b) - 1
        self.set_n_channels(1)

    @staticmethod
    def from_filter(filt: Filter, warping_factor: WarpingFactorType) -> "WarpedFIR":
        """Instantiate with the coefficients of a filter. It must be FIR

        Parameters
        ----------
        filt : Filter
            Filter with coefficients.
        warping_factor : WarpingFactor
            Factor to use for warping.

        Returns
        -------
        WarpedFIR

        """
        assert filt.is_fir, "This is only valid for a FIR filter"
        b, _ = filt.get_coefficients(FilterCoefficientsType.Ba)
        return WarpedFIR(b, warping_factor, filt.sampling_rate_hz)

    def set_n_channels(self, n_channels: int) -> None:
        assert n_channels > 0
        self.buffer = np.zeros((self.N, n_channels))

    def reset_state(self) -> None:
        self.buffer.fill(0.0)

    def process_sample(self, x: float, channel: int) -> float:
        if (
            _warped_fir_filtering_sample_rust is not None
            and self.b.dtype == np.float64
            and self.buffer.dtype == np.float64
            and 0 <= channel < self.buffer.shape[1]
        ):
            return _warped_fir_filtering_sample_rust(
                self.b, self.warp, x, self.buffer, channel
            )
        return self._process_sample_python(x, channel)

    def _process_sample_python(self, x: float, channel: int) -> float:
        # Start delay-free output
        output = x * self.b[0]
        residue = x

        # Update states and accumulate in output
        for nn in range(self.order):
            # New value
            new_residue = (
                self.buffer[nn + 1, channel] - residue
            ) * self.warp + self.buffer[nn, channel]
            # Accumulate old in buffer
            self.buffer[nn, channel] = residue
            # Swap
            residue = new_residue
            # Accumulate output
            if nn + 1 < len(self.b):
                output += new_residue * self.b[nn + 1]

        # Save last residue
        self.buffer[-1, channel] = residue

        return output

    def process_block(
        self, block: NDArray[np.float64], channel: int
    ) -> NDArray[np.float64]:
        output: NDArray[np.float64] = np.empty(len(block), dtype=np.float64)
        if (
            _warped_fir_filtering_block_rust is not None
            and self.b.dtype == np.float64
            and getattr(block, "dtype", None) == np.float64
            and 0 <= channel < self.buffer.shape[1]
        ):
            _warped_fir_filtering_block_rust(
                self.b, self.warp, block, output, self.buffer, channel
            )
        else:
            for index in range(len(block)):
                output[index] = self.process_sample(block[index], channel)
        return output

    def filter_signal(self, signal: Signal) -> Signal:
        """Filter a whole signal with the warped FIR filter. The existing
        buffers are left unmodified in this operation.

        Parameters
        ----------
        signal : Signal
            Signal to be filtered.

        """
        assert self.sampling_rate_hz == signal.sampling_rate_hz, (
            "Sampling rates do not match"
        )
        buffer_prior = self.buffer.copy()
        self.set_n_channels(signal.number_of_channels)
        new_signal = signal.copy_with_new_time_data(
            self._process_time_data_vector(signal.time_data)
        )
        self.buffer = buffer_prior
        return new_signal

    def _process_time_data_vector(
        self, time_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        output = np.array(time_data, copy=True)
        if (
            _warped_fir_filtering_rust is not None
            and self.b.dtype == np.float64
            and output.dtype == np.float64
        ):
            _warped_fir_filtering_rust(self.b, self.warp, output, self.buffer)
            return output
        _warped_fir_filtering_python(self.b, self.warp, output, self.buffer)
        return output


class WarpedIIR(WarpedFIR):
    """The Warped IIR filter has a structure which warps the input during the
    filtering stage. This implementation is done according to [1].

    References
    ----------
    - [1]: Karjalainen, M. & Härmä, Aki & Laine, Unto & Huopaniemi, J.. (1997).
      Warped filters and their audio applications. 4 pp..
      10.1109/ASPAA.1997.625615.

    """

    def __init__(
        self,
        b: NDArray[np.float64],
        a: NDArray[np.float64],
        warping_factor: WarpingFactorType,
        sampling_rate_hz: int,
    ) -> None:
        """Instantiate a warped IIR filter with its coefficients and a warping
        factor. See [1] for details on use and implementation.

        Parameters
        ----------
        b : NDArray[np.float64]
            Feedforward filter coefficients.
        a : NDArray[np.float64]
            Feedbackward filter coefficients.
        warping_factor : WarpingFactor
            Factor to use for warping. Use
            `WarpingFactor.Custom.with_factor()` to pass an explicit value in
            ]-1; 1[.
        sampling_rate_hz : int
            Sampling rate of the filter. It is only relevant when filtering a
            whole signal and not in the sample-by-sample processing.

        References
        ----------
        - [1]: Karjalainen, M. & Härmä, Aki & Laine, Unto & Huopaniemi, J..
          (1997). Warped filters and their audio applications. 4 pp..
          10.1109/ASPAA.1997.625615.

        """
        assert b.ndim == 1, "Coefficients can only have a single dimension"
        assert a.ndim == 1, "Coefficients can only have a single dimension"

        self.N = max(len(a), len(b))
        self.order = self.N - 1

        # Normalize coefficients
        self.b = b / a[0]
        self.a = a / a[0]

        # Prepare rest data
        self.warp = warping_factor.get_factor(sampling_rate_hz)
        self.sampling_rate_hz = sampling_rate_hz
        self.set_n_channels(1)
        self.__compute_sigmas()

    @staticmethod
    def from_filter(filt: Filter, warping_factor: WarpingFactorType) -> "WarpedIIR":
        """Instantiate with the coefficients of a filter. It must be IIR

        Parameters
        ----------
        filt : Filter
            Filter with coefficients.
        warping_factor : WarpingFactor
            Factor to use for warping.

        Returns
        -------
        WarpedFIR

        """
        assert filt.is_iir, "This is only valid for a IIR filter"
        b, a = filt.get_coefficients(FilterCoefficientsType.Ba)
        return WarpedIIR(b, a, warping_factor, filt.sampling_rate_hz)

    def __compute_sigmas(self) -> None:
        """Computation from Karjalainen, M. & Härmä, Aki & Laine, Unto &
        Huopaniemi, J.. (1997). Warped filters and their audio applications.
        4 pp.. 10.1109/ASPAA.1997.625615.

        """
        # Start vector from the end
        N = len(self.a)
        self.sigmas = np.zeros(N + 1)
        self.sigmas[-1] = self.warp * self.a[-1]
        S = self.a[-1]

        for i in range(N - 1, 1, -1):
            S_new = self.a[i - 1] - self.warp * S
            self.sigmas[i] = self.warp * S_new + S
            S = S_new

        # Compute first entries
        self.sigmas[1] = S

        # Prepare for realtime application
        self.sigmas[0] = 1.0 / (1.0 - self.warp * S)
        self.sigmas[1:] *= -1.0

    def process_sample(self, x: float, channel: int) -> float:
        if (
            _warped_iir_filtering_sample_rust is not None
            and self.b.dtype == np.float64
            and self.sigmas.dtype == np.float64
            and self.buffer.dtype == np.float64
            and 0 <= channel < self.buffer.shape[1]
        ):
            return _warped_iir_filtering_sample_rust(
                self.b, self.sigmas, self.warp, x, self.buffer, channel
            )

        # IIR section
        x += (self.sigmas[1:] @ self.buffer[: len(self.sigmas) - 1, channel]).item()
        x *= self.sigmas[0]
        # FIR section
        return self._process_sample_python(x, channel)

    def process_block(
        self, block: NDArray[np.float64], channel: int
    ) -> NDArray[np.float64]:
        output: NDArray[np.float64] = np.empty(len(block), dtype=np.float64)
        if (
            _warped_iir_filtering_block_rust is not None
            and self.b.dtype == np.float64
            and getattr(block, "dtype", None) == np.float64
            and 0 <= channel < self.buffer.shape[1]
        ):
            _warped_iir_filtering_block_rust(
                self.b,
                self.sigmas,
                self.warp,
                block,
                output,
                self.buffer,
                channel,
            )
        else:
            for index in range(len(block)):
                output[index] = self.process_sample(block[index], channel)
        return output

    def _process_time_data_vector(
        self, time_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        output = np.array(time_data, copy=True)
        if (
            _warped_iir_filtering_rust is not None
            and self.b.dtype == np.float64
            and output.dtype == np.float64
        ):
            _warped_iir_filtering_rust(
                self.b, self.sigmas, self.warp, output, self.buffer
            )
            return output
        _warped_iir_filtering_python(
            self.b, self.sigmas, self.warp, output, self.buffer
        )
        return output
