"""
This file contains alternative filter implementations.
"""

from warnings import warn
import numpy as np
from numpy.typing import NDArray

from .signal import Signal
from .realtime_filter import RealtimeFilter


class LatticeLadderFilter(RealtimeFilter):
    """This is a class that handles a Lattice/Ladder filter representation.
    Depending on the `k` (reflection) or `c` (feedforward) coefficients, it
    might be a lattice or lattice/ladder filter structure.

    The filtering is done on a pure-python implementation and is considerably
    slower than `scipy.signal.lfilter`.

    References
    ----------
    - Oppenheim, A. V., Schafer, R. W.,, Buck, J. R. (1999). Discrete-Time
      Signal Processing. Prentice-hall Englewood Cliffs.

    """

    def __init__(
        self,
        k_coefficients: NDArray[np.float64],
        c_coefficients: NDArray[np.float64] | None = None,
        sampling_rate_hz: int | None = None,
    ):
        """Constructs a lattice or lattice/ladder filter. If `k_coefficients`
        and `c_coefficients` are passed, it is assumed that it is an IIR
        filter. In case no `c_coefficients` are passed, it is assumed to be an
        FIR filter.

        Filtering can also be done with second-order sections by passing
        2d-arrays as coefficients (only IIR filters are supported).

        Parameters
        ----------
        k_coefficients : NDArray[np.float64]
            Reflection coefficients. It can be a 1d array or a 2d-array for
            second-order sections with shape (section, coefficients).
        c_coefficients : NDArray[np.float64], optional
            Feedforward coefficients. It can be a 1d-array or a 2d-array for
            second-order sections. Default: `None`.
        sampling_rate_hz : int
            Sampling rate of the filter.

        Notes
        -----
        Assuming the input x[n] is on the left of a block diagram and y[n]
        on the right, the coefficients are used in each case as follows:

        - IIR filter: first k (and c) coefficient is the first from right
            to left.
        - FIR filter: first k coefficient is the first from left to right.

        - For second-order sections, the coefficients should be passed as
          2d-arrays. For instance, `k` can have shape (4, 2), meaning that it
          has 4 second-order sections.

        """
        assert sampling_rate_hz is not None, "Sampling rate cannot be None"
        assert k_coefficients.ndim in (2, 1), (
            "k_coefficients should be a " + "vector or a matrix"
        )

        if k_coefficients.ndim == 2:
            assert c_coefficients is not None, (
                "Second-order sections are only valid for IIR filters. "
                + "C coefficients cannot be None"
            )
            assert k_coefficients.shape[1] == 2, (
                "When k has two dimensions, it is assumed that the "
                + "second one has length 2 (second-order section)"
            )
            assert (
                c_coefficients.shape[1] == 3
            ), "Second-order sections should have 3 c coefficients"
            assert (
                c_coefficients.shape[0] == k_coefficients.shape[0]
            ), "Number of second-order sections do not match"
            self.iir_filter = True
            self.sos_filtering = True
        else:
            self.sos_filtering = False
            if c_coefficients is not None and k_coefficients.ndim == 1:
                assert len(c_coefficients) == len(k_coefficients) + 1, (
                    "c_coefficients must have the length "
                    + "len(k_coefficients) + 1"
                )
                self.iir_filter = True
            else:
                self.iir_filter = False
        self.k = k_coefficients
        self.c = c_coefficients
        self.state: NDArray[np.float64] | None = None
        self.sampling_rate_hz = sampling_rate_hz
        self.set_n_channels(1)

    def set_n_channels(self, n_channels: int):
        assert n_channels > 0, "At least one channel must be initialized"

        self.state = np.zeros((len(self.k), n_channels))
        if self.iir_filter:
            if self.sos_filtering:
                self.state = np.zeros((self.k.shape[0], 2, n_channels))
        self.n_channels = n_channels

    def reset_state(self):
        self.state.fill(0.0)

    def filter_signal(self, signal: Signal) -> Signal:
        """Filtering using a lattice ladder structure (general IIR filter). The
        implementation follows [1].

        Parameters
        ----------
        signal : Signal
            Signal to filter.

        Returns
        -------
        Signal
            Filtered signal.

        References
        ----------
        - [1]: Oppenheim, A. V., Schafer, R. W.,, Buck, J. R. (1999).
          Discrete-Time Signal Processing. Prentice-hall Englewood Cliffs.

        """
        assert (
            signal.sampling_rate_hz == self.sampling_rate_hz
        ), "Sampling rates do not match"

        td = signal.time_data.copy()

        if self.n_channels != signal.number_of_channels:
            warn(
                """Number of channels did not match the filter's """
                + "state. The right number of channels are automatically"
                + "initiated"
            )
            self.set_n_channels(signal.number_of_channels)

        if self.iir_filter:
            if self.sos_filtering:
                td, self.state = _lattice_ladder_filtering_sos(
                    self.k, self.c, td, self.state
                )
            else:
                td, self.state = _lattice_ladder_filtering_iir(
                    self.k, self.c, td, self.state
                )
        elif not self.iir_filter:
            td, self.state = _lattice_filtering_fir(self.k, td, self.state)
        elif self.iir_filter and self.c is None:
            raise NotImplementedError(
                "No implementation for all-pole IIR filtering"
            )

        return signal.copy_with_new_time_data(td)

    def process_sample(self, x: float, channel: int):
        """Filtering using a lattice ladder structure (general IIR filter). The
        implementation follows [1].

        Parameters
        ----------
        x : float
            Input sample
        channel : int
            Channel to use (filter state)

        Returns
        -------
        float
            Filtered sample

        References
        ----------
        - [1]: Oppenheim, A. V., Schafer, R. W.,, Buck, J. R. (1999).
          Discrete-Time Signal Processing. Prentice-hall Englewood Cliffs.

        """
        if self.iir_filter:
            if self.sos_filtering:
                return self.__lattice_ladder_filtering_sos_sample(x, channel)
            else:
                return self.__lattice_ladder_filtering_iir_sample(x, channel)
        else:
            return self.__lattice_filtering_fir_sample(x, channel)

    def __lattice_ladder_filtering_sos_sample(
        self, x: float, channel: int
    ) -> float:
        for section in range(self.k.shape[0]):
            x_low = 0

            x += self.state[section, 1, channel] * self.k[section, 1]
            s = x * -self.k[section, 1] + self.state[section, 1, channel]
            x_low += s * self.c[section, 2]

            x += self.state[section, 0, channel] * self.k[section, 0]
            s = x * -self.k[section, 0] + self.state[section, 0, channel]
            self.state[section, 1, channel] = s
            x_low += s * self.c[section, 1]
            self.state[section, 0, channel] = x

            x = x * self.c[section, 0] + x_low
        return x

    def __lattice_ladder_filtering_iir_sample(
        self,
        x: float,
        channel: int,
    ) -> float:
        order_iterations = len(self.k) - 1
        x_low = 0
        for i in range(order_iterations, -1, -1):
            x += self.state[i, channel] * self.k[i]
            s = x * -self.k[i] + self.state[i, channel]
            if i != order_iterations:
                self.state[i + 1, channel] = s
            x_low += s * self.c[i + 1]
        self.state[0, channel] = x
        return x * self.c[0] + x_low

    def __lattice_filtering_fir_sample(self, x: float, channel: int) -> float:
        x_o = x
        s0 = x_o
        for i_k in range(len(self.k)):
            s1 = -x_o * self.k[i_k] + self.state[i_k, channel]
            x_o -= self.state[i_k, channel] * self.k[i_k]
            self.state[i_k, channel] = s0
            s0 = s1
        return x_o


def _lattice_ladder_filtering_sos(
    k: NDArray[np.float64],
    c: NDArray[np.float64],
    td: NDArray[np.float64],
    state: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Filtering using a lattice/ladder structure of second-order sections. See
    `_lattice_ladder_filtering` for the parameter explanation.

    """
    assert k.shape[1] == 2, "Invalid second-order sections"
    assert c.shape[1] == 3, "Invalid second-order sections"

    assert state.shape[0] == k.shape[0], (
        "State first dimension must "
        + "match the number of second-order sections"
    )

    for ch in range(td.shape[1]):
        for i_ch in np.arange(td.shape[0]):
            for section in range(k.shape[0]):
                x = td[i_ch, ch]
                x_low = 0

                x += state[section, 1, ch] * k[section, 1]
                s = x * -k[section, 1] + state[section, 1, ch]
                x_low += s * c[section, 2]

                x += state[section, 0, ch] * k[section, 0]
                s = x * -k[section, 0] + state[section, 0, ch]
                state[section, 1, ch] = s
                x_low += s * c[section, 1]
                state[section, 0, ch] = x

                td[i_ch, ch] = x * c[section, 0] + x_low

    return td, state


def _lattice_filtering_fir(
    k: NDArray[np.float64],
    td: NDArray[np.float64],
    state: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """FIR Filtering using a lattice structure."""
    assert state.shape[0] == k.shape[0], "State length must match filter order"

    for ch in range(td.shape[1]):
        for i_ch in np.arange(td.shape[0]):
            x_o = td[i_ch, ch]
            s0 = x_o
            for i_k in range(len(k)):
                s1 = -x_o * k[i_k] + state[i_k, ch]
                x_o -= state[i_k, ch] * k[i_k]
                state[i_k, ch] = s0
                s0 = s1
            td[i_ch, ch] = x_o
    return td, state


def _lattice_ladder_filtering_iir(
    k: NDArray[np.float64],
    c: NDArray[np.float64],
    td: NDArray[np.float64],
    state: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Filtering using a lattice ladder structure (general IIR filter). The
    implementation follows [1].

    Parameters
    ----------
    k : NDArray[np.float64]
        Reflection coefficients.
    c : NDArray[np.float64]
        Feedforward coefficients.
    td : NDArray[np.float64]
        Time data assumed to have shape (time samples, channel).
    state : NDArray[np.float64], optional
        Initial state for each channel as a 2D-matrix with shape
        (filter order, channel). State of the filter in the beginning. The last
        state corresponds to the last reflection coefficient (furthest to the
        left). If `None`, it is initialized to zero. Default: `None`.

    Returns
    -------
    new_td : NDArray[np.float64]
        Filtered time data.
    state : NDArray[np.float64]
        Filter's state after filtering. It can be `None` if `None` was
        originally passed for `state`.

    References
    ----------
    - [1]: Oppenheim, A. V., Schafer, R. W.,, Buck, J. R. (1999). Discrete-Time
      Signal Processing. Prentice-hall Englewood Cliffs.

    """
    order_iterations = len(k) - 1

    for ch in range(td.shape[1]):
        for i_ch in np.arange(td.shape[0]):
            x = td[i_ch, ch]
            x_low = 0
            for i in range(order_iterations, -1, -1):
                x += state[i, ch] * k[i]
                s = x * -k[i] + state[i, ch]
                if i != order_iterations:
                    state[i + 1, ch] = s
                x_low += s * c[i + 1]
            state[0, ch] = x
            td[i_ch, ch] = x * c[0] + x_low

    return td, state


def _get_lattice_ladder_coefficients_iir(
    b: NDArray[np.float64], a: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute reflection coefficients `k` and ladder coefficients `c` from
    feedforward `b` and feedbackward `a` coefficients according to the
    equations presented in [1].

    Parameters
    ----------
    b : NDArray[np.float64]
        Feedforward coefficients of a filter.
    a : NDArray[np.float64]
        Feedbackward coefficients.

    Returns
    -------
    k : NDArray[np.float64]
        Reflection coefficients with the length of the order .
    c : NDArray[np.float64]
        Ladder coefficients.

    References
    ----------
    - [1]: Oppenheim, A. V., Schafer, R. W.,, Buck, J. R. (1999). Discrete-Time
      Signal Processing. Prentice-hall Englewood Cliffs.

    """
    N = len(a) - 1
    k = np.zeros(N)
    a_s = np.zeros((N, N))

    k[-1] = -a[-1]
    a_s[-1, :] = -a[1:]
    for i in range(N - 2, -1, -1):
        for m in range(i, -1, -1):
            a_s[i, m] = (a_s[i + 1, m] + k[i + 1] * a_s[i + 1, i - m]) / (
                1 - k[i + 1] ** 2
            )
        k[i] = a_s[i, i]

    c = np.zeros(len(b))
    for m in range(len(b) - 1, -1, -1):
        summed = 0
        for i in range(m + 1, len(b)):
            summed += c[i] * a_s[i - 1, i - 1 - m]
        c[m] = b[m] + summed
    return k, c


def _get_lattice_ladder_coefficients_iir_sos(
    sos: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the lattice/ladder coefficients for second-order IIR sections.

    Parameters
    ----------
    sos : NDArray[np.float64]
        Second-order sections with shape (..., 6) as used by `scipy.signal`.

    Returns
    -------
    k_sos : NDArray[np.float64]
        Reflection coefficients for second-order sections.
    c_sos : NDArray[np.float64]
        Ladder coefficients for second-order sections.

    """
    # Normalize second-order sections individually
    if not np.all(sos[:, 3] == 1.0):
        sos /= sos[:, 3]

    n_sections = sos.shape[0]
    k = np.zeros((n_sections, 2))

    k[:, 1] = -sos[:, -1]
    a12 = -sos[:, -2]
    k[:, 0] = (a12 + k[:, 1] * a12) / (1 - k[:, 1] ** 2)

    c = np.zeros((n_sections, 3))
    c[:, 2] = sos[:, 2]
    c[:, 1] = sos[:, 1] + c[:, 2] * a12
    c[:, 0] = sos[:, 0] + c[:, 1] * k[:, 0] + c[:, 2] * k[:, 1]
    return k, c


def _get_lattice_coefficients_fir(
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute reflection coefficients `k` for an FIR filter according to the
    equations presented in [1].

    Parameters
    ----------
    b : NDArray[np.float64]
        Feedforward coefficients of a filter.

    Returns
    -------
    k : NDArray[np.float64]
        Reflection coefficients.

    References
    ----------
    - [1]: Oppenheim, A. V., Schafer, R. W.,, Buck, J. R. (1999). Discrete-Time
      Signal Processing. Prentice-hall Englewood Cliffs.

    """
    N = len(b) - 1
    k = np.zeros(N)
    a_s = np.zeros((N, N))

    k[-1] = -b[-1]
    a_s[-1, :] = -b[1:]
    for i in range(N - 2, -1, -1):
        for m in range(i, -1, -1):
            a_s[i, m] = (a_s[i + 1, m] + k[i + 1] * a_s[i + 1, i - m]) / (
                1 - k[i + 1] ** 2
            )
        k[i] = a_s[i, i]
    return k
