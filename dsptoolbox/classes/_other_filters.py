"""
This file contains alternative filter implementations.
"""
from .signal_class import Signal
from warnings import warn
import numpy as np


class LatticeLadderFilter():
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
    def __init__(self, k_coefficients: np.ndarray,
                 c_coefficients: np.ndarray = None,
                 iir_filter: bool = None, sampling_rate_hz: int = None):
        """Construct a lattice or lattice/ladder filter. If `k_coefficients`
        and `c_coefficients` are passed, it is assumed that it is an IIR
        filter. In case no `c_coefficients` are passed, the user must define
        if it is an IIR or FIR filter.

        Parameters
        ----------
        k_coefficients : `np.ndarray`
            Reflection coefficients (1d array).
        c_coefficients : `np.ndarray`, optional
            Feedforward coefficients (1d array). Default: `None`.
        iir_filter : bool, optional
            This can be set to `None` if both k and c coefficients are passed
            (general IIR filter). In case there are only k coefficients, `True`
            assumes an all-pole IIR filter while `False` gives a lattice FIR
            representation. Default: `None`.
        sampling_rate_hz : int
            Sampling rate of the filter.

        Notes
        -----
        Assuming the input x[n] is on the left of a block diagram and y[n]
        on the right, the coefficients are used in each case as follows:

        - IIR filter: first k (and c) coefficient is the first from right
            to left.
        - FIR filter: first k coefficient is the first from left to right.

        """
        assert sampling_rate_hz is not None, 'Sampling rate cannot be None'
        assert k_coefficients.ndim == 1, 'k_coefficients should be a vector'
        if c_coefficients is not None:
            assert len(c_coefficients) == len(k_coefficients) + 1, \
                'c_coefficients must have the length len(k_coefficients) + 1'
            self.iir_filter = True
        else:
            assert iir_filter is not None, \
                'It must be specificied whether it is an IIR or FIR filter'
            self.iir_filter = iir_filter
        self.k = k_coefficients
        self.c = c_coefficients
        self.state = None
        self.sampling_rate_hz = sampling_rate_hz

    def initialize_zi(self, n_channels: int):
        """Initialize the filter's state values for a number of channels.

        Parameters
        ----------
        n_channels : int
            Number of channels for which to initialize the filter's states.

        """
        self.state = np.zeros((len(self.k), n_channels))

    def filter_signal(self, signal: Signal, channels=None,
                      activate_zi: bool = False) -> Signal:
        """Filter the selected channels of a signal.

        Parameters
        ----------
        signal : `Signal`
            Signal to filter.
        channels : `np.ndarray`, int, optional
            Channels to filter. If `None`, all channels of the signal are
            filtered.
        activate_zi : bool, optional
            When `True`, the current filter state is reloaded and updated
            after filtering.

        Returns
        -------
        filtered_signal : `Signal`
            Filtered signal.

        """
        assert signal.sampling_rate_hz == self.sampling_rate_hz, \
            'Sampling rates do not match'
        if channels is None:
            channels = np.arange(signal.number_of_channels)
        else:
            channels = np.atleast_1d(channels)
            assert np.all(signal.number_of_channels > channels), \
                'Requested channel to filter does not exist'

        td = signal.time_data[:, channels]

        if activate_zi:
            if self.state.shape[1] != len(channels):
                warn('''Number of channels did not match the filter's ''' +
                     'state. The right number of channels are automatically' +
                     'initiated')
                self.initialize_zi(len(channels))

        if self.iir_filter and self.c is not None:
            td, self.state = \
                _lattice_ladder_filtering(self.k, self.c, td, self.state)
        elif not self.iir_filter:
            raise NotImplementedError('No implementation for FIR filtering')
        elif self.iir_filter and self.c is None:
            raise NotImplementedError(
                'No implementation for all-pole IIR filtering')

        filtered_signal = signal.copy()
        new_td = filtered_signal.time_data
        new_td[:, channels] = td
        filtered_signal.time_data = new_td
        return filtered_signal


def _lattice_ladder_filtering(
        k: np.ndarray, c: np.ndarray, td: np.ndarray,
        state: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Filtering using a lattice ladder structure (general IIR filter). The
    implementation follows [1].

    Parameters
    ----------
    k : `np.ndarray`
        Reflection coefficients.
    c : `np.ndarray`
        Feedforward coefficients.
    td : `np.ndarray`
        Time data assumed to have shape (time samples, channel).
    state : `np.ndarray`, optional
        Initial state for each channel as a 2D-matrix with shape
        (filter order, channel). State of the filter in the beginning.
        If `None`, it is initialized to zero. Default: `None`.

    Returns
    -------
    new_td : `np.ndarray`
        Filtered time data.
    state : `np.ndarray`
        Filter's state after filtering. It can be `None` if `None` was
        originally passed for `state`.

    References
    ----------
    - [1]: Oppenheim, A. V., Schafer, R. W.,, Buck, J. R. (1999). Discrete-Time
      Signal Processing. Prentice-hall Englewood Cliffs.

    """
    passed_state = True
    if state is None:
        passed_state = False
        state = np.zeros((len(k), td.shape[1]))
    order_iterations = len(k)-1

    for ch in range(td.shape[1]):
        for i_ch in np.arange(td.shape[0]):
            x = td[i_ch, ch]
            x_low = 0
            for i in range(order_iterations, -1, -1):
                x += state[i, ch]*k[i]
                s = x * -k[i] + state[i, ch]
                if i != order_iterations:
                    state[i+1, ch] = s
                x_low += s*c[i+1]
            state[0, ch] = x
            td[i_ch, ch] = x*c[0] + x_low

    if not passed_state:
        state = None
    return td, state


def _get_lattice_ladder_coefficients_iir(b: np.ndarray, a: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray]:
    """Compute reflection coefficients `k` and ladder coefficients `c` from
    feedforward `b` and feedbackward `a` coefficients according to the
    equations presented in [1].

    Parameters
    ----------
    b : `np.ndarray`
        Feedforward coefficients of a filter.
    a : `np.ndarray`
        Feedbackward coefficients.

    Returns
    -------
    k : `np.ndarray`
        Reflection coefficients with the length of the order .
    c : `np.ndarray`
        Ladder coefficients.

    References
    ----------
    - [1]: Oppenheim, A. V., Schafer, R. W.,, Buck, J. R. (1999). Discrete-Time
      Signal Processing. Prentice-hall Englewood Cliffs.

    """
    N = len(a)-1
    k = np.zeros(N)
    a_s = np.zeros((N, N))

    k[-1] = -a[-1]
    a_s[-1, :] = -a[1:]
    for i in range(N-2, -1, -1):
        for m in range(i, -1, -1):
            a_s[i, m] = (a_s[i+1, m]+k[i+1]*a_s[i+1, i-m])/(1-k[i+1]**2)
        k[i] = a_s[i, i]

    c = np.zeros(len(b))
    for m in range(len(b)-1, -1, -1):
        summed = 0
        for i in range(m+1, len(b)):
            summed += c[i]*a_s[i-1, i-1-m]
        c[m] = b[m] + summed
    return k, c


def _get_lattice_coefficients_fir(b: np.ndarray) \
        -> np.ndarray:
    """Compute reflection coefficients `k` for an FIR filter according to the
    equations presented in [1].

    Parameters
    ----------
    b : `np.ndarray`
        Feedforward coefficients of a filter.

    Returns
    -------
    k : `np.ndarray`
        Reflection coefficients.

    References
    ----------
    - [1]: Oppenheim, A. V., Schafer, R. W.,, Buck, J. R. (1999). Discrete-Time
      Signal Processing. Prentice-hall Englewood Cliffs.

    """
    N = len(b)-1
    k = np.zeros(N)
    a_s = np.zeros((N, N))

    k[-1] = -b[-1]
    a_s[-1, :] = -b[1:]
    for i in range(N-2, -1, -1):
        for m in range(i, -1, -1):
            a_s[i, m] = (a_s[i+1, m]+k[i+1]*a_s[i+1, i-m])/(1-k[i+1]**2)
        k[i] = a_s[i, i]
    return k
