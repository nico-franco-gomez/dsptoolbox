import numpy as np
from numpy.typing import NDArray
from .realtime_filter import RealtimeFilter


class StateSpaceFilter(RealtimeFilter):
    def __init__(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        C: NDArray[np.float64],
        D: NDArray[np.float64],
    ):
        """Instantiate a state space filter from the system's matrices. State
        space filters have the structure:

        ```
            y[n] = Cx[n] + Du[n]
            x[n+1] = Ax[n] + Bu[n]
        ```

        where `y` is the output, `u` the input, `x` the internal state vector.
        `A` is the state transition matrix, which determines the dynamics of
        the system. The matrices are expected to be in the controller canonical
        form, which is the output of `scipy.signal.tf2ss`.

        The filter is initiated for 1 channel by default.

        Parameters
        ----------
        A : NDArray[np.float64]
            State transition.
        B : NDArray[np.float64]
            Input influence.
        C : NDArray[np.float64]
            Output coupling.
        D : NDArray[np.float64]
            Direct transmission.

        References
        ----------
        - https://www.dsprelated.com/freebooks/filters/State_Space_Filters.html

        """
        assert A.ndim == 2, "Matrix A should have exactly 2 dimensions"
        assert len(B) == A.shape[1], "Matrix B dimensions are not valid"
        self.A = A.squeeze()
        self.B = B.squeeze()
        self.C = C.squeeze()
        self.D = D.squeeze()
        self.set_n_channels(1)

    def reset_state(self):
        self.x.fill(0.0)

    def set_n_channels(self, n_channels):
        self.x = np.zeros((self.A.shape[0], n_channels))

    def process_sample(self, x, channel):
        y = self.C @ self.x[:, channel] + self.D * x
        self.x[:, channel] = self.A @ self.x[:, channel] + self.B * x
        return y
