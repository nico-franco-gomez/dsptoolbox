from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import tf2ss

from ..classes.filter import Filter
from ..standard.enums import FilterCoefficientsType
from .realtime_filter import RealtimeFilter

_state_space_filtering_sample_rust: Callable[..., float] | None = None
try:
    from .._rust import (
        state_space_filtering_sample as _state_space_filtering_sample_rust,
    )
except ImportError:
    pass


class StateSpaceFilter(RealtimeFilter[float]):
    def __init__(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        C: NDArray[np.float64],
        D: NDArray[np.float64],
    ) -> None:
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

    @staticmethod
    def from_filter(filt: Filter) -> "StateSpaceFilter":
        """Get a state-space filter from a common IIR representation. This
        function converts always to b and a coefficients before going into
        A, B, C, D matrices. For better numerical stability in high order
        filters, refer to `from_filter_as_sos_list`.

        Parameters
        ----------
        filt : Filter

        Returns
        -------
        StateSpaceFilter

        """
        b, a = filt.get_coefficients(FilterCoefficientsType.Ba)
        return StateSpaceFilter(*tf2ss(b, a))

    @staticmethod
    def from_filter_as_sos_list(filt: Filter) -> list["StateSpaceFilter"]:
        """Get a state-space filter from a common IIR representation. This
        function converts each SOS of the original filter into A, B, C, D
        matrices and returns a list of second-order StateSpaceFilter.

        Parameters
        ----------
        filt : Filter

        Returns
        -------
        list[StateSpaceFilter]

        """
        sos = filt.get_coefficients(FilterCoefficientsType.Sos)
        n_sections = sos.shape[0]
        return [
            StateSpaceFilter(*tf2ss(sos[n, :3], sos[n, 3:])) for n in range(n_sections)
        ]

    def reset_state(self) -> None:
        self.x.fill(0.0)

    def set_n_channels(self, n_channels: int) -> None:
        self.x = np.zeros((self.A.shape[0], n_channels))

    def process_sample(self, x: float, channel: int) -> float:
        if (
            _state_space_filtering_sample_rust is not None
            and self.A.ndim == 2
            and self.A.dtype == np.dtype(np.float64)
            and self.B.dtype == np.dtype(np.float64)
            and self.C.dtype == np.dtype(np.float64)
            and np.asarray(self.D).dtype == np.dtype(np.float64)
            and self.x.dtype == np.dtype(np.float64)
            and 0 <= channel < self.x.shape[1]
        ):
            return _state_space_filtering_sample_rust(
                self.A,
                self.B,
                self.C,
                float(self.D),
                x,
                self.x,
                channel,
            )

        y = self.C @ self.x[:, channel] + self.D * x
        self.x[:, channel] = self.A @ self.x[:, channel] + self.B * x
        return y
