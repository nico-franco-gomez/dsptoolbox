import numpy as np
from numpy.typing import NDArray

def lattice_filtering_fir(
    k: NDArray[np.float64],
    time_data: NDArray[np.float64],
    state: NDArray[np.float64],
) -> None: ...
def warp_time_series(
    time_data: NDArray[np.float64], warping_factor: float
) -> NDArray[np.float64]: ...
