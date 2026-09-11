import numpy as np
from numpy.typing import NDArray

def laguerre(
    time_data: NDArray[np.float64], warping_factor: float
) -> NDArray[np.float64]: ...
def lattice_filtering_fir(
    k: NDArray[np.float64],
    time_data: NDArray[np.float64],
    state: NDArray[np.float64],
) -> None: ...
def warp_time_series(
    time_data: NDArray[np.float64], warping_factor: float
) -> NDArray[np.float64]: ...
def squeeze_scalogram(
    scalogram: NDArray[np.complex128],
    freqs: NDArray[np.float64],
    fs: float,
    delta_w: float,
    apply_frequency_normalization: bool,
    gradient: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
