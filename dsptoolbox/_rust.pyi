import numpy as np
from numpy.typing import NDArray

def warp_time_series(
	time_data: NDArray[np.float64], warping_factor: float
) -> NDArray[np.float64]: ...
