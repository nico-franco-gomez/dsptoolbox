import numpy as np
from numpy.typing import NDArray

# Realtime filter structures
def kautz_filtering_sample(
    real_poles: NDArray[np.float64],
    real_coefficients: NDArray[np.float64],
    complex_q: NDArray[np.float64],
    complex_r: NDArray[np.float64],
    complex_coefficients: NDArray[np.float64],
    input: float,
    real_state: NDArray[np.float64],
    real_advance_state: NDArray[np.float64],
    complex_state: NDArray[np.float64],
    complex_advance_state: NDArray[np.float64],
    channel: int,
) -> float: ...
def parallel_filtering_sample(
    iir_b: NDArray[np.float64],
    iir_a: NDArray[np.float64],
    fir_b: NDArray[np.float64],
    delay_b: NDArray[np.float64],
    input: float,
    iir_state: NDArray[np.float64],
    fir_state: NDArray[np.float64],
    fir_index: NDArray[np.int64],
    delay_state: NDArray[np.float64],
    delay_index: NDArray[np.int64],
    channel: int,
) -> float: ...
def fir_filtering_sample(
    b: NDArray[np.float64],
    input: float,
    state: NDArray[np.float64],
    current_state_ind: NDArray[np.int64],
    channel: int,
) -> float: ...
def iir_filtering_sample(
    b: NDArray[np.float64],
    a: NDArray[np.float64],
    input: float,
    state: NDArray[np.float64],
    channel: int,
) -> float: ...
def lattice_filtering_fir(
    k: NDArray[np.float64],
    time_data: NDArray[np.float64],
    state: NDArray[np.float64],
) -> None: ...
def lattice_filtering_fir_sample(
    k: NDArray[np.float64],
    input: float,
    state: NDArray[np.float64],
    channel: int,
) -> float: ...
def lattice_filtering_iir(
    k: NDArray[np.float64],
    c: NDArray[np.float64],
    time_data: NDArray[np.float64],
    state: NDArray[np.float64],
) -> None: ...
def lattice_filtering_iir_sample(
    k: NDArray[np.float64],
    c: NDArray[np.float64],
    input: float,
    state: NDArray[np.float64],
    channel: int,
) -> float: ...
def lattice_filtering_sos(
    k: NDArray[np.float64],
    c: NDArray[np.float64],
    time_data: NDArray[np.float64],
    state: NDArray[np.float64],
) -> None: ...
def lattice_filtering_sos_sample(
    k: NDArray[np.float64],
    c: NDArray[np.float64],
    input: float,
    state: NDArray[np.float64],
    channel: int,
) -> float: ...
def warped_fir_filtering(
    b: NDArray[np.float64],
    warp: float,
    time_data: NDArray[np.float64],
    state: NDArray[np.float64],
) -> None: ...
def warped_fir_filtering_sample(
    b: NDArray[np.float64],
    warp: float,
    input: float,
    state: NDArray[np.float64],
    channel: int,
) -> float: ...
def warped_fir_filtering_block(
    b: NDArray[np.float64],
    warp: float,
    input: NDArray[np.float64],
    output: NDArray[np.float64],
    state: NDArray[np.float64],
    channel: int,
) -> None: ...
def warped_iir_filtering(
    b: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    warp: float,
    time_data: NDArray[np.float64],
    state: NDArray[np.float64],
) -> None: ...
def warped_iir_filtering_sample(
    b: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    warp: float,
    input: float,
    state: NDArray[np.float64],
    channel: int,
) -> float: ...
def warped_iir_filtering_block(
    b: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    warp: float,
    input: NDArray[np.float64],
    output: NDArray[np.float64],
    state: NDArray[np.float64],
    channel: int,
) -> None: ...

# Transforms
def laguerre(
    time_data: NDArray[np.float64], warping_factor: float
) -> NDArray[np.float64]: ...
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
