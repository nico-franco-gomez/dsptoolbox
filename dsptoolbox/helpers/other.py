from os import sep
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import toeplitz as toeplitz_scipy

from .gain_and_level import to_db


def find_nearest_points_index_in_vector(points, vector) -> NDArray[np.int_]:
    """Gives back the indexes with the nearest points in vector

    Parameters
    ----------
    points : float or array_like
        Points to look for nearest index in vector.
    vector : array_like
        Vector in which to look for points.

    Returns
    -------
    indexes : `NDArray[np.int_]`
        Indexes of the points.

    """
    points = np.array(points)
    if np.ndim(points) == 0:
        points = points[..., None]
    indexes = np.zeros(len(points), dtype=np.int_)
    for ind, p in enumerate(points):
        indexes[ind] = np.argmin(np.abs(p - vector))
    return indexes


def find_frequencies_above_threshold(
    spec, f, threshold_db, normalize=True
) -> list:
    """Finds frequencies above a certain threshold in a given (amplitude)
    spectrum."""
    denum_db = to_db(spec, True)
    if normalize:
        denum_db -= np.max(denum_db)
    freqs = f[denum_db > threshold_db]
    return [freqs[0], freqs[-1]]


def _toeplitz(
    h: NDArray[np.float64], length_of_input: int
) -> NDArray[np.float64]:
    """Creates a toeplitz matrix from a system response given an input length.

    Parameters
    ----------
    h : NDArray[np.float64]
        System's impulse response.
    length_of_input : int
        Input length needed for the shape of the toeplitz matrix.

    Returns
    -------
    NDArray[np.float64]
        Toeplitz matrix with shape (len(h)+length_of_input-1, length_of_input).
        Convolution is done by using dot product from the right::

            convolve_result = toeplitz_matrix @ input_vector

    """
    column = np.hstack([h, np.zeros(length_of_input - 1)])
    row = np.zeros((length_of_input))
    row[0] = h[0]
    return toeplitz_scipy(c=column, r=row)


def _check_format_in_path(path: str, desired_format: str) -> str:
    """Checks if a given path already has a format and it matches the desired
    format. If not, an assertion error is raised. If the path does not have
    any format, the desired one is added.

    Parameters
    ----------
    path : str
        Path of file.
    desired_format : str
        Format that the file should have.

    Returns
    -------
    str
        Path with the desired format.

    """
    format = path.split(sep)[-1].split(".")
    if len(format) != 1:
        assert (
            format[-1] == desired_format
        ), f"{format[-1]} is not the desired format"
    else:
        path += f".{desired_format}"
    return path


def _get_next_power_2(number, mode: str = "closest") -> int:
    """This function returns the power of 2 closest to the given number.

    Parameters
    ----------
    number : int, float
        Number for which to find the closest power of 2.
    mode : str {'closest', 'floor', 'ceil'}, optional
        `'closest'` gives the closest value. `'floor'` returns the next smaller
        power of 2 and `'ceil'` the next larger. Default: `'closest'`.

    Returns
    -------
    number_2 : int
        Next power of 2 according to the selected mode.

    """
    assert number > 0, "Only positive numbers are valid"
    mode = mode.lower()
    assert mode in (
        "closest",
        "floor",
        "ceil",
    ), "Mode must be either closest, floor or ceil"

    p = np.log2(number)
    if mode == "closest":
        remainder = p - int(p)
        mode = "floor" if remainder < 0.5 else "ceil"
    if mode == "floor":
        p = np.floor(p).astype(int)
    elif mode == "ceil":
        p = np.ceil(p).astype(int)
    return int(2**p)


def _euclidean_distance_matrix(x: NDArray[np.float64], y: NDArray[np.float64]):
    """Compute the euclidean distance matrix between two vectors efficiently.

    Parameters
    ----------
    x : NDArray[np.float64]
        First vector or matrix with shape (Point x, Dimensions).
    y : NDArray[np.float64]
        Second vector or matrix with shape (Point y, Dimensions).

    Returns
    -------
    dist : NDArray[np.float64]
        Euclidean distance matrix with shape (Point x, Point y).

    """
    assert (
        x.ndim == 2 and y.ndim == 2
    ), "Inputs must have exactly two dimensions"
    assert x.shape[1] == y.shape[1], "Dimensions do not match"
    return np.sqrt(
        np.sum(x**2, axis=1, keepdims=True)
        + np.sum(y.T**2, axis=0, keepdims=True)
        - 2 * x @ y.T
    )


def _get_fractional_octave_bandwidth(
    f_c: float, fraction: int = 1
) -> NDArray[np.float64]:
    """Returns an array with lower and upper bounds for a given center
    frequency with (1/fraction)-octave width.

    Parameters
    ----------
    f_c : float
        Center frequency.
    fraction : int, optional
        Octave fraction to define bandwidth. Passing 0 just returns the center
        frequency as lower and upper bounds. Default: 1.

    Returns
    -------
    f_bounds : NDArray[np.float64]
        Array of length 2 with lower and upper bounds.

    """
    if fraction == 0:
        return np.array([f_c, f_c])
    return np.array(
        [f_c * 2 ** (-1 / fraction / 2), f_c * 2 ** (1 / fraction / 2)]
    )


def _compute_number_frames(
    window_length: int, step: int, signal_length: int, zero_padding: bool
) -> tuple[int, int]:
    """Gives back the number of frames that will be computed.

    Parameters
    ----------
    window_length : int
        Length of the window to be used.
    step : int
        Step size in samples. It is defined as `window_length - overlap`.
    signal_length : int
        Total signal length.
    zero_padding : bool
        When `True`, it is assumed that the signal will be zero padded in the
        end to make use of all time samples. `False` will effectively discard
        the blocks where zero-padding would be needed.

    Returns
    -------
    n_frames : int
        Number of frames to be observed in the signal.
    padding_samples : int
        Number of samples with which the signal should be padded.

    """
    if zero_padding:
        n_frames = int(np.ceil(signal_length / step))
        padding_samples = window_length - int(signal_length % step)
    else:
        padding_samples = 0
        n_frames = int(np.ceil((signal_length - window_length) / step))
    return n_frames, padding_samples


def _pad_trim(
    vector: NDArray,
    desired_length: int,
    axis: int = 0,
    in_the_end: bool = True,
) -> NDArray:
    """Pads (with zeros) or trim (depending on size and desired length)."""
    if vector.shape[axis] == desired_length:
        return vector.copy()

    throw_axis = False
    if vector.ndim < 2:
        assert axis == 0, "You can only pad along the 0 axis"
        vector = vector[..., None]
        throw_axis = True
    elif vector.ndim > 2:
        vector = vector.squeeze()
        if vector.ndim > 2:
            raise ValueError(
                "This function is only implemented for 1D and 2D arrays"
            )
    type_of_data = vector.dtype
    diff = desired_length - vector.shape[axis]
    if axis == 1:
        vector = vector.T
    if diff > 0:
        if not in_the_end:
            vector = np.flip(vector, axis=0)
        new_vec = np.concatenate(
            [vector, np.zeros((diff, vector.shape[1]), dtype=type_of_data)]
        )
        if not in_the_end:
            new_vec = np.flip(new_vec, axis=0)
    elif diff < 0:
        if not in_the_end:
            vector = np.flip(vector, axis=0)
        new_vec = vector[:desired_length, :]
        if not in_the_end:
            new_vec = np.flip(new_vec, axis=0)
    else:
        new_vec = vector.copy()
    if axis == 1:
        new_vec = new_vec.T
    if throw_axis:
        new_vec = new_vec[:, 0]
    return new_vec
