from os.path import splitext

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import toeplitz as toeplitz_scipy

from ..standard.enums import Power2Rounding
from .gain_and_level import to_db


def find_nearest_points_index_in_vector(
    points: float | ArrayLike, vector: NDArray[np.float64]
) -> NDArray[np.int_]:
    """Gives back the indexes with the nearest points in vector

    Parameters
    ----------
    points : float or array_like
        Points to look for nearest index in vector.
    vector : array_like
        Vector in which to look for points.

    Returns
    -------
    indexes : ``NDArray[np.int_]``
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
    spec: NDArray[np.float64],
    f: NDArray[np.float64],
    threshold_db: float,
    normalize: bool = True,
) -> list[float]:
    """Finds the first and last frequency above a certain threshold in a given
    (amplitude) spectrum."""
    denum_db = to_db(spec, True)
    if normalize:
        denum_db -= np.max(denum_db)
    freqs = f[denum_db > threshold_db]
    if len(freqs) == 0:
        raise ValueError(
            f"No frequency bin lies above the threshold of {threshold_db} dB"
        )
    return [freqs[0], freqs[-1]]


def _toeplitz(h: NDArray[np.float64], length_of_input: int) -> NDArray[np.float64]:
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
    row = np.zeros(length_of_input)
    row[0] = h[0]
    return toeplitz_scipy(c=column, r=row)


def _get_path_extension(path: str) -> str:
    """Return the lower-case extension of a path, without the leading dot."""
    return splitext(path)[1].lower().lstrip(".")


def _check_path_format(path: str, expected_format: str) -> None:
    """Ensure that a path carries the expected file extension.

    Parameters
    ----------
    path : str
        Path of file.
    expected_format : str
        Extension that the path must have, without the leading dot.

    Raises
    ------
    ValueError
        When the path has no extension or a different one.

    """
    extension = _get_path_extension(path)
    if extension != expected_format:
        raise ValueError(
            f"The path must end in '.{expected_format}', but it "
            + (f"ends in '.{extension}'" if extension else "has no extension")
        )


def _get_next_power_2(
    number: float, mode: Power2Rounding = Power2Rounding.Closest
) -> int:
    """This function returns the power of 2 closest to the given number.

    Parameters
    ----------
    number : int, float
        Number for which to find the closest power of 2.
    mode : Power2Rounding, optional
        Rounding direction. Default: Closest.

    Returns
    -------
    number_2 : int
        Next power of 2 according to the selected mode.

    """
    assert number > 0, "Only positive numbers are valid"

    p = np.log2(number)
    if mode == Power2Rounding.Closest:
        remainder = p - int(p)
        mode = Power2Rounding.Floor if remainder < 0.5 else Power2Rounding.Ceil
    if mode == Power2Rounding.Floor:
        p = np.floor(p).astype(int)
    elif mode == Power2Rounding.Ceil:
        p = np.ceil(p).astype(int)
    return int(2**p)


def _euclidean_distance_matrix(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
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
    assert x.ndim == 2 and y.ndim == 2, "Inputs must have exactly two dimensions"
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
    return np.array([f_c * 2 ** (-1 / fraction / 2), f_c * 2 ** (1 / fraction / 2)])


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
            raise ValueError("This function is only implemented for 1D and 2D arrays")
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
