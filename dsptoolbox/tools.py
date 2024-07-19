"""
This module contains general math and dsp utilities. These functions are solely
based on arrays and primitive data types.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any
from scipy.interpolate import interp1d

from ._general_helpers import (
    _fractional_octave_smoothing as fractional_octave_smoothing,
    _wrap_phase as wrap_phase,
    _get_smoothing_factor_ema as get_smoothing_factor_ema,
    _interpolate_fr as interpolate_fr,
    _time_smoothing as time_smoothing,
)


def log_frequency_vector(
    frequency_range_hz: list[float], n_bins_per_octave: int
) -> NDArray[np.float64]:
    """Obtain a logarithmically spaced frequency vector with a specified number
    of frequency bins per octave.

    Parameters
    ----------
    frequency_range_hz : list[float]
        Frequency with length 2 for defining the frequency range. The lowest
        frequency should be above 0.
    n_bins_per_octave : int
        Number of frequency bins in each octave.

    Returns
    -------
    NDArray[np.float64]
        Log-spaced frequency vector

    """
    assert frequency_range_hz[0] > 0, "The first frequency bin should not be 0"

    n_octave = np.log2(frequency_range_hz[1] / frequency_range_hz[0])
    return frequency_range_hz[0] * 2 ** (
        np.arange(0, n_octave, 1 / n_bins_per_octave)
    )


def to_db(
    x: NDArray[np.float64],
    amplitude_input: bool,
    dynamic_range_db: float | None = None,
    min_value: float | None = float(np.finfo(np.float64).smallest_normal),
) -> NDArray[np.float64]:
    """Convert to dB from amplitude or power representation. Clipping small
    values can be activated in order to avoid -inf dB outcomes.

    Parameters
    ----------
    x : NDArray[np.float64]
        Array to convert to dB.
    amplitude_input : bool
        Set to True if the values in x are in their linear form. False means
        they have been already squared, i.e., in their power form.
    dynamic_range_db : float, None, optional
        If specified, a dynamic range in dB for the vector is applied by
        finding its largest value and clipping to `max - dynamic_range_db`.
        This will always overwrite `min_value` if specified. Pass None to
        ignore. Default: None.
    min_value : float, None, optional
        Minimum value to clip `x` before converting into dB in order to avoid
        `np.nan` or `-np.inf` in the output. Pass None to ignore. Default:
        `np.finfo(np.float64).smallest_normal`.

    Returns
    -------
    NDArray[np.float64]
        New array or float in dB.

    """
    factor = 20.0 if amplitude_input else 10.0

    if min_value is None and dynamic_range_db is None:
        return factor * np.log10(np.abs(x))

    x_abs = np.abs(x)

    if dynamic_range_db is not None:
        min_value = np.max(x_abs) * 10.0 ** (-abs(dynamic_range_db) / factor)

    return factor * np.log10(np.clip(x_abs, a_min=min_value, a_max=None))


def from_db(x: float | NDArray[np.float64], amplitude_output: bool):
    """Get the values in their amplitude or power form from dB.

    Parameters
    ----------
    x : float, NDArray[np.float64]
        Values in dB.
    amplitude_output : bool
        When True, the values are returned in their linear form. Otherwise,
        the squared (power) form is returned.

    Returns
    -------
    float NDArray[np.float64]
        Converted values

    """
    factor = 20.0 if amplitude_output else 10.0
    return 10 ** (x / factor)


def get_exact_value_at_frequency(
    freqs_hz: NDArray[np.float64], y: NDArray[Any], f: float = 1e3
):
    """Return the exact value at 1 kHz extracted by using linear interpolation.

    Parameters
    ----------
    freqs_hz : NDArray[np.float64]
        Frequency vector in Hz. It is assumed to be in ascending order.
    y : NDArray[np.float64]
        Values to use for the interpolation.
    f : float, optional
        Frequency to query. Default: 1000.

    Returns
    -------
    float
        Queried value.

    """
    assert (
        freqs_hz[0] <= f and freqs_hz[-1] >= f
    ), "Frequency vector does not contain 1 kHz"
    assert freqs_hz.ndim == 1, "Frequency vector can only have one dimension"
    assert len(freqs_hz) == len(y), "Lengths do not match"

    # Single value in vector or last value matches
    if freqs_hz[-1] == f:
        return y[-1]

    ind = np.searchsorted(freqs_hz, f)
    if freqs_hz[ind] > f:
        ind -= 1
    return (f - freqs_hz[ind]) * (y[ind + 1] - y[ind]) / (
        freqs_hz[ind + 1] - freqs_hz[ind]
    ) + y[ind]


def log_mean(x: NDArray[np.float64], axis: int = 0):
    """Get the mean value while using a logarithmic x-axis. It is assumed that
    `x` is initially linearly-spaced.

    Parameters
    ----------
    x : NDArray[np.float64]
        Vector for which to obtain the mean.
    axis : int, optional
        Axis along which to compute the mean.

    Returns
    -------
    float or NDArray[np.float64]
        Logarithmic mean along the selected axis.

    """
    # Linear and logarithmic frequency vector
    N = x.shape[axis]
    l1 = np.arange(N)
    k_log = (N) ** (l1 / (N - 1))
    # Interpolate to logarithmic scale
    vec_log = interp1d(
        l1 + 1, x, kind="linear", copy=False, assume_sorted=True, axis=axis
    )(k_log)
    return np.mean(vec_log, axis=axis)


def frequency_crossover(
    crossover_region_hz: list[float],
    logarithmic: bool = True,
):
    """Return a callable that can be used to extract values from a crossover
    to use on frequency data. This uses a hann window function to generate the
    crossover. It is a "fade-in", i.e., the values are 0 before the low
    frequency and rise up to 1 at the high frequency of the crossover.

    Parameters
    ----------
    crossover_region_hz : list with length 2
        Frequency range for which to create the crossover.
    logarithmic : bool, optional
        When True, the crossover is defined logarithmically on the frequency
        axis. Default: True.

    Returns
    -------
    callable
        Callable that produces values from the crossover function. The input
        should always be in Hz. It can take float or NDArray[np.float64] and
        returns the same type.

    """
    f = (
        log_frequency_vector(crossover_region_hz, 250)
        if logarithmic
        else np.linspace(
            crossover_region_hz[0],
            crossover_region_hz[1],
            int(crossover_region_hz[1] - crossover_region_hz[0]),
        )
    )
    length = len(f)
    w = np.hanning(length * 2)[:length]
    i = interp1d(
        f,
        w,
        kind="cubic",
        copy=False,
        bounds_error=False,
        fill_value=(0.0, 1.0),
        assume_sorted=True,
    )

    def func(x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        return i(x)

    return func


__all__ = [
    "fractional_octave_smoothing",
    "wrap_phase",
    "get_smoothing_factor_ema",
    "interpolate_fr",
    "time_smoothing",
    "log_frequency_vector",
    "to_db",
    "from_db",
    "get_exact_value_at_frequency",
    "log_mean",
    "frequency_crossover",
]
