"""
Tools
-----
This module contains general dsp utilities. These functions use exclusively
arrays and primitive data types instead of custom classes.

Levels and conversions:

- `to_db()`, `from_db()`
- `convert_sample_representation()`

Frequency vectors:

- `log_frequency_vector()`
- `erb_frequencies()`
- `fractional_octave_frequencies()`
- `frequency_crossover()`
- `warp_frequency()`

Spectra:

- `interpolate_fr()`
- `scale_spectrum()`
- `fractional_octave_smoothing()`
- `wrap_phase()`
- `get_exact_value_at_frequency()`
- `log_mean()`

Time series:

- `framed_signal()`, `reconstruct_from_framed_signal()`
- `time_smoothing()`, `get_smoothing_factor_ema()`
- `next_power_2()`

"""

from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .helpers.bytes_conversion import (
    _array_to_bytes_24bits,
    _bytes_to_array_24bits,
)
from .helpers.gain_and_level import from_db, to_db
from .helpers.other import _get_next_power_2 as next_power_2
from .helpers.smoothing import (
    _fractional_octave_smoothing as fractional_octave_smoothing,
)
from .helpers.smoothing import (
    _get_smoothing_factor_ema as get_smoothing_factor_ema,
)
from .helpers.smoothing import (
    _time_smoothing as time_smoothing,
)
from .helpers.spectrum_utilities import (
    _interpolate_fr as interpolate_fr,
)
from .helpers.spectrum_utilities import (
    _scale_spectrum as scale_spectrum,
)
from .helpers.spectrum_utilities import (
    _warp_frequency_vector as warp_frequency,
)
from .helpers.spectrum_utilities import (
    _wrap_phase as wrap_phase,
)
from .standard._framed_signal_representation import (
    _get_framed_signal as framed_signal,
)
from .standard._framed_signal_representation import (
    _reconstruct_framed_signal as reconstruct_from_framed_signal,
)
from .standard._standard_backend import (
    _center_frequencies_fractional_octaves_iec,
    _exact_center_frequencies_fractional_octaves,
)
from .standard.enums import SampleFormat


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
    return frequency_range_hz[0] * 2 ** (np.arange(0, n_octave, 1 / n_bins_per_octave))


def get_exact_value_at_frequency(
    freqs_hz: NDArray[np.float64], y: NDArray[Any], f: float = 1e3
) -> NDArray[Any]:
    """Return the exact value at a given frequency by using linear
    interpolation.

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
    assert freqs_hz[0] <= f and freqs_hz[-1] >= f, (
        f"Frequency vector does not contain {f} Hz"
    )
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


def log_mean(x: NDArray[np.float64], axis: int = 0) -> NDArray[np.float64]:
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
    crossover_region_hz: tuple[float, float],
    logarithmic: bool = True,
) -> Callable[[float | NDArray[np.float64]], float | NDArray[np.float64]]:
    """Return a callable that can be used to extract values from a crossover
    to use on frequency data. This uses a hann window function to generate the
    crossover. It is a "fade-in", i.e., the values are 0 before the low
    frequency and rise up to 1 at the high frequency of the crossover.

    Parameters
    ----------
    crossover_region_hz : tuple[float, float]
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


@overload
def fractional_octave_frequencies(
    num_fractions: int = 1,
    frequency_range: tuple[float, float] = (20.0, 20e3),
    return_cutoff: Literal[False] = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def fractional_octave_frequencies(
    num_fractions: int,
    frequency_range: tuple[float, float],
    return_cutoff: Literal[True],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]: ...


def fractional_octave_frequencies(
    num_fractions: int = 1,
    frequency_range: tuple[float, float] = (20.0, 20e3),
    return_cutoff: bool = False,
) -> (
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ]
    | tuple[NDArray[np.float64], NDArray[np.float64]]
):
    """Return the octave center frequencies according to the IEC 61260:1:2014
    standard. This implementation has been taken from the pyfar package. See
    references.

    For numbers of fractions other than `1` and `3`, only the
    exact center frequencies are returned, since nominal frequencies are not
    specified by corresponding standards.

    Parameters
    ----------
    num_fractions : int, optional
        The number of bands an octave is divided into. Eg., `1` refers to
        octave bands and `3` to third octave bands. The default is `1`.
    frequency_range : array, tuple
        The lower and upper frequency limits, the default is
        `frequency_range=(20, 20e3)`.
    return_cutoff : bool, optional
        When `True`, the lower and upper cutoff frequencies of each band are
        returned as a third output. Default: `False`.

    Returns
    -------
    nominal : array, float
        The nominal center frequencies in Hz specified in the standard.
        Nominal frequencies are only returned for octave bands and third octave
        bands. Otherwise, an empty array is returned.
    exact : array, float
        The exact center frequencies in Hz, resulting in a uniform distribution
        of frequency bands over the frequency range.
    cutoff_freq : tuple, array, float
        The lower and upper critical frequencies in Hz of the bandpass filters
        for each band as a tuple corresponding to `(f_lower, f_upper)`.

    References
    ----------
    - The pyfar package: https://github.com/pyfar/pyfar

    """
    nominal = np.array([])

    f_lims = np.asarray(frequency_range)
    if f_lims.size != 2:
        raise ValueError("You need to specify a lower and upper limit frequency.")
    if f_lims[0] > f_lims[1]:
        raise ValueError("The second frequency needs to be higher than the first.")

    if num_fractions in [1, 3]:
        nominal, exact = _center_frequencies_fractional_octaves_iec(
            nominal, num_fractions
        )

        mask = (nominal >= f_lims[0]) & (nominal <= f_lims[1])
        nominal = nominal[mask]
        exact = exact[mask]

    else:
        exact = _exact_center_frequencies_fractional_octaves(num_fractions, f_lims)

    if return_cutoff:
        octave_ratio = 10 ** (3 / 10)
        freqs_upper = exact * octave_ratio ** (1 / 2 / num_fractions)
        freqs_lower = exact * octave_ratio ** (-1 / 2 / num_fractions)
        f_crit = (freqs_lower, freqs_upper)
        return nominal, exact, f_crit
    else:
        return nominal, exact


def erb_frequencies(
    freq_range_hz: tuple[float, float] | None = (20, 20000),
    resolution: float = 1,
    reference_frequency_hz: float = 1000,
) -> NDArray[np.float64]:
    """Get frequencies that are linearly spaced on the ERB frequency scale.
    This implementation was taken and adapted from the pyfar package. See
    references.

    Parameters
    ----------
    freq_range_hz : array-like, optional
        The upper and lower frequency limits in Hz between which the frequency
        vector is computed. Default: [20, 20e3].
    resolution : float, optional
        The frequency resolution in ERB units. 1 returns frequencies that are
        spaced by 1 ERB unit, a value of 0.5 would return frequencies that are
        spaced by 0.5 ERB units. Default: 1.
    reference_frequency_hz : float, optional
        The reference frequency in Hz relative to which the frequency vector
        is constructed. Default: 1000.

    Returns
    -------
    frequencies : NDArray[np.float64]
        The frequencies in Hz that are linearly distributed on the ERB scale
        with a spacing given by `resolution` ERB units.

    References
    ----------
    - The pyfar package: https://github.com/pyfar/pyfar
    - B. C. J. Moore, An introduction to the psychology of hearing,
      (Leiden, Boston, Brill, 2013), 6th ed.
    - V. Hohmann, “Frequency analysis and synthesis using a gammatone
      filterbank,” Acta Acust. united Ac. 88, 433-442 (2002).
    - P. L. Søndergaard, and P. Majdak, “The auditory modeling toolbox,”
      in The technology of binaural listening, edited by J. Blauert
      (Heidelberg et al., Springer, 2013) pp. 33-56.

    """

    # check input
    if (
        not isinstance(freq_range_hz, (list, tuple, np.ndarray))
        or len(freq_range_hz) != 2
    ):
        raise ValueError("freq_range must be an array like of length 2")
    if freq_range_hz[0] > freq_range_hz[1]:
        freq_range_hz = [freq_range_hz[1], freq_range_hz[0]]
    if resolution <= 0:
        raise ValueError("Resolution must be larger than zero")

    # convert the frequency range and reference to ERB scale
    # (Hohmann 2002, Eq. 16)
    erb_range = (
        9.2645 * np.sign(freq_range_hz) * np.log(1 + np.abs(freq_range_hz) * 0.00437)
    )
    erb_ref = (
        9.2645
        * np.sign(reference_frequency_hz)
        * np.log(1 + np.abs(reference_frequency_hz) * 0.00437)
    )

    # get the referenced range
    erb_ref_range = np.array([erb_ref - erb_range[0], erb_range[1] - erb_ref])

    # construct the frequencies on the ERB scale
    n_points = np.floor(erb_ref_range / resolution).astype(int)
    erb_points = np.arange(-n_points[0], n_points[1] + 1) * resolution + erb_ref

    # convert to frequencies in Hz
    frequencies = (
        1 / 0.00437 * np.sign(erb_points) * (np.exp(np.abs(erb_points) / 9.2645) - 1)
    )

    return frequencies


def convert_sample_representation(
    values: NDArray | bytes,
    input_format: SampleFormat,
    output_format: SampleFormat,
    cast_output: bool = True,
    output_in_bytes: bool = False,
) -> tuple[NDArray | bytes, float, float]:
    """This function takes in an array of audio samples and turns it into the
    desired sample output format. It always clips the input to the maximum
    allowed range.

    Parameters
    ----------
    values : NDArray, bytes
        Values to convert. If in bytes, the output will be a flat array as the
        input without reordering.
    input_format : SampleFormat
        Input format for the samples. If the input is a byte array, the samples
        will be read using `numpy.frombuffer()`. In the case of Int24 and
        Uint24, the input is expected to have 3-bytes samples and the
        endianness of the current platform.
    output_format : SampleFormat
        Output format for the samples.
    cast_output : bool, optional
        When True, the output vector is casted to the equivalent data type of
        the output format. This throws an assertion error if the casting is not
        supported by numpy (for Int24 and Uint24) AND the output is not in
        bytes. When avoiding casting, the data type of the output is always
        np.float64. Default: True.
    output_in_bytes : bool, optional
        When True, the array is returned with its bytes representation as
        produced by `numpy.tobytes()`. In the case of Int24 and Uint24 and
        `cast_output=True`, the
        bytes are always produced with the endianness of the current platform
        and the size is 3 per sample bytes and in c ordering. Default: False.

    Returns
    -------
    output : NDArray or bytes
        Vector with samples in the desired format.
    equilibrium : float
        Value that represents equilibrium in the output sample format.
    span_value : float
        Maximum distance from the equilibrium to the ends of the dynamic range
        in the output sample format. Use equilibrium ± span_value to find the
        range of values.

    Notes
    -----
    - Dithering is advised when lowering the bit depth, this is not done
      within this function.
    - Passing the same format as input and output will raise an AssertionError.

    """
    if input_format == output_format:
        raise AssertionError("No conversion is necessary")

    input_is_24_bits = input_format.bit_depth() == 24
    output_is_24_bits = output_format.bit_depth() == 24

    if type(values) is bytes:
        if input_is_24_bits:
            values = _bytes_to_array_24bits(values, input_format.is_signed())
        else:
            values = np.frombuffer(values, dtype=input_format.to_numpy_dtype())

    # ==== Input (convert always to double precision)
    if not input_format.is_float():
        max_value_input = 2.0 ** (input_format.bit_depth() - 1) - 1
        values = values.astype(np.float64) / max_value_input
        if not input_format.is_signed():
            values -= 1.0
    values = np.clip(values, -1.0, 1.0)

    # ==== Output (from double precision to desired format)
    if output_format == SampleFormat.Float32:
        return values.astype(np.float32), 0.0, 1.0
    elif output_format == SampleFormat.Float64:
        return values, 0.0, 1.0

    # Fixed-point output
    max_value_output = 2.0 ** (output_format.bit_depth() - 1) - 1
    output = values * max_value_output
    equilibrium = 0.0

    if not output_format.is_signed():
        output += max_value_output
        equilibrium += max_value_output

    if cast_output:
        if output_is_24_bits:
            assert output_in_bytes, (
                "This format is only valid for casting when " + "the output is in bytes"
            )
            # Held in a 32-bit type until `_array_to_bytes_24bits` packs it
            sample_type = np.int32 if output_format.is_signed() else np.uint32
        else:
            sample_type = output_format.to_numpy_dtype()
        output = output.astype(sample_type)
    else:
        output = np.trunc(output)

    if not output_in_bytes:
        return output, equilibrium, max_value_output

    if output_is_24_bits and cast_output:
        return (
            _array_to_bytes_24bits(output),
            equilibrium,
            max_value_output,
        )
    else:
        return output.tobytes(), equilibrium, max_value_output


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
    "erb_frequencies",
    "fractional_octave_frequencies",
    "scale_spectrum",
    "framed_signal",
    "reconstruct_from_framed_signal",
    "convert_sample_representation",
    "next_power_2",
    "warp_frequency",
]
