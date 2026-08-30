from warnings import warn

import numpy as np
from numpy.typing import NDArray

from ..classes import MultiBandSignal, Signal
from ..helpers.latency import (
    _fractional_latency,
    _get_correlation_of_latencies,
)
from ._standard_backend import _latency


def latency(
    in1: Signal | MultiBandSignal,
    in2: Signal | MultiBandSignal | None = None,
    polynomial_points: int = 0,
) -> tuple[NDArray, NDArray[np.float64]]:
    """Computes latency between two signals using the correlation method.
    If there is no second signal, the latency between the first and the other
    channels is computed. `in1` is to be understood as a delayed version
    of `in2` for the latency to be positive. The other way around will give
    the same result but negative.

    This function can compute the sub-sample latency between two signals using
    Zero-Crossing of the analytic (hilbert transformed) correlation function.
    See [1] for more details. The number of polynomial points taken around the
    correlation maximum can be arbitrarily set, although some polynomial orders
    might fail to compute the root. In that case, integer latency will be
    returned for the respective channel. To avoid fractional latency, use
    `polynomial_points = 0`.

    The quality of the estimation is assessed by computing the pearson
    correlation coefficient between the two time series after compensating the
    delay. See notes for details.

    Parameters
    ----------
    in1 : `Signal` or `MultiBandSignal`
        First signal.
    in2 : `Signal` or `MultiBandSignal`, optional
        Second signal. If it is `None`, the first channel of `in1` will be
        taken as `in2`, i.e., the "undelayed" version. Default: `None`.
    polynomial_points : int, optional
        This corresponds to the number of points taken around the root in order
        to fit a polynomial for the fractional latency. Accuracy might improve
        with higher orders but it could also lead to ill-conditioned
        polynomials. In case root finding is not successful, integer latency
        values are returned. Default: 0.

    Returns
    -------
    lags : NDArray[np.float64]
        Delays in samples. For `Signal`, the output shape is (channel).
        In case in2 is `None`, the length is `channels - 1`. In the case of
        `MultiBandSignal`, output shape is (band, channel).
    correlations : NDArray[np.float64]
        Correlation for computed delays with the same shape as lags.

    Notes
    -----
    - The correlation coefficients have values between [-1, 1]. The closer the
      absolute value is to 1, the better the latency estimation. This is always
      computed using the integer latency for performance.

    References
    ----------
    - [1]: N. S. M. Tamim and F. Ghani, "Hilbert transform of FFT pruned cross
      correlation function for optimization in time delay estimation," 2009
      IEEE 9th Malaysia International Conference on Communications (MICC),
      Kuala Lumpur, Malaysia, 2009, pp. 809-814,
      doi: 10.1109/MICC.2009.5431382.

    """
    assert polynomial_points >= 0, "Polynomial points has to be at least 0"
    if polynomial_points == 0:
        latency_func = _latency
        data_type: type[int | float] = int
    else:
        latency_func = _fractional_latency
        data_type = float

    if isinstance(in1, Signal):
        if in2 is not None:
            assert isinstance(in2, Signal), "Both signals must be of type Signal"
            assert in1.sampling_rate_hz == in2.sampling_rate_hz, (
                "Sampling rates must match"
            )
            assert in1.number_of_channels == in2.number_of_channels, (
                "Number of channels between the two signals must match"
            )
            td2 = in2.time_data
        else:
            assert in1.number_of_channels > 1, (
                "Signal must have at least 2 channels to compare"
            )
            td2 = None
        latencies = latency_func(
            in1.time_data, td2, polynomial_points=polynomial_points
        )
        try:
            return latencies, _get_correlation_of_latencies(
                td2 if td2 is not None else in1.time_data[:, 0][..., None],
                in1.time_data if td2 is not None else in1.time_data[:, 1:],
                np.round(latencies, 0).astype(np.int_),
            )
        except (ValueError, IndexError) as e:
            warn(
                "An error occurred while computing the correlations. "
                + f"They are set to 0. Original error: {e}",
                stacklevel=2,
            )
            return latencies, np.zeros(len(latencies))

    elif isinstance(in1, MultiBandSignal):
        if in2 is not None:
            assert isinstance(in2, MultiBandSignal), (
                "Both signals must be of type Signal"
            )
            assert in1.sampling_rate_hz == in2.sampling_rate_hz, (
                "Sampling rates must match"
            )
            pass_in2 = True
        else:
            pass_in2 = False

        if pass_in2:
            lags = np.zeros(
                (in1.number_of_bands, in1.number_of_channels), dtype=data_type
            )
            correlations = np.zeros(
                (in1.number_of_bands, in1.number_of_channels), dtype=np.float64
            )
            for band in range(in1.number_of_bands):
                lags[band, :], correlations[band, :] = latency(
                    in1.bands[band],
                    in2.bands[band],
                    polynomial_points=polynomial_points,
                )
        else:
            lags = np.zeros(
                (in1.number_of_bands, in1.number_of_channels - 1),
                dtype=data_type,
            )
            correlations = np.zeros(
                (in1.number_of_bands, in1.number_of_channels - 1),
                dtype=np.float64,
            )
            for band in range(in1.number_of_bands):
                lags[band, :], correlations[band, :] = latency(
                    in1.bands[band], None, polynomial_points=polynomial_points
                )
        return lags, correlations
    else:
        raise TypeError("Signals must either be type Signal or MultiBandSignal")
