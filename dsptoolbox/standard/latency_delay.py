import numpy as np
from numpy.typing import NDArray
from scipy.signal import oaconvolve
from warnings import warn

from ..classes import Signal, MultiBandSignal
from ._standard_backend import _latency, _fractional_delay_filter
from .._general_helpers import (
    _pad_trim,
    _fractional_latency,
    _get_correlation_of_latencies,
)


def latency(
    in1: Signal | MultiBandSignal,
    in2: Signal | MultiBandSignal | None = None,
    polynomial_points: int = 0,
) -> tuple[NDArray[np.float64] | NDArray[np.int_], NDArray[np.float64]]:
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
            assert (
                in1.sampling_rate_hz == in2.sampling_rate_hz
            ), "Sampling rates must match"
            assert (
                in1.number_of_channels == in2.number_of_channels
            ), "Number of channels between the two signals must match"
            assert isinstance(
                in2, Signal
            ), "Both signals must be of type Signal"
            td2 = in2.time_data
        else:
            assert (
                in1.number_of_channels > 1
            ), "Signal must have at least 2 channels to compare"
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
        except Exception as e:
            print(e)
            warn(
                "An error occured while computing the correlations. "
                + "They are set to 0."
            )
            return latencies, np.zeros(len(latencies))

    elif isinstance(in1, MultiBandSignal):
        if in2 is not None:
            assert isinstance(
                in2, MultiBandSignal
            ), "Both signals must be of type Signal"
            assert (
                in1.sampling_rate_hz == in2.sampling_rate_hz
            ), "Sampling rates must match"
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
        return lags
    else:
        raise TypeError(
            "Signals must either be type Signal or MultiBandSignal"
        )


def fractional_delay(
    sig: Signal | MultiBandSignal,
    delay_seconds: float,
    channels=None,
    keep_length: bool = False,
    order: int = 30,
    side_lobe_suppression_db: float = 60,
) -> Signal | MultiBandSignal:
    """Apply fractional time delay to a signal.

    Parameters
    ----------
    sig : `Signal` or `MultiBandSignal`
        Signal to be delayed.
    delay_seconds : float
        Delay in seconds.
    channels : int or array-like, optional
        Channels to be delayed. Pass `None` to delay all channels.
        Default: `None`.
    keep_length : bool, optional
        When `True`, the signal retains its original length and loses
        information for the latest samples. If only specific channels are to be
        delayed, and keep_length is set to `False`, the remaining channels are
        zero-padded in the end. Default: `False`.
    order : int, optional
        Order of the sinc filter, higher order yields better results at the
        expense of computation time. Default: 30.
    side_lobe_suppression_db : float, optional
        Side lobe suppresion in dB for the Kaiser window. Default: 60.

    Returns
    -------
    out_sig : `Signal` or `MultiBandSignal`
        Delayed signal.

    """
    assert delay_seconds >= 0, "Delay must be positive"
    if isinstance(sig, Signal):
        if delay_seconds == 0:
            return sig.copy()
        if sig.time_data_imaginary is not None:
            warn(
                "Imaginary time data will be ignored in this function. "
                + "Delay it manually by creating another signal object, if "
                + "needed."
            )
        delay_samples = delay_seconds * sig.sampling_rate_hz
        if keep_length:
            assert (
                delay_samples < sig.time_data.shape[0]
            ), "Delay too large for the given signal"
        if channels is None:
            channels = np.arange(sig.number_of_channels)
        channels = np.atleast_1d(np.asarray(channels).squeeze())
        assert np.all(channels < sig.number_of_channels) and len(
            np.unique(channels)
        ) == len(channels), "There is at least an invalid channel number"

        # Get filter and integer delay
        delay_int, frac_delay_filter = _fractional_delay_filter(
            delay_samples, order, side_lobe_suppression_db
        )

        # Copy data
        new_time_data = sig.time_data

        # Create space for the filter in the end of signal
        new_time_data = _pad_trim(
            new_time_data, sig.time_data.shape[0] + len(frac_delay_filter) - 1
        )

        # Delay channels
        new_time_data[:, channels] = oaconvolve(
            sig.time_data[:, channels],
            frac_delay_filter[..., None],
            mode="full",
            axes=0,
        )

        # Handle delayed and undelayed channels
        channels_not = np.setdiff1d(
            channels, np.arange(new_time_data.shape[1])
        )
        not_delayed = new_time_data[:, channels_not]
        delayed = new_time_data[:, channels]

        # Delay respective channels in the beginning and add zeros in the end
        # to the others
        delayed = _pad_trim(
            delayed, delay_int + new_time_data.shape[0], in_the_end=False
        )
        not_delayed = _pad_trim(
            not_delayed, delay_int + new_time_data.shape[0], in_the_end=True
        )

        new_time_data = _pad_trim(
            new_time_data, delay_int + new_time_data.shape[0], in_the_end=True
        )
        new_time_data[:, channels_not] = not_delayed
        new_time_data[:, channels] = delayed

        # =========== handle length ===========================================
        if keep_length:
            new_time_data = new_time_data[: sig.time_data.shape[0], :]

        # =========== give out object =========================================
        out_sig = sig.copy()
        out_sig.clear_time_window()
        out_sig.time_data = new_time_data

    elif isinstance(sig, MultiBandSignal):
        new_bands = []
        out_sig = sig.copy()
        for b in sig.bands:
            new_bands.append(
                fractional_delay(
                    b,
                    delay_seconds,
                    channels,
                    keep_length,
                    order,
                    side_lobe_suppression_db,
                )
            )
        out_sig.bands = new_bands
    else:
        raise TypeError(
            "Passed signal should be either type Signal or "
            + "MultiBandSignal"
        )
    return out_sig


def delay(
    sig: Signal | MultiBandSignal,
    delay_samples: int,
    channels=None,
    keep_length: bool = False,
) -> Signal | MultiBandSignal:
    """Apply a time delay to a signal. This function is faster than
    `fractional_delay` because it only applies integer delay by zero-padding.

    Parameters
    ----------
    sig : `Signal` or `MultiBandSignal`
        Signal to be delayed.
    delay_samples : int
        Delay in samples.
    channels : int or array-like, optional
        Channels to be delayed. Pass `None` to delay all channels.
        Default: `None`.
    keep_length : bool, optional
        When `True`, the signal retains its original length and loses
        information for the latest samples. If only specific channels are to be
        delayed, and keep_length is set to `False`, the remaining channels are
        zero-padded in the end. Default: `False`.

    Returns
    -------
    out_sig : `Signal` or `MultiBandSignal`
        Delayed signal.

    """
    if isinstance(sig, Signal):
        if delay_samples == 0:
            return sig.copy()
        if keep_length:
            assert (
                delay_samples < sig.time_data.shape[0]
            ), "Delay too large for the given signal"
        if channels is None:
            channels = np.arange(sig.number_of_channels)
        channels = np.atleast_1d(np.asarray(channels).squeeze())
        assert np.all(channels < sig.number_of_channels) and len(
            np.unique(channels)
        ) == len(channels), "There is at least an invalid channel number"

        # Copy data
        new_time_data = sig.time_data

        # Handle delayed and undelayed channels
        channels_not = np.setdiff1d(
            channels, np.arange(new_time_data.shape[1])
        )
        not_delayed = new_time_data[:, channels_not]
        delayed = new_time_data[:, channels]

        delayed = _pad_trim(
            delayed, delay_samples + new_time_data.shape[0], in_the_end=False
        )
        not_delayed = _pad_trim(
            not_delayed,
            delay_samples + new_time_data.shape[0],
            in_the_end=True,
        )

        new_time_data = _pad_trim(
            new_time_data,
            delay_samples + new_time_data.shape[0],
            in_the_end=True,
        )
        new_time_data[:, channels_not] = not_delayed
        new_time_data[:, channels] = delayed
        if keep_length:
            new_time_data = new_time_data[: sig.time_data.shape[0], :]

        out_sig = sig.copy()
        out_sig.clear_time_window()
        out_sig.time_data = new_time_data
    elif isinstance(sig, MultiBandSignal):
        new_bands = []
        out_sig = sig.copy()
        for b in sig.bands:
            new_bands.append(delay(b, delay_samples, channels, keep_length))
        out_sig.bands = new_bands
    else:
        raise TypeError(
            "Passed signal should be either type Signal or "
            + "MultiBandSignal"
        )
    return out_sig
