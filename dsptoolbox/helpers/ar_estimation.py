import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate


def _levison_durbin_recursion(
    autocorrelation: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Levinson-Durbin recursion to be applied to the autocorrelation estimate.
    It is always computed along the first (most outer) axis.

    Parameters
    ----------
    autocorrelation : NDArray[np.float64]
        Autocorrelation function with only positive lags and length of
        `order + 1`, where `order` corresponds to the order of the AR
        estimation. It can have any shape, but the AR parameters are always
        computed along the outer axis.

    Returns
    -------
    reflection_coefficients : NDArray[np.float64]
        Denominator coefficients with shape (coefficient, ...).
    prediction_error : NDArray[np.float64]
        Variance of the remaining error.

    """
    prediction_error = autocorrelation[0, ...].copy()  # Signal variance
    autocorr_coefficients = autocorrelation[1:, ...].copy()

    num_coefficients = autocorr_coefficients.shape[0]
    ar_parameters = np.zeros_like(autocorr_coefficients)

    for order in range(num_coefficients):
        reflection_value = autocorr_coefficients[order].copy()
        if order == 0:
            reflection_coefficient = -reflection_value / prediction_error
        else:
            for lag in range(order):
                reflection_value += (
                    ar_parameters[lag] * autocorr_coefficients[order - lag - 1]
                )
            reflection_coefficient = -reflection_value / prediction_error
        prediction_error *= 1.0 - reflection_coefficient**2.0
        if np.any(prediction_error <= 0):
            raise ValueError("Invalid prediction error: Singular Matrix")
        ar_parameters[order] = reflection_coefficient

        if order == 0:
            continue

        half_order = (order + 1) // 2
        for lag in range(half_order):
            reverse_lag = order - lag - 1
            save_value = ar_parameters[lag].copy()
            ar_parameters[lag] = (
                save_value
                + reflection_coefficient * ar_parameters[reverse_lag]
            )
            if lag != reverse_lag:
                ar_parameters[reverse_lag] += (
                    reflection_coefficient * save_value
                )

    # Add first coefficient a0
    ndim = ar_parameters.ndim
    pad_width = tuple([(1, 0)] + [(0, 0)] * (ndim - 1))
    return (
        np.pad(ar_parameters, pad_width, mode="constant", constant_values=1.0),
        prediction_error,
    )


def _yw_ar_estimation(
    time_data: NDArray[np.float64], order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the autoregressive coefficients for an AR process using the
    Levinson-Durbin recursion to solve the Yule-Walker equations. This is done
    from the biased autocorrelation.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Time data with up to three dimensions. The AR parameters are always
        computed along the first (outer) axis.
    order : int
        Recursion order.

    Returns
    -------
    NDArray[np.float64]
        Reflection coefficients with shape (coefficient, ...).
    NDArray[np.float64]
        Variance of the remaining error.

    """
    assert (
        time_data.ndim <= 3
    ), "This function only accepts a signal with one, two or three dimensions"

    length_td = time_data.shape[0]
    if time_data.ndim == 1:
        autocorrelation = (
            correlate(time_data, time_data, "full")[
                length_td - 1 : length_td + order
            ]
            / length_td
        )
    elif time_data.ndim == 2:
        autocorrelation = np.zeros((order + 1, time_data.shape[1]))
        for i in range(time_data.shape[1]):
            # Biased autocorrelation (only positive lags)
            autocorrelation[:, i] = (
                correlate(time_data[:, i], time_data[:, i], "full")[
                    length_td - 1 : length_td + order
                ]
                / length_td
            )
    else:
        autocorrelation = np.zeros(
            (order + 1, time_data.shape[1], time_data.shape[2])
        )
        for ii in range(time_data.shape[2]):
            for i in range(time_data.shape[1]):
                # Biased autocorrelation (only positive lags)
                autocorrelation[:, i, ii] = (
                    correlate(
                        time_data[:, i, ii], time_data[:, i, ii], "full"
                    )[length_td - 1 : length_td + order]
                    / length_td
                )

    return _levison_durbin_recursion(autocorrelation)


def _burg_ar_estimation(
    time_data: NDArray[np.float64], order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Burg's method to estimate the AR parameters. This is done always along
    the first axis. This implementation is taken from [2] and can take any
    shape of input vector.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Time data to estimate.
    order : int
        Order of the estimation.

    Returns
    -------
    NDArray[np.float64]
        Denominator (reflection) coefficients with shape (coefficient,
        channel).
    NDArray[np.float64]
        Variances of the prediction error.

    References
    ----------
    - [1]: Larry Marple. A New Autoregressive Spectrum Analysis Algorithm. IEEE
      Transactions on Acoustics, Speech, and Signal Processing vol 28, no. 4,
      1980.
    - [2]: McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt
      McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music
      signal analysis in python.” In Proceedings of the 14th python in science
      conference, pp. 18-25. 2015.

    """
    onedim = time_data.ndim == 1
    if onedim:
        time_data = time_data[:, None]
        shape = list(time_data.shape)
        ar_coeffs = np.zeros((order + 1, 1))
    else:
        shape = list(time_data.shape)
        shape[0] += 1
        ar_coeffs = np.zeros(tuple(shape))

    ar_coeffs[0] = 1.0
    ar_coeffs_prev = ar_coeffs.copy()

    shape[0] = 1
    reflect_coeff = np.zeros(shape)
    den = reflect_coeff.copy()

    epsilon = np.finfo(np.float64).eps

    fwd_pred_error = time_data[1:]
    bwd_pred_error = time_data[:-1]
    den[0] = np.sum(fwd_pred_error**2 + bwd_pred_error**2, axis=0)

    for i in range(order):
        reflect_coeff[0] = (
            -2.0 * np.sum(bwd_pred_error * fwd_pred_error, axis=0)
        ) / (den[0] + epsilon)
        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            ar_coeffs[j] = (
                ar_coeffs_prev[j]
                + reflect_coeff[0] * ar_coeffs_prev[i - j + 1]
            )

        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

        q = 1.0 - reflect_coeff[0] ** 2
        den[0] = q * den[0] - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2

        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]

    return ar_coeffs.squeeze() if onedim else ar_coeffs, den[0]
