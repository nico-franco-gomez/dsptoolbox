import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate, lfilter
from scipy.linalg import convolution_matrix, lstsq, solve, toeplitz
from enum import Enum, auto


class ArmaMethod(Enum):
    """Method to use for computing estimating the ARMA parameters.

    `YuleWalker` and `Burg` deliver AR parameters, while the MA parameters are
    subsequently fitted using a least-squares approximation.

    `Prony` and `SteiglitzMcBride` deliver directly both AR and MA parameters.
    `SteiglitzMcBride` utilizes `Prony` as initial estimate for the AR parameters.

    """

    YuleWalker = auto()
    Burg = auto()
    Prony = auto()
    SteiglitzMcBride = auto()


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
                save_value + reflection_coefficient * ar_parameters[reverse_lag]
            )
            if lag != reverse_lag:
                ar_parameters[reverse_lag] += reflection_coefficient * save_value

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
            correlate(time_data, time_data, "full")[length_td - 1 : length_td + order]
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
        autocorrelation = np.zeros((order + 1, time_data.shape[1], time_data.shape[2]))
        for ii in range(time_data.shape[2]):
            for i in range(time_data.shape[1]):
                # Biased autocorrelation (only positive lags)
                autocorrelation[:, i, ii] = (
                    correlate(time_data[:, i, ii], time_data[:, i, ii], "full")[
                        length_td - 1 : length_td + order
                    ]
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
        reflect_coeff[0] = (-2.0 * np.sum(bwd_pred_error * fwd_pred_error, axis=0)) / (
            den[0] + epsilon
        )
        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            ar_coeffs[j] = (
                ar_coeffs_prev[j] + reflect_coeff[0] * ar_coeffs_prev[i - j + 1]
            )

        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

        q = 1.0 - reflect_coeff[0] ** 2
        den[0] = q * den[0] - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2

        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]

    return ar_coeffs.squeeze() if onedim else ar_coeffs, den[0]


def _prony(
    h: NDArray[np.float64], order_b: int, order_a: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate ARMA model coefficients using Prony's method.

    Fits a system B(z)/A(z) to match the first samples of the impulse response `h`,
    with `order_b` zeros and `order_a` poles.

    Parameters
    ----------
    h : NDArray[np.float64]
        Impulse response to be modelled.
    order_b : int
        Number of zeros (numerator order).
    order_a : int
        Number of poles (denominator order).

    Returns
    -------
    b : NDArray[np.float64]
        Numerator (MA) coefficients of length `order_b + 1`.
    a : NDArray[np.float64]
        Denominator (AR) coefficients of length `order_a + 1`, with leading 1.

    """
    h = np.array(h, dtype=np.float64)
    n_samples = len(h) - 1

    # Ensure we have enough samples for the requested model order
    if n_samples <= max(order_b, order_a):
        n_samples = max(order_b, order_a) + 1
        h = np.concatenate([h, np.zeros(n_samples + 1 - len(h))])

    # Normalize impulse response by the first sample
    scale = h[0] if h[0] != 0.0 else 1.0

    # Build Toeplitz matrix of the normalized impulse response
    H = toeplitz(h / scale, np.hstack((1.0, np.zeros(n_samples))))

    # Trim columns to denominator order + 1
    if n_samples > order_a:
        H = H[:, : order_a + 1]

    # Split into the top block (for MA recovery) and the bottom block
    # (for AR estimation via least squares)
    H_top = H[: order_b + 1, :]
    h_rhs = H[order_b + 1 : n_samples + 1, 0]
    H_bottom = H[order_b:n_samples, :order_a]

    # Solve overdetermined system for AR coefficients (skip leading 1)
    a = np.concatenate(([1.0], lstsq(-H_bottom, h_rhs, cond=None)[0]))

    # Recover MA coefficients from the top block
    b = scale * (a @ H_top.T)

    return b, a


def _steiglitz_mcbride(
    h: NDArray[np.float64], order_b: int, order_a: int, n_iterations: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute linear model via Steiglitz-McBride iteration.

    Finds coefficients of the system B(z)/A(z) with approximate impulse response `h`,
    `order_a` poles and `order_b` zeros.

    Parameters
    ----------
    h : NDArray[np.float64]
        Impulse response.
    order_b : int
        Number of zeros (numerator order).
    order_a : int
        Number of poles (denominator order).
    n_iterations : int
        Number of iterations.

    Returns
    -------
    b : NDArray[np.float64]
        Numerator coefficients (length `order_b + 1`).
    a : NDArray[np.float64]
        Denominator coefficients (length `order_a + 1`, with leading 1).

    """
    N = len(h)

    # Initialize denominator coefficients via Prony
    _, a = _prony(h, 0, order_a)

    # Unit impulse used as the input signal for the all-pole inverse filter
    impulse: NDArray[np.float64] = np.zeros(N)
    impulse[0] = 1.0

    for _ in range(n_iterations):
        # Filter the impulse response and the unit impulse through 1/A(z)
        u = lfilter([1.0], a, h)
        v = lfilter([1.0], a, impulse)

        # Build convolution matrices (truncated to N rows)
        C1 = convolution_matrix(u, order_a + 1, mode="full")[:N, :]
        C2 = convolution_matrix(v, order_b + 1, mode="full")[:N, :]

        # Assemble the system:  [-C1[:,1:] | C2] @ c = C1[:,0]
        # where c = [a_1..a_na, b_0..b_nb]
        T = np.hstack((-C1[:, 1:], C2))
        rhs = C1[:, 0]

        # Use direct solve for square systems, least-squares otherwise
        if T.shape[0] == T.shape[1]:
            c = solve(T, rhs)
        else:
            c = lstsq(T, rhs)[0]

        # Extract updated AR and MA coefficients from solution vector
        a = np.concatenate(([1.0], c[:order_a]))
        b = c[order_a : order_a + order_b + 1]

    return b, a
