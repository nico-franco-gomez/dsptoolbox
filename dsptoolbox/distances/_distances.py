"""
Backend for distance measures
"""

import numpy as np
from scipy.integrate import simpson
from numpy.typing import NDArray

from ..helpers.gain_and_level import _rms
from ..helpers.other import _pad_trim, _compute_number_frames


def _log_spectral_distance(
    x: NDArray[np.float64], y: NDArray[np.float64], f: NDArray[np.float64]
) -> float:
    """Computes log spectral distance between two signals.

    Parameters
    ----------
    x : NDArray[np.float64]
        First power spectrum.
    y : NDArray[np.float64]
        Second power spectrum.
    f : NDArray[np.float64]
        Frequency vector.

    Returns
    -------
    log_spec : float
        Log spectral distance.

    """
    assert x.shape == y.shape, "Power spectra have different lengths"
    integral = simpson((10 * np.log10(x / y)) ** 2, x=f)
    log_spec = np.sqrt(integral)
    return log_spec


def _itakura_saito_measure(
    x: NDArray[np.float64], y: NDArray[np.float64], f: NDArray[np.float64]
) -> float:
    """Computes log spectral distance between two signals.

    Parameters
    ----------
    x : NDArray[np.float64]
        First power spectrum.
    y : NDArray[np.float64]
        Second power spectrum.
    f : NDArray[np.float64]
        Frequency vector.

    Returns
    -------
    ism : float
        Itakura Saito measure.

    """
    assert x.shape == y.shape, "Power spectra have different lengths"
    ism = simpson(x / y - np.log10(x / y) - 1, x=f)
    return ism


def _snr(
    s: NDArray[np.float64], n: NDArray[np.float64]
) -> float | NDArray[np.float64]:
    """Computes SNR from the passed numpy arrays.

    Parameters
    ----------
    s : NDArray[np.float64]
        Signal
    n : NDArray[np.float64]
        Noise

    Returns
    -------
    snr : float or NDArray[np.float64]
        SNR between signals. It can be an array if signals are multichannel.

    """
    return 20 * np.log10(_rms(s) / _rms(n))


def _sisdr(s: NDArray[np.float64], shat: NDArray[np.float64]) -> float:
    """Scale-invariant signal-to-distortion ratio

    Parameters
    ----------
    s : NDArray[np.float64]
        Target signal.
    shat : NDArray[np.float64]
        Modified or approximated signal.

    Returns
    -------
    sisdr : float
        SI-SDR value between two signals.

    """
    alpha = (s @ shat) / (s @ s)
    sisdr = 10 * np.log10(
        np.sum((alpha * s) ** 2) / np.sum((alpha * s - shat) ** 2)
    )
    return sisdr


def _fw_snr_seg_per_channel(
    x: NDArray[np.float64],
    xhat: NDArray[np.float64],
    snr_range_db: NDArray[np.float64],
    gamma: float,
    time_window: NDArray[np.float64],
    step_samples: int,
) -> float:
    """This function gets an original signal x and a modified signal xhat
    and computes the frequency-weighted segmental SNR according to
    Y. Hu and P. C. Loizou,
    "Evaluation of Objective Quality Measures for Speech Enhancement". See
    references.

    Parameters
    ----------
    x : NDArray[np.float64]
        Original signal with shape (time_samples, bands).
    xhat : NDArray[np.float64]
        Modified signal with shape (time_samples, bands).
    snr_range_db : NDArray[np.float64] with length 2
        SNR range in dB.
    gamma : float
        Gamma exponent for the weighting function. See reference for details.
    time_window : NDArray[np.float64]
        Time window to be used.
    step : int
        Hop length between each time frame.

    Returns
    -------
    snr : float
        SNR value.

    References
    ----------
    - Y. Hu and P. C. Loizou, "Evaluation of Objective Quality Measures for
      Speech Enhancement," in IEEE Transactions on Audio, Speech, and Language
      Processing, vol. 16, no. 1, pp. 229-238, Jan. 2008,
      doi: 10.1109/TASL.2007.911054.
    - https://ieeexplore.ieee.org/document/4389058

    """
    eps = 1e-30  # Some small number for the logarithm function
    length_signal = len(x)
    length_window = len(time_window)
    n_frames, pad_samples = _compute_number_frames(
        length_window, step_samples, length_signal, True
    )
    x = _pad_trim(x, length_signal + pad_samples)
    xhat = _pad_trim(xhat, length_signal + pad_samples)

    fw_snr_seg = 0
    position = 0
    # Loop for time frames
    for _ in range(n_frames):
        # Time signals
        x_m = x[position : position + length_window, :]
        xhat_m = xhat[position : position + length_window, :]
        # "Collectors"
        weights_jm = np.zeros(length_window // 2 + 1)
        snr_jm = np.zeros_like(weights_jm)
        # Loop for bands
        for ib in range(x.shape[1]):
            X_jm = np.abs(np.fft.rfft(x_m[:, ib] * time_window))
            Xhat_jm = np.abs(np.fft.rfft(xhat_m[:, ib] * time_window))
            # Weightning function, gamma parameter can range between 0.1 and 2
            W_jm = X_jm**gamma

            # Normalization of spectra: probably for avoiding scaling
            # inconsistencies when total energy in the signals is not the same
            X_jm /= np.sum(X_jm)
            Xhat_jm /= np.sum(Xhat_jm)

            snr_jm += np.log10(X_jm**2 / (X_jm - Xhat_jm + eps) ** 2) * W_jm
            weights_jm += W_jm

        # Mean SNR over all frequencies
        snr_frame = np.mean(10 * snr_jm / weights_jm)

        # Range dB
        if snr_frame < snr_range_db[0]:
            snr_frame = snr_range_db[0]
        if snr_frame > snr_range_db[1]:
            snr_frame = snr_range_db[1]
        fw_snr_seg += snr_frame

        # Updating position
        position += step_samples

    fw_snr_seg /= n_frames
    return fw_snr_seg
