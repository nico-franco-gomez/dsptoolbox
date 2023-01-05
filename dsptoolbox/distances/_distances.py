"""
Backend for distance measures
"""
import numpy as np
from scipy.integrate import simpson


def _log_spectral_distance(x: np.ndarray, y: np.ndarray, f):
    """Computes log spectral distance between two signals.

    Parameters
    ----------
    x : `np.ndarray`
        First power spectrum.
    y : `np.ndarray`
        Second power spectrum.
    f : `np.ndarray`
        Frequency vector.

    Returns
    -------
    log_spec : float
        Log spectral distance.

    """
    assert x.shape == y.shape, \
        'Power spectra have different lengths'
    integral = simpson((10*np.log10(x/y))**2, f)
    log_spec = np.sqrt(integral)
    return log_spec


def _itakura_saito_measure(x: np.ndarray, y: np.ndarray, f):
    """Computes log spectral distance between two signals.

    Parameters
    ----------
    x : `np.ndarray`
        First power spectrum.
    y : `np.ndarray`
        Second power spectrum.
    f : `np.ndarray`
        Frequency vector.

    Returns
    -------
    ism : float
        Itakura Saito measure.

    """
    assert x.shape == y.shape, \
        'Power spectra have different lengths'
    ism = simpson(x/y - np.log10(x/y) - 1, f)
    return ism


def _snr(s: np.ndarray, n: np.ndarray):
    """Computes SNR from the passed numpy arrays.

    Parameters
    ----------
    s : `np.ndarray`
        Signal
    n : `np.ndarray`
        Noise

    Returns
    -------
    snr : float
        SNR between signals.

    """
    return 20*np.log10(rms(s)/rms(n))


def rms(x: np.ndarray):
    """Root mean squared value of a discrete time series

    Parameters
    ----------
    x : `np.ndarray`
        Time series

    Returns
    -------
    rms : float
        Root mean squared value

    """
    return np.sqrt(np.sum(x**2)/len(x))


def _sisdr(s: np.ndarray, shat: np.ndarray):
    """Scale-invariant signal-to-distortion ratio

    Parameters
    ----------
    s : `np.ndarray`
        Target signal.
    shat : `np.ndarray`
        Modified or approximated signal.

    Returns
    -------
    sisdr : float
        SI-SDR value between two signals.

    """
    alpha = (s @ shat)/(s @ s)
    sisdr = 10*np.log10(np.sum((alpha*s)**2) / np.sum((alpha*s - shat)**2))
    return sisdr
