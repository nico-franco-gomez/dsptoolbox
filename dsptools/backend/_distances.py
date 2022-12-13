'''
Backend for distance measures
'''
import numpy as np
from scipy.integrate import simpson


def _log_spectral_distance(x: np.ndarray, y: np.ndarray, f):
    '''
    Computes log spectral distance between two signals.

    Parameters
    ----------
    x : np.ndarray
        First power spectrum
    y : np.ndarray
        Second power spectrum
    f : np.ndarray
        Frequency vector

    Returns
    -------
    log_spec : float
        Log spectral distance
    '''
    assert x.shape == y.shape, \
        'Power spectra have different lengths'
    integral = simpson((10*np.log10(x/y))**2, f)
    log_spec = np.sqrt(integral)
    return log_spec


def _itakura_saito_measure(x: np.ndarray, y: np.ndarray, f):
    '''
    Computes log spectral distance between two signals.

    Parameters
    ----------
    x : np.ndarray
        First power spectrum
    y : np.ndarray
        Second power spectrum
    f : np.ndarray
        Frequency vector

    Returns
    -------
    ism : float
        Itakura Saito measure
    '''
    assert x.shape == y.shape, \
        'Power spectra have different lengths'
    ism = simpson(x/y - np.log10(x/y) - 1, f)
    return ism
