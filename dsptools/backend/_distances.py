'''
Backend for distance measures
'''
from numpy import (ndarray, sqrt, log10, )
from scipy.integrate import simpson


def _log_spectral_distance(x: ndarray, y: ndarray, f):
    '''
    Computes log spectral distance between two signals.

    Parameters
    ----------
    x : ndarray
        First power spectrum
    y : ndarray
        Second power spectrum
    f : ndarray
        Frequency vector

    Returns
    -------
    log_spec : float
        Log spectral distance
    '''
    assert x.shape == y.shape, \
        'Power spectra have different lengths'
    integral = simpson((10*log10(x/y))**2, f)
    log_spec = sqrt(integral)
    return log_spec


def _itakura_saito_measure(x: ndarray, y: ndarray, f):
    '''
    Computes log spectral distance between two signals.

    Parameters
    ----------
    x : ndarray
        First power spectrum
    y : ndarray
        Second power spectrum
    f : ndarray
        Frequency vector

    Returns
    -------
    ism : float
        Itakura Saito measure
    '''
    assert x.shape == y.shape, \
        'Power spectra have different lengths'
    ism = simpson(x/y - log10(x/y) - 1, f)
    return ism
