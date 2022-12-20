"""
General use filter banks to be created and given back as a filter bank
object
"""
# import numpy as np
# from .signal_class import Signal, MultiBandSignal
# from .filter_class import Filter, FilterBank
from ._filterbank import LRFilterBank


def linkwitz_riley_crossovers(freqs, order, sampling_rate_hz: int = 48000):
    """Returns a linkwitz-riley crossovers filter bank.

    Parameters
    ----------
    freqs : array-like
        Frequencies at which to set the crossovers.
    order : array-like
        Order of the crossovers. The higher, the steeper.
    sampling_rate_hz : int, optional
        Sampling rate for the filterbank. Default: 48000.

    Returns
    -------
    fb : LRFilterBank
        Filter bank in form of LRFilterBank class which contains the same
        methods as the FilterBank class but is generated with different
        internal methods.
    """
    return LRFilterBank(freqs, order, sampling_rate_hz)
