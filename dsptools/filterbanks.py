'''
General use filter banks to be created and given back as a filter bank
object
'''
# import numpy as np
# from .signal_class import Signal, MultiBandSignal
# from .filter_class import Filter, FilterBank
from .backend._filterbank import LRFilterBank


def linkwitz_riley_crossovers(freqs, order, sampling_rate_hz):
    '''
    Returns a linkwitz-riley crossovers filter bank
    '''
    return LRFilterBank(freqs, order, sampling_rate_hz)
