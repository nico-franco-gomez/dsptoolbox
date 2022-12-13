'''
Functions for filtering Signal objects with filt objects
'''
# import numpy as np
import scipy.signal as sig
from .signal_class import Signal
from .filter_class import Filter


def filter_on_signal(signal: Signal, filt: Filter, channel: int = None,
                     zi: bool = False):
    '''
    Takes in a Signal object and filters selected channels. Exports a new
    Signal object

    Parameters
    ----------
    signal : Signal
        Signal to be filtered.
    filt : Filter
        Filter to be used on the signal.
    channel : int or array-like, optional
        Channel or array of channels to be filtered. When `None`, all
        channels are filtered. Default: `None`.
    zi : bool, optional
        When `True`, the filter state values are updated after filtering.
        Default: `False`.

    Returns
    -------
    new_signal : Signal
        New Signal object.
    '''
    new_time_data = signal.time_data.copy()
    if channel is None:
        channels = range(signal.number_of_channels)
    else:
        channels = [int(i) for i in channel]
    for ch in channels:
        if hasattr(filt, 'sos'):
            if zi:
                y, filt.zi = \
                    sig.sosfilt(
                        filt.sos, signal.time_data[:, ch], zi=filt.zi)
            else:
                y = sig.sosfilt(filt.sos, signal.time_data[:, ch])
        elif hasattr(filt, 'ba'):
            if zi:
                y, filt.zi = \
                    sig.lfilter(
                        b=filt.ba[0], a=filt.ba[1],
                        x=signal.time_data[:, ch], zi=filt.zi)
            else:
                y = sig.lfilter(
                        b=filt.ba[0], a=filt.ba[1],
                        x=signal.time_data[:, ch])
        new_time_data[:, ch] = y

    new_signal = Signal(None, new_time_data, signal.sampling_rate_hz)
    return new_signal
