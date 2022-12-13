'''
Distance measures between signals
'''
import numpy as np
from .signal_class import Signal
from .backend._general_helpers import _find_nearest
from .backend._distances import (_log_spectral_distance,
                                 _itakura_saito_measure)


__all__ = ['log_spectral', 'itakura_saito']


def log_spectral(insig1: Signal, insig2: Signal, method: str = 'welch',
                 f_range_hz=[20, 20000], **kwargs):
    '''
    Computes log spectral distance between two signals.

    Parameters
    ----------
    insig1 : Signal
        Signal 1
    insig2 : Signal
        Signal 2
    f_range_hz : array, optional
        Range of frequencies in which to compute the distance. When `None`,
        it is computed in all frequencies. Default: [20, 20000]
    '''
    assert insig1.sampling_rate_hz == insig2.sampling_rate_hz,\
        'Sampling rates do not match'
    assert insig1.number_of_channels == insig2.number_of_channels,\
        'Signals have different channel numbers'

    fs_hz = insig1.sampling_rate_hz
    if f_range_hz is None:
        f_range_hz = [0, fs_hz//2]
    else:
        assert len(f_range_hz) == 2, 'f_range_hz must only have a lower' +\
            ' and an upper limit'
        f_range_hz = np.sort(f_range_hz)
        assert f_range_hz[1] <= fs_hz//2, 'Upper bound for ' +\
            'frequency must be smaller than the nyquist frequency'
        assert not any(f_range_hz < 0), 'Frequencies in range must be ' +\
            'positive'
    # spec1 = np.abs(insig1.get_spectrum()['spectrum'])**2
    # spec2 = np.abs(insig2.get_spectrum()['spectrum'])**2
    insig1.set_spectrum_parameters(method=method, **kwargs)
    insig2.set_spectrum_parameters(method=method, **kwargs)
    f, spec1 = insig1.get_spectrum()
    f, spec2 = insig2.get_spectrum()

    psd1 = np.abs(spec1)
    psd2 = np.abs(spec2)
    if method == 'standard':
        psd1 = psd1**2
        psd2 = psd2**2

    ids = _find_nearest(f_range_hz, f)
    f = f[ids[0]:ids[1]]

    distances = np.zeros(insig1.number_of_channels)
    for n in range(insig1.number_of_channels):
        distances[n] = \
            _log_spectral_distance(
                psd1[ids[0]:ids[1], n], psd2[ids[0]:ids[1], n], f)
    return distances


def itakura_saito(insig1: Signal, insig2: Signal, method: str = 'welch',
                  f_range_hz=[20, 20000], **kwargs):
    '''
    Computes log spectral distance between two signals.

    Parameters
    ----------
    insig1 : Signal
        Signal 1
    insig2 : Signal
        Signal 2
    f_range_hz : array, optional
        Range of frequencies in which to compute the distance. When `None`,
        it is computed in all frequencies. Default: [20, 20000]
    '''
    assert insig1.sampling_rate_hz == insig2.sampling_rate_hz,\
        'Sampling rates do not match'
    assert insig1.number_of_channels == insig2.number_of_channels,\
        'Signals have different channel numbers'

    fs_hz = insig1.sampling_rate_hz
    if f_range_hz is None:
        f_range_hz = [0, fs_hz//2]
    else:
        assert len(f_range_hz) == 2, 'f_range_hz must only have a lower' +\
            ' and an upper limit'
        f_range_hz = np.sort(f_range_hz)
        assert f_range_hz[1] <= fs_hz//2, 'Upper bound for ' +\
            'frequency must be smaller than the nyquist frequency'
        assert not any(f_range_hz < 0), 'Frequencies in range must be ' +\
            'positive'
    insig1.set_spectrum_parameters(method=method, **kwargs)
    insig2.set_spectrum_parameters(method=method, **kwargs)
    f, spec1 = insig1.get_spectrum()
    f, spec2 = insig2.get_spectrum()

    psd1 = np.abs(spec1)
    psd2 = np.abs(spec2)
    if method == 'standard':
        psd1 = psd1**2
        psd2 = psd2**2

    ids = _find_nearest(f_range_hz, f)
    f = f[ids[0]:ids[1]]

    distances = np.zeros(insig1.number_of_channels)
    for n in range(insig1.number_of_channels):
        distances[n] = \
            _itakura_saito_measure(
                psd1[ids[0]:ids[1], n], psd2[ids[0]:ids[1], n], f)
    return distances
