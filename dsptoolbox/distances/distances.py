"""
Distance measures between signals
"""
import numpy as np
from dsptoolbox import Signal
from dsptoolbox._general_helpers import _find_nearest
from ._distances import (_log_spectral_distance,
                         _itakura_saito_measure,
                         _snr, _sisdr)


__all__ = ['log_spectral', 'itakura_saito']


def log_spectral(insig1: Signal, insig2: Signal, method: str = 'welch',
                 f_range_hz=[20, 20000], **kwargs):
    """Computes log spectral distance between two signals.

    Parameters
    ----------
    insig1 : Signal
        Signal 1.
    insig2 : Signal
        Signal 2.
    f_range_hz : array, optional
        Range of frequencies in which to compute the distance. When `None`,
        it is computed in all frequencies. Default: [20, 20000].

    """
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
            _log_spectral_distance(
                psd1[ids[0]:ids[1], n], psd2[ids[0]:ids[1], n], f)
    return distances


def itakura_saito(insig1: Signal, insig2: Signal, method: str = 'welch',
                  f_range_hz=[20, 20000], **kwargs):
    """Computes itakura-saito measure between two signals. Beware that this
    measure is not symmetric (x, y) != (y, x).

    Parameters
    ----------
    insig1 : Signal
        Signal 1.
    insig2 : Signal
        Signal 2.
    f_range_hz : array, optional
        Range of frequencies in which to compute the distance. When `None`,
        it is computed in all frequencies. Default: [20, 20000].

    """
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


def snr(signal: Signal, noise: Signal):
    """Classical Signal-to-noise ratio. If noise only has one channel,
    it is assumed to be the noise for all channels of signal.

    Parameters
    ----------
    signal : `Signal`
        Signal.
    noise : `Signal`
        Noise.

    Returns
    -------
    snr_per_channel : `np.ndarray`
        SNR value per channel

    """
    assert signal.sampling_rate_hz == noise.sampling_rate_hz,\
        'Sampling rates do not match'
    if noise.number_of_channels != 1:
        assert signal.number_of_channels == noise.number_of_channels,\
            'Signals have different channel numbers'
        multichannel = False
    else:
        multichannel = True

    snr_per_channel = np.empty(signal.number_of_channels)
    for n in range(signal.number_of_channels):
        if multichannel:
            n_noise = 0
        else:
            n_noise = n
        snr_per_channel[n] = _snr(
            signal.time_data[:, n], noise.time_data[:, n_noise])
    return snr_per_channel


def si_sdr(target_signal: Signal, modified_signal: Signal):
    """Computes scale-invariant signal to distortion ratio from an original
    and a modified signal. If target signal only has one channel, it is
    assumed to be the target one for all the channels in the modified signal.
    See reference for details.

    Parameters
    ----------
    tartget_signal : `Signal`
        Original signal.
    modified_signal : `Signal`
        Signal after modification/enhancement.

    Returns
    -------
    sdr : `np.ndarray`
        SI-SDR per channel.

    References
    ----------
    - https://arxiv.org/abs/1811.02508

    """
    assert modified_signal.sampling_rate_hz == target_signal.sampling_rate_hz,\
        'Sampling rates do not match'
    if target_signal.number_of_channels != 1:
        assert modified_signal.number_of_channels == \
            target_signal.number_of_channels, \
                'Signals have different channel numbers'
        multichannel = False
    else:
        multichannel = True
    assert modified_signal.time_data.shape[0] == \
        target_signal.time_data.shape[0], 'Length of signals do not match'
    
    sdr = np.empty(modified_signal.number_of_channels)
    for n in range(modified_signal.number_of_channels):
        if multichannel:
            n_1 = 0
        else:
            n_1 = n
        sdr[n] = _sisdr(
            target_signal.time_data[:, n_1], modified_signal.time_data[:, n])
    return sdr
