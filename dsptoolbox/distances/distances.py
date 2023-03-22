"""
Distance measures between signals.
Beware that these distances have not been yet validated with other tools.
"""
import numpy as np
from scipy.signal import windows

from dsptoolbox import Signal
from dsptoolbox.filterbanks import auditory_filters_gammatone
from dsptoolbox._general_helpers import _find_nearest
from ._distances import (_log_spectral_distance,
                         _itakura_saito_measure,
                         _snr, _sisdr, _fw_snr_seg_per_channel)


def log_spectral(insig1: Signal, insig2: Signal, method: str = 'welch',
                 f_range_hz=[20, 20000], energy_normalization: bool = True,
                 spectrum_parameters: dict = None) -> np.ndarray:
    """Computes log spectral distance between two signals.

    Parameters
    ----------
    insig1 : Signal
        Signal 1.
    insig2 : Signal
        Signal 2.
    method : str, optional
        Method to compute the spectrum. Choose from `'welch'` or `'standard'`.
        Default: `'welch'`.
    f_range_hz : array-like with length 2, optional
        Range of frequencies in which to compute the distance. When `None`,
        it is computed in all frequencies. Default: [20, 20000].
    energy_normalization : bool, optional
        When `True`, the observed part of the spectrum is energy-normalized.
        Default: `True`.
    spectrum_parameters : dict, optional
        Additional parameters to be used in the computation of spectrum. Pass
        `None` to use default parameters in the
        `Signal.set_spectrum_parameters()` method. Default: `None`.

    Returns
    -------
    distances : `np.ndarray`
        Log spectral distance per channel for the given signals.

    References
    ----------
    - https://en.wikipedia.org/wiki/Log-spectral_distance

    """
    assert insig1.sampling_rate_hz == insig2.sampling_rate_hz,\
        'Sampling rates do not match'
    assert insig1.number_of_channels == insig2.number_of_channels,\
        'Signals have different channel numbers'
    if spectrum_parameters is None:
        spectrum_parameters = {}

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
    insig1.set_spectrum_parameters(method=method, **spectrum_parameters)
    insig2.set_spectrum_parameters(method=method, **spectrum_parameters)
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
        x = psd1[ids[0]:ids[1], n]
        y = psd2[ids[0]:ids[1], n]
        if energy_normalization:
            x /= np.sum(x)
            y /= np.sum(y)
        distances[n] = _log_spectral_distance(x, y, f)
    return distances


def itakura_saito(insig1: Signal, insig2: Signal, method: str = 'welch',
                  f_range_hz=[20, 20000], energy_normalization: bool = True,
                  spectrum_parameters: dict = None) -> np.ndarray:
    """Computes itakura-saito measure between two signals. Beware that this
    measure is not symmetric (x, y) != (y, x).

    Parameters
    ----------
    insig1 : Signal
        Signal 1.
    insig2 : Signal
        Signal 2.
    method : str, optional
        Method to compute the spectrum. Choose from `'welch'` or `'standard'`.
        Default: `'welch'`.
    f_range_hz : array-like with length 2, optional
        Range of frequencies in which to compute the distance. When `None`,
        it is computed in all frequencies. Default: [20, 20000].
    energy_normalization : bool, optional
        When `True`, the observed part of the spectrum is energy-normalized.
        Default: `True`.
    spectrum_parameters : dict, optional
        Additional parameters to be used in the computation of spectrum. Pass
        `None` to use default parameters in the
        `Signal.set_spectrum_parameters()` method. Default: `None`.

    Returns
    -------
    distances : `np.ndarray`
        Itakura-saito measure for the given signals.

    References
    ----------
    - https://en.wikipedia.org/wiki/Itakura–Saito_distance

    """
    assert insig1.sampling_rate_hz == insig2.sampling_rate_hz,\
        'Sampling rates do not match'
    assert insig1.number_of_channels == insig2.number_of_channels,\
        'Signals have different channel numbers'
    if spectrum_parameters is None:
        spectrum_parameters = {}

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
    insig1.set_spectrum_parameters(method=method, **spectrum_parameters)
    insig2.set_spectrum_parameters(method=method, **spectrum_parameters)
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
        x = psd1[ids[0]:ids[1], n]
        y = psd2[ids[0]:ids[1], n]
        if energy_normalization:
            x /= np.sum(x)
            y /= np.sum(y)
        distances[n] = _itakura_saito_measure(x, y, f)
    return distances


def snr(signal: Signal, noise: Signal) -> np.ndarray:
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

    References
    ----------
    - https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    """
    assert signal.sampling_rate_hz == noise.sampling_rate_hz,\
        'Sampling rates do not match'
    if noise.number_of_channels != 1:
        assert signal.number_of_channels == noise.number_of_channels,\
            'Signals have different channel numbers'
    return _snr(signal.time_data, noise.time_data)


def si_sdr(target_signal: Signal, modified_signal: Signal) -> np.ndarray:
    """Computes scale-invariant signal to distortion ratio from a target
    and a modified signal. If target signal only has one channel, it is
    assumed to be the target for all the channels in the modified signal.
    See reference for details.

    Parameters
    ----------
    target_signal : `Signal`
        Original signal. If it only has one channel and the modified signal
        has multiple, it is assumed to be the target signal for all channels.
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
        sdr[n] = _sisdr(target_signal.time_data[:, n_1],
                        modified_signal.time_data[:, n])
    return sdr


def fw_snr_seg(x: Signal, xhat: Signal, f_range_hz=[20, 10e3],
               snr_range_db=[-10, 35], gamma: float = 0.2) -> np.ndarray:
    """Frequency-weighted segmental SNR (fwSNRseg) computation between two
    signals.

    This distance measure divides the signal into auditory frequency
    bands (using the auditory gammatone filters) and splits the signal in time
    frames to further compute SNR. This distance was shown to correlate
    relatively well with results from listening tests and other objective
    measures. See references for more information.

    NOTE: the time window is fixed to be a 75 ms Hamming window with 50%
    overlap instead as gaussian window (as in the paper) since no length
    and beta parameter were specified in the publication.

    Parameters
    ----------
    x : `Signal`
        Original clean signal. If this signal only contains one channel and
        `xhat` more, it is assumed that this channel is the original of all the
        others.
    xhat : `Signal`
        Enhanced/modified signal.
    f_range_hz : array-like with length of 2, optional
        Frequency range in which to analyze the signals. Default: [20, 10e3].
    snr_range_db : array-like with length of 2, optional
        SNR range to be regarded. If any frame throws a value outside this
        range, it is set to the boundary. Default: [-10, 35].
    gamma : float, optional
        Gamma parameter to be used for the frame weightning. See paper for
        more information about it. Its recommended range is (according to
        reference) constrained to [0.1, 2]. Default: 0.2.

    Returns
    -------
    snr_per_channel : `np.ndarray`
        Frequency-weighted, time-segmented SNR per channel.

    References
    ----------
    - Y. Hu and P. C. Loizou, "Evaluation of Objective Quality Measures for
      Speech Enhancement," in IEEE Transactions on Audio, Speech, and Language
      Processing, vol. 16, no. 1, pp. 229-238, Jan. 2008,
      doi: 10.1109/TASL.2007.911054.
    - https://ieeexplore.ieee.org/document/4389058

    """
    # Sampling rates
    assert x.sampling_rate_hz == xhat.sampling_rate_hz, \
        'Sampling rates do not match'
    fs_hz = x.sampling_rate_hz
    # Lengths
    assert x.time_data.shape[0] == xhat.time_data.shape[0], \
        'Signal lengths do not match'
    # Number of channels
    multichannel = False
    if x.number_of_channels != xhat.number_of_channels:
        assert x.number_of_channels == 1, \
            'Invalid number of channels for this measurement'
        multichannel = True
    # Frequency range
    assert len(f_range_hz) == 2, \
        'Frequency range must have lower and upper bounds'
    f_range = np.asarray(f_range_hz)
    f_range.sort()
    assert f_range[1] < fs_hz//2, \
        f'Upper frequency range {f_range[1]} must be smaller than nyquist ' +\
        f'frequency {fs_hz//2}'
    assert f_range[0] > 0, \
        'Frequency range must be positive'
    # SNR range
    assert len(snr_range_db) == 2, \
        'SNR range must have lower and upper bounds'
    snr_range_db = np.asarray(snr_range_db)
    snr_range_db.sort()
    # Time window
    length_samp = int(75e-3*fs_hz)
    if length_samp % 2 == 1:
        length_samp += 1
    window = windows.hamming(length_samp, sym=False)
    step = len(window)//2  # 50% overlap
    # Gamma
    assert gamma >= 0.1 and gamma <= 2, \
        f'{gamma} is not in the valid range for gamma [0.1, 5]'
    # Generate filter bank
    aud_fb = auditory_filters_gammatone(
        frequency_range_hz=f_range, resolution=1, sampling_rate_hz=fs_hz)
    x = aud_fb.filter_signal(x, mode='parallel')
    xhat = aud_fb.filter_signal(xhat, mode='parallel')
    # SNR time-segmented with weighting function
    snr_per_channel = np.empty(xhat.number_of_channels)
    for n in range(xhat.number_of_channels):
        xhat_ = xhat.get_all_bands(n).time_data
        if multichannel:
            n_original = 0
        else:
            n_original = n
        x_ = x.get_all_bands(n_original).time_data
        snr_per_channel[n] = _fw_snr_seg_per_channel(
            x_, xhat_, snr_range_db, gamma,
            time_window=window, step_samples=step)
    return snr_per_channel
