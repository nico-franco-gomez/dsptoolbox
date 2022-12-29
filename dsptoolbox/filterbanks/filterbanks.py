"""
General use filter banks to be created and given back as a filter bank
object
"""
import numpy as np
from scipy.signal import windows
import warnings
from dsptoolbox import Filter, FilterBank
from ._filterbank import LRFilterBank, fractional_octave_frequencies


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


def reconstructing_fractional_octave_bands(
        num_fractions: int = 1, frequency_range=[63, 16000],
        overlap: float = 1, slope: int = 0, n_samples: int = 2**12,
        sampling_rate_hz: int = 48000):
    """Create and/or apply an amplitude preserving fractional octave filter
    bank. This implementation is taken from the pyfar package.
    See references for more information about it.

    Parameters
    ----------
    num_fractions : int, optional
        Octave fraction, e.g., ``3`` for third-octave bands. The default is
        ``1``.
    frequency_range : tuple, optional
        Frequency range for fractional octave in Hz. The default is
        ``(63, 16000)``
    overlap : float
        Band overlap of the filter slopes between 0 and 1. Smaller values yield
        wider pass-bands and steeper filter slopes. The default is ``1``.
    slope : int, optional
        Number > 0 that defines the width and steepness of the filter slopes.
        Larger values yield wider pass-bands and steeper filter slopes. The
        default is ``0``.
    n_samples : int, optional
        Length of the filter in samples. Longer filters yield more exact
        filters. The default is ``2**12``.
    sampling_rate : int
        Sampling frequency in Hz. The default is ``None``. Only required if
        ``signal=None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterFIR
        FIR Filter object. Only returned if ``signal = None``.
    frequencies : np.ndarray
        Center frequencies of the filters.

    References
    ----------
    - https://pubmed.ncbi.nlm.nih.gov/20136211/
    - https://github.com/pyfar/pyfar

    """
    valid_lengths = 2**(np.arange(5, 18))
    assert n_samples in valid_lengths, \
        'Only lengths between 2**5 and 2**17 are allowed'

    if overlap < 0 or overlap > 1:
        raise ValueError("overlap must be between 0 and 1")

    if not isinstance(slope, int) or slope < 0:
        raise ValueError("slope must be a positive integer.")

    # number of frequency bins
    n_bins = int(n_samples // 2 + 1)

    # fractional octave frequencies
    _, f_m, f_cut_off = fractional_octave_frequencies(
        num_fractions, frequency_range, return_cutoff=True)

    # discard fractional octaves, if the center frequency exceeds
    # half the sampling rate
    f_id = f_m < sampling_rate_hz / 2
    if not np.all(f_id):
        warnings.warn("Skipping bands above the Nyquist frequency")

    # DFT lines of the lower cut-off and center frequency as in
    # Antoni, Eq. (14)
    k_1 = \
        np.round(n_samples * f_cut_off[0][f_id] / sampling_rate_hz).astype(int)
    k_m = \
        np.round(n_samples * f_m[f_id] / sampling_rate_hz).astype(int)
    k_2 = \
        np.round(n_samples * f_cut_off[1][f_id] / sampling_rate_hz).astype(int)

    # overlap in samples (symmetrical around the cut-off frequencies)
    P = np.round(overlap / 2 * (k_2 - k_m)).astype(int)
    # initialize array for magnitude values
    g = np.ones((len(k_m), n_bins))

    # calculate the magnitude responses
    # (start at 1 to make the first fractional octave band as the low-pass)
    for b_idx in range(1, len(k_m)):

        if P[b_idx] > 0:
            # calculate phi_l for Antoni, Eq. (19)
            p = np.arange(-P[b_idx], P[b_idx] + 1)
            # initialize phi_l in the range [-1, 1]
            # (Antoni suggest to initialize this in the range of [0, 1] but
            # that yields wrong results and might be an error in the original
            # paper)
            phi = p / P[b_idx]
            # recursion if slope>0 as in Antoni, Eq. (20)
            for _ in range(slope):
                phi = np.sin(np.pi / 2 * phi)
            # shift range to [0, 1]
            phi = .5 * (phi + 1)

            # apply fade out to current channel
            g[b_idx - 1, k_1[b_idx] - P[b_idx]:k_1[b_idx] + P[b_idx] + 1] = \
                np.cos(np.pi / 2 * phi)
            # apply fade in in to next channel
            g[b_idx, k_1[b_idx] - P[b_idx]:k_1[b_idx] + P[b_idx] + 1] = \
                np.sin(np.pi / 2 * phi)

        # set current and next channel to zero outside their range
        g[b_idx - 1, k_1[b_idx] + P[b_idx]:] = 0.
        g[b_idx, :k_1[b_idx] - P[b_idx]] = 0.

    # Force -6 dB at the cut-off frequencies. This is not part of Antony (2010)
    g = g**2

    # generate linear phase
    frequencies = np.fft.rfftfreq(n_samples, 1/sampling_rate_hz)
    group_delay = n_samples / 2 / sampling_rate_hz
    g = g.astype(complex) * np.exp(-1j * 2 * np.pi * frequencies * group_delay)

    # get impulse responses
    time = np.fft.irfft(g)

    # window
    time *= windows.hann(time.shape[-1])

    filters = []
    for i in range(time.shape[0]):
        config = {}
        config['ba'] = [time[i, :], [1]]
        filters.append(
            Filter('other', config, sampling_rate_hz=sampling_rate_hz))
    filt_bank = FilterBank(filters=filters)

    return filt_bank
