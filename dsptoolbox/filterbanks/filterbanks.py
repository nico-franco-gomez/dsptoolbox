"""
General use filter banks to be created and given back as a filter bank
object
"""
import numpy as np
from scipy.signal import windows
import warnings
from dsptoolbox import (Filter, FilterBank, fractional_octave_frequencies,
                        erb_frequencies)
from ._filterbank import LRFilterBank, GammaToneFilterBank


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
    LRFilterBank
        Filter bank in form of LRFilterBank class which contains the same
        methods as the FilterBank class but is generated with different
        internal methods.

    """
    return LRFilterBank(freqs, order, sampling_rate_hz)


def reconstructing_fractional_octave_bands(
        num_fractions: int = 1, frequency_range=[63, 16000],
        overlap: float = 1, slope: int = 0, n_samples: int = 2**11,
        sampling_rate_hz: int = 48000) -> FilterBank:
    """Create and/or apply an amplitude preserving fractional octave filter
    bank. This implementation is taken directly from the pyfar package.
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
        filters. The default is ``2**11``.
    sampling_rate : int
        Sampling frequency in Hz. The default is ``None``. Only required if
        ``signal=None``.

    Returns
    -------
    filt_bank : `FilterBank`
        Filter Bank object with FIR filters.

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


def auditory_filters_gammatone(freq_range_hz=[20, 20000],
                               resolution: float = 1,
                               sampling_rate_hz: int = 48000) \
        -> GammaToneFilterBank:
    """Generate an auditory filter bank for analysis purposes. This code was
    taken and adapted from the pyfar package. In this implementation, the
    reference frequency is fixed to 1000 Hz and delay to 4 ms.

    For a more general implementation of this filter bank please refer to the
    pyfar package or the octave/matlab auditory modelling toolbox.
    See references.

    Parameters
    ----------
    freq_range : array-like
        The upper and lower frequency in Hz between which the filter bank is
        constructed. Values must be larger than 0 and not exceed half the
        sampling rate.
    resolution : number
        The frequency resolution of the filter bands in equivalent rectangular
        bandwidth (ERB) units. The bands of the filter bank are distributed
        linearly on the ERB scale. The default value of ``1`` results in one
        filter band per ERB. A value of ``0.5`` would result in two filter
        bands per ERB.
    sampling_rate_hz : int, optional
        The sampling rate of the filter bank in Hz. Default: 48000.

    Returns
    -------
    gammatone_fb : GammaToneFilterBank
        Auditory filters, gamma tone filter bank.

    References
    ----------
    - pyfar: https://github.com/pyfar/pyfar
    - auditory modelling toolbox: https://www.amtoolbox.org

    """
    # Create frequencies
    frequencies_hz = erb_frequencies(freq_range_hz, resolution)
    n_bands = len(frequencies_hz)

    # Eq. (13) in Hohmann 2002
    erb_aud = 24.7 + frequencies_hz / 9.265

    # Eq. (14.3) in Hohmann 2002 (precomputed values for order=4)
    a_gamma = np.pi * 720 * 2**(-6) / 36
    # Eq. (14.2) in Hohmann 2002
    b = erb_aud / a_gamma
    # Eq. (14.1) in Hohmann 2002
    lam = np.exp(-2 * np.pi * b / sampling_rate_hz)
    # Eq. (10) in Hohmann 2002
    beta = 2 * np.pi * frequencies_hz / sampling_rate_hz
    # Eq. (1) in Hohmann 2002 (these are the a_1 coefficients)
    coefficients = lam * np.exp(1j * beta)
    # normalization from Sec. 2.2 in Hohmann 2002
    # (this is the b_0 coefficient)
    normalizations = 2 * (1-np.abs(coefficients))**4

    filters = []
    for bb in range(n_bands):
        sos_section = np.tile(np.atleast_2d(
            [1, 0, 0, 1, -coefficients[bb], 0]),
            (4, 1))
        sos_section[3, 0] = normalizations[bb]
        f = Filter('other', {'sos': sos_section}, sampling_rate_hz)
        f.warning_if_complex = False
        filters.append(f)

    gammatone_fb = GammaToneFilterBank(
        filters,
        info={'Type of filter bank': 'Gammatone filter bank'},
        frequencies=frequencies_hz,
        coefficients=coefficients,
        normalizations=normalizations)
    return gammatone_fb
