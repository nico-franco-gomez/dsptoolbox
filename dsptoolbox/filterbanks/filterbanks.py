"""
General use filter banks to be created and given back as a filter bank
object
"""

import numpy as np
from scipy.signal import windows, bilinear_zpk
import warnings
from .. import (
    Filter,
    FilterBank,
    fractional_octave_frequencies,
    erb_frequencies,
)
from ._filterbank import LRFilterBank, GammaToneFilterBank, QMFCrossover


def linkwitz_riley_crossovers(
    crossover_frequencies_hz, order, sampling_rate_hz: int
) -> LRFilterBank:
    """Returns a linkwitz-riley crossovers filter bank.

    Parameters
    ----------
    crossover_frequencies_hz : array-like
        Frequencies at which to set the crossovers.
    order : array-like
        Order of the crossovers. The higher, the steeper.
    sampling_rate_hz : int
        Sampling rate for the filterbank.

    Returns
    -------
    LRFilterBank
        Filter bank in form of LRFilterBank class which contains the same
        methods as the FilterBank class but is generated with different
        internal methods.

    """
    return LRFilterBank(crossover_frequencies_hz, order, sampling_rate_hz)


def reconstructing_fractional_octave_bands(
    frequency_range_hz=[63, 16000],
    octave_fraction: int = 1,
    overlap: float = 1,
    slope: int = 0,
    n_samples: int = 2**11,
    sampling_rate_hz: int | None = None,
) -> FilterBank:
    """Create a perfect reconstruction filter bank with linear-phase
    characteristics. According to (Antoni J., 2010). This implementation is
    taken from the pyfar package. See references for more information about it.

    Parameters
    ----------
    frequency_range_hz : tuple, optional
        Frequency range in Hz. Default:[ 63, 16e3].
    octave_fraction : int, optional
        Octave fraction used to define bandwidth. Default: 1.
    overlap : float, optional
        Band overlap of the filter slopes between 0 and 1. Smaller values yield
        wider pass-bands and steeper filter slopes. Default: 1.
    slope : int, optional
        Number > 0 that defines the width and steepness of the filter slopes.
        Larger values yield wider pass-bands and steeper filter slopes.
        Default: 0.
    n_samples : int, optional
        Length of the filter in samples. Longer filters are more precise.
        Default: 2**11 = 2048.
    sampling_rate : int
        Sampling frequency in Hz. Default: `None`.

    Returns
    -------
    filt_bank : `FilterBank`
        Filter Bank object with FIR filters.

    References
    ----------
    - Antoni J. Orthogonal-like fractional-octave-band filters. J Acoust Soc
      Am. 2010 Feb;127(2):884-95. doi: 10.1121/1.3273888. PMID: 20136211.
      https://pubmed.ncbi.nlm.nih.gov/20136211/
    - https://github.com/pyfar/pyfar

    """
    assert sampling_rate_hz is not None, "Sampling rate should not be None"
    valid_lengths = 2 ** (np.arange(5, 18))
    assert (
        n_samples in valid_lengths
    ), "Only lengths between 2**5 and 2**17 are allowed"

    if overlap < 0 or overlap > 1:
        raise ValueError("overlap must be between 0 and 1")

    if not isinstance(slope, int) or slope < 0:
        raise ValueError("slope must be a positive integer.")

    # fractional octave frequencies
    _, f_m, f_cut_off = fractional_octave_frequencies(
        octave_fraction, frequency_range_hz, return_cutoff=True
    )

    # number of frequency bins
    n_bins = int(n_samples // 2 + 1)

    # discard fractional octaves, if the center frequency exceeds
    # half the sampling rate
    f_id = f_m < sampling_rate_hz / 2
    if not np.all(f_id):
        warnings.warn("Skipping bands above the Nyquist frequency")

    # DFT lines of the lower cut-off and center frequency as in
    # Antoni, Eq. (14)
    k_1 = np.round(n_samples * f_cut_off[0][f_id] / sampling_rate_hz).astype(
        int
    )
    k_m = np.round(n_samples * f_m[f_id] / sampling_rate_hz).astype(int)
    k_2 = np.round(n_samples * f_cut_off[1][f_id] / sampling_rate_hz).astype(
        int
    )

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
            phi = 0.5 * (phi + 1)

            # apply fade out to current channel
            g[b_idx - 1, k_1[b_idx] - P[b_idx] : k_1[b_idx] + P[b_idx] + 1] = (
                np.cos(np.pi / 2 * phi)
            )
            # apply fade in in to next channel
            g[b_idx, k_1[b_idx] - P[b_idx] : k_1[b_idx] + P[b_idx] + 1] = (
                np.sin(np.pi / 2 * phi)
            )

        # set current and next channel to zero outside their range
        g[b_idx - 1, k_1[b_idx] + P[b_idx] :] = 0.0
        g[b_idx, : k_1[b_idx] - P[b_idx]] = 0.0

    # Force -6 dB at the cut-off frequencies. This is not part of Antony (2010)
    g = g**2

    # generate linear phase
    frequencies = np.fft.rfftfreq(n_samples, 1 / sampling_rate_hz)
    group_delay = n_samples / 2 / sampling_rate_hz
    g = g.astype(complex) * np.exp(-1j * 2 * np.pi * frequencies * group_delay)

    # get impulse responses
    time = np.fft.irfft(g)

    # window
    time *= windows.hann(time.shape[-1])

    filters = []
    for i in range(time.shape[0]):
        config = {}
        config["ba"] = [time[i, :], [1]]
        filters.append(
            Filter("other", config, sampling_rate_hz=sampling_rate_hz)
        )
    filt_bank = FilterBank(filters=filters)

    return filt_bank


def auditory_filters_gammatone(
    frequency_range_hz=[20, 20000],
    resolution: float = 1,
    sampling_rate_hz: int | None = None,
) -> GammaToneFilterBank:
    """Generate an auditory filter bank for analysis purposes. This code was
    taken and adapted from the pyfar package. In this implementation, the
    reference frequency is fixed to 1000 Hz and delay to 4 ms.

    For a more general implementation of this filter bank please refer to the
    pyfar package or the octave/matlab auditory modelling toolbox.
    See references.

    Parameters
    ----------
    frequency_range_hz : array-like, optional
        The upper and lower frequency in Hz between which the filter bank is
        constructed. Values must be larger than 0 and not exceed half the
        sampling rate. Default: [20, 20e3].
    resolution : number, optional
        The frequency resolution of the filter bands in equivalent rectangular
        bandwidth (ERB) units. The bands of the filter bank are distributed
        linearly on the ERB scale. The default value of 1 results in one
        filter band per ERB. Default: 1.
    sampling_rate_hz : int
        The sampling rate of the filter bank in Hz. Default: `None`.

    Returns
    -------
    gammatone_fb : GammaToneFilterBank
        Auditory filters, gamma tone filter bank.

    Methods
    -------
    Apart from all the methods of the `FilterBank` class, there is also the
    `reconstruct()` method, which takes a `MultiBandSignal` and recreates
    the original `Signal` from it.

    References
    ----------
    - pyfar: https://github.com/pyfar/pyfar
    - auditory modelling toolbox: https://www.amtoolbox.org

    """
    assert (
        sampling_rate_hz is not None
    ), "A sampling rate must be passed to create the filter bank"
    assert (
        np.max(frequency_range_hz) <= sampling_rate_hz // 2
    ), "Highest frequency should not be higher than the nyquist frequency"
    # Create frequencies
    frequencies_hz = erb_frequencies(frequency_range_hz, resolution)
    n_bands = len(frequencies_hz)

    # Eq. (13) in Hohmann 2002
    erb_aud = 24.7 + frequencies_hz / 9.265

    # Eq. (14.3) in Hohmann 2002 (precomputed values for order=4)
    a_gamma = np.pi * 720 * 2 ** (-6) / 36
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
    normalizations = 2 * (1 - np.abs(coefficients)) ** 4

    filters = []
    for bb in range(n_bands):
        sos_section = np.tile(
            np.atleast_2d([1, 0, 0, 1, -coefficients[bb], 0]), (4, 1)
        )
        sos_section[3, 0] = normalizations[bb]
        f = Filter("other", {"sos": sos_section}, sampling_rate_hz)
        f.warning_if_complex = False
        filters.append(f)

    gammatone_fb = GammaToneFilterBank(
        filters,
        info={"Type of filter bank": "Gammatone filter bank"},
        frequencies=frequencies_hz,
        coefficients=coefficients,
        normalizations=normalizations,
    )
    return gammatone_fb


def qmf_crossover(lowpass: Filter) -> QMFCrossover:
    """This creates quadrature mirror filters that work as a two band,
    maximally decimated filter bank. For a 1st order FIR filter, this filter
    bank is ensured to have perfect reconstruction capabilities.

    Parameters
    ----------
    lowpass : `Filter`
        Lowpass filter prototype with which to create the other filters.

    Returns
    -------
    fb : `QMFCrossover`
        Quadrature mirror filters crossover.

    References
    ----------
    - https://ccrma.stanford.edu/~jos/sasp/Quadrature_Mirror_Filters_QMF.html

    """
    return QMFCrossover(lowpass)


def fractional_octave_bands(
    frequency_range_hz=[31.5, 16e3],
    octave_fraction: int = 1,
    filter_order: int = 6,
    sampling_rate_hz: int | None = None,
):
    """Create and return a filter bank containing a set of of fractional
    octave filters that are compliant with the specifications presented in [1].
    These are butterworth filters with at least order 3. For offline
    applications where phase or time information is also needed, filtering
    with `zero_phase=True` is recommended (the filter order will be doubled).

    For even filter orders, this filter bank is energy preserving.

    Parameters
    ----------
    frequency_range_hz : array-like, optional
        Frequency range in Hz for which to compute the filter bank.
        Default: [31.5, 16000].
    octave_fraction : int, optional
        Octave fraction to use. Default: 1.
    filter_order : int, optional
        Filter order to be used. Less than 4 is not recommended. Default: 4.
    sampling_rate_hz : int
        Sampling rate in Hz.

    Returns
    -------
    octave_filter_bank : `FilterBank`
        Filter bank containing fractional octave band filters.

    References
    ----------
    - [1]: ANSI S1.11:2004.

    """
    assert (
        sampling_rate_hz is not None
    ), "A sampling rate must be passed for the filter bank"
    frequency_range_hz = np.atleast_1d(np.squeeze(frequency_range_hz))
    frequency_range_hz.sort()
    assert (
        len(frequency_range_hz) == 2
    ), "Frequency range must contain exactly two entries"
    assert frequency_range_hz[-1] < sampling_rate_hz // 2, (
        "The highest frequency in the range is higher than the nyquist "
        + "frequency"
    )

    # fractional octave frequencies
    lower, upper = fractional_octave_frequencies(
        octave_fraction, frequency_range_hz, return_cutoff=True
    )[2]

    octave_filter_bank = FilterBank()

    for ind in range(len(lower)):
        top = "bandpass"
        freqs = [lower[ind], upper[ind]]
        if upper[ind] > sampling_rate_hz // 2:
            top = "highpass"
            freqs = lower[ind]

        f = Filter(
            "iir",
            dict(
                type_of_pass=top,
                filter_design_method="butter",
                order=filter_order,
                freqs=freqs,
            ),
            sampling_rate_hz=sampling_rate_hz,
        )
        octave_filter_bank.add_filter(f)

    return octave_filter_bank


def weightning_filter(
    weightning: str = "a", sampling_rate_hz: int | None = None
):
    """Returns a digital IIR weightning filter according to [1]. The
    approximation is based on the coefficients given in [2].

    Parameters
    ----------
    weightning : {'a', 'c'} str, optional
        Type of weightning. Choose between `'a'` or `'c'`. Default: `'a'`.
    sampling_rate_hz : int
        Sampling rate for the digital filter.

    Returns
    -------
    weightning_filter : `Filter`
        Weightning filter.

    References
    ----------
    - [1]: IEC 61672:2003.
    - [2]: https://en.wikipedia.org/wiki/A-weighting

    """
    weightning = weightning.lower()
    assert weightning in (
        "a",
        "c",
    ), "Invalid type of weightning. Use either a or c"
    if weightning == "a":
        z = [0, 0, 0, 0]
        k = 7.39705e9
        p = [-129.4, -129.4, -676.7, -4636, -76655, -76655]
    else:
        z = [0, 0]
        k = 5.91797e9
        p = [-129.4, -129.4, -76655, -76655]
    coeff = bilinear_zpk(z, p, k, sampling_rate_hz)
    weightning_filter = Filter(
        "other", dict(zpk=coeff), sampling_rate_hz=sampling_rate_hz
    )
    return weightning_filter


def complementary_fir_filter(fir: Filter) -> Filter:
    """Returns a complementary filter for an FIR filter with linear phase.
    It returns for instance a lowpass from a highpass prototype. Combined,
    they deliver perfect magnitude reconstruction (with some quantization
    error) and a constant group delay. This method might only return meaningful
    results if the filter prototype has an odd number of taps and linear phase.

    Parameters
    ----------
    fir : `Filter`
        Filter prototype with linear phase.

    Returns
    -------
    fir_complementary : `Filter`
        Complementary filter.

    """
    assert fir.filter_type == "fir", "Filter prototype must be an FIR filter"

    b = fir.ba[0].copy()
    impulse_index = np.argmax(np.abs(b))
    b *= -1
    b[impulse_index] += 1
    fir_complementary = Filter("other", {"ba": [b, [1]]}, fir.sampling_rate_hz)
    return fir_complementary
