"""
Backend for standard functions
"""

import numpy as np
from scipy.signal import correlate, hilbert
from scipy.special import iv as bessel_first_mod
from numpy.typing import NDArray

from ..helpers.spectrum_utilities import _wrap_phase


def _latency(
    in1: NDArray[np.float64],
    in2: NDArray[np.float64] | None,
    polynomial_points: int,
):
    """Computes the latency between two functions using the correlation method.
    The variable polynomial_points is only a dummy to share the same function
    signature as the `_fractional_latency` function.

    """
    if in2 is None:
        in2_ = in1[:, 0][..., None]
        in1_ = np.atleast_2d(in1[:, 1:])
        xcorr = correlate(in2_, in1_, mode="full")
        peak_inds = np.argmax(np.abs(xcorr), axis=0)
    else:
        peak_inds = np.zeros(in1.shape[1], dtype=int)
        for i in range(in1.shape[1]):
            xcorr = correlate(in2[:, i].flatten(), in1[:, i].flatten())
            peak_inds[i] = int(np.argmax(np.abs(xcorr)))
    return in1.shape[0] - peak_inds - 1


def _group_delay_direct(
    phase: NDArray[np.float64 | np.complex128], delta_f: float = 1
) -> NDArray[np.float64]:
    """Computes group delay by differentiation of the unwrapped phase.

    Parameters
    ----------
    phase : NDArray[np.float64]
        Complex spectrum or phase for the direct method
    delta_f : float, optional
        Frequency step for the phase. If it equals 1, group delay is computed
        in samples and not in seconds. Default: 1.

    Returns
    -------
    gd : NDArray[np.float64]
        Group delay vector either in s or in samples if no
        frequency step is given.

    """
    if np.iscomplexobj(phase):
        phase = np.angle(phase)
    if delta_f != 1:
        gd = -np.gradient(np.unwrap(phase), delta_f) / np.pi / 2
    else:
        gd = -np.gradient(np.unwrap(phase))
    return gd


def _minimum_phase(
    magnitude: NDArray[np.float64],
    whole_spectrum: bool = False,
    unwrapped: bool = True,
    odd_length: bool = False,
) -> NDArray[np.float64]:
    """Computes minimum phase system from magnitude spectrum.

    Parameters
    ----------
    magnitude : NDArray[np.float64]
        Spectrum for which to compute the minimum phase. If real, it is assumed
        to be already the magnitude.
    whole_spectrum : bool, optional
        When `True`, it is assumed that the spectrum is passed with both
        positive and negative frequencies. Otherwise, the negative frequencies
        are obtained by mirroring the spectrum. Default: `False`.
    uwrapped : bool, optional
        If `True`, the unwrapped phase is given. Default: `True`.
    odd_length : bool, optional
        When `True`, it is assumed that the underlying time data of the half
        spectrum had an odd length. Default: `False`.

    Returns
    -------
    minimum_phase : NDArray[np.float64]
        Minimal phase of the system.

    """
    if np.iscomplexobj(magnitude):
        magnitude = np.abs(magnitude)

    original_length = magnitude.shape[0]
    if not whole_spectrum:
        if odd_length:
            # Nyquist is not contained in the spectrum
            magnitude = np.concatenate(
                [magnitude, magnitude[1:][::-1]], axis=0
            )
        else:
            magnitude = np.concatenate(
                [magnitude, magnitude[1:-1][::-1]], axis=0
            )

    minimum_phase = -np.imag(
        hilbert(np.log(np.clip(magnitude, a_min=1e-40, a_max=None)), axis=0)
    )[:original_length]

    if not unwrapped:
        minimum_phase = _wrap_phase(minimum_phase)
    return minimum_phase


def _center_frequencies_fractional_octaves_iec(
    nominal, num_fractions
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns the exact center frequencies for fractional octave bands
    according to the IEC 61260:1:2014 standard.
    octave ratio
    .. G = 10^{3/10}
    center frequencies
    .. f_m = f_r G^{x/b}
    .. f_m = f_e G^{(2x+1)/(2b)}
    where b is the number of octave fractions, f_r is the reference frequency
    chosen as 1000Hz and x is the index of the frequency band.

    Parameters
    ----------
    num_fractions : 1, 3
        The number of octave fractions. 1 returns octave center frequencies,
        3 returns third octave center frequencies.

    Returns
    -------
    nominal : array, float
        The nominal (rounded) center frequencies specified in the standard.
        Nominal frequencies are only returned for octave bands and third octave
        bands
    exact : array, float
        The exact center frequencies, resulting in a uniform distribution of
        frequency bands over the frequency range.

    References
    ----------
    - This implementation is taken from the pyfar package. See
      https://github.com/pyfar/pyfar

    """
    if num_fractions == 1:
        nominal = np.array(
            [31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3], dtype=float
        )
    elif num_fractions == 3:
        nominal = np.array(
            [
                25,
                31.5,
                40,
                50,
                63,
                80,
                100,
                125,
                160,
                200,
                250,
                315,
                400,
                500,
                630,
                800,
                1000,
                1250,
                1600,
                2000,
                2500,
                3150,
                4000,
                5000,
                6300,
                8000,
                10000,
                12500,
                16000,
                20000,
            ],
            dtype=float,
        )

    reference_freq = 1e3
    octave_ratio = 10 ** (3 / 10)

    iseven = np.mod(num_fractions, 2) == 0
    if ~iseven:
        indices = np.around(
            num_fractions
            * np.log(nominal / reference_freq)
            / np.log(octave_ratio)
        )
        exponent = indices / num_fractions
    else:
        indices = (
            np.around(
                2.0
                * num_fractions
                * np.log(nominal / reference_freq)
                / np.log(octave_ratio)
                - 1
            )
            / 2
        )
        exponent = (2 * indices + 1) / num_fractions / 2

    exact = reference_freq * octave_ratio**exponent

    return nominal, exact


def _exact_center_frequencies_fractional_octaves(
    num_fractions, frequency_range
) -> NDArray[np.float64]:
    """Calculate the center frequencies of arbitrary fractional octave bands.

    Parameters
    ----------
    num_fractions : int
        The number of fractions
    frequency_range
        The upper and lower frequency limits

    Returns
    -------
    exact : array, float
        An array containing the center frequencies of the respective fractional
        octave bands

    References
    ----------
    - This implementation is taken from the pyfar package. See
      https://github.com/pyfar/pyfar

    """
    ref_freq = 1e3
    Nmax = np.around(num_fractions * (np.log2(frequency_range[1] / ref_freq)))
    Nmin = np.around(num_fractions * (np.log2(ref_freq / frequency_range[0])))

    indices = np.arange(-Nmin, Nmax + 1)
    exact = ref_freq * 2 ** (indices / num_fractions)

    return exact


def _kaiser_window_beta(A):
    """Return a shape parameter beta to create kaiser window based on desired
    side lobe suppression in dB.

    This function has been taken from the pyfar package. See references.

    Parameters
    ----------
    A : float
        Side lobe suppression in dB

    Returns
    -------
    beta : float
        Shape parameter beta after [#]_, Eq. 7.75

    References
    ----------
    - A. V. Oppenheim and R. W. Schafer, Discrete-time signal processing,
      Third edition, Upper Saddle, Pearson, 2010.
    - The pyfar package: https://github.com/pyfar/pyfar

    """
    A = np.abs(A)
    if A > 50:
        return 0.1102 * (A - 8.7)
    if A >= 21:
        return 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    return 0.0


def _kaiser_window_fractional(
    length: int, side_lobe_suppression_db: float, fractional_delay: float
) -> NDArray[np.float64]:
    """Create a kaiser window with a fractional offset.

    Parameters
    ----------
    length : int
        Window length.
    side_lobe_suppression_db : float
        Expected side lobe suppression in dB.
    fractional_delay : float
        Decimal sample offset.

    Returns
    -------
    NDArray[np.float64]
        Kaiser window.

    """
    filter_order = length - 1
    alpha = filter_order / 2
    beta = _kaiser_window_beta(np.abs(side_lobe_suppression_db))
    L = np.arange(length).astype(float) - fractional_delay

    if filter_order % 2:
        L += 0.5
    else:
        if fractional_delay > 0.5:
            L += 1
    Z = beta * np.sqrt(
        np.array(1 - ((L - alpha) / alpha) ** 2, dtype="complex")
    )
    return np.real(bessel_first_mod(0, Z)) / bessel_first_mod(0, beta)


def _indices_above_threshold_dbfs(
    time_vec: NDArray[np.float64],
    threshold_dbfs: float,
    attack_smoothing_coeff: int,
    release_smoothing_coeff: int,
    normalize: bool = True,
):
    """Returns indices with power above a given power threshold (in dBFS) in a
    time series. time_vec can be normalized to peak value prior to computation.

    Parameters
    ----------
    time_vec : NDArray[np.float64]
        Time series for which to find indices above power threshold. Can only
        take one channel.
    threshold_dbfs : float
        Threshold in dBFS to be regarded for activation.
    attack_smoothing_coeff : int
        Coefficient for attack smoothing for level computation.
    release_smoothing_coeff : int
        Coefficient for release smoothing for level computation.
    normalize : bool, optional
        When `True`, signal is normalized such that the threshold is relative
        to peak level and not absolute. Default: `True`.

    Returns
    -------
    indices_above : NDArray[np.float64]
        Array of type boolean with length of time_vec indicating indices
        above threshold with `True` and below with `False`.

    """
    time_vec = np.asarray(time_vec).squeeze()
    assert time_vec.ndim == 1, "Function is implemented for 1D-arrays only"

    # Normalization
    if normalize:
        time_vec /= np.abs(time_vec).max()

    # Power in dB
    time_power = time_vec.squeeze() ** 2

    momentary_gain = np.zeros(len(time_power), dtype=np.float64)
    for i in np.arange(1, len(time_power)):
        if momentary_gain[i] > time_power[i - 1]:
            coeff = attack_smoothing_coeff
        elif momentary_gain[i] < time_power[i - 1]:
            coeff = release_smoothing_coeff
        else:
            coeff = 0
        momentary_gain[i] = (
            coeff * time_power[i] + (1 - coeff) * momentary_gain[i - 1]
        )
    momentary_gain = 10.0 * np.log10(momentary_gain)

    # Get Indices above threshold
    indices_above = momentary_gain > threshold_dbfs
    return indices_above


def _detrend(
    time_data: NDArray[np.float64], polynomial_order: int
) -> NDArray[np.float64]:
    """Compute and return detrended signal.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Time data of the signal with shape (time samples, channels).
    polynomial_order : int
        Polynomial order of the fitted polynomial that will be removed
        from time data. 0 is equal to mean removal.

    Returns
    -------
    new_time_data : NDArray[np.float64]
        Detrended time data with shape (time samples, channels).

    """
    time_indexes = np.arange(len(time_data))
    linear_trend = np.polyfit(time_indexes, time_data, deg=polynomial_order)
    for n in range(time_data.shape[1]):
        time_data[:, n] -= np.polyval(linear_trend[:, n], time_indexes)
    return time_data


def _get_window_envelope(
    window: NDArray[np.float64],
    total_length_samples: int,
    step_size_samples: int,
    number_frames: int,
    squared: bool = True,
):
    """Compute the window envelope for a given window with step size and total
    length. The window can be squared or not.

    """
    if squared:
        window **= 2
    envelope = np.zeros(total_length_samples)

    start = 0
    for _ in range(number_frames):
        envelope[start : start + len(window)] += window
        start += step_size_samples
    return envelope


def _fractional_delay_filter(
    delay_samples: float,
    filter_order: int,
    side_lobe_suppression_db: float,
) -> tuple[int, NDArray[np.float64]]:
    """This function delivers fractional delay filters according to
    specifications. Besides, additional integer delay, that might be necessary
    to compute the output, is returned as well.

    The implementation was taken and adapted from the pyfar package. See
    references.

    Parameters
    ---------
    delay_samples : float
        Amount of delay in samples.
    filter_order : int
        Order for the sinc-filter. Higher orders deliver better results but
        require more computational resources.
    side_lobe_suppression_db : float
        A kaiser window can be applied to the sinc-filter. Its beta parameter
        will be computed according to the required side lobe suppression (
        a common value would be 60 dB). Pass `None` to avoid any windowing
        on the filter.

    Returns
    -------
    integer_delay : int
        Additional integer delay necessary to achieve total desired delay.
    h : NDArray[np.float64]
        Filter's impulse response for fractional delay.

    References
    ----------
    - The pyfar package: https://github.com/pyfar/pyfar
    - T. I. Laakso, V. Välimäki, M. Karjalainen, and U. K. Laine,
      'Splitting the unit delay,' IEEE Signal Processing Magazine 13,
      30-60 (1996). doi:10.1109/79.482137
    - A. V. Oppenheim and R. W. Schafer, Discrete-time signal processing,
      (Upper Saddle et al., Pearson, 2010), Third edition.

    """
    # Separate delay in integer and fractional
    delay_int = int(delay_samples)
    delay_frac = delay_samples - delay_int

    # =========== Sinc function ===============================================
    if filter_order % 2:
        M_opt = int(delay_frac) - (filter_order - 1) / 2
    else:
        M_opt = np.round(delay_frac) - filter_order / 2
    n = np.arange(filter_order + 1) + M_opt - delay_frac
    sinc = np.sinc(n)

    # =========== Kaiser window ===============================================
    kaiser = _kaiser_window_fractional(
        filter_order + 1, side_lobe_suppression_db, delay_frac
    )

    # Compute filter and final integer delay
    frac_delay_filter = sinc * kaiser
    integer_delay = int(delay_int + M_opt)

    return integer_delay, frac_delay_filter
