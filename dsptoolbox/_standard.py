"""
Backend for standard functions
"""

import numpy as np
from scipy.signal import correlate, check_COLA, windows, hilbert, convolve
from scipy.special import iv as bessel_first_mod
from ._general_helpers import (
    _pad_trim,
    _compute_number_frames,
    _wrap_phase,
)
from warnings import warn


def _latency(
    in1: np.ndarray, in2: np.ndarray | None = None, polynomial_points: int = 0
):
    """Computes the latency between two functions using the correlation method.
    The variable polynomial_points is only a dummy to share the same function
    signature as the `_fractional_latency` function.

    """
    if in2 is None:
        in2 = in1[:, 0][..., None]
        in1 = np.atleast_2d(in1[:, 1:])
        xcorr = correlate(in2, in1, mode="full")
        peak_inds = np.argmax(np.abs(xcorr), axis=0)
    else:
        peak_inds = np.zeros(in1.shape[1], dtype=int)
        for i in range(in1.shape[1]):
            xcorr = correlate(in2[:, i].flatten(), in1[:, i].flatten())
            peak_inds[i] = int(np.argmax(np.abs(xcorr)))
    return in1.shape[0] - peak_inds - 1


def _welch(
    x: np.ndarray,
    y: np.ndarray | None,
    fs_hz: int,
    window_type: str = "hann",
    window_length_samples: int = 1024,
    overlap_percent=50,
    detrend: bool = True,
    average: str = "mean",
    scaling: str | None = "power spectral density",
) -> np.ndarray:
    """Cross spectral density computation with Welch's method.

    Parameters
    ----------
    x : `np.ndarray`
        First signal with shape (time samples, channel).
    y : `np.ndarray` or `None`
        Second signal with shape (time samples, channel). If `None`, the auto-
        spectrum of `x` will be computed.
    fs_hz : int
        Sampling rate in Hz.
    window_type : str, optional
        Window type to be used. Refer to scipy.signal.windows for available
        ones. Default: `'hann'`
    window_length_samples : int, optional
        Window length to be used. Determines frequency resolution in the end.
        Only powers of 2 are accepted. Default: 1024.
    overlap_percent : int, optional
        Overlap in percentage. Default: 50.
    detrend : bool, optional
        Detrending from each time segment (removing mean). Default: True.
    average : str, optional
        Type of mean to be computed. Take `'mean'` or `'median'`.
        Default: `'mean'`
    scaling : str, optional
        Scaling. Use `'power spectrum'`, `'power spectral density'`,
        `'amplitude spectrum'` or `'amplitude spectral density'`. Pass `None`
        to avoid any scaling. See references for details about scaling.
        Default: `'power spectral density'`.

    Returns
    -------
    csd : `np.ndarray`
        Complex cross spectral density vector if x and y are different.
        Alternatively, the (real) autocorrelation power spectral density when
        x and y are the same. If density or spectrum depends on scaling.
        Depending on the input, the output shape is (time samples) or
        (time samples, channel).

    References
    ----------
    - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and spectral
      density estimation by the Discrete Fourier transform (DFT), including a
      comprehensive list of window functions and some new at-top windows.
    - Allen, B., Anderson, W. G., Brady, P. R., Brown, D. A., & Creighton,
      J. D. E. (2005). FINDCHIRP: an algorithm for detection of gravitational
      waves from inspiraling compact binaries.
      See http://arxiv.org/abs/gr-qc/0509116.

    """
    autospectrum = y is None

    if type(x) is not np.ndarray:
        x = np.asarray(x).squeeze()

    if not autospectrum:
        if type(y) is not np.ndarray:
            y = np.asarray(y).squeeze()
        assert x.shape == y.shape, "Shapes of data do not match"
        # NOTE: Computing the spectrum in a vectorized manner for all channels
        # simultaneously does not seem to be faster than doing it sequentally
        # for each channel. Maybe parallelizing with something like numba could
        # be advantageous...

    if x.ndim == 2:
        multi_channel = True
    else:
        multi_channel = False

    assert len(x.shape) <= 2, (
        f"{x.shape} are too many dimensions. Use flat"
        + " arrays or 2D-Arrays instead"
    )

    valid_window_sizes = np.array([int(2**x) for x in range(3, 19)])
    assert window_length_samples in valid_window_sizes, (
        "Window length should be a power of 2 between [8, 262_144] or "
        + "[2**3, 2**18]"
    )
    assert (
        overlap_percent >= 0 and overlap_percent < 100
    ), "overlap_percent should be between 0 and 100"
    valid_average = ["mean", "median"]
    assert average in valid_average, (
        f"{average} is not valid. Use " + "either mean or median"
    )
    valid_scaling = [
        "power spectrum",
        "power spectral density",
        "amplitude spectrum",
        "amplitude spectral density",
        None,
    ]
    assert scaling in valid_scaling, (
        f"{scaling} is not valid. Use "
        + "power spectrum, power spectral density, amplitude spectrum, "
        + "amplitude spectral density or None"
    )
    if scaling is None:
        scaling = ""

    # Window and step
    window = windows.get_window(
        window_type, window_length_samples, fftbins=True
    )
    overlap_samples = int(overlap_percent / 100 * window_length_samples)
    step = window_length_samples - overlap_samples

    # Check COLA
    if not check_COLA(window, nperseg=len(window), noverlap=overlap_samples):
        warn(
            "Selected window type and overlap do not meet the constant "
            + "overlap and add constraint! Results might be distorted"
        )

    if not multi_channel:
        x = x[..., None]
        if not autospectrum:
            y = y[..., None]

    x_frames = _get_framed_signal(x, window_length_samples, step)
    if not autospectrum:
        y_frames = _get_framed_signal(y, window_length_samples, step)

    # Window
    x_frames *= window[:, np.newaxis, np.newaxis]
    if not autospectrum:
        y_frames *= window[:, np.newaxis, np.newaxis]

    if not multi_channel:
        x_frames = np.squeeze(x_frames)
        if not autospectrum:
            y_frames = np.squeeze(y_frames)

    # Detrend
    if detrend:
        x_frames -= np.mean(x_frames, axis=0)
        if not autospectrum:
            y_frames -= np.mean(y_frames, axis=0)

    if not autospectrum:
        sp_frames = np.fft.rfft(x_frames, axis=0).conjugate() * np.fft.rfft(
            y_frames, axis=0
        )
    else:
        sp_frames = np.abs(np.fft.rfft(x_frames, axis=0)) ** 2.0

    # Direct averaging (much faster than averaging magnitude and phase)
    if average == "mean":
        csd = np.mean(sp_frames, axis=1)
    else:
        csd = np.median(sp_frames.real, axis=1) + 1j * np.median(
            sp_frames.imag, axis=1
        )
        # Bias according to reference
        n = (
            sp_frames.shape[1]
            if sp_frames.shape[1] % 2 == 1
            else sp_frames.shape[1] - 1
        )
        bias = np.sum((-1) ** (n + 1) / n)
        csd /= bias

    # Weightning (with 2 because one-sided)
    if scaling in ("power spectrum", "amplitude spectrum"):
        factor = 2 / np.sum(window) ** 2
    elif scaling in ("power spectral density", "amplitude spectral density"):
        factor = 2 / (window @ window) / fs_hz
        # With this factor, energy can be regained by integrating the psd
        # while taking into account the frequency step
    else:
        factor = 1

    csd *= factor
    if factor != 1:
        csd[0, ...] /= 2
        csd[-1, ...] /= 2

    if "amplitude" in scaling:
        csd = np.sqrt(csd)

    return csd


def _group_delay_direct(phase: np.ndarray, delta_f: float = 1):
    """Computes group delay by differentiation of the unwrapped phase.

    Parameters
    ----------
    phase : `np.ndarray`
        Complex spectrum or phase for the direct method
    delta_f : float, optional
        Frequency step for the phase. If it equals 1, group delay is computed
        in samples and not in seconds. Default: 1.

    Returns
    -------
    gd : `np.ndarray`
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
    magnitude: np.ndarray,
    whole_spectrum: bool = False,
    unwrapped: bool = True,
    odd_length: bool = False,
) -> np.ndarray:
    """Computes minimum phase system from magnitude spectrum.

    Parameters
    ----------
    magnitude : `np.ndarray`
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
    minimum_phase : `np.ndarray`
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


def _stft(
    x: np.ndarray,
    fs_hz: int,
    window_length_samples: int = 2048,
    window_type: str = "hann",
    overlap_percent=50,
    fft_length_samples: int | None = None,
    detrend: bool = True,
    padding: bool = False,
    scaling: str | None = None,
):
    """Computes the STFT of a signal. Output matrix has (freqs_hz, seconds_s).

    Parameters
    ----------
    x : `np.ndarray`
        First signal
    fs_hz : int
        Sampling rate in Hz.
    window_length_samples : int, optional
        Window length to be used. Determines frequency resolution in the end.
        Only powers of 2 are accepted. Default: 1024.
    window_type : str, optional
        Window type to be used. Refer to scipy.signal.windows for available
        ones. Default: `'hann'`
    overlap_percent : int, optional
        Overlap in percentage. Default: 50.
    fft_length_samples : int, optional
        Length of the FFT window for each time window. This affects
        the frequency resolution and can also crop the time window. Pass
        `None` to use the window length. Default: `None`.
    detrend : bool, optional
        Detrending from each time segment (removing mean). Default: True.
    padding : bool, optional
        When `True`, the original signal is padded in the beginning and ending
        so that no energy is lost due to windowing when the COLA constraint is
        met. Default: `False`.
    scaling : str, optional
        Scale as `"amplitude spectrum"`, `"amplitude spectral density"`,
        `"power spectrum"` or `"power spectral density"`. Pass `None`
        to avoid any scaling. See references for details. Default: `None`.

    Returns
    -------
    time_s : `np.ndarray`
        Time vector in seconds for each frame.
    freqs_hz : `np.ndarray`
        Frequency vector.
    stft : `np.ndarray`
        STFT matrix with shape (frequency, time, channel).

    References
    ----------
    - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and spectral
      density estimation by the Discrete Fourier transform (DFT), including a
      comprehensive list of window functions and some new at-top windows.

    """
    valid_window_sizes = np.array([int(2**x) for x in range(4, 17)])
    assert window_length_samples in valid_window_sizes, (
        "Window length should be a power of 2 between [16, 65536] or "
        + "[2**4, 2**16]"
    )
    assert overlap_percent >= 0 and overlap_percent < 100, (
        "overlap_percent" + " should be between 0 and 100"
    )
    valid_scaling = [
        "power spectrum",
        "power spectral density",
        "amplitude spectrum",
        "amplitude spectral density",
        None,
    ]
    assert scaling in valid_scaling, (
        f"{scaling} is not valid. Use "
        + "power spectrum, power spectral density, amplitude spectrum, "
        + "amplitude spectral density or None"
    )

    if scaling is None:
        scaling = ""

    if fft_length_samples is None:
        fft_length_samples = window_length_samples

    # Window and step
    window = windows.get_window(
        window_type, window_length_samples, fftbins=True
    )
    overlap_samples = int(overlap_percent / 100 * window_length_samples + 0.5)
    step = window_length_samples - overlap_samples

    # Check COLA
    if not check_COLA(window, nperseg=len(window), noverlap=overlap_samples):
        warn(
            "Selected window type and overlap do not meet the constant "
            + "overlap and add constraint! Results might be distorted"
        )

    # Padding
    if padding:
        x = np.pad(x, ((overlap_samples, overlap_samples), (0, 0)))
    # Framed signal
    time_x = _get_framed_signal(x, window_length_samples, step, True)
    # Windowing
    time_x *= window[..., np.newaxis, np.newaxis]
    # Detrend
    if detrend:
        time_x -= np.mean(time_x, axis=0)
    # Spectra
    stft = np.fft.rfft(time_x, axis=0, n=fft_length_samples)
    # Scaling
    if scaling in ("power spectrum", "amplitude spectrum"):
        factor = 2**0.5 / np.sum(window)
    elif scaling in ("power spectral density", "amplitude spectral density"):
        factor = (2 / (window @ window) / fs_hz) ** 0.5
    else:
        factor = 1

    stft *= factor

    if scaling:
        stft[0, ...] /= 2**0.5

    if scaling and fft_length_samples % 2 == 0:
        stft[-1, ...] /= 2**0.5

    if "power" in scaling:
        stft = np.abs(stft) ** 2

    time_s = np.linspace(0, len(x) / fs_hz, stft.shape[1])
    freqs_hz = np.fft.rfftfreq(len(window), 1 / fs_hz)
    return time_s, freqs_hz, stft


def _csm(
    time_data: np.ndarray,
    sampling_rate_hz: int,
    window_length_samples: int = 1024,
    window_type: str = "hann",
    overlap_percent: int = 50,
    detrend: bool = True,
    average: str = "mean",
    scaling: str = "power",
):
    """Computes the cross spectral matrix of a multichannel signal.
    Output matrix has (frequency, channels, channels).

    Parameters
    ----------
    time_data : `np.ndarray`
        Signal
    sampling_rate_hz : int
        Sampling rate in Hz.
    window_length_samples : int, optional
        Window length to be used. Determines frequency resolution in the end.
        Only powers of 2 are accepted. Default: 1024.
    window_type : str, optional
        Window type to be used. Refer to scipy.signal.windows for available
        ones. Default: `'hann'`
    overlap_percent : int, optional
        Overlap in percentage. Default: 50.
    detrend : bool, optional
        Detrending from each time segment (removing mean). Default: True.
    average : str, optional
        Type of mean to be computed. Take `'mean'` or `'median'`.
        Default: `'mean'`
    scaling : str, optional
        Scaling. Use `'power spectrum'`, `'power spectral density'`,
        `'amplitude spectrum'` or `'amplitude spectral density'`. Pass `None`
        to avoid any scaling. See references for details about scaling.
        Default: `'power spectral density'`.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector
    csm : `np.ndarray`
        Cross spectral matrix with shape (frequency, channels, channels).

    References
    ----------
    - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and spectral
      density estimation by the Discrete Fourier transform (DFT), including a
      comprehensive list of window functions and some new at-top windows.

    """
    # ===== Remarks on speed =============
    # It has been tried to vectorize the whole computation of the CSM by using
    # a multi-channel approach in the _welch() function or with a class. This
    # leads to a dramatic drop in performance and an elevated memory cost.
    # Maybe using some parallel computing framework like numba would make it
    # faster, but less readable and it would be a new dependency of the
    # package... So far the double loop has been best solution
    # =====================================
    number_of_channels = time_data.shape[1]
    csm = np.zeros(
        (
            window_length_samples // 2 + 1,
            number_of_channels,
            number_of_channels,
        ),
        dtype="cfloat",
    )
    for ind1 in range(number_of_channels):
        for ind2 in range(ind1, number_of_channels):
            # Complex conjugate second signal and not first (like transposing
            # the matrix)
            csm[:, ind2, ind1] = _welch(
                time_data[:, ind1],
                time_data[:, ind2] if ind1 != ind2 else None,
                sampling_rate_hz,
                window_length_samples=window_length_samples,
                window_type=window_type,
                overlap_percent=overlap_percent,
                detrend=detrend,
                average=average,
                scaling=scaling,
            )
            if ind1 == ind2:
                csm[:, ind1, ind2] *= 0.5
    csm += np.swapaxes(csm, 1, 2).conjugate()
    f = np.fft.rfftfreq(window_length_samples, 1 / sampling_rate_hz)
    return f, csm


def _center_frequencies_fractional_octaves_iec(
    nominal, num_fractions
) -> tuple[np.ndarray, np.ndarray]:
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
) -> np.ndarray:
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
) -> np.ndarray:
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
    `np.ndarray`
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
    time_vec: np.ndarray,
    threshold_dbfs: float,
    attack_smoothing_coeff: int,
    release_smoothing_coeff: int,
    normalize: bool = True,
):
    """Returns indices with power above a given power threshold (in dBFS) in a
    time series. time_vec can be normalized to peak value prior to computation.

    Parameters
    ----------
    time_vec : `np.ndarray`
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
    indices_above : `np.ndarray`
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

    momentary_gain = np.zeros(len(time_power))
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
    momentary_gain = 10 * np.log10(momentary_gain)

    # Get Indices above threshold
    indices_above = momentary_gain > threshold_dbfs
    return indices_above


def _detrend(time_data: np.ndarray, polynomial_order: int) -> np.ndarray:
    """Compute and return detrended signal.

    Parameters
    ----------
    time_data : np.ndarray
        Time data of the signal with shape (time samples, channels).
    polynomial_order : int
        Polynomial order of the fitted polynomial that will be removed
        from time data. 0 is equal to mean removal.

    Returns
    -------
    new_time_data : np.ndarray
        Detrended time data with shape (time samples, channels).

    """
    time_indexes = np.arange(len(time_data))
    linear_trend = np.polyfit(time_indexes, time_data, deg=polynomial_order)
    for n in range(time_data.shape[1]):
        time_data[:, n] -= np.polyval(linear_trend[:, n], time_indexes)
    return time_data


def _rms(x: np.ndarray) -> float | np.ndarray:
    """Root mean squared value of a discrete time series.

    Parameters
    ----------
    x : `np.ndarray`
        Time series.

    Returns
    -------
    rms : float or `np.ndarray`
        Root mean squared of a signal. Float or np.ndarray depending on input.

    """
    single_dim = False
    if x.ndim < 2:
        single_dim = True
        x = x[..., None]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(
            "Shape of array is not valid. Only 2D-Arrays " + "are valid"
        )
    rms_vals = np.sqrt(np.mean(x**2, axis=0))
    if single_dim:
        rms_vals = np.squeeze(rms_vals)
    return rms_vals


def _get_framed_signal(
    td: np.ndarray,
    window_length_samples: int,
    step_size: int,
    keep_last_frames: bool = True,
) -> np.ndarray:
    """This method computes a framed version of a signal and returns it.

    Parameters
    ----------
    td : `np.ndarray`
        Signal with shape (time samples, channels).
    window_length_samples : int
        Window length in samples.
    step_size : int
        Step size (also called hop length) in samples.
    keep_last_frames : bool, optional
        When `True`, the last frames (probably with zero-padding) are kept.
        Otherwise, no frames with zero padding are included. Default: `True`.

    Returns
    -------
    td_framed : `np.ndarray`
        Framed signal with shape (time samples, frames, channels).

    """
    # Force casting to integers
    if type(window_length_samples) is not int:
        window_length_samples = int(window_length_samples)
    if type(step_size) is not int:
        step_size = int(step_size)

    # Start Parameters
    n_frames, padding_samp = _compute_number_frames(
        window_length_samples,
        step_size,
        td.shape[0],
        zero_padding=keep_last_frames,
    )
    td = _pad_trim(td, td.shape[0] + padding_samp)
    td_framed = np.zeros(
        (window_length_samples, n_frames, td.shape[1]), dtype="float"
    )

    # Create time frames
    start = 0
    for n in range(n_frames):
        td_framed[:, n, :] = td[
            start : start + window_length_samples, :
        ].copy()
        start += step_size

    return td_framed


def _reconstruct_framed_signal(
    td_framed: np.ndarray,
    step_size: int,
    window: str | np.ndarray | None = None,
    original_signal_length: int | None = None,
    safety_threshold: float = 1e-4,
) -> np.ndarray:
    """Gets and returns a framed signal into its vector representation.

    Parameters
    ----------
    td_framed : `np.ndarray`
        Framed signal with shape (time samples, frame, channel).
    step_size : int
        Step size in samples between frames (also known as hop length).
    window : str, `np.ndarray`, optional
        Window (if applies). Pass `None` to avoid using a window during
        reconstruction. Default: `None`.
    original_signal_length : int, optional
        When different than `None`, the output is padded or trimmed to this
        length. Default: `None`.
    safety_threshold : float, optional
        When reconstructing the signal with a window, very small values can
        lead to instabilities. This safety threshold avoids dividing with
        samples beneath this value. Default: 1e-4.

        Dividing by 1e-4 is the same as amplifying by 80 dB.

    Returns
    -------
    td : `np.ndarray`
        Reconstructed signal.

    """
    assert (
        td_framed.ndim == 3
    ), "Framed signal must contain exactly three dimensions"
    if window is not None:
        if type(window) is str:
            window = windows.get_window(window, td_framed.shape[0])
        elif type(window) is np.ndarray:
            assert window.ndim == 1, "Window must be a 1D-array"
            assert (
                window.shape[0] == td_framed.shape[0]
            ), "Window length does not match signal length"
        td_framed *= window[:, np.newaxis, np.newaxis]

    total_length = int(
        step_size * td_framed.shape[1]
        + td_framed.shape[0] * (1 - step_size / td_framed.shape[0])
    )
    td = np.zeros((total_length, td_framed.shape[-1]))

    start = 0
    for i in range(td_framed.shape[1]):
        td[start : start + td_framed.shape[0], :] += td_framed[:, i, :]
        start += step_size

    if window is not None:
        envelope = _get_window_envelope(
            window, total_length, step_size, td_framed.shape[1], True
        )
        if safety_threshold is not None:
            envelope = np.clip(envelope, a_min=safety_threshold, a_max=None)
        non_zero = envelope > np.finfo(td.dtype).tiny
        td[non_zero, ...] /= envelope[non_zero, np.newaxis]

    if original_signal_length is not None:
        td = _pad_trim(td, original_signal_length)
    return td


def _get_window_envelope(
    window: np.ndarray,
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
) -> tuple[int, np.ndarray]:
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
    h : `np.ndarray`
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
