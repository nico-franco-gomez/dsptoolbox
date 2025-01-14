import numpy as np
from scipy.signal import check_COLA, windows
from warnings import warn
from numpy.typing import NDArray

from ..standard._framed_signal_representation import _get_framed_signal
from ..standard.enums import SpectrumScaling


def _welch(
    x: NDArray[np.float64],
    y: NDArray[np.float64] | None,
    fs_hz: int,
    window_type: str,
    window_length_samples: int,
    overlap_percent: float,
    detrend: bool,
    average: str,
    scaling: SpectrumScaling,
) -> NDArray[np.float64]:
    """Cross spectral density computation with Welch's method.

    Parameters
    ----------
    x : NDArray[np.float64]
        First signal with shape (time samples, channel).
    y : NDArray[np.float64] or `None`
        Second signal with shape (time samples, channel). If `None`, the auto-
        spectrum of `x` will be computed.
    fs_hz : int
        Sampling rate in Hz.
    window_type : str
        Window type to be used. Refer to `scipy.signal.windows` for available
        ones.
    window_length_samples : int
        Window length to be used. Determines frequency resolution in the end.
        Only powers of 2 are accepted.
    overlap_percent : float
        Overlap in percentage.
    detrend : bool
        Detrending from each time segment (removing mean).
    average : str
        Type of mean to be computed. Take `'mean'` or `'median'`.
    scaling : SpectrumScaling
        Scaling. See references for details about scaling.

    Returns
    -------
    csd : NDArray[np.float64]
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

    if type(x) is not NDArray[np.float64]:
        x = np.asarray(x).squeeze()

    if not autospectrum:
        if type(y) is not NDArray[np.float64]:
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
        sp_frames = np.fft.rfft(
            x_frames, axis=0, norm=scaling.fft_norm()
        ).conjugate() * np.fft.rfft(y_frames, axis=0, norm=scaling.fft_norm())
    else:
        sp_frames = (
            np.abs(np.fft.rfft(x_frames, axis=0, norm=scaling.fft_norm()))
            ** 2.0
        )

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
    if scaling.has_physical_units():
        factor = scaling.get_scaling_factor(
            window_length_samples, fs_hz, window
        )
        csd *= factor
        csd[np.array([0, -1]), ...] /= 2

    if scaling.is_amplitude_scaling():
        csd = np.sqrt(csd)

    return csd


def _stft(
    x: NDArray[np.float64],
    fs_hz: int,
    window_length_samples: int,
    window_type: str,
    overlap_percent: float,
    fft_length_samples: int | None,
    detrend: bool,
    padding: bool,
    scaling: SpectrumScaling,
):
    """Computes the STFT of a signal. Output matrix has (freqs_hz, seconds_s).

    Parameters
    ----------
    x : NDArray[np.float64]
        (Multichannel) Time series.
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
    time_s : NDArray[np.float64]
        Time vector in seconds for each frame.
    freqs_hz : NDArray[np.float64]
        Frequency vector.
    stft : NDArray[np.float64]
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
    stft = np.fft.rfft(
        time_x, axis=0, n=fft_length_samples, norm=scaling.fft_norm()
    )

    # Scaling
    if scaling.has_physical_units():
        stft[0, ...] /= 2**0.5
        if fft_length_samples % 2 == 0:
            stft[-1, ...] /= 2**0.5
        factor = scaling.get_scaling_factor(fft_length_samples, fs_hz, window)
        if not scaling.is_amplitude_scaling():
            stft = np.abs(stft) ** 2.0
        stft *= factor

    time_s = np.linspace(0, len(x) / fs_hz, stft.shape[1])
    freqs_hz = np.fft.rfftfreq(len(window), 1 / fs_hz)
    return time_s, freqs_hz, stft


def _csm_welch(
    time_data: NDArray[np.float64],
    sampling_rate_hz: int,
    window_length_samples: int,
    window_type: str,
    overlap_percent: int,
    detrend: bool,
    average: str,
    scaling: str,
):
    """Computes the cross spectral matrix of a multichannel signal using
    welch's method for a periodogram. Output matrix has (frequency, channels,
    channels).

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Signal
    sampling_rate_hz : int
        Sampling rate in Hz.
    window_length_samples : int
        Window length to be used. Determines frequency resolution in the end.
        Only powers of 2 are accepted.
    window_type : str
        Window type to be used. Refer to scipy.signal.windows for available
        ones.
    overlap_percent : int
        Overlap in percentage.
    detrend : bool
        Detrending from each time segment (removing mean).
    average : str
        Type of mean to be computed. Take `'mean'` or `'median'`.
    scaling : SpectrumScaling
        Scaling. See references for details about scaling.

    Returns
    -------
    f : NDArray[np.float64]
        Frequency vector
    csm : NDArray[np.float64]
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
        dtype=np.complex128,
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
            # Half for summing the csm
            if ind1 == ind2:
                csm[:, ind1, ind2] *= 0.5
    csm += np.swapaxes(csm, 1, 2).conjugate()
    f = np.fft.rfftfreq(window_length_samples, 1 / sampling_rate_hz)
    return f, csm


def _csm_fft(
    spectrum: NDArray[np.complex128],
    scaling: SpectrumScaling,
    window: NDArray[np.float64] | None,
    sampling_rate_hz: int,
) -> np.complex128:
    """Compute the cross-spectral matrix from a multichannel complex spectrum.
    It is assumed that the input spectrum has no scaling, i.e., FFTBackward
    normalization.

    Parameters
    ----------
    spectrum : NDArray[np.complex128]
        Complex spectrum from which to compute the cross-spectral matrix.
    scaling : SpectrumScaling
        Output scaling.
    window : NDArray[np.float64], None
        Time window if it has been applied.
    sampling_rate_hz : int
        Sampling rate.

    Returns
    -------
    np.complex128
        Complex spectrum.

    Notes
    -----
    - Output scalings from FFTNorms always deliver a squared, i.e., cross-
      spectrum.

    """
    if window is not None:
        assert (
            window.shape[1] == spectrum.shape[1]
        ), "Number of channels does not match"

    number_of_channels = spectrum.shape[1]
    csm = np.zeros(
        (
            spectrum.shape[0],
            number_of_channels,
            number_of_channels,
        ),
        dtype=np.complex128,
    )
    for ind1 in range(number_of_channels):
        for ind2 in range(ind1, number_of_channels):
            # Complex conjugate second signal and not first (like transposing
            # the matrix)
            csm[:, ind2, ind1] = (
                spectrum[:, ind1].conjugate() * spectrum[:, ind2]
            )
            # Half for summing the csm
            if ind1 == ind2:
                csm[:, ind1, ind2] *= 0.5
    csm += np.swapaxes(csm, 1, 2).conjugate()

    # Handle scaling
    if scaling == SpectrumScaling.FFTBackward:
        return csm

    # Fix DC and Nyquist because one-sided
    csm[np.array([0, -1]), ...] /= 2.0

    # Conversion
    factor = SpectrumScaling.FFTBackward.conversion_factor(
        scaling, spectrum.shape[0] // 2 + 1, sampling_rate_hz, window
    )[:, None]
    factor = np.repeat(factor, number_of_channels, axis=-1)
    csm *= factor[None, ...]
    if scaling.is_amplitude_scaling():
        csm = np.sqrt(csm)
    return csm
