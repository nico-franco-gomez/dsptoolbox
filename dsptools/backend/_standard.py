"""
Backend for standard functions
"""
from numpy import (ndarray, zeros, argmax, array, zeros_like, mean, fft,
                   conjugate, unwrap, angle, median, exp, iscomplexobj,
                   gradient, pi, imag, log, linspace, complex64)
from scipy.signal import correlate, check_COLA, windows, hilbert
from ._general_helpers import _pad_trim, _compute_number_frames


def _latency(in1: ndarray, in2: ndarray):
    """Computes the latency between two functions using the correlation method.
    """
    if len(in1.shape) < 2:
        in1 = in1[..., None]
    if len(in2.shape) < 2:
        in2 = in2[..., None]
    latency_per_channel_samples = zeros(in1.shape[1], dtype=int)
    for i in range(in1.shape[1]):
        xcorr = correlate(in2[:, i].flatten(), in1[:, i].flatten())
        latency_per_channel_samples[i] = \
            int(in1.shape[0] - argmax(abs(xcorr)) - 1)
    return latency_per_channel_samples


def _welch(x, y, fs_hz, window_type: str = 'hann',
           window_length_samples: int = 1024, overlap_percent=50,
           detrend: bool = True, average: str = 'mean',
           scaling: str = 'power'):
    """Cross spectral density computation with Welch's method.

    Parameters
    ----------
    x : ndarray
        First signal
    y : ndarray
        Second signal
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
        Scaling for spectral power density or spectrum. Takes `'power'` or
        `'spectrum'`. Default: `'power'`.

    Returns
    -------
    csd : ndarray
        Cross spectral density vector. Complex-valued if x and y are different.
    """
    if type(x) != ndarray:
        x = array(x).squeeze()
    if type(y) != ndarray:
        y = array(y).squeeze()
    assert x.shape == y.shape, \
        'Shapes of data do not match'
    assert len(x.shape) < 2, f'{x.shape} are too many dimensions. Use flat' +\
        ' arrays instead'
    valid_window_sizes = array([int(2**x) for x in range(7, 17)])
    assert window_length_samples in valid_window_sizes, \
        'Window length should be a power of 2 between [128, 65536] or ' +\
        '[2**7, 2**16]'
    assert overlap_percent > 0 and overlap_percent < 100, 'overlap_percent ' +\
        'should be between 0 and 100'
    valid_average = ['mean', 'median']
    assert average in valid_average, f'{average} is not valid. Use ' +\
        'either mean or median'
    valid_scaling = ['power', 'spectrum']
    assert scaling in valid_scaling, f'{scaling} is not valid. Use ' +\
        'either power or spectrum'

    # Window and step
    window = \
        windows.get_window(window_type, window_length_samples, fftbins=True)
    overlap_samples = int(overlap_percent/100 * window_length_samples)
    step = window_length_samples - overlap_samples

    # Check COLA
    assert check_COLA(window, nperseg=len(window), noverlap=overlap_samples),\
        'Selected window type and overlap do not meet the constant overlap ' +\
        'and add constraint. Please select other.'

    # Start Parameters
    n_frames, padding_samp = \
        _compute_number_frames(window_length_samples, step, len(x))
    x = _pad_trim(x, len(x) + padding_samp)
    y = _pad_trim(y, len(y) + padding_samp)
    magnitude = zeros((window_length_samples//2+1, n_frames), dtype='float')
    phase = zeros_like(magnitude)

    start = 0
    for n in range(n_frames):
        time_x = x[start:start+window_length_samples].copy()
        time_y = y[start:start+window_length_samples].copy()
        # Windowing
        time_x *= window
        time_y *= window
        # Detrend
        if detrend:
            time_x -= mean(time_x)
            time_y -= mean(time_y)
        # Spectra
        sp_x = fft.rfft(time_x)
        sp_y = fft.rfft(time_y)
        m = conjugate(sp_x) * sp_y
        magnitude[:, n] = abs(m)
        phase[:, n] = unwrap(angle(m))
        start += step

    # Mean without first and last arrays
    if average == 'mean':
        magnitude = mean(magnitude[:, 1:-2], axis=-1)
        phase = mean(phase[:, 1:-2], axis=-1)
    else:
        magnitude = median(magnitude[:, 1:-2], axis=-1)
        phase = median(phase[:, 1:-2], axis=-1)

    # Cross spectral density
    if not all(x == y):
        csd = magnitude * exp(1j*phase)
    else:
        csd = magnitude

    # Zero frequency fix when detrending
    if detrend:
        csd[0] = csd[1]

    # Weightning (with 2 because one-sided)
    if scaling == 'power':
        factor = 2 / (window @ window) / fs_hz
    else:
        factor = 2 / sum(window)**2 / fs_hz
    csd[1:] = csd[1:]*factor
    return csd


def _group_delay_direct(phase: ndarray, delta_f: float = 1):
    """Computes group delay by differentiation of the unwrapped phase.

    Parameters
    ----------
    phase : ndarray
        Complex spectrum or phase for the direct method
    delta_f : float, optional
        Frequency step for the phase. If it equals 1, group delay is computed
        in samples and not in seconds. Default: 1.

    Returns
    -------
    gd : ndarray
        Group delay vector either in s or in samples if no
        sampling rate is given
    """
    if iscomplexobj(phase):
        phase = angle(phase)
    if delta_f != 1:
        gd = -gradient(unwrap(phase), delta_f)/pi/2
    else:
        gd = -gradient(unwrap(phase))
    return gd


def _minimal_phase(magnitude: ndarray, unwrapped: bool = True):
    """Computes minimal phase from magnitude spectrum.

    Parameters
    ----------
    magnitude : ndarray
        Spectrum for which to compute the minimal phase. If real, it is assumed
        to be already the magnitude.
    uwrapped : bool, optional
        If `True`, the unwrapped phase is given. Default: `True`.

    Returns
    -------
    minimal_phase : ndarray
        Minimal phase of the system.
    """
    if iscomplexobj(magnitude):
        magnitude = abs(magnitude)
    minimal_phase = -imag(hilbert(log(magnitude)))
    if not unwrapped:
        minimal_phase = angle(exp(1j*minimal_phase))
    return minimal_phase


def _stft(x: ndarray, fs_hz: int, window_length_samples: int = 2048,
          window_type: str = 'hann', overlap_percent=50,
          detrend: bool = True, padding: bool = True, scaling: bool = False):
    """Computes the STFT of a signal. Output matrix has (freqs_hz, seconds_s).

    Parameters
    ----------
    x : ndarray
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
    detrend : bool, optional
        Detrending from each time segment (removing mean). Default: True.
    padding : bool, optional
        When `True`, the original signal is padded in the beginning and ending
        so that no energy is lost due to windowing. Default: `True`.
    scaling : bool, optional
        When `True`, the output is scaled with sampling rate. Default: `False`.

    Returns
    -------
    time_s : ndarray
        Time vector in seconds for each frame.
    freqs_hz : ndarray
        Frequency vector.
    stft : ndarray
        STFT matrix.
    """
    valid_window_sizes = array([int(2**x) for x in range(7, 17)])
    assert window_length_samples in valid_window_sizes, \
        'Window length should be a power of 2 between [128, 65536] or ' +\
        '[2**7, 2**16]'
    assert overlap_percent > 0 and overlap_percent < 100, 'overlap_percent ' +\
        'should be between 0 and 100'

    # Window and step
    window = \
        windows.get_window(window_type, window_length_samples, fftbins=True)
    overlap_samples = int(overlap_percent/100 * window_length_samples)
    step = window_length_samples - overlap_samples

    # Check COLA
    assert check_COLA(window, nperseg=len(window), noverlap=overlap_samples),\
        'Selected window type and overlap do not meet the constant overlap ' +\
        'and add constraint. Please select other.'

    # Padding
    if padding:
        x = _pad_trim(x, len(x)+overlap_samples, in_the_end=False)
        x = _pad_trim(x, len(x)+overlap_samples, in_the_end=True)

    # Number of samples and padding
    n_frames, padding_samp = \
        _compute_number_frames(window_length_samples, step, len(x))

    x = _pad_trim(x, len(x) + padding_samp)
    stft = zeros((window_length_samples//2+1, n_frames), dtype='cfloat')

    start = 0
    for n in range(n_frames):
        time_x = x[start:start+window_length_samples].copy()
        # Windowing
        time_x *= window
        # Detrend
        if detrend:
            time_x -= mean(time_x)
        # Spectra
        stft[:, n] = fft.rfft(time_x)
        start += step

    # Scaling
    if scaling:
        factor = 2 / sum(window)**2
    else:
        factor = 1
    stft *= factor

    time_s = linspace(0, len(x)/fs_hz, stft.shape[1])
    freqs_hz = fft.rfftfreq(len(window), 1/fs_hz)
    return time_s, freqs_hz, stft


def _csm(time_data: ndarray, sampling_rate_hz: int,
         window_length_samples: int = 1024, window_type: str = 'hann',
         overlap_percent: int = 50, detrend: bool = True,
         average: str = 'mean', scaling: str = 'power'):
    """Computes the cross spectral matrix of a multichannel signal.
    Output matrix has (frequency, channels, channels).

    Parameters
    ----------
    time_data : ndarray
        Signal
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
    detrend : bool, optional
        Detrending from each time segment (removing mean). Default: True.
    average : str, optional
        Type of mean to be computed. Take `'mean'` or `'median'`.
        Default: `'mean'`
    scaling : str, optional
        Scaling for spectral power density or spectrum. Takes `'power'` or
        `'spectrum'`. Default: `'power'`.

    Returns
    -------
    f : ndarray
        Frequency vector
    csm : ndarray
        Cross spectral matrix with shape (frequency, channels, channels).
    """
    number_of_channels = time_data.shape[1]
    csm = zeros((window_length_samples//2+1,
                 number_of_channels,
                 number_of_channels), dtype=complex64)

    for ind1 in range(number_of_channels):
        for ind2 in range(ind1, number_of_channels):
            csm[:, ind1, ind2] = \
                _welch(time_data[:, ind1],
                       time_data[:, ind2],
                       sampling_rate_hz,
                       window_length_samples=window_length_samples,
                       window_type=window_type,
                       overlap_percent=overlap_percent,
                       detrend=detrend,
                       average=average,
                       scaling=scaling)
            if ind1 == ind2:
                csm[:, ind1, ind2] /= 2
    for nfreq in range(csm.shape[0]):
        csm[nfreq, :, :] = \
            csm[nfreq, :, :] + csm[nfreq, :, :].T.conjugate()
    f = fft.rfftfreq(
        window_length_samples,
        1/sampling_rate_hz)
    return f, csm
