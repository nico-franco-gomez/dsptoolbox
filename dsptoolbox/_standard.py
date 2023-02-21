"""
Backend for standard functions
"""
import numpy as np
from scipy.signal import correlate, check_COLA, windows, hilbert
from ._general_helpers import _pad_trim, _compute_number_frames
from warnings import warn


def _latency(in1: np.ndarray, in2: np.ndarray):
    """Computes the latency between two functions using the correlation method.

    """
    if len(in1.shape) < 2:
        in1 = in1[..., None]
    if len(in2.shape) < 2:
        in2 = in2[..., None]
    latency_per_channel_samples = np.zeros(in1.shape[1], dtype=int)
    for i in range(in1.shape[1]):
        xcorr = correlate(in2[:, i].flatten(), in1[:, i].flatten())
        latency_per_channel_samples[i] = \
            int(in1.shape[0] - np.argmax(abs(xcorr)) - 1)
    return latency_per_channel_samples


def _welch(x, y, fs_hz: int, window_type: str = 'hann',
           window_length_samples: int = 1024, overlap_percent=50,
           detrend: bool = True, average: str = 'mean',
           scaling: str = 'power spectral density') -> np.ndarray:
    """Cross spectral density computation with Welch's method.

    Parameters
    ----------
    x : `np.ndarray`
        First signal
    y : `np.ndarray`
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
    # from time import time
    if type(x) != np.ndarray:
        x = np.array(x).squeeze()
    if type(y) != np.ndarray:
        y = np.array(y).squeeze()
    assert x.shape == y.shape, \
        'Shapes of data do not match'
    assert len(x.shape) < 2, f'{x.shape} are too many dimensions. Use flat' +\
        ' arrays instead'
    valid_window_sizes = np.array([int(2**x) for x in range(4, 17)])
    assert window_length_samples in valid_window_sizes, \
        'Window length should be a power of 2 between [16, 65536] or ' +\
        '[2**4, 2**16]'
    assert overlap_percent >= 0 and overlap_percent < 100, \
        'overlap_percent should be between 0 and 100'
    valid_average = ['mean', 'median']
    assert average in valid_average, f'{average} is not valid. Use ' +\
        'either mean or median'
    valid_scaling = ['power spectrum', 'power spectral density',
                     'amplitude spectrum', 'amplitude spectral density', None]
    assert scaling in valid_scaling, f'{scaling} is not valid. Use ' +\
        'power spectrum, power spectral density, amplitude spectrum, ' +\
        'amplitude spectral density or None'

    # Window and step
    window = windows.get_window(
        window_type, window_length_samples, fftbins=True)
    overlap_samples = int(overlap_percent/100 * window_length_samples)
    step = window_length_samples - overlap_samples

    # Check COLA
    if not check_COLA(window, nperseg=len(window), noverlap=overlap_samples):
        warn('Selected window type and overlap do not meet the constant ' +
             'overlap and add constraint! Results might be distorted')

    # Start Parameters
    n_frames, padding_samp = \
        _compute_number_frames(window_length_samples, step, len(x))
    x = _pad_trim(x, len(x) + padding_samp)
    y = _pad_trim(y, len(y) + padding_samp)
    x_frames = np.zeros((window_length_samples, n_frames), dtype='float')
    y_frames = np.zeros_like(x_frames)

    # Create time frames
    start = 0
    for n in range(n_frames):
        x_frames[:, n] = x[start:start+window_length_samples].copy()
        y_frames[:, n] = y[start:start+window_length_samples].copy()
        start += step

    # Window
    x_frames *= window[..., None]
    y_frames *= window[..., None]

    # Detrend
    if detrend:
        x_frames -= np.mean(x_frames, axis=0)
        y_frames -= np.mean(y_frames, axis=0)

    # Combine
    sp_frames = np.fft.rfft(x_frames, axis=0).conjugate() * \
        np.fft.rfft(y_frames, axis=0)

    # ==== Averaging of magnitude and unwrapped phase much slower...
    # # Get magnitude and phase
    # magnitude = np.abs(sp_frames)
    # phase = np.unwrap(np.angle(sp_frames), axis=0)

    # # mean without first and last arrays
    # if average == 'mean':
    #     magnitude = np.mean(magnitude[:, 1:-1], axis=-1)
    #     phase = np.mean(phase[:, 1:-1], axis=-1)
    # else:
    #     magnitude = np.median(magnitude[:, 1:-1], axis=-1)
    #     phase = np.median(phase[:, 1:-1], axis=-1)

    # # Cross spectral density
    # csd = magnitude * np.exp(1j*phase)

    # Direct averaging much faster
    if average == 'mean':
        csd = np.mean(sp_frames, axis=-1)
    else:
        csd = np.median(sp_frames.real, axis=-1) + 1j * \
            np.median(sp_frames.imag, axis=-1)
        # Bias according to reference
        n = sp_frames.shape[1] if sp_frames.shape[1] % 2 == 1 else \
            sp_frames.shape[1] - 1
        bias = np.sum((-1)**(n+1)/n)
        # This is taken from scipy, it is somewhat different to the paper
        # ii_2 = 2 * np.arange(1., (n-1) // 2 + 1)
        # bias = 1 + np.sum(1. / (ii_2 + 1) - 1. / ii_2)
        csd /= bias

    # Weightning (with 2 because one-sided)
    if scaling in ('power spectrum', 'amplitude spectrum'):
        factor = 2 / np.sum(window)**2
    elif scaling in ('power spectral density', 'amplitude spectral density'):
        factor = 2 / (window @ window) / fs_hz
        # With this factor, energy can be regained by integrating the psd
        # while taking into account the frequency step
        # =====================================
        # This is not consistent with scipy or the reference paper but does
        # arrive at the right energy when summing over the psd (without taking
        # frequency step into account)
        # factor = 2 / (window @ window) / window_length_samples
    else:
        factor = 1

    # Zero frequency fix when detrending
    if detrend:
        csd[0] = csd[1]

    csd *= factor
    csd[0] /= 2
    csd[-1] /= 2

    if 'amplitude' in scaling:
        csd = np.sqrt(csd)

    # Cast to real output if there is no imaginary part
    if np.all(csd.imag == 0):
        csd = csd.real

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
        gd = -np.gradient(np.unwrap(phase), delta_f)/np.pi/2
    else:
        gd = -np.gradient(np.unwrap(phase))
    return gd


def _minimum_phase(magnitude: np.ndarray, unwrapped: bool = True):
    """Computes minimum phase system from magnitude spectrum.

    Parameters
    ----------
    magnitude : `np.ndarray`
        Spectrum for which to compute the minimum phase. If real, it is assumed
        to be already the magnitude.
    uwrapped : bool, optional
        If `True`, the unwrapped phase is given. Default: `True`.

    Returns
    -------
    minimum_phase : `np.ndarray`
        Minimal phase of the system.

    """
    if np.iscomplexobj(magnitude):
        magnitude = np.abs(magnitude)
    minimum_phase = -np.imag(hilbert(np.log(np.clip(
        magnitude, a_min=1e-25, a_max=None)), axis=0))
    # Scale to pi
    minimum_phase = minimum_phase / np.max(np.abs(minimum_phase)) * np.pi
    if not unwrapped:
        minimum_phase = np.angle(np.exp(1j*minimum_phase))
    return minimum_phase


def _stft(x: np.ndarray, fs_hz: int, window_length_samples: int = 2048,
          window_type: str = 'hann', overlap_percent=50,
          detrend: bool = True, padding: bool = True, scaling: bool = False):
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
    detrend : bool, optional
        Detrending from each time segment (removing mean). Default: True.
    padding : bool, optional
        When `True`, the original signal is padded in the beginning and ending
        so that no energy is lost due to windowing. Default: `True`.
    scaling : bool, optional
        When `True`, the output is scaled as an amplitude spectrum, otherwise
        no scaling is applied. See references for details. Default: `False`.

    Returns
    -------
    time_s : `np.ndarray`
        Time vector in seconds for each frame.
    freqs_hz : `np.ndarray`
        Frequency vector.
    stft : `np.ndarray`
        STFT matrix with shape (frequency, time).

    References
    ----------
    - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and spectral
      density estimation by the Discrete Fourier transform (DFT), including a
      comprehensive list of window functions and some new at-top windows.

    """
    valid_window_sizes = np.array([int(2**x) for x in range(4, 17)])
    assert window_length_samples in valid_window_sizes, \
        'Window length should be a power of 2 between [16, 65536] or ' +\
        '[2**4, 2**16]'
    assert overlap_percent >= 0 and overlap_percent < 100, 'overlap_percent' +\
        ' should be between 0 and 100'

    # Window and step
    window = \
        windows.get_window(window_type, window_length_samples, fftbins=True)
    overlap_samples = int(overlap_percent/100 * window_length_samples)
    step = window_length_samples - overlap_samples

    # Check COLA
    if not check_COLA(window, nperseg=len(window), noverlap=overlap_samples):
        warn('Selected window type and overlap do not meet the constant ' +
             'overlap and add constraint! Results might be distorted')

    # Padding
    if padding:
        x = _pad_trim(x, len(x)+overlap_samples, in_the_end=False)
        x = _pad_trim(x, len(x)+overlap_samples, in_the_end=True)

    # Number of samples and padding
    n_frames, padding_samp = \
        _compute_number_frames(window_length_samples, step, len(x))

    x = _pad_trim(x, len(x) + padding_samp)
    time_x = np.zeros((window_length_samples, n_frames), dtype='float')

    # Create time frames
    start = 0
    for n in range(n_frames):
        time_x[:, n] = x[start:start+window_length_samples].copy()
        start += step
    # Windowing
    time_x *= window[..., None]
    # Detrend
    if detrend:
        time_x -= np.mean(time_x, axis=0)
    # Spectra
    stft = np.fft.rfft(time_x, axis=0)
    # Scaling
    if scaling:
        factor = np.sqrt(2 / np.sum(window)**2)
        # factor = 2 / np.sum(window**2) / window_length_samples
        # factor = 2 / np.sum(window)**2 / fs_hz * window_length_samples
    else:
        factor = 1
    stft *= factor

    time_s = np.linspace(0, len(x)/fs_hz, stft.shape[1])
    freqs_hz = np.fft.rfftfreq(len(window), 1/fs_hz)
    return time_s, freqs_hz, stft


def _csm(time_data: np.ndarray, sampling_rate_hz: int,
         window_length_samples: int = 1024, window_type: str = 'hann',
         overlap_percent: int = 50, detrend: bool = True,
         average: str = 'mean', scaling: str = 'power'):
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
    number_of_channels = time_data.shape[1]
    csm = np.zeros((window_length_samples//2+1,
                    number_of_channels,
                    number_of_channels), dtype='cfloat')
    for ind1 in range(number_of_channels):
        for ind2 in range(ind1, number_of_channels):
            # Complex conjugate second signal and not first (like transposing
            # the matrix)
            # csm[:, ind1, ind2] = \
            csm[:, ind2, ind1] = \
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
                csm[:, ind1, ind2] *= 0.5
    csm += np.swapaxes(csm, 1, 2).conjugate()
    f = np.fft.rfftfreq(window_length_samples, 1/sampling_rate_hz)
    return f, csm


# =============================================================================
# It was shown that this helper class was not faster than the _csm function.
# Vectorizing some of the computations did not seem to bring any advantages
# to the speed. Especially, np.fft.rfft was a little slower and np.unwrap was
# much slower when applied to large matrices along specific dimensions...
# =============================================================================
# class _CSMHelper():
#     """This is a helper class to compute the csm in an efficient manner.

#     """
#     def __init__(self, time_data: np.ndarray, sampling_rate_hz: int,
#                  window_length_samples: int = 1024,
#                  window_type: str = 'hann', overlap_percent: int = 50,
#                  detrend: bool = True, average: str = 'mean',
#                  scaling: str = 'power'):
#         """Computes the cross spectral matrix of a multichannel signal.
#         Output matrix has (frequency, channels, channels).

#         Parameters
#         ----------
#         time_data : `np.ndarray`
#             Signal
#         sampling_rate_hz : int
#             Sampling rate in Hz.
#         window_length_samples : int, optional
#             Window length to be used. Determines frequency resolution in the
#             end. Only powers of 2 are accepted. Default: 1024.
#         window_type : str, optional
#             Window type to be used. Refer to scipy.signal.windows for
#             available ones. Default: `'hann'`
#         overlap_percent : int, optional
#             Overlap in percentage. Default: 50.
#         detrend : bool, optional
#             Detrending from each time segment (removing mean). Default: True.
#         average : str, optional
#             Type of mean to be computed. Take `'mean'` or `'np.median'`.
#             Default: `'mean'`
#         scaling : str, optional
#             Scaling for spectral power density or spectrum. Takes `'power'`
#             or `'spectrum'`. Default: `'power'`.

#         Attributes
#         ----------
#         All passed parameters are saved as attributes.

#         - `get_csm()`: Method to trigger the computation of csm.

#         """
#         assert time_data.shape[1] > 1, \
#             'There has to be at least 2 channels to compute the csm'
#         valid_window_sizes = np.array([int(2**x) for x in range(7, 17)])
#         assert window_length_samples in valid_window_sizes, \
#             'Window length should be a power of 2 between [128, 65536] ' +\
#             'or [2**7, 2**16]'
#         assert overlap_percent > 0 and overlap_percent < 100, \
#             'overlap_percent should be between 0 and 100'
#         valid_average = ['mean', 'median']
#         assert average in valid_average, f'{average} is not valid. Use ' +\
#             'either mean or median'
#         valid_scaling = ['power', 'spectrum']
#         assert scaling in valid_scaling, f'{scaling} is not valid. Use ' +\
#             'either power or spectrum'

#         # Window and step
#         window = windows.get_window(
#             window_type, window_length_samples, fftbins=True)
#         overlap_samples = int(overlap_percent/100 * window_length_samples)
#         step = window_length_samples - overlap_samples

#         # Check COLA
#         assert check_COLA(
#             window, nperseg=len(window), noverlap=overlap_samples),\
#             'Selected window type and overlap do not meet the constant ' +\
#             'overlap and add constraint. Please select other.'

#         # Save all attributes in object
#         self.time_data = time_data
#         self.window = window
#         self.window_length_samples = window_length_samples
#         self.sampling_rate_hz = sampling_rate_hz
#         self.overlap_samples = overlap_samples
#         self.step = step
#         self.detrend = detrend

#         if average == 'mean':
#             self.average_func = np.mean
#         else:
#             self.average_func = np.median

#         if scaling == 'power':
#             self.factor = 2 / (window @ window) / sampling_rate_hz
#         else:
#             self.factor = 2 / sum(window)**2 / sampling_rate_hz
#         self.f_vec = \
#             np.fft.rfftfreq(window_length_samples, 1/sampling_rate_hz)

#     def _compute_framed_spectra(self):
#         """Computes the framed spectra.

#         """
#         # Compute number of needed frames and padding
#         original_length = self.time_data.shape[0]
#         n_frames, padding_samp = _compute_number_frames(
#             self.window_length_samples, self.step, original_length)
#         self.time_data = _pad_trim(
#             self.time_data, original_length + padding_samp)
#         td_frames = np.zeros(
#             (self.window_length_samples, n_frames, self.time_data.shape[1]),
#             dtype='float')
#         self.n_frames = n_frames

#         # Create time frames (time samples, frame, channel)
#         start = 0
#         for n in range(n_frames):
#             td_frames[:, n, :] = \
#                 self.time_data[
#                     start:start+self.window_length_samples, :].copy()
#             start += self.step

#         # Window
#         td_frames *= self.window[:, np.newaxis, np.newaxis]
#         # Detrend
#         if self.detrend:
#             td_frames -= np.mean(td_frames, axis=0)
#         # Spectra (frequency bins, frames, channel)
#         self.sp_frames = np.fft.rfft(td_frames, axis=0)

#     def get_csm(self):
#         """Computes and returns the CSM.

#         Returns
#         -------
#         f : `np.ndarray`
#             Frequency vector.
#         csm : `np.ndarray`
#             Cross-spectral density matrix with shape
#             (frequency, channel, channel).

#         """
#         self._compute_framed_spectra()
#         number_of_channels = self.time_data.shape[1]
#         csm_framed = np.zeros((self.window_length_samples//2+1,
#                                self.n_frames,
#                                number_of_channels,
#                                number_of_channels), dtype=np.complex64)

#         # Maybe second loop can be vectorized by using np.roll along one
#         # channel axis. There would be unnecessary computations but it
#         # might be faster
#         for ind1 in range(number_of_channels):
#             for ind2 in range(ind1, number_of_channels):
#                 csm_framed[:, :, ind1, ind2] = \
#                     np.conjugate(self.sp_frames[:, :, ind1]) * \
#                     self.sp_frames[:, :, ind2]
#                 if ind1 == ind2:
#                     csm_framed[:, :, ind1, ind2] *= 0.5

#         # Get magnitude and phase
#         magnitude = np.abs(csm_framed)
#         # phase = np.angle(csm_framed)
#         phase = np.unwrap(np.angle(csm_framed), axis=0)
#         del csm_framed
#         # Average along frames (axis=1)
#         magnitude = self.average_func(magnitude, axis=1)
#         phase = self.average_func(phase, axis=1)

#         # Redo matrix with factor
#         csm = magnitude * np.exp(1j*phase) * self.factor
#         # csm = np.mean(csm_framed, axis=1) * self.factor
#         # Complete lower triangle csm matrix
#         csm += np.swapaxes(csm, 1, 2).conjugate()
#         return self.f_vec, csm


def _center_frequencies_fractional_octaves_iec(nominal, num_fractions):
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
        nominal = np.array([
            31.5, 63, 125, 250, 500, 1e3,
            2e3, 4e3, 8e3, 16e3], dtype=float)
    elif num_fractions == 3:
        nominal = np.array([
            25, 31.5, 40, 50, 63, 80, 100, 125, 160,
            200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000,
            6300, 8000, 10000, 12500, 16000, 20000], dtype=float)

    reference_freq = 1e3
    octave_ratio = 10**(3/10)

    iseven = np.mod(num_fractions, 2) == 0
    if ~iseven:
        indices = np.around(
            num_fractions * np.log(nominal/reference_freq)
            / np.log(octave_ratio))
        exponent = (indices/num_fractions)
    else:
        indices = np.around(
            2.0*num_fractions *
            np.log(nominal/reference_freq) / np.log(octave_ratio) - 1)/2
        exponent = ((2*indices + 1) / num_fractions / 2)

    exact = reference_freq * octave_ratio**exponent

    return nominal, exact


def _exact_center_frequencies_fractional_octaves(
        num_fractions, frequency_range):
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
    Nmax = np.around(num_fractions*(np.log2(frequency_range[1]/ref_freq)))
    Nmin = np.around(num_fractions*(np.log2(ref_freq/frequency_range[0])))

    indices = np.arange(-Nmin, Nmax+1)
    exact = ref_freq * 2**(indices / num_fractions)

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
        beta = 0.1102 * (A - 8.7)
    elif A >= 21:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.0

    return beta


def _indexes_above_threshold_dbfs(time_vec: np.ndarray, threshold_dbfs: float,
                                  attack_samples: int, release_samples: int):
    """Returns indexes with power above a passed threshold (in dBFS) in a time
    series. time_vec is normalized prior to computation.

    Parameters
    ----------
    time_vec : `np.ndarray`
        Time series for which to find indexes above power threshold.
    threshold_dbfs : float
        Threshold to be used.
    attack_samples : int
        Number of samples representing attack time signal has surpassed
        power threshold.
    release_samples : int
        Number of samples representing release time after signal has decayed
        below power threshold.

    Returns
    -------
    indexes_above : `np.ndarray`
        Array of type boolean with length of time_vec indicating indexes
        above threshold with `True` and below with `False`.

    """
    time_vec = np.asarray(time_vec).squeeze()
    assert time_vec.ndim == 1, \
        'Function is implemented for 1D-arrays only'

    # Find peak value index
    max_ind = np.argmax(np.abs(time_vec))

    # Power in dB
    time_power = 20*np.log10(np.abs(time_vec))

    # Normalization
    time_power -= time_power[max_ind]

    # All indexes above threshold
    indexes_above_0 = time_power > threshold_dbfs
    indexes_above = np.zeros_like(indexes_above_0).astype(bool)

    # Apply release and attack
    for ind in range(len(indexes_above)):
        # Attack after certain amount of samples surpass threshold
        ind_attack = 0 if ind-attack_samples < 0 else ind-attack_samples
        if np.all(indexes_above_0[ind_attack:ind]):
            indexes_above[ind:ind+release_samples] = True
    return indexes_above
