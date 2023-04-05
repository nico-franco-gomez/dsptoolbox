"""
Here are methods considered as somewhat special or less common.
"""
from dsptoolbox.classes.signal_class import Signal
from dsptoolbox.plots import general_matrix_plot
from dsptoolbox._standard import _reconstruct_framed_signal
from dsptoolbox._general_helpers import _hz2mel, _mel2hz, _pad_trim

import numpy as np
from scipy.signal.windows import get_window
from scipy.fft import dct
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from seaborn import set_style
set_style('whitegrid')


def cepstrum(signal: Signal, mode='power') -> np.ndarray:
    """Returns the cepstrum of a given signal in the Quefrency domain.

    Parameters
    ----------
    signal : Signal
        Signal to compute the cepstrum from.
    mode : str, optional
        Type of cepstrum. Supported modes are `'power'`, `'real'` and
        `'complex'`. Default: `'power'`.

    Returns
    -------
    ceps : `np.ndarray`
        Cepstrum.

    References
    ----------
    https://de.wikipedia.org/wiki/Cepstrum

    """
    mode = mode.lower()
    assert mode in ('power', 'complex', 'real'), \
        f'{mode} is not a supported mode'

    signal.set_spectrum_parameters(method='standard')
    _, sp = signal.get_spectrum()

    if mode in ('power', 'real'):
        ceps = np.abs(np.fft.irfft((2*np.log(np.abs(sp))), axis=0))**2
    else:
        phase = np.unwrap(np.angle(sp), axis=0)
        ceps = np.fft.irfft(np.log(np.abs(sp)) + 1j*phase, axis=0).real
    if mode == 'real':
        ceps = (ceps**0.5)/2
    return ceps


def log_mel_spectrogram(s: Signal, channel: int = 0, range_hz=None,
                        n_bands: int = 40, generate_plot: bool = True,
                        stft_parameters: dict = None):
    """Returns the log mel spectrogram of the specific signal and channel.

    Parameters
    ----------
    s : `Signal`
        Signal to generate the spectrogram.
    channel : int, optional
        Channel of the signal to be used for the plot generation. Only one
        channel can be passed. Default: 0.
    range_hz : array-like with length 2, optional
        Range of frequencies to use. Pass `None` to analyze the whole spectrum.
        Default: `None`.
    n_bands : int, optional
        Number of mel bands to generate. Default: 40.
    generate_plot : bool, optional
        Plots the obtained results. Use ``dsptoolbox.plots.show()`` to show
        the plot. Default: `True`.
    stft_parameters : dict, optional
        Pass arguments to define computation of STFT. If `None` is passed, the
        parameters already set in the signal will be used. Refer to
        `Signal.set_spectrogram_parameters()` for details. Default: `None`.

    Returns
    -------
    time_s : `np.ndarray`
        Time vector.
    f_mel : `np.ndarray`
        Frequency vector in Mel.
    log_mel_sp : `np.ndarray`
        Log mel spectrogram with shape (frequency, time frame, channel).

    When `generate_plot=True`:

    time_s : `np.ndarray`
        Time vector.
    f_mel : `np.ndarray`
        Frequency vector in Mel.
    log_mel_sp : `np.ndarray`
        Log mel spectrogram with shape (frequency, time frame, channel).
    fig : `matplotlib.figure.Figure`
        Figure.
    ax : `matplotlib.axes.Axes`
        Axes.

    """
    if stft_parameters is not None:
        s.set_spectrogram_parameters(**stft_parameters)
    time_s, f_hz, sp = s.get_spectrogram()
    mfilt, f_mel = mel_filterbank(f_hz, range_hz, n_bands, normalize=True)
    log_mel_sp = np.tensordot(mfilt, np.abs(sp), axes=[-1, 0])
    log_mel_sp = 20*np.log10(np.clip(log_mel_sp, a_min=1e-20, a_max=None))
    if generate_plot:
        fig, ax = general_matrix_plot(
            log_mel_sp[..., channel], range_x=[time_s[0], time_s[-1]],
            range_y=[f_mel[0], f_mel[-1]], range_z=50,
            ylabel='Frequency / Mel', xlabel='Time / s',
            ylog=False, returns=True)
        return time_s, f_mel, log_mel_sp, fig, ax
    return time_s, f_mel, log_mel_sp


def mel_filterbank(f_hz: np.ndarray, range_hz=None, n_bands: int = 40,
                   normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Creates equidistant mel triangle filters in a given range. The returned
    matrix can be used to convert Hz into Mel in a spectrogram.

    NOTE: This is not a filter bank in the usual sense, thus it does not create
    a FilterBank object to be applied to a signal. Its intended use is in the
    frequency domain.

    Parameters
    ----------
    f_hz : `np.ndarray`
        Frequency vector.
    range_hz : array-like with length 2, optional
        Range (in Hz) in which to create the filters. If `None`, the whole
        available spectrum is used. Default: `None`.
    n_bands : int, optional
        Number of bands to create. Default: 40.
    normalize : bool, optional
        When `True`, the bands are area normalized for preserving approximately
        same energy in each band. Default: `True`.

    Returns
    -------
    mel_filters : `np.ndarray`
        Mel filters matrix with shape (bands, frequency).
    mel_center_freqs : `np.ndarray`
        Vector containing mel center frequencies.

    """
    f_hz = np.squeeze(f_hz)
    assert f_hz.ndim == 1, \
        'f_hz should be a 1D-array'
    n_bands = int(n_bands)

    # Create range
    if range_hz is None:
        range_hz = f_hz[[0, -1]]
    else:
        range_hz = np.atleast_1d(np.asarray(range_hz).squeeze())
        assert len(range_hz) == 2, \
            'range_hz should be an array with exactly two values!'
        range_hz = np.sort(range_hz)
        assert range_hz[-1] <= f_hz[-1], \
            f'Upper frequency in range {range_hz[-1]} is bigger than ' +\
            f'nyquist frequency {f_hz[-1]}'
        assert range_hz[0] >= 0, \
            'Lower frequency in range must be positive'

    # Compute band center frequencies in mel
    range_mel = _hz2mel(range_hz)
    mel_center_freqs = np.linspace(
        range_mel[0], range_mel[1], n_bands+2, endpoint=True)

    # Center frequencies in Hz
    bands_hz = _mel2hz(mel_center_freqs)

    # Find indexes for frequencies
    inds = np.empty_like(bands_hz, dtype=int)
    for ind, b in enumerate(bands_hz):
        inds[ind] = np.argmin(np.abs(b - f_hz))

    # Create triangle filters
    mel_filters = np.zeros((n_bands, len(f_hz)))
    for n in range(n_bands):
        ni = n+1
        mel_filters[n, inds[ni-1]:inds[ni]] = \
            np.linspace(0, 1, inds[ni]-inds[ni-1], endpoint=False)
        mel_filters[n, inds[ni]:inds[ni+1]] = \
            np.linspace(1, 0, inds[ni+1] - inds[ni], endpoint=False)
        if normalize:
            mel_filters[n, :] /= np.sum(mel_filters[n, :])
    return mel_filters, mel_center_freqs[1:-1]


def plot_waterfall(sig: Signal, channel: int = 0,
                   dynamic_range_db: float = 40,
                   stft_parameters: dict = None) -> tuple[Figure, Axes]:
    """Generates and returns a waterfall plot from a signal. The settings
    for the spectrogram saved in the signal are the ones used for the plot
    generation.

    Parameters
    ----------
    sig : `Signal`
        Signal to plot waterfall diagramm for.
    channel : int, optional
        Channel to take for the waterfall plot.
    dynamic_range_db : float, optional
        Sets the maximum dynamic range in dB to show in the plot. Pass `None`
        to avoid setting any dynamic range. Default: 40.
    stft_parameters : dict, optional
        Dictionary containing settings for the stft. If `None` is passed,
        the parameters already set in `Signal` object are used. Refer to
        `Signal.set_spectrogram_parameters()` for details. Default: `None`.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure.
    ax : `matplotlib.axes.Axes`
        Axes.

    """
    assert dynamic_range_db > 0, \
        'Dynamic range has to be more than 0'
    sig = sig.get_channels(channel)
    if stft_parameters is not None:
        sig.set_spectrogram_parameters(**stft_parameters)
    t, f, stft = sig.get_spectrogram()

    stft = np.abs(stft[..., 0])
    z_label_extra = ''
    if dynamic_range_db is not None:
        stft /= np.max(stft)
        clip_val = 10**(-dynamic_range_db/20)
        stft = np.clip(stft, a_min=clip_val, a_max=None)
        z_label_extra = 'FS (normalized @ peak)'

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='3d'))
    tt, ff = np.meshgrid(t, f)
    ax.plot_surface(tt, ff, 20*np.log10(stft), cmap='magma')
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Frequency / Hz')
    ax.set_zlabel('dB'+z_label_extra)
    fig.tight_layout()
    # fig.colorbar(surface, ax=ax, shrink=0.4, aspect=10)
    return fig, ax


def mfcc(signal: Signal, channel: int = 0,
         mel_filters: np.ndarray = None, generate_plot: bool = True,
         stft_parameters: dict = None):
    """Mel-frequency cepstral coefficients for a windowed signal are computed
    and returned using the discrete cosine transform of type 2 (see
    `scipy.fft.dct` for more details).

    Parameters
    ----------
    signal : `Signal`
        The signal for which to compute the mel-frequency cepstral
        coefficients.
    channel : int, optional
        Channel of the signal for which to plot the MFCC when
        `generate_plot=True`. Default: 0.
    mel_filters : `np.ndarray`, optional
        Hz-to-Mel transformation matrix with shape (mel band, frequency Hz).
        It can be created using `mel_filterbank`. If `None` is passed, the
        filters are automatically computed regarding the whole
        available spectrum and dividing it in 40 bands (with normalized
        amplitudes for energy preserving filters, see `mel_filterbank` for
        details). Default: `None`.
    generate_plot : bool, optional
        When `True`, a plot of the MFCC is generated and returned.
        Default: `True`.
    stft_parameters : dict, optional
        Pass arguments to define computation of STFT. If `None` is passed, the
        parameters already set in the signal will be used. Refer to
        `Signal.set_spectrogram_parameters()` for details. Default: `None`.

    Returns
    -------
    time_s : `np.ndarray`
        Time vector.
    f_mel : `np.ndarray`
        Frequency vector in mel. If `mel_filters` is passed, this is only a
        list with entries [0, n_mel_filters].
    mfcc : `np.ndarray`
        Mel-frequency cepstral coefficients with shape (cepstral coefficients,
        time frame, channel).

    When `generate_plot=True`:

    time_s : `np.ndarray`
        Time vector.
    f_mel : `np.ndarray`
        Frequency vector in mel. If `mel_filters` is passed, this is only a
        list with entries [0, n_mel_filters].
    mfcc : `np.ndarray`
        Mel-frequency cepstral coefficients with shape (cepstral coefficients,
        time frame, channel).
    fig : `matplotlib.figure.Figure`
        Figure.
    ax : `matplotlib.axes.Axes`
        Axes.

    """
    if stft_parameters is not None:
        signal.set_spectrogram_parameters(**stft_parameters)
    time_s, f, sp = signal.get_spectrogram()

    # Get Log power spectrum
    log_sp = 2*np.log(np.abs(sp))

    # Mel filters
    if mel_filters is None:
        mel_filters, f_mel = mel_filterbank(f, None, n_bands=40)
    else:
        assert mel_filters.shape[1] == log_sp.shape[0], \
            f'Shape of the mel filter matrix {mel_filters.shape} does ' +\
            f'not match the STFT {log_sp.shape}'
        f_mel = [0, mel_filters.shape[0]]

    # Convert from Hz to Mel
    log_sp = np.tensordot(mel_filters, log_sp, axes=[-1, 0])

    # Discrete cosine transform
    mfcc = np.abs(dct(log_sp, type=2, axis=0))

    # Prune nans
    np.nan_to_num(mfcc, copy=False, nan=0)

    # Plot and return
    if generate_plot:
        fig, ax = general_matrix_plot(
            mfcc[..., channel], range_x=[time_s[0], time_s[-1]],
            range_y=[f_mel[0], f_mel[-1]],
            xlabel='Time / s', ylabel='Cepstral coefficients', returns=True)
        return time_s, f_mel, mfcc, fig, ax
    return time_s, f_mel, mfcc


def istft(stft: np.ndarray, original_signal: Signal = None,
          parameters: dict = None, sampling_rate_hz: int = None,
          window_length_samples: int = None, window_type: str = None,
          overlap_percent: int = None, fft_length_samples: int = None,
          padding: bool = None, scaling: bool = None) -> Signal:
    """This function transforms a complex STFT back into its respective time
    signal using the method presented in [1]. For this to be possible, it is
    necessary to know the parameters that were used while converting the signal
    into its STFT representation. A dictionary containing the parameters
    corresponding can be passed, as well as the original `Signal` in which
    these parameters are saved. Alternatively, it is possible to pass them
    explicitely.

    Parameters
    ----------
    stft : `np.ndarray`
        Complex STFT with shape (frequency, time frame, channel). It is assumed
        that only positive frequencies (including 0) are present.
    original_signal : `Signal`, optional
        Initial signal from which the STFT matrix was generated.
        Default: `None`.
    parameters : dict, optional
        Dictionary containing the parameters used to compute the STFT matrix.
        Default: `None`.
    sampling_rate_hz : int, optional
        Sampling rate of the original signal.
    window_length_samples : int, optional
        Window length in samples. Default: `None`.
    window_type : str, optional
        Window type. It must be supported by `scipy.signal.windows.get_window`.
        Default: `None`.
    overlap_percent : int, optional
        Window overlap in percent (between 0 and 100). Default: `None`.
    fft_length_samples : int, optional
        Length of the FFT applied to the time frames. Default: `None`.
    padding : bool, optional
        `True` means that the original signal was zero-padded in the beginning
        and end in order to avoid losing energy due to window effects.
        Default: `None`.
    scaling : bool, optional
        When `True`, it is assumed that the STFT matrix was scaled as an
        amplitude spectrum. Default: `None`.

    Returns
    -------
    reconstructed_signal : `Signal`
        Reconstructed signal from the complex STFT.

    Notes
    -----
    - In order to get the STFT (framed signal representation), it is probable
      that the original signal was zero-padded in the end. If the original
      signal is passed, the output will have the same length. If not, it might
      be longer by an amount of samples smaller than a window size.
    - It is important to notice that if the original signal was detrended,
      this can not be recovered and might lead to small distortions in the
      reconstructed one.
    - Instabilities when the original STFT was not zero-padded are avoided by
      padding during reconstruction at the expense of small amplitude
      distortion at the edges.

    References
    ----------
    - [1]: D. Griffin and Jae Lim, "Signal estimation from modified short-time
      Fourier transform," in IEEE Transactions on Acoustics, Speech, and Signal
      Processing, vol. 32, no. 2, pp. 236-243, April 1984,
      doi: 10.1109/TASSP.1984.1164317.

    """
    assert stft.ndim == 3, \
        f'{stft.ndim} is not a valid number of dimensions. It must be 3'

    if original_signal is not None:
        assert parameters is None, \
            'A signal was passed. No parameters dictionary should be passed'
        parameters = original_signal._spectrogram_parameters.copy()
    elif parameters is not None:
        pass
    else:
        assert (window_length_samples is not None) and \
            (window_type is not None) and (overlap_percent is not None) and \
            (padding is not None) and (scaling is not None), \
            'At least one of the needed parameters needed was passed as None'
        parameters = {'window_length_samples': window_length_samples,
                      'window_type': window_type,
                      'overlap_percent': overlap_percent,
                      'fft_length_samples': fft_length_samples,
                      'padding': padding, 'scaling': scaling}

    window = get_window(parameters['window_type'],
                        parameters['window_length_samples'])

    if parameters['scaling']:
        stft /= np.sqrt(2 / np.sum(window)**2)

    td_framed = np.fft.irfft(stft, axis=0, n=parameters['fft_length_samples'])

    # Reconstruct from framed representation to continuous
    step = int((1 - parameters['overlap_percent']/100) * len(window))

    if parameters['padding']:
        td = _reconstruct_framed_signal(td_framed, step_size=step,
                                        window=window)
        overlap = int(parameters['overlap_percent']/100 * len(window))
        td = td[overlap:-overlap, :]
    else:
        extra_window = np.zeros_like(td_framed[:, 0, :])[:, np.newaxis, :]
        td_framed = np.append(extra_window, td_framed, axis=1)
        td_framed = np.append(td_framed, extra_window, axis=1)
        td = _reconstruct_framed_signal(td_framed, step_size=step,
                                        window=window)
        td = td[step:-step, :]

    if original_signal is not None:
        td = _pad_trim(td, original_signal.time_data.shape[0])
        reconstructed_signal = original_signal.copy()
        reconstructed_signal.time_data = td
    else:
        reconstructed_signal = Signal(None, time_data=td,
                                      sampling_rate_hz=sampling_rate_hz)
    return reconstructed_signal
