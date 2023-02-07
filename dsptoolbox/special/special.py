"""
Here are methods considered as somewhat special or less common.
"""
from dsptoolbox.classes.signal_class import Signal
from dsptoolbox.plots import general_matrix_plot
from dsptoolbox._general_helpers import _hz2mel, _mel2hz

import numpy as np
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

    ceps = np.zeros_like(signal.time_data)
    signal.set_spectrum_parameters(method='standard')
    _, sp = signal.get_spectrum()

    for n in range(signal.number_of_channels):
        if mode in ('power', 'real'):
            cp = np.abs(np.fft.irfft((2*np.log(np.abs(sp[:, n])))))**2
        else:
            phase = np.unwrap(np.angle(sp[:, n]))
            cp = np.fft.irfft(np.log(np.abs(sp[:, n])) + 1j*phase).real
        if mode == 'real':
            cp = (cp**0.5)/2
        ceps[:, n] = cp
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
        Channel of the signal to be used. Default: 0.
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
        Log mel spectrogram.

    When `generate_plot=True`:

    time_s : `np.ndarray`
        Time vector.
    f_mel : `np.ndarray`
        Frequency vector in Mel.
    log_mel_sp : `np.ndarray`
        Log mel spectrogram.
    fig : `matplotlib.figure.Figure`
        Figure.
    ax : `matplotlib.axes.Axes`
        Axes.

    """
    if stft_parameters is not None:
        s.set_spectrogram_parameters(**stft_parameters)
    time_s, f_hz, sp = s.get_spectrogram(channel)
    mfilt, f_mel = mel_filterbank(f_hz, range_hz, n_bands, normalize=True)
    log_mel_sp = mfilt @ np.abs(sp)
    log_mel_sp = 20*np.log10(np.clip(log_mel_sp, a_min=1e-20, a_max=None))
    if generate_plot:
        fig, ax = general_matrix_plot(
            log_mel_sp, range_x=[time_s[0], time_s[-1]],
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
    bands_mel : `np.ndarray`
        Vector containing mel bands that correspond to the filters.

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
    bands_mel = np.linspace(
        range_mel[0], range_mel[1], n_bands+2, endpoint=True)

    # Center frequencies in Hz
    bands_hz = _mel2hz(bands_mel)

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
    return mel_filters, bands_mel


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
    if stft_parameters is not None:
        sig.set_spectrogram_parameters(**stft_parameters)
    t, f, stft = sig.get_spectrogram(channel_number=channel)

    stft = np.abs(stft)
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
