"""
Here are methods considered as somewhat special or less common.
"""

from ..classes.signal import Signal
from ..classes.filter import Filter
from ..classes.impulse_response import ImpulseResponse
from ..classes.multibandsignal import MultiBandSignal
from ..plots import general_matrix_plot
from .._standard import _reconstruct_framed_signal, _get_framed_signal
from .._general_helpers import (
    _hz2mel,
    _mel2hz,
    _pad_trim,
    __yw_ar_estimation,
    __burg_ar_estimation,
)
from ..room_acoustics._room_acoustics import _find_ir_start
from ..transforms._transforms import (
    _pitch2frequency,
    Wavelet,
    MorletWavelet,
    _squeeze_scalogram,
    _get_kernels_vqt,
    _warp_time_series,
    _get_warping_factor,
    _dft_backend,
)
from ..tools import to_db

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import get_window
from scipy.fft import dct
from scipy.signal import oaconvolve, resample_poly, lfilter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

try:
    from seaborn import set_style

    set_style("whitegrid")
except ModuleNotFoundError as e:
    print("Seaborn will not be used for plotting: ", e)
    pass


def cepstrum(
    signal: Signal, mode="power"
) -> NDArray[np.float64] | NDArray[np.complex128]:
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
    ceps : NDArray[np.float64] or NDArray[np.complex128]
        Cepstrum.

    References
    ----------
    https://de.wikipedia.org/wiki/Cepstrum

    """
    mode = mode.lower()
    assert mode in (
        "power",
        "complex",
        "real",
    ), f"{mode} is not a supported mode"

    signal.set_spectrum_parameters(method="standard")
    _, sp = signal.get_spectrum()

    if mode in ("power", "real"):
        ceps = np.abs(np.fft.irfft((2 * np.log(np.abs(sp))), axis=0)) ** 2.0
    else:
        phase = np.unwrap(np.angle(sp), axis=0)
        ceps = np.fft.irfft(np.log(np.abs(sp)) + 1j * phase, axis=0).real
    if mode == "real":
        ceps = (ceps**0.5) / 2.0
    return ceps


def log_mel_spectrogram(
    s: Signal,
    channel: int = 0,
    range_hz=None,
    n_bands: int = 40,
    generate_plot: bool = True,
    stft_parameters: dict | None = None,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    | tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        plt.Figure,
        plt.Axes,
    ]
):
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
    time_s : NDArray[np.float64]
        Time vector.
    f_mel : NDArray[np.float64]
        Frequency vector in Mel.
    log_mel_sp : NDArray[np.float64]
        Log mel spectrogram with shape (frequency, time frame, channel).

    When `generate_plot=True`:

    time_s : NDArray[np.float64]
        Time vector.
    f_mel : NDArray[np.float64]
        Frequency vector in Mel.
    log_mel_sp : NDArray[np.float64]
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
    log_mel_sp = np.tensordot(mfilt, np.abs(sp) ** 2.0, axes=(-1, 0))

    log_mel_sp = to_db(log_mel_sp, False)

    if generate_plot:
        fig, ax = general_matrix_plot(
            log_mel_sp[..., channel],
            range_x=[time_s[0], time_s[-1]],
            range_y=[f_mel[0], f_mel[-1]],
            range_z=50,
            ylabel="Frequency / Mel",
            xlabel="Time / s",
            ylog=False,
            returns=True,
        )
        return time_s, f_mel, log_mel_sp, fig, ax
    return time_s, f_mel, log_mel_sp


def mel_filterbank(
    f_hz: NDArray[np.float64],
    range_hz=None,
    n_bands: int = 40,
    normalize: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Creates equidistant mel triangle filters in a given range. The returned
    matrix can be used to convert Hz into Mel in a spectrogram.

    NOTE: This is not a filter bank in the usual sense, thus it does not create
    a FilterBank object to be applied to a signal. Its intended use is in the
    frequency domain.

    Parameters
    ----------
    f_hz : NDArray[np.float64]
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
    mel_filters : NDArray[np.float64]
        Mel filters matrix with shape (bands, frequency). These are to be
        applied to a power response (squared spectrum).
    mel_center_freqs : NDArray[np.float64]
        Vector containing mel center frequencies.

    """
    f_hz = np.squeeze(f_hz)
    assert f_hz.ndim == 1, "f_hz should be a 1D-array"
    n_bands = int(n_bands)

    # Create range
    if range_hz is None:
        range_hz = f_hz[[0, -1]]
    else:
        range_hz = np.atleast_1d(np.asarray(range_hz).squeeze())
        assert (
            len(range_hz) == 2
        ), "range_hz should be an array with exactly two values!"
        range_hz = np.sort(range_hz)
        assert range_hz[-1] <= f_hz[-1], (
            f"Upper frequency in range {range_hz[-1]} is bigger than "
            + f"nyquist frequency {f_hz[-1]}"
        )
        assert range_hz[0] >= 0, "Lower frequency in range must be positive"

    # Compute band center frequencies in mel
    range_mel = _hz2mel(range_hz)
    mel_center_freqs = np.linspace(
        range_mel[0], range_mel[1], n_bands + 2, endpoint=True
    )

    # Center frequencies in Hz
    bands_hz = _mel2hz(mel_center_freqs)

    # Find indexes for frequencies
    inds = np.empty_like(bands_hz, dtype=int)
    for ind, b in enumerate(bands_hz):
        inds[ind] = np.argmin(np.abs(b - f_hz))

    # Create triangle filters
    mel_filters = np.zeros((n_bands, len(f_hz)))
    for n in range(n_bands):
        ni = n + 1
        mel_filters[n, inds[ni - 1] : inds[ni]] = np.linspace(
            0, 1, inds[ni] - inds[ni - 1], endpoint=False
        )
        mel_filters[n, inds[ni] : inds[ni + 1]] = np.linspace(
            1, 0, inds[ni + 1] - inds[ni], endpoint=False
        )
        if normalize:
            mel_filters[n, :] /= np.sum(mel_filters[n, :])
    return mel_filters, mel_center_freqs[1:-1]


def plot_waterfall(
    sig: Signal,
    channel: int = 0,
    dynamic_range_db: float = 40,
    stft_parameters: dict | None = None,
) -> tuple[Figure, Axes]:
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
    assert dynamic_range_db > 0, "Dynamic range has to be more than 0"
    sig = sig.get_channels(channel)
    if stft_parameters is not None:
        sig.set_spectrogram_parameters(**stft_parameters)
    t, f, stft = sig.get_spectrogram()

    if sig._spectrum_parameters["scaling"] is None:
        amplitude_scaling = True
    else:
        amplitude_scaling = "amplitude" in sig._spectrum_parameters["scaling"]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection="3d"))
    tt, ff = np.meshgrid(t, f)
    ax.plot_surface(
        tt,
        ff,
        to_db(stft[..., channel], amplitude_scaling, dynamic_range_db),
        cmap="magma",
    )
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Frequency / Hz")
    ax.set_zlabel("dB")
    fig.tight_layout()
    return fig, ax


def mfcc(
    signal: Signal,
    channel: int = 0,
    mel_filters: NDArray[np.float64] | None = None,
    generate_plot: bool = True,
    stft_parameters: dict | None = None,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    | tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        plt.Figure,
        plt.Axes,
    ]
):
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
    mel_filters : NDArray[np.float64], optional
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
    time_s : NDArray[np.float64]
        Time vector.
    f_mel : NDArray[np.float64]
        Frequency vector in mel. If `mel_filters` is passed, this is only a
        list with entries [0, n_mel_filters].
    mfcc : NDArray[np.float64]
        Mel-frequency cepstral coefficients with shape (cepstral coefficients,
        time frame, channel).

    When `generate_plot=True`:

    time_s : NDArray[np.float64]
        Time vector.
    f_mel : NDArray[np.float64]
        Frequency vector in mel. If `mel_filters` is passed, this is only a
        list with entries [0, n_mel_filters].
    mfcc : NDArray[np.float64]
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

    # Mel filters
    if mel_filters is None:
        mel_filters, f_mel = mel_filterbank(f, None, n_bands=40)
    else:
        assert mel_filters.shape[1] == sp.shape[0], (
            f"Shape of the mel filter matrix {mel_filters.shape} does "
            + f"not match the STFT {sp.shape}"
        )
        f_mel = np.array([0, mel_filters.shape[0]])

    # Convert from Hz to Mel
    sp = np.tensordot(mel_filters, np.abs(sp) ** 2.0, axes=(-1, 0))

    # Get Log power spectrum
    log_sp = to_db(sp, False)

    # Discrete cosine transform
    mfcc = np.abs(dct(log_sp, type=2, axis=0))

    # Prune nans
    np.nan_to_num(mfcc, copy=False, nan=0)

    # Plot and return
    if generate_plot:
        fig, ax = general_matrix_plot(
            mfcc[..., channel],
            range_x=[time_s[0], time_s[-1]],
            range_y=[f_mel[0], f_mel[-1]],
            xlabel="Time / s",
            ylabel="Cepstral coefficients",
            returns=True,
        )
        return time_s, f_mel, mfcc, fig, ax
    return time_s, f_mel, mfcc


def istft(
    stft: NDArray[np.complex128],
    original_signal: Signal | None = None,
    parameters: dict | None = None,
    sampling_rate_hz: int | None = None,
    window_length_samples: int | None = None,
    window_type: str | None = None,
    overlap_percent: int | None = None,
    fft_length_samples: int | None = None,
    padding: bool | None = None,
    scaling: bool | None = None,
) -> Signal:
    """This function transforms a complex STFT back into its respective time
    signal using the method presented in [1]. For this to be possible, it is
    necessary to know the parameters that were used while converting the signal
    into its STFT representation. A dictionary containing the parameters
    corresponding can be passed, as well as the original `Signal` in which
    these parameters are saved. Alternatively, it is possible to pass them
    explicitely.

    Parameters
    ----------
    stft : NDArray[np.complex128]
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
    assert (
        stft.ndim == 3
    ), f"{stft.ndim} is not a valid number of dimensions. It must be 3"

    if original_signal is not None:
        assert (
            parameters is None
        ), "A signal was passed. No parameters dictionary should be passed"
        parameters = original_signal._spectrogram_parameters.copy()
    elif parameters is not None:
        pass
    else:
        assert (
            (window_length_samples is not None)
            and (window_type is not None)
            and (overlap_percent is not None)
            and (padding is not None)
            and (scaling is not None)
        ), "At least one of the needed parameters needed was passed as None"
        parameters = {
            "window_length_samples": window_length_samples,
            "window_type": window_type,
            "overlap_percent": overlap_percent,
            "fft_length_samples": fft_length_samples,
            "padding": padding,
            "scaling": scaling,
        }

    window = get_window(
        parameters["window_type"], parameters["window_length_samples"]
    )

    if parameters["scaling"]:
        stft /= np.sqrt(2 / np.sum(window) ** 2)

    td_framed = np.fft.irfft(stft, axis=0, n=parameters["fft_length_samples"])
    td_framed = td_framed[: parameters["window_length_samples"], ...]

    # Reconstruct from framed representation to continuous
    step = int((1 - parameters["overlap_percent"] / 100) * len(window))

    if parameters["padding"]:
        td = _reconstruct_framed_signal(
            td_framed, step_size=step, window=window
        )
        overlap = int(parameters["overlap_percent"] / 100 * len(window))
        td = td[overlap:-overlap, :]
    else:
        extra_window = np.zeros_like(td_framed[:, 0, :])[:, np.newaxis, :]
        td_framed = np.append(extra_window, td_framed, axis=1)
        td_framed = np.append(td_framed, extra_window, axis=1)
        td = _reconstruct_framed_signal(
            td_framed, step_size=step, window=window
        )
        td = td[step:-step, :]

    if original_signal is not None:
        td = _pad_trim(td, original_signal.time_data.shape[0])
        reconstructed_signal = original_signal.copy()
        reconstructed_signal.time_data = td
    else:
        reconstructed_signal = Signal(
            None, time_data=td, sampling_rate_hz=sampling_rate_hz
        )
    return reconstructed_signal


def chroma_stft(
    signal: Signal,
    tuning_a_hz: float = 440,
    compression: float = 0.5,
    plot_channel: int = -1,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    | tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        plt.Figure,
        plt.Axes,
    ]
):
    """This computes the Chroma Features and Pitch STFT. See [1] for details.

    Parameters
    ----------
    signal : `Signal`
        Signal for which to compute the chroma features. The saved parameters
        for the spectrogram are used to compute the STFT.
    tuning_a_hz : float, optional
        Tuning in Hz for the A4. Default: 440.
    compression : float, optional
        Compression factor as explained in [1]. Default: 0.5.
    plot_channel : int, optional
        When different than -1, a chroma plot for the corresponding channel is
        generated and returned. Default: -1.

    Returns
    -------
    t : NDArray[np.float64]
        Time vector corresponding to each time frame.
    chroma_stft : NDArray[np.float64]
        Chroma Features with shape (note, time frame, channel). First index
        is C, second C#, etc. (Until B).
    pitch_stft : NDArray[np.float64]
        Pitch log-STFT with shape (pitch, time frame, channel). First index
        is note 0 (MIDI), i.e., C0.
    When `plot_channel != -1`:
        Figure and Axes.

    References
    ----------
    - [1]: Short-Time Fourier Transform and Chroma Features. Müller.

    """
    assert tuning_a_hz > 0, "Tuning A4 must be greater than zero"
    assert compression > 0, "Compression factor must be greater than zero"

    t, f, stft = signal.get_spectrogram()

    # Energy
    stft = np.abs(stft) ** 2

    # Get frequencies
    pitch_frequencies = _pitch2frequency(tuning_a_hz)

    pitch_transformation = np.zeros((len(pitch_frequencies), len(f)))
    # This is the pitch representation
    for ind, fn in enumerate(pitch_frequencies):
        inds = (f >= fn * 2 ** (-1 / 24)) & (f < fn * 2 ** (1 / 24))
        pitch_transformation[ind, inds] = 1

    # Chroma sums over all octaves
    n_notes = 12
    chroma_transformation = np.zeros((n_notes, len(pitch_frequencies)))
    for i in range(n_notes):
        chroma_transformation[i, i::n_notes] = 1

    # Get Pitch STFT Frequency Scaling
    pitch_stft = np.tensordot(pitch_transformation, stft, (1, 0))
    # Sum over all octaves for chroma
    chroma_stft = np.tensordot(chroma_transformation, pitch_stft, (1, 0))

    pitch_stft = np.log(1 + compression * pitch_stft)
    chroma_stft = np.log(1 + compression * chroma_stft)

    if plot_channel != -1:
        fig, ax = plt.subplots(1, 1)
        image = ax.imshow(
            chroma_stft[..., plot_channel], aspect="auto", origin="lower"
        )
        ax.set_yticks(
            np.arange(12),
            ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        )
        time_step = int(1 / t[1])
        ax.set_xticks(
            np.arange(0, chroma_stft.shape[1], time_step),
            np.round(t[::time_step]),
        )
        ax.set_xlabel("Time / s")
        ax.set_ylabel("Note")
        fig.colorbar(image)
        return t, chroma_stft, pitch_stft, fig, ax
    return t, chroma_stft, pitch_stft


def cwt(
    signal: Signal,
    frequencies: NDArray[np.float64],
    wavelet: Wavelet | MorletWavelet,
    channel: NDArray[np.float64] | None = None,
    synchrosqueezed: bool = False,
    apply_synchrosqueezed_normalization: bool = False,
) -> NDArray[np.complex128]:
    """Returns a scalogram by means of the continuous wavelet transform.

    Parameters
    ----------
    signal : `Signal`
        Signal for which to compute the cwt.
    frequencies : NDArray[np.float64]
        Frequencies to query with the wavelet.
    wavelet : `Wavelet` or `MorletWavelet`
        Type of wavelet to use. It must be a class inherited from the
        `Wavelet` class.
    channel : NDArray[np.float64], optional
        Channel for which to compute the cwt. If `None`, all channels are
        computed. Default: `None`.
    synchrosqueezed : bool, optional
        When `True`, the scalogram is synchrosqueezed using the phase
        transform. Default: `False`.
    apply_synchrosqueezed_normalization : bool, optional
        When `True`, each scale is scaled by taking into account the
        normalization as shown in Eq. (2.4) of [1]. `False` does not apply
        any normalization. This is only done for synchrosqueezed scalograms.
        Default: `False`.

    Returns
    -------
    scalogram : NDArray[np.complex128]
        Complex scalogram scalogram with shape (frequency, time sample,
        channel).

    Notes
    -----
    - Zero-padding in the beginning is done for reducing boundary effects.

    References
    ----------
    - [1]: Ingrid Daubechies, Jianfeng Lu, Hau-Tieng Wu. Synchrosqueezed
      wavelet transforms: An empirical mode decomposition-like tool. 2011.
    - General information about synchrosqueezing:
      https://dsp.stackexchange.com/questions/71398/synchrosqueezing-wavelet
      -transform-explanation

    """
    if channel is None:
        channel = np.arange(signal.number_of_channels)
    channel = np.atleast_1d(channel)
    td = signal.time_data[:, channel]

    scalogram = np.zeros(
        (len(frequencies), td.shape[0], td.shape[1]), dtype=np.complex128
    )

    for ind_f, f in enumerate(frequencies):
        wv = np.array(wavelet.get_wavelet(f, signal.sampling_rate_hz))
        wv /= np.abs(wv).sum()

        scalogram[ind_f, ...] = oaconvolve(
            td, wv[..., None], axes=0, mode="same"
        )

    if synchrosqueezed:
        scalogram = _squeeze_scalogram(
            scalogram,
            frequencies,
            signal.sampling_rate_hz,
            apply_frequency_normalization=apply_synchrosqueezed_normalization,
        )

    return scalogram


def hilbert(
    signal: Signal | ImpulseResponse | MultiBandSignal,
) -> Signal | ImpulseResponse | MultiBandSignal:
    """Compute the analytic signal using the hilbert transform of the real
    signal.

    Parameters
    ----------
    signal : `Signal`, `MultiBandSignal`
        Signal to convert.

    Returns
    -------
    analytic : `Signal`, `MultiBandSignal`
        Analytical signal.

    Notes
    -----
    - Since it is not causal, the whole time series must be passed
      through an FFT. This could take long or be too memory intensive depending
      on the size of the original signal and the computer.
    - The new `Signal` has the real part saved in `self.time_data` and the
      imaginary in `self.time_data_imaginary`. Complex time series can
      therefore be constructed with::

        complex_ts = Signal.time_data + Signal.time_data_imaginary*1j

    """
    if isinstance(signal, Signal):
        td = signal.time_data

        sp = np.fft.fft(td, axis=0)
        if len(td) % 2 == 0:
            nyquist = len(td) // 2
            sp[1:nyquist, :] *= 2
            sp[nyquist + 1 :, :] = 0
        else:
            sp[1 : (len(td) + 1) // 2, :] *= 2
            sp[(len(td) + 1) // 2 :, :] = 0

        analytic = signal.copy()
        analytic.time_data = np.fft.ifft(sp, axis=0)
    elif type(signal) is MultiBandSignal:
        new_mb = signal.copy()
        for ind, b in enumerate(new_mb):
            new_mb.bands[ind] = hilbert(b)
        return new_mb
    else:
        raise TypeError("Signal does not have a valid type")
    return analytic


def vqt(
    signal: Signal,
    channel: NDArray[np.int_] | None = None,
    q: float = 1,
    gamma: float = 50,
    octaves: list = [1, 5],
    bins_per_octave: int = 24,
    a4_tuning: int = 440,
    window: str | tuple = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Variable-Q Transform. This is a special case of the continuous wavelet
    transform with complex morlet wavelets for the time-frequency analysis.
    Constant-Q Transform can be obtained by setting `gamma = 0`.

    Parameters
    ----------
    signal : `Signal`
        Signal for which to compute the cqt coefficients.
    channel : NDArray[np.float64] or int, optional
        Channel(s) for which to compute the cqt coefficients. If `None`,
        all channels are computed. Default: `None`.
    q : float, optional
        Q-factor. 1 would be the optimal value but setting it to something
        lower might increase temporal resolution at the expense of frequency
        resolution. Default: 1.
    gamma : float, optional
        This is the factor for the bandwidth adaptation (in Hz). This extends
        the bandwidth of each kernel. Set to 0 for obtaining the Constant-Q
        Transform. Default: 50.
    octaves : list with length 2, optional
        Range of musical octaves for which to compute the cqt coefficients.
        [1, 4] computes all corresponding frequencies from C1 up until B4.
        Exact frequencies are adapted from the a4-tuning parameter.
        Default: [1, 5].
    bins_per_octave : int, optional
        Number of frequency bins to divide an octave. Default: 24.
    a4_tuning : int, optional
        Frequency for A4 in Hz. Default: 440.
    window : str or tuple, optional
        Type of window to use for the kernels. This is directly passed
        to `scipy.signal.get_window()`, so that a tuple containing a window
        type and an additional parameter can be used. Default: `'hann'`.

    Returns
    -------
    f : NDArray[np.float64]
        Frequency vector.
    vqt : NDArray[np.complex128]
        VQT coefficients with shape (frequency, time samples, channel).

    References
    ----------
    - Schörkhuber and Klapuri: CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC
      PROCESSING.
    - Schörkhuber et. al.: A Matlab Toolbox for Efficient Perfect
      Reconstruction Time-Frequency Transforms with Log-Frequency Resolution.

    """
    if channel is None:
        channel = np.arange(signal.number_of_channels)
    channel = np.atleast_1d(channel)

    td = signal.time_data[:, channel]

    # Highest frequency corresponds to the B of the last octave
    highest_f = a4_tuning * 2 ** (octaves[1] - 4 + 2 / 12)

    # Find necessary sampling rate, i.e., one with nyquist just above
    # highest frequency. This delivers the necessary decimation factor.
    decimation = int((signal.sampling_rate_hz // 2) / (highest_f * 1.1))
    mid_fs = signal.sampling_rate_hz // decimation
    td = resample_poly(td, up=1, down=decimation, axis=0)

    # Gamma adaptation
    gamma = gamma / signal.sampling_rate_hz * mid_fs

    kernels = _get_kernels_vqt(
        q, highest_f, bins_per_octave, mid_fs, window, gamma
    )

    octs = octaves[1] - octaves[0] + 1
    cqt = np.zeros(
        (0, signal.time_data.shape[0], signal.number_of_channels),
        dtype=np.complex128,
    )

    for oc in np.arange(octs):
        # Accumulator for octave
        acc = np.zeros((0, td.shape[0], td.shape[1]), dtype=np.complex128)

        for k in kernels:
            out = oaconvolve(td, k[..., None], mode="same", axes=0)
            acc = np.append(acc, out[None, ...], axis=0)

        # Resample back to original sampling rate and save
        if oc != 0:
            acc = resample_poly(acc, up=2**oc, down=1, axis=1)
        acc = resample_poly(acc, up=decimation, down=1, axis=1)

        length_diff = acc.shape[1] - cqt.shape[1]
        if length_diff > 0:
            acc = acc[:, : cqt.shape[1], :]
        elif length_diff < 0:
            acc = np.pad(acc, ((0, 0), (0, -length_diff), (0, 0)))
        cqt = np.append(cqt, acc, axis=0)
        # Decimate for further computation
        td = resample_poly(td, up=1, down=2, axis=0)

    # Invert frequency axis
    cqt = np.flip(cqt, axis=0)
    f = a4_tuning * 2 ** (
        np.arange(octaves[0] - 4 - 9 / 12, octaves[1] - 4 + 2 / 12, 1 / 12)
    )
    return f, cqt


def stereo_mid_side(signal: Signal, forward: bool) -> Signal:
    """This function converts a left-right stereo signal to its mid-side
    representation or the other way around. It is only available for
    two-channels signals.

    Parameters
    ----------
    signal : `Signal`
        Signal with two channels, i.e., left and right (in that order) or
        mid and side.
    forward : bool
        When `True`, left-right is converted to mid-side. When `False`,
        mid-side is turned into left-right.

    Returns
    -------
    new_sig : `Signal`
        Converted signal. Left (or mid) are always the first channel.

    """
    assert (
        signal.number_of_channels == 2
    ), "Signal must have exactly two channels"
    new_sig = signal.copy()
    td = signal.time_data
    td[:, 0] = signal.time_data[:, 0] + signal.time_data[:, 1]
    td[:, 1] = signal.time_data[:, 0] - signal.time_data[:, 1]
    if forward:
        td /= 2
    new_sig.time_data = td
    return new_sig


def laguerre(signal: Signal, warping_factor: float) -> Signal:
    """This function implements the discrete Laguerre Transform in the time
    domain according to [1]. It is mainly used for frequency warping. See notes
    for details.

    This is a resource-intensive operation that should be applied to signals
    that are not very long.

    Parameters
    ----------
    signal : Signal
        Signal to be transformed.
    warping_factor : float
        Warping factor. It must be in the range ]-1; 1[.

    Returns
    -------
    Signal
        Transformed signal.

    Notes
    -----
    - This transform can be reversed by applying it once with `warping_factor`
      and then with `-warping_factor`.
    - It is an alternative, more general formulation to Warping and a special
      case of using Kautz filters for a fixed pole. `warping_factor` here leads
      to the same frequency mapping as warping.
    - This can be used for frequency-dependent windowing of an impulse
      response. Since this is not shift-invariant, the start of the IR should
      be placed at `t=0`.
    - In general, `warping_factor < 0.` shifts the frequency axis towards
      nyquist, i.e., increases the resolution of the lower frequencies while
      lowering that of higher frequencies. See [2] for the
      resolution/frequency-mapping of warping.

    References
    ----------
    - [1]: Zölzer, Battista. Digital Audio Effects DAFX. Chapter 11, second
      edition.
    - [2]: Bank, B. (2022). Warped, Kautz, and Fixed-Pole Parallel Filters: A
      Review. Journal of the Audio Engineering Society.

    """
    assert (
        np.abs(warping_factor) < 1.0
    ), "Warping factor cannot be larger than 1."

    xx = signal.time_data[::-1, ...]  # Time reversal
    output = np.zeros_like(xx)

    b = np.array([warping_factor, 1.0])
    a = np.array([1.0, warping_factor])
    b_normalized = (1.0 - warping_factor**2.0) ** 0.5

    # First filtering stage with normalization
    xx = lfilter(b_normalized, a, xx, axis=0)
    output[0, :] = xx[-1, :]

    # Rest filters
    for i in range(1, xx.shape[0]):
        xx = lfilter(b, a, xx, axis=0)
        output[i, :] = xx[-1, :]

    out = signal.copy()
    out.time_data = output
    return out


def warp(
    ir: Signal,
    warping_factor: float | str,
    shift_ir: bool,
    total_length: int | None = None,
) -> Signal | tuple[Signal, float]:
    r"""Compute a warped signal as explained by [1]. This operation
    corresponds to computing a warped FIR-Filter (WFIR).

    To pre-warp a signal, pass a negative `warping_factor`. To de-warp it, use
    the same positive `warping_factor`. See notes for details.

    Parameters
    ----------
    ir : `Signal`
        Impulse response to (de)warp.
    warping_factor : float, str, {"bark", "erb", "bark-", "erb-"}
        Warping factor. It has to be in the range ]-1; 1[. If a string is
        provided, warping the frequency axis to (or from) an approximation
        of the psychoacoustically motivated Bark or ERB scales is performed
        according to [4]. Pass "-" in the end for the dewarping (backwards)
        stage.
    shift_ir : bool
        Since the warping of an IR is not shift-invariant (see [2]), it is
        recommended to place the start of the IR at the first index. When
        `True`, the first sample to surpass -20 dBFS (relative to peak) is
        shifted to the beginning and the previous samples are sent to the
        end of the signal. `False` avoids any manipulation.
    total_length : int, optional
        Total length to use for the warped signal. If `None`, the original
        length is maintained. Default: `None`.

    Returns
    -------
    warped_ir : `Signal`
        The same IR with warped or dewarped time vector.
    float
        Warping factor. Only returned in case "bark" or "erb" was passed.

    Notes
    -----
    - Depending on the signal length, this might be a slow computation.
    - Frequency-dependent windowing can be easily done in the warped domain.
      This is not the approach used in `window_frequency_dependent()`, but
      it can be achieved with this function. See [2] for more details.
    - In general, `warping_factor < 0.` shifts the frequency axis towards
      nyquist, i.e., increases the resolution of the lower frequencies while
      lowering that of higher frequencies. See [1] and [3] for the frequency
      mapping of warping.
    - `warping_factor` will have a frequency-warping where a single frequency
      point remains unwarped. The formula for this is [1]:

        .. math::
            \frac{f_s}{2\pi}\arccos(\lambda)

      where lambda is the `warping_factor` and f_s the sampling rate.
    - Warping poles and zeros in the rational transfer function can be done
      by replacing the z^-1 with (z^-1 - lambda)/(1 - lambda*z^-1). This leads,
      for instance, to transforming a pole p0 to a new pole p with

        .. math::
            p = \frac{\lambda + p_0}{1 + p_0 \lambda}

      while appending the factor

        .. math::
            \left(1 + \lambda z^{-1}\right)^{M_p - N_z}

      to the transfer function, where Mp is the total number of poles and Nz
      the total number of zeros.
    - The frequency scale approximation to the Bark scale presented in [4]
      is more accurate than for the ERB scale.

    References
    ----------
    - [1]: Härmä, Aki & Karjalainen, Matti & Avioja, Lauri & Välimäki, Vesa &
      Laine, Unto & Huopaniemi, Jyri. (2000). Frequency-Warped Signal
      Processing for Audio Applications. Journal of the Audio Engineering
      Society. 48. 1011-1031.
    - [2]: M. Karjalainen and T. Paatero, "Frequency-dependent signal
      windowing," Proceedings of the 2001 IEEE Workshop on the Applications of
      Signal Processing to Audio and Acoustics (Cat. No.01TH8575), New Platz,
      NY, USA, 2001, pp. 35-38, doi: 10.1109/ASPAA.2001.969536.
    - [3]: Bank, B. (2022). Warped, Kautz, and Fixed-Pole Parallel Filters: A
      Review. Journal of the Audio Engineering Society.
    - [4]: III, J.O. & Abel, Jonathan. (1999). Bark and ERB Bilinear
      Transforms. Speech and Audio Processing, IEEE Transactions on. 7.
      697 - 708. 10.1109/89.799695.

    """
    approximation_warping_factor = type(warping_factor) is str
    warping_factor = _get_warping_factor(warping_factor, ir.sampling_rate_hz)

    td = ir.time_data
    if shift_ir:
        for ch in range(ir.number_of_channels):
            start = _find_ir_start(td[:, ch], -20)
            td[:, ch] = np.roll(td[:, ch], -start)

    if total_length is None:
        total_length = td.shape[0]

    td = _warp_time_series(td[:total_length, ...], warping_factor)
    warped_ir = ir.copy()
    warped_ir.time_data = td

    if approximation_warping_factor:
        return warped_ir, warping_factor

    return warped_ir


def warp_filter(filter: Filter, warping_factor: float) -> Filter:
    r"""Apply warping to a filter by transforming its poles and zeros. See
    references for details on warping.

    Parameters
    ----------
    filter : Filter
        Filter to be warped.
    warping_factor : float
        Warping factor. See `warp()` for details.

    Returns
    -------
    Filter
        Warped filter.

    Notes
    -----
    - The overall filter gain of the filter is not modified by this function.
    - Warping poles and zeros in the rational transfer function can be done
      by replacing the z^-1 with (z^-1 - lambda)/(1 - lambda*z^-1). This leads,
      for instance, to transforming a pole p0 to a new pole p with

        .. math::
            p = \frac{\lambda + p_0}{1 + p_0 \lambda}

      while appending the factor

        .. math::
            \left(1 + \lambda z^{-1}\right)^{M_p - N_z}

      to the transfer function, where Mp is the total number of poles and Nz
      the total number of zeros.
    - Warping filters with orders above 100 is not recommended due to numerical
      errors when finding their polynomial roots. This does not apply if the
      filter has the zeros and poles from which its coefficients were computed.

    References
    ----------
    - [1]: Härmä, Aki & Karjalainen, Matti & Avioja, Lauri & Välimäki, Vesa &
      Laine, Unto & Huopaniemi, Jyri. (2000). Frequency-Warped Signal
      Processing for Audio Applications. Journal of the Audio Engineering
      Society. 48. 1011-1031.
    - [2]: M. Karjalainen and T. Paatero, "Frequency-dependent signal
      windowing," Proceedings of the 2001 IEEE Workshop on the Applications of
      Signal Processing to Audio and Acoustics (Cat. No.01TH8575), New Platz,
      NY, USA, 2001, pp. 35-38, doi: 10.1109/ASPAA.2001.969536.
    - [3]: Bank, B. (2022). Warped, Kautz, and Fixed-Pole Parallel Filters: A
      Review. Journal of the Audio Engineering Society.
    - [4]: III, J.O. & Abel, Jonathan. (1999). Bark and ERB Bilinear
      Transforms. Speech and Audio Processing, IEEE Transactions on. 7.
      697 - 708. 10.1109/89.799695.

    """
    assert abs(warping_factor) < 1.0, "Warping factor must be less than 1."
    z, p, k = filter.get_coefficients("zpk")
    p = (warping_factor + p) / (1 + warping_factor * p)
    z = (warping_factor + z) / (1 + warping_factor * z)
    if len(p) > len(z):
        z = np.hstack([z, [warping_factor] * (len(p) - len(z))])
    elif len(z) > len(p):
        p = np.hstack([p, [warping_factor] * (len(z) - len(p))])
    return Filter.from_zpk(z, p, k, filter.sampling_rate_hz)


def lpc(
    signal: Signal,
    order: int,
    window_length_samples: int,
    synthesize_encoded_signal: bool = False,
    method_ar: str = "burg",
    hop_size_samples: int | None = None,
    window_type: str = "hann",
):
    """Encode an input signal into its linear-predictive coding coefficients.
    This transforms the signal into source-filter representation and works
    best with inputs that can be modeled through the all-pole model.

    Parameters
    ----------
    signal : Signal
        Input to encode.
    order : int
        Order of the coefficients to use.
    window_length_samples : int
        Window length in samples.
    synthesize_encoded_signal : bool, optional
        When True, the encoded signal is synthesized and returned. To this end,
        white noise is always used as source. Pass False to avoid this
        computation. Default: False.
    method_ar : str, {"yw", "burg"}, optional
        Method to use for obtaining the LP coefficients. Choose from "yw"
        (Yule-Walker) or "burg". Default: "burg".
    hop_size_samples : int, None, optional
        Hop size to use from window to window. If None is passed, a hop size
        corresponding to 50% of the window length will be used. Default: None.
    window_type : str, optional
        Window type to use. It is recommended that a window type that satifies
        the COLA-condition with length and hop size is chosen. Default: "hann".

    Returns
    -------
    a_coefficients : NDArray[np.float64]
        LP coefficients with shape (time window, coefficient, channel).
    variances : NDArray[np.float64]
        Variances (quadratic) of the source with shape (time window, channel).
    reconstructed_signal : Signal
        Signal reconstructed from the estimated LP coefficients using white
        noise as the source. This is only returned if
        `synthesize_encoded_signal=True`.

    References
    ----------
    - https://en.wikipedia.org/wiki/Linear_predictive_coding
    - https://ccrma.stanford.edu/~hskim08/lpc/

    """
    method_ar = method_ar.lower()
    assert method_ar in ("burg", "yw"), "AR method is not supported"

    # Get windowed signal
    if hop_size_samples is None:
        hop_size_samples = window_length_samples // 2
    td = _get_framed_signal(
        signal.time_data, window_length_samples, hop_size_samples, True
    )
    window = get_window(window_type, window_length_samples, fftbins=True)
    td *= window[:, None, None]

    a, var = (
        __burg_ar_estimation(td, order)
        if method_ar == "burg"
        else __yw_ar_estimation(td, order)
    )

    if not synthesize_encoded_signal:
        return a, var

    synthesized_signal = np.zeros_like(td)
    for channel in range(td.shape[2]):
        for n_window in range(td.shape[1]):
            source = np.random.normal(
                0.0, var[n_window, channel] ** 0.5, td.shape[0]
            )
            synthesized_signal[:, n_window, channel] = lfilter(
                [1.0],
                a[:, n_window, channel],
                source,
            )
    synthesized_signal = _reconstruct_framed_signal(
        synthesized_signal, hop_size_samples, window, len(signal)
    )
    return Signal.from_time_data(synthesized_signal, signal.sampling_rate_hz)


def dft(signal: Signal, frequency_vector_hz: NDArray[np.float64]):
    """DFT for any set of frequencies. This is a direct computation of the DFT,
    so it is significantly slower than an FFT, but it can be used to obtain any
    desired frequency resolution.

    Parameters
    ----------
    signal : Signal
        Signal for which to compute the spectrum.
    frequency_vector_hz : NDArray[np.float64]
        Frequency vector to query.

    Returns
    -------
    spectrum : NDArray[np.complex128]
        Spectrum with the defined frequency resolution. It has shape
        (frequency bin, channel).

    Notes
    -----
    - This function uses a parallelized computation of the DFT bins with numba,
      its performance might differ significantly from one computer to the
      other.
    - Frequency resolution different than linear can be obtained from the FFT
      via warping, FFTLog (fast hankel transform) or the Chirp-Z transform.
      None of these transforms allow for a completely arbitrary spacing of the
      frequency bins.

    """
    time_data = signal.time_data.astype(np.complex128)
    f_normalized = (
        frequency_vector_hz * (time_data.shape[0] / signal.sampling_rate_hz)
    ).astype(np.complex128)
    dft_factor = (
        -2j * np.pi * np.linspace(0.0, 1.0, time_data.shape[0], endpoint=False)
    )
    spectrum = np.zeros(
        (len(frequency_vector_hz), time_data.shape[1]), dtype=np.complex128
    )
    return _dft_backend(time_data, f_normalized, dft_factor, spectrum)
