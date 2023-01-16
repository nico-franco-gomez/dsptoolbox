"""
Here are methods considered as somewhat special or less common.
"""
import numpy as np
from dsptoolbox.classes.signal_class import Signal
from dsptoolbox._standard import _minimum_phase, _group_delay_direct
from dsptoolbox.plots import general_matrix_plot
from dsptoolbox._general_helpers import _hz2mel, _mel2hz


def cepstrum(signal: Signal, mode='power'):
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


def min_phase_from_mag(spectrum: np.ndarray, sampling_rate_hz: int,
                       signal_type: str = 'ir'):
    """Returns a minimal phase signal from a magnitude spectrum using
    the hilbert transform.

    Parameters
    ----------
    spectrum : `np.ndarray`
        Spectrum with only positive frequencies and 0.
    sampling_rate_hz : int
        Signal's sampling rate in Hz.
    signal_type : str, optional
        Type of signal to be returned. Default: `'ir'`.

    Returns
    -------
    sig_min_phase : `Signal`
        Signal with same magnitude spectrum but minimal phase.

    References
    ----------
    - https://en.wikipedia.org/wiki/Minimum_phase

    """
    if spectrum.ndim < 2:
        spectrum = spectrum[..., None]
    assert spectrum.ndim < 3, \
        'Spectrum should have shape (bins, channels)'
    if spectrum.shape[0] < spectrum.shape[1]:
        spectrum = spectrum.T
    spectrum = np.abs(spectrum)
    min_spectrum = np.empty(spectrum.shape, dtype='cfloat')
    for n in range(spectrum.shape[1]):
        phase = _minimum_phase(spectrum[:, n], False)
        min_spectrum[:, n] = spectrum[:, n]*np.exp(1j*phase)
    time_data = np.fft.irfft(min_spectrum, axis=0)
    sig_min_phase = Signal(
        None, time_data=time_data,
        sampling_rate_hz=sampling_rate_hz, signal_type=signal_type)
    return sig_min_phase


def lin_phase_from_mag(spectrum: np.ndarray, sampling_rate_hz: int,
                       group_delay_ms='minimal',
                       check_causality: bool = True,
                       signal_type: str = 'ir'):
    """Returns a linear phase signal from a magnitude spectrum. It is possible
    to return the smallest causal group delay by checking the minimal phase
    version of the signal and choosing a constant group delay that is never
    lower than minimum group delay (for each channel). A value for the group
    delay can be also passed directly and applied to all channels. If check
    causility is activated, it is assessed that the given group delay is not
    less than each minimal group delay. If deactivated, the generated phase
    could lead to a non-causal system!

    Parameters
    ----------
    spectrum : `np.ndarray`
        Spectrum with only positive frequencies and 0.
    sampling_rate_hz : int
        Signal's sampling rate in Hz.
    group_delay_ms : str or float, optional
        Constant group delay that the phase should have for all channels
        (in ms). Pass `'minimal'` to create a signal with the minimum linear
        phase possible (that is different for each channel).
        Default: `'minimal'`.
    check_causality : bool, optional
        When `True`, it is assessed for each channel that the given group
        delay is not lower than the minimal group delay. Default: `True`.
    signal_type : str, optional
        Type of signal to be returned. Default: `'ir'`.

    Returns
    -------
    sig_lin_phase : `Signal`
        Signal with same magnitude spectrum but linear phase.

    """
    # Check spectrum
    if spectrum.ndim < 2:
        spectrum = spectrum[..., None]
    assert spectrum.ndim < 3, \
        'Spectrum should have shape (bins, channels)'
    if spectrum.shape[0] < spectrum.shape[1]:
        spectrum = spectrum.T
    spectrum = np.abs(spectrum)

    # Check group delay ms parameter
    minimum_group_delay = False
    if type(group_delay_ms) == str:
        group_delay_ms = group_delay_ms.lower()
        assert group_delay_ms == 'minimal', \
            'Group delay should be set to minimal'
        minimum_group_delay = True
    elif type(group_delay_ms) in (float, int):
        group_delay_ms /= 1000
    else:
        raise TypeError('group_delay_ms must be either str, float or int')

    # Frequency vector
    f_vec = np.fft.rfftfreq(spectrum.shape[0]*2-1, 1/sampling_rate_hz)
    delta_f = f_vec[1]-f_vec[0]

    # New spectrum
    lin_spectrum = np.empty(spectrum.shape, dtype='cfloat')
    for n in range(spectrum.shape[1]):
        if check_causality or minimum_group_delay:
            min_phase = _minimum_phase(spectrum[:, n], False)
            min_gd = _group_delay_direct(min_phase, delta_f)
            gd = np.max(min_gd) + 1e-3  # add 1 ms as safety factor
            if check_causality and type(group_delay_ms) != str:
                assert gd <= group_delay_ms, \
                    f'Given group delay {group_delay_ms*1000} ms is lower ' +\
                    f'than minimal group delay {gd*1000} ms for channel {n}'
                gd = group_delay_ms
        else:
            gd = group_delay_ms
        lin_spectrum[:, n] = spectrum[:, n]*np.exp(
            -1j * 2 * np.pi * f_vec * gd)
    time_data = np.fft.irfft(lin_spectrum, axis=0)
    sig_lin_phase = Signal(
        None, time_data=time_data,
        sampling_rate_hz=sampling_rate_hz, signal_type=signal_type)
    return sig_lin_phase


def log_mel_spectrogram(s: Signal, channel: int = 0, range_hz=None,
                        n_bands: int = 40, generate_plot: bool = True,
                        **kwargs):
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
    **kwargs : dict, optional
        Pass arguments to define computation of STFT. If nothing is passed, the
        parameters set in the signal will be used.

    Returns
    -------
    time_s : `np.ndarray`
        Time vector.
    f_mel : `np.ndarray`
        Frequency vector in Mel.
    log_mel_sp : `np.ndarray`
        Log mel spectrogram.

    """
    if kwargs:
        s.set_spectrogram_parameters(**kwargs)
    time_s, f_hz, sp = s.get_spectrogram(channel)
    mfilt, f_mel = mel_filterbank(f_hz, range_hz, n_bands, normalize=True)
    log_mel_sp = mfilt @ np.abs(sp)
    log_mel_sp = 20*np.log10(log_mel_sp+1e-30)
    if generate_plot:
        general_matrix_plot(
            log_mel_sp, range_x=[time_s[0], time_s[-1]],
            range_y=[f_mel[0], f_mel[-1]], range_z=50,
            ylabel='Frequency / Mel', xlabel='Time / s',
            ylog=False)
    return time_s, f_mel, log_mel_sp


def mel_filterbank(f_hz: np.ndarray, range_hz=None, n_bands: int = 40,
                   normalize: bool = True):
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
