"""
Methods used for acquiring and windowing transfer functions
"""
import numpy as np
from dsptoolbox.classes.signal_class import Signal
from dsptoolbox._general_helpers import (_find_frequencies_above_threshold)
from ._transfer_functions import (_spectral_deconvolve,
                                  _window_this_ir)
from dsptoolbox._standard import (
    _welch, _minimum_phase, _group_delay_direct, _pad_trim)


def spectral_deconvolve(num: Signal, denum: Signal,
                        mode: str = 'regularized', start_stop_hz=None,
                        threshold_db=-30, padding: bool = False,
                        keep_original_length: bool = False) -> Signal:
    """Deconvolution by spectral division of two signals. If the denominator
    signal only has one channel, the deconvolution is done using that channel
    for all channels of the numerator.

    Parameters
    ----------
    num : `Signal`
        Signal to deconvolve from.
    denum : `Signal`
        Signal to deconvolve.
    mode : str, optional
        `'window'` uses a spectral window in the numerator. `'regularized'`
        uses a regularized inversion. `'standard'` uses direct deconvolution.
        Default: `'regularized'`.
    start_stop_hz : array, None, optional
        `'automatic'` uses a threshold dBFS to create a spectral
        window for the numerator or regularized inversion. Array of 2 or
        4 frequency points can be also manually given. `None` uses no
        spectral window. If mode is standard, start_stop_hz has to be set
        to `None`. Default: `None`.
    threshold_db : int, optional
        Threshold in dBFS for the automatic creation of the window.
        Default: -30.
    padding : bool, optional
        Pads the time data with 2 length. Done for separating distortion
        in negative time bins when deconvolving sweep measurements.
        Default: `False`.
    keep_original_length : bool, optional
        Only regarded when padding is `True`. It trims the newly deconvolved
        data to its original length. Default: `False`.

    Returns
    -------
    new_sig : `Signal`
        Deconvolved signal.

    """
    num = num.copy()
    denum = denum.copy()
    assert num.time_data.shape[0] == denum.time_data.shape[0], \
        'Lengths do not match for spectral deconvolution'
    if denum.number_of_channels != 1:
        assert num.number_of_channels == denum.number_of_channels, \
            'The number of channels do not match.'
        multichannel = False
    else:
        multichannel = True
    assert num.sampling_rate_hz == denum.sampling_rate_hz, \
        'Sampling rates do not match'
    mode = mode.lower()
    assert mode in ('regularized', 'window', 'standard'),\
        f'{mode} is not supported. Use regularized, window or None'
    if mode == 'standard':
        assert start_stop_hz is None, \
            'No start_stop_hz vector can be passed when using standard mode'

    original_length = num.time_data.shape[0]

    if padding:
        num.time_data = _pad_trim(num.time_data, original_length*2)
        denum.time_data = _pad_trim(denum.time_data, original_length*2)
    fft_length = original_length*2 if padding else original_length

    denum.set_spectrum_parameters(method='standard')
    _, denum_fft = denum.get_spectrum()
    num.set_spectrum_parameters(method='standard')
    freqs_hz, num_fft = num.get_spectrum()
    fs_hz = num.sampling_rate_hz

    new_time_data = np.zeros_like(num.time_data)

    for n in range(num.number_of_channels):
        n_denum = 0 if multichannel else n
        if mode != 'standard':
            if start_stop_hz is None:
                start_stop_hz = _find_frequencies_above_threshold(
                    denum_fft[:, n_denum], freqs_hz, threshold_db)
            if len(start_stop_hz) == 2:
                temp = []
                temp.append(start_stop_hz[0]/np.sqrt(2))
                temp.append(start_stop_hz[0])
                temp.append(start_stop_hz[1])
                temp.append(np.min([start_stop_hz[1]*np.sqrt(2), fs_hz/2]))
                start_stop_hz = temp
            elif len(start_stop_hz) == 4:
                pass
            else:
                raise ValueError('start_stop_hz vector should have 2 or 4' +
                                 ' values')
        new_time_data[:, n] = _spectral_deconvolve(
                num_fft[:, n], denum_fft[:, n_denum], freqs_hz,
                fft_length,
                start_stop_hz=start_stop_hz,
                mode=mode)
    new_sig = Signal(None, new_time_data, num.sampling_rate_hz,
                     signal_type='ir')
    if padding:
        if keep_original_length:
            new_sig.time_data = _pad_trim(new_sig.time_data, original_length)
    return new_sig


def window_ir(signal: Signal, constant_percentage=0.75, exp2_trim: int = 13,
              window_type='hann', at_start: bool = True) -> Signal:
    """Windows an IR with trimming and selection of constant valued length.

    Parameters
    ----------
    signal: `Signal`
        Signal to window
    constant_percentage: float, optional
        Percentage (between 0 and 1) of the window's length that should be
        constant value. Default: 0.75.
    exp2_trim: int, optional
        Exponent of two defining the length to which the IR should be
        trimmed. For avoiding trimming set to `None`. Default: 13.
    window_type: str, optional
        Window function to be used. Available selection from
        scipy.signal.windows: `barthann`, `bartlett`, `blackman`,
        `boxcar`, `cosine`, `hamming`, `hann`, `flattop`, `nuttall` and
        others. Pass a tuple with window type and extra parameters if needed.
        Default: `hann`.
    at_start: bool, optional
        Windows the start with a rising window as well as the end.
        Default: `True`.

    Returns
    -------
    new_sig : `Signal`
        Windowed signal. The used window is also saved under `new_sig.window`.

    """
    assert signal.signal_type in ('rir', 'ir'), \
        f'{signal.signal_type} is not a valid signal type. Use rir or ir.'
    if exp2_trim is not None:
        total_length = int(2**exp2_trim)
    else:
        total_length = len(signal.time_data)
    new_time_data = np.zeros((total_length, signal.number_of_channels))

    window = np.zeros((total_length, signal.number_of_channels))
    for n in range(signal.number_of_channels):
        new_time_data[:, n], window[:, n] = \
            _window_this_ir(
                signal.time_data[:, n],
                total_length,
                window_type,
                exp2_trim,
                constant_percentage,
                at_start)

    new_sig = Signal(
        None, new_time_data, signal.sampling_rate_hz,
        signal_type=signal.signal_type)
    new_sig.set_window(window)
    return new_sig


def compute_transfer_function(output: Signal, input: Signal, mode='h2',
                              window_length_samples: int = 1024,
                              spectrum_parameters: dict = None) -> \
        Signal:
    """Gets transfer function H1, H2 or H3 (for stochastic signals).
    H1: for noise in the output signal. `Gxy/Gxx`.
    H2: for noise in the input signal. `Gyy/Gyx`.
    H3: for noise in both signals. `G_xy / abs(G_xy) * (G_yy/G_xx)**0.5`.
    If the input signal only has one channel, it is assumed to be the input
    for all of the channels of the output.

    Parameters
    ----------
    output : `Signal`
        Signal with output channels.
    input : `Signal`
        Signal with input channels.
    mode : str, optional
        Type of transfer function. `'h1'`, `'h2'` and `'h3'` are available.
        Default: `'h2'`.
    window_length_samples : int, optional
        Window length for the IR. Spectrum has the length
        window_length_samples//2 + 1. Default: 1024.
    spectrum_parameters : dict, optional
        Extra parameters for the computation of the cross spectral densities
        using welch's method. See `Signal.set_spectrum_parameters()`
        for details. Default: empty dictionary.

    Returns
    -------
    tf : `Signal`
        Transfer functions as `Signal` object. Coherences are also computed
        and saved in the `Signal` object.

    """
    mode = mode.casefold()
    assert mode in \
        ('h1'.casefold(), 'h2'.casefold(), 'h3'.casefold()), \
        f'{mode} is not a valid mode. Use H1, H2 or H3'
    assert input.sampling_rate_hz == output.sampling_rate_hz, \
        'Sampling rates do not match'
    assert input.time_data.shape[0] == output.time_data.shape[0], \
        'Signal lengths do not match'
    if input.number_of_channels != 1:
        assert input.number_of_channels == output.number_of_channels, \
            'Channel number does not match between signals'
        multichannel = False
    else:
        multichannel = True
    if spectrum_parameters is None:
        spectrum_parameters = {}
    assert type(spectrum_parameters) == dict, \
        'Spectrum parameters should be passed as a dictionary'

    H_time = np.zeros((window_length_samples, output.number_of_channels))
    coherence = np.zeros((window_length_samples//2 + 1,
                          output.number_of_channels))
    if multichannel:
        G_xx = _welch(
            input.time_data[:, 0],
            input.time_data[:, 0],
            input.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **spectrum_parameters)
    for n in range(output.number_of_channels):
        G_yy = _welch(
            output.time_data[:, n],
            output.time_data[:, n],
            input.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **spectrum_parameters)
        if multichannel:
            n_input = 0
        else:
            n_input = n
            G_xx = _welch(
                input.time_data[:, n_input],
                input.time_data[:, n_input],
                input.sampling_rate_hz,
                window_length_samples=window_length_samples,
                **spectrum_parameters)
        if mode == 'h2'.casefold():
            G_yx = _welch(
                    output.time_data[:, n],
                    input.time_data[:, n_input],
                    output.sampling_rate_hz,
                    window_length_samples=window_length_samples,
                    **spectrum_parameters)
        G_xy = _welch(
            input.time_data[:, n_input],
            output.time_data[:, n],
            output.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **spectrum_parameters)

        if mode == 'h1'.casefold():
            H_time[:, n] = np.fft.irfft(G_xy / G_xx)
        elif mode == 'h2'.casefold():
            H_time[:, n] = np.fft.irfft(G_yy / G_yx)
        elif mode == 'h3'.casefold():
            H_time[:, n] = np.fft.irfft(
                G_xy / np.abs(G_xy) * (G_yy/G_xx)**0.5)
        coherence[:, n] = np.abs(G_xy)**2 / G_xx / G_yy
    tf = Signal(None, H_time, output.sampling_rate_hz,
                signal_type=mode.lower())
    tf.set_coherence(coherence)
    return tf


def spectral_average(signal: Signal) -> Signal:
    """Averages all channels of a given IR using their magnitude and
    phase spectra and returns the averaged IR.

    Parameters
    ----------
    signal : `Signal`
        Signal with channels to be averaged over.

    Returns
    -------
    avg_sig : `Signal`
        Averaged signal.

    """
    assert signal.signal_type in ('rir', 'ir'), \
        'Averaging is valid for signal types rir or ir and not ' +\
        f'{signal.signal_type}'
    assert signal.number_of_channels > 1, \
        'Signal has only one channel so no meaningful averaging can be done'

    l_samples = signal.time_data.shape[0]

    # Obtain channel magnitude and phase spectra
    _, sp = signal.get_spectrum()
    mag = np.abs(sp)
    pha = np.unwrap(np.angle(sp), axis=0)

    # Build averages
    new_mag = np.mean(mag, axis=1)
    new_pha = np.mean(pha, axis=1)
    # New signal
    new_sp = new_mag * np.exp(1j*new_pha)

    # New time data and signal object
    new_time_data = np.fft.irfft(new_sp[..., None], n=l_samples, axis=0)
    avg_sig = signal.copy()
    avg_sig.time_data = new_time_data
    return avg_sig


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
                       group_delay_ms: str | float = 'minimum',
                       check_causality: bool = True,
                       signal_type: str = 'ir') -> Signal:
    """Returns a linear phase signal from a magnitude spectrum. It is possible
    to return the smallest causal group delay by checking the minimum phase
    version of the signal and choosing a constant group delay that is never
    lower than minimum group delay (for each channel). A value for the group
    delay can be also passed directly and applied to all channels. If check
    causility is activated, it is assessed that the given group delay is not
    less than each minimum group delay. If deactivated, the generated phase
    could lead to a non-causal system!

    Parameters
    ----------
    spectrum : `np.ndarray`
        Spectrum with only positive frequencies and 0.
    sampling_rate_hz : int
        Signal's sampling rate in Hz.
    group_delay_ms : str or float, optional
        Constant group delay that the phase should have for all channels
        (in ms). Pass `'minimum'` to create a signal with the minimum linear
        phase possible (that is different for each channel).
        Default: `'minimum'`.
    check_causality : bool, optional
        When `True`, it is assessed for each channel that the given group
        delay is not lower than the minimum group delay. Default: `True`.
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
        assert group_delay_ms == 'minimum', \
            'Group delay should be set to minimum'
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
