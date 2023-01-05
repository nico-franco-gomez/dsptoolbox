"""
Methods used for acquiring and windowing transfer functions
"""
import numpy as np
from dsptoolbox import Signal
from dsptoolbox._general_helpers import (_find_frequencies_above_threshold)
from ._transfer_functions import (_spectral_deconvolve,
                                  _window_this_ir)
from dsptoolbox._standard import _welch
from dsptoolbox.standard_functions import pad_trim


def spectral_deconvolve(num: Signal, denum: Signal,
                        mode='regularized', start_stop_hz=None,
                        threshold_db=-30, padding: bool = False,
                        keep_original_length: bool = False):
    """Deconvolution by spectral division of two signals. If the denominator
    signal only has one channel, the deconvolution is done using it for all
    channels of the numerator.

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
        spectral window.
    threshold : int, optional
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
    assert mode in ('regularized', 'window', 'standard'),\
        f'{mode} is not supported. Use regularized, window or None'

    original_length = num.time_data.shape[0]

    if padding:
        num = pad_trim(num, original_length*2)
        denum = pad_trim(denum, original_length*2)

    denum.set_spectrum_parameters(method='standard')
    _, denum_fft = denum.get_spectrum()
    num.set_spectrum_parameters(method='standard')
    freqs_hz, num_fft = num.get_spectrum()
    fs_hz = num.sampling_rate_hz

    new_time_data = np.zeros_like(num.time_data)

    for n in range(num.number_of_channels):
        if multichannel:
            n_denum = 0
        else:
            n_denum = n
        if mode != 'standard':
            #
            if start_stop_hz is None:
                start_stop_hz = \
                    _find_frequencies_above_threshold(
                        denum_fft[:, n_denum], freqs_hz, threshold_db)
            #
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
        new_time_data[:, n] = \
            _spectral_deconvolve(
                num_fft[:, n], denum_fft[:, n_denum], freqs_hz,
                start_stop_hz=start_stop_hz,
                mode=mode)
    new_sig = Signal(None, new_time_data, num.sampling_rate_hz,
                     signal_type='ir')
    if padding:
        if keep_original_length:
            new_sig = pad_trim(new_sig, original_length)
    return new_sig


def window_ir(signal: Signal, constant_percentage=0.75, exp2_trim: int = 13,
              window_type='hann', at_start: bool = True):
    """Windows an IR with trimming and selection of constant valued length.

    Parameters
    ----------
    signal: Signal
        Signal to window
    constant_percentage: float, optional
        Percentage (between 0 and 1) of the window that should be
        constant value. Default: 0.75
    exp2_trim: int, optional
        Exponent of two defining the length to which the IR should be
        trimmed. For avoiding trimming set to `None`. Default: 13.
    window_type: str, optional
        Window function to be used. Available selection from
        scipy.signal.windows: `barthann`, `bartlett`, `blackman`,
        `boxcar`, `cosine`, `hamming`, `hann`, `flattop`, `nuttall` and
        others without extra parameters. Default: `hann`.
    at_start: bool, optional
        Windows the start with a rising window as well as the end.
        Default: `True`.

    Returns
    -------
    new_sig : Signal
        Windowed signal. The used window is also saved under `new_sig.window`.

    """
    assert signal.signal_type in ('rir', 'ir'), \
        f'{signal.signal_type} is not a valid signal type. Use rir or ir.'
    if exp2_trim is not None:
        total_length = int(2**exp2_trim)
    else:
        total_length = len(signal.time_data)
    new_time_data = np.zeros((total_length, signal.number_of_channels))

    for n in range(signal.number_of_channels):
        new_time_data[:, n], window = \
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
                              window_length_samples: int = 1024, **kwargs):
    """Gets transfer function H1, H2 or H3.
    H1: for noise in the output signal. `Gxy/Gxx`.
    H2: for noise in the input signal. `Gyy/Gyx`.
    H3: for noise in both signals. `G_xy / np.abs(G_xy) * (G_yy/G_xx)**0.5`.
    If the input signal only has one channel, it is assumed to be the input
    for all of the channels of the output.

    Parameters
    ----------
    output : Signal
        Signal with output channels.
    input : Signal
        Signal with input channels.
    mode : str, optional
        Type of transfer function. `'h1'`, `'h2'` and `'h3'` are available.
        Default: `'h2'`.
    window_length_samples : int, optional
        Window length for the IR. Spectrum has the length
        window_length_samples//2 + 1. Default: 1024.
    **kwargs : dict, optional
        Extra parameters for the computation of the cross spectral densities
        using welch's method.

    Returns
    -------
    tf : Signal
        Transfer functions. Coherences are also computed and saved in the
        Signal object.

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
    H_time = np.zeros((window_length_samples, output.number_of_channels))
    coherence = np.zeros((window_length_samples//2 + 1,
                          output.number_of_channels))
    if multichannel:
        G_xx = _welch(
            input.time_data[:, 0],
            input.time_data[:, 0],
            input.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **kwargs)
    for n in range(output.number_of_channels):
        G_yy = _welch(
            output.time_data[:, n],
            output.time_data[:, n],
            input.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **kwargs)
        if multichannel:
            n_input = 0
        else:
            n_input = n
            G_xx = _welch(
                input.time_data[:, n_input],
                input.time_data[:, n_input],
                input.sampling_rate_hz,
                window_length_samples=window_length_samples,
                **kwargs)
        if mode == 'h2'.casefold():
            G_yx = _welch(
                    output.time_data[:, n],
                    input.time_data[:, n_input],
                    output.sampling_rate_hz,
                    window_length_samples=window_length_samples,
                    **kwargs)
        G_xy = _welch(
            input.time_data[:, n_input],
            output.time_data[:, n],
            output.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **kwargs)

        if mode == 'h1'.casefold():
            H_time[:, n] = np.fft.irfft(G_xy / G_xx)
        elif mode == 'h2'.casefold():
            H_time[:, n] = np.fft.irfft(G_yy / G_yx)
        elif mode == 'h3'.casefold():
            H_time[:, n] = np.fft.irfft(G_xy / np.abs(G_xy) * (G_yy/G_xx)**0.5)
        coherence[:, n] = np.abs(G_xy)**2 / G_xx / G_yy
    tf = Signal(None, H_time, output.sampling_rate_hz,
                signal_type=mode.lower())
    tf.set_coherence(coherence)
    return tf
