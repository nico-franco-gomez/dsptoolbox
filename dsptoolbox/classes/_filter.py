"""
Backend for filter class and general filtering functions.
"""
import numpy as np
from warnings import warn
from enum import Enum
import scipy.signal as sig
from .signal_class import Signal
from .multibandsignal import MultiBandSignal


def _get_biquad_type(number: int = None, name: str = None):
    """Helper method that handles string inputs for the biquad filters.

    """
    if name is not None:
        valid_names = ('peaking', 'lowpass', 'highpass', 'bandpass_skirt',
                       'bandpass_peak', 'notch', 'allpass', 'lowshelf',
                       'highshelf')
        name = name.lower()
        assert name in valid_names, f'{name} is not a valid name. Please ' +\
            '''select from the ('peaking', 'lowpass', 'highpass',
            'bandpass_skirt', 'bandpass_peak', 'notch', 'allpass', 'lowshelf',
            'highshelf')'''

    class biquad(Enum):
        peaking = 0
        lowpass = 1
        highpass = 2
        bandpass_skirt = 3
        bandpass_peak = 4
        notch = 5
        allpass = 6
        lowshelf = 7
        highshelf = 8

    if number is None:
        assert name is not None, \
            'Either number or name must be given, not both'
        r = eval(f'biquad.{name}')
        r = r.value
    else:
        assert name is None, 'Either number or name must be given, not both'
        r = biquad(number).name
    return r


def _biquad_coefficients(eq_type=0, fs_hz: int = 48000,
                         frequency_hz: float = 1000, gain_db: float = 1,
                         q: float = 1):
    """Creates the filter coefficients for biquad filters.
    https://www.musicdsp.org/en/latest/_downloads/3e1dc886e7849251d6747b194d482272/Audio-EQ-Cookbook.txt
    eq_type: 0 PEAKING, 1 LOWPASS, 2 HIGHPASS, 3 BANDPASS_SKIRT,
        4 BANDPASS_PEAK, 5 NOTCH, 6 ALLPASS, 7 LOWSHELF, 8 HIGHSHELF.

    """
    A = np.sqrt(10**(gain_db / 20.0))
    Omega = 2.0 * np.pi * (frequency_hz / fs_hz)
    sn = np.sin(Omega)
    cs = np.cos(Omega)
    alpha = sn / (2.0 * q)
    a = np.ones(3, dtype=np.float32)
    b = np.ones(3, dtype=np.float32)
    if eq_type == 0:  # Peaking
        b[0] = 1 + alpha * A
        b[1] = -2*cs
        b[2] = 1 - alpha * A
        a[0] = 1 + alpha / A
        a[1] = -2 * cs
        a[2] = 1 - alpha / A
    elif eq_type == 1:  # Lowpass
        b[0] = (1 - cs) / 2
        b[1] = 1 - cs
        b[2] = b[0]
        a[0] = 1 + alpha
        a[1] = -2 * cs
        a[2] = 1 - alpha
    elif eq_type == 2:  # Highpass
        b[0] = (1 + cs) / 2.0
        b[1] = -1 * (1 + cs)
        b[2] = b[0]
        a[0] = 1 + alpha
        a[1] = -2 * cs
        a[2] = 1 - alpha
    elif eq_type == 3:  # Bandpass skirt
        b[0] = sn / 2
        b[1] = 0
        b[2] = -b[0]
        a[0] = 1 + alpha
        a[1] = -2 * cs
        a[2] = 1 - alpha
    elif eq_type == 4:  # Bandpass peak
        b[0] = alpha
        b[1] = 0
        b[2] = -b[0]
        a[0] = 1 + alpha
        a[1] = -2 * cs
        a[2] = 1 - alpha
    elif eq_type == 5:  # Notch
        b[0] = 1
        b[1] = -2 * cs
        b[2] = b[0]
        a[0] = 1 + alpha
        a[1] = -2 * cs
        a[2] = 1 - alpha
    elif eq_type == 6:  # Allpass
        b[0] = 1 - alpha
        b[1] = -2 * cs
        b[2] = 1 + alpha
        a[0] = 1 + alpha
        a[1] = -2 * cs
        a[2] = 1 - alpha
    elif eq_type == 7:  # Lowshelf
        b[0] = A*((A+1) - (A-1)*cs + 2*np.sqrt(A)*alpha)
        b[1] = 2*A*((A-1) - (A+1)*cs)
        b[2] = A*((A+1) - (A-1)*cs - 2*np.sqrt(A)*alpha)
        a[0] = (A+1) + (A-1)*cs + 2*np.sqrt(A)*alpha
        a[1] = -2*((A-1) + (A+1)*cs)
        a[2] = (A+1) + (A-1)*cs - 2*np.sqrt(A)*alpha
    elif eq_type == 8:  # Highshelf
        b[0] = A*((A+1) + (A-1)*cs + 2*np.sqrt(A)*alpha)
        b[1] = -2*A*((A-1) + (A+1)*cs)
        b[2] = A*((A+1) + (A-1)*cs - 2*np.sqrt(A)*alpha)
        a[0] = (A+1) - (A-1)*cs + 2*np.sqrt(A)*alpha
        a[1] = 2*((A-1) - (A+1)*cs)
        a[2] = (A+1) - (A-1)*cs - 2*np.sqrt(A)*alpha
    else:
        raise Exception('eq_type not supported')
    return b, a


def _impulse(length_samples: int = 512):
    """Creates an impulse with the given length

    Parameters
    ----------
    length_samples : int, optional
        Length for the impulse.

    Returns
    -------
    imp : `np.ndarray`
        Impulse.

    """
    imp = np.zeros(length_samples)
    imp[0] = 1
    return imp


def _group_delay_filter(ba, length_samples: int = 512, fs_hz: int = 48000):
    """Computes group delay using the method in
    https://www.dsprelated.com/freebooks/filters/Phase_Group_Delay.html.
    The implementation is mostly taken from `scipy.signal.group_delay` !

    Parameters
    ----------
    ba : array-like
        Array containing b (numerator) and a (denominator) for filter.
    length_samples : int, optional
        Length for the final vector. Default: 512.
    fs_hz : int, optional
        Sampling frequency rate in Hz. Default: 48000.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    gd : `np.ndarray`
        Group delay in seconds.

    """
    # Frequency vector at which to evaluate
    omega = np.linspace(0, np.pi, length_samples)
    # Turn always to FIR
    c = np.convolve(ba[0], np.conjugate(ba[1][::-1]))
    cr = c * np.arange(len(c))  # Ramped coefficients
    # Evaluation
    num = np.polyval(cr, np.exp(1j*omega))
    denum = np.polyval(c, np.exp(1j*omega))

    # Group delay
    gd = np.real(num/denum) - len(ba[1]) + 1

    # Look for infinite values
    gd[~np.isfinite(gd)] = 0
    f = omega/np.pi*(fs_hz/2)
    gd /= fs_hz
    return f, gd


def _filter_on_signal(signal: Signal, sos, channels=None,
                      zi=None, zero_phase: bool = False,
                      warning_on_complex_output: bool = True):
    """Takes in a `Signal` object and filters selected channels. Exports a new
    `Signal` object.

    Parameters
    ----------
    signal : `Signal`
        Signal to be filtered.
    sos : array-like
        SOS coefficients of filter.
    channels : int or array-like, optional
        Channel or array of channels to be filtered. When `None`, all
        channels are filtered. Default: `None`.
    zi : array-like, optional
        When not `None`, the filter state values are updated after filtering.
        Default: `None`.
    zero_phase : bool, optional
        Uses zero-phase filtering on signal. Be aware that the filter
        is doubled in this case. Default: `False`.
    warning_on_complex_output: bool, optional
        When `True`, there is a warning when the output is complex. Either way,
        only the real part is regarded. Default: `True`.

    Returns
    -------
    new_signal : `Signal`
        New Signal object.

    """
    # Time Data
    new_time_data = signal.time_data

    # zi unpacking
    if zi is not None:
        zi = np.moveaxis(np.asarray(zi), 0, -1)

    # Channels
    if channels is None:
        channels = np.arange(signal.number_of_channels)

    # Filtering
    if zi is not None:
        y, zi[:, :, channels] = \
            sig.sosfilt(
                sos, signal.time_data[:, channels],
                zi=zi[:, :, channels], axis=0)
    else:
        if zero_phase:
            y = sig.sosfiltfilt(sos, signal.time_data[:, channels], axis=0)
        else:
            y = sig.sosfilt(sos, signal.time_data[:, channels], axis=0)

    # Cast to real if complex
    if np.iscomplexobj(y):
        if warning_on_complex_output:
            warn('Filter output is complex. Imaginary part is saved in ' +
                 'Signal as time_data_imaginary')
        new_time_data = new_time_data.astype('cfloat')

    # Create new signal
    new_time_data[:, channels] = y
    new_signal = signal.copy()
    new_signal.time_data = new_time_data

    # zi packing
    if zi is not None:
        zi_new = []
        for n in range(signal.number_of_channels):
            zi_new.append(zi[:, :, n])
    return new_signal, zi


def _filter_on_signal_ba(signal: Signal, ba, channels=None,
                         zi=None, zero_phase: bool = False,
                         warning_on_complex_output: bool = True):
    """Takes in a `Signal` object and filters selected channels. Exports a new
    `Signal` object.

    Parameters
    ----------
    signal : `Signal`
        Signal to be filtered.
    ba : list
        List with ba coefficients of filter. Form ba=[b, a] where b and a
        are of type `np.ndarray`.
    channels : array-like, optional
        Channel or array of channels to be filtered. When `None`, all
        channels are filtered. Default: `None`.
    zi : list, optional
        When not `None`, the filter state values are updated after filtering.
        They should be passed as a list with the zi 1D-arrays.
        Default: `None`.
    zero_phase : bool, optional
        Uses zero-phase filtering on signal. Be aware that the filter
        is doubled in this case. Default: `False`.
    warning_on_complex_output: bool, optional
        When `True`, there is a warning when the output is complex. Either way,
        only the real part is regarded. Default: `True`.

    Returns
    -------
    new_signal : `Signal`
        New Signal object.

    """
    # Take lfilter function, might be a different one depending if filter is
    # FIR or IIR
    lfilter = sig.lfilter
    # See if it is FIR Filter and normalize to a[0] = 1
    ba[0] = np.atleast_1d(ba[0])
    ba[1] = np.atleast_1d(np.asarray(ba[1]).squeeze())
    if len(ba[1]) == 1:
        ba[0] = np.asarray(ba[0]) / np.squeeze(ba[1])
        ba[1] = 1
        lfilter = _lfilter_fir

    # Time Data
    new_time_data = signal.time_data

    # zi unpacking
    if zi is not None:
        zi = np.asarray(zi).T

    # Channels
    if channels is None:
        channels = np.arange(signal.number_of_channels)

    # Filtering
    if zi is not None:
        y, zi[:, channels] = lfilter(
                ba[0], a=ba[1], x=signal.time_data[:, channels],
                zi=zi[:, channels])
    else:
        if zero_phase:
            y = sig.filtfilt(
                b=ba[0], a=ba[1], x=signal.time_data[:, channels], axis=0)
        else:
            y = lfilter(
                ba[0], a=ba[1], x=signal.time_data[:, channels])

    # Take only real part if output is complex
    if np.iscomplexobj(y):
        if warning_on_complex_output:
            warn('Filter output is complex. Imaginary part is saved in ' +
                 'Signal as time_data_imaginary')
        new_time_data = new_time_data.astype('cfloat')

    # Create new signal
    new_time_data[:, channels] = y
    new_signal = signal.copy()
    new_signal.time_data = new_time_data

    # zi packing
    if zi is not None:
        zi_new = []
        for n in range(zi.shape[1]):
            zi_new.append(zi[:, n])
    return new_signal, zi


def _filterbank_on_signal(signal: Signal, filters, activate_zi: bool = False,
                          mode: str = 'parallel', zero_phase: bool = False,
                          same_sampling_rate: bool = True):
    """Applies filter bank on a given signal.

    Parameters
    ----------
    signal : `Signal`
        Signal to be filtered.
    filters : list
        List containing filters to be applied to signal.
    activate_zi : bool, optional
        When `True`, the filter initial values for each channel are updated
        while filtering. Default: `None`.
    mode : str, optional
        Mode of filtering. Choose from `'parallel'`, `'sequential'` and
        `'summed'`. Default: `'parallel'`.
    zero_phase : bool, optional
        Uses zero-phase filtering on signal. Be aware that the filter order
        is doubled in this case. Default: `False`.
    same_sampling_rate : bool, optional
        When `True`, the output MultiBandSignal (parallel filtering) has
        same sampling rate for all bands. Default: `True`.

    Returns
    -------
    new_signal : `Signal` or `MultiBandSignal`
        New Signal object.

    """
    n_filt = len(filters)
    if mode == 'parallel':
        ss = []
        for n in range(n_filt):
            ss.append(
                filters[n].filter_signal(
                    signal, activate_zi=activate_zi, zero_phase=zero_phase))
        out_sig = MultiBandSignal(
            ss, same_sampling_rate=same_sampling_rate)
    elif mode == 'sequential':
        out_sig = signal.copy()
        for n in range(n_filt):
            out_sig = \
                filters[n].filter_signal(
                    out_sig, activate_zi=activate_zi, zero_phase=zero_phase)
    else:
        new_time_data = \
            np.zeros((signal.time_data.shape[0],
                      signal.number_of_channels, n_filt))
        for n in range(n_filt):
            s = filters[n].filter_signal(
                    signal, activate_zi=activate_zi, zero_phase=zero_phase)
            new_time_data[:, :, n] = s.time_data
        new_time_data = np.sum(new_time_data, axis=-1)
        out_sig = signal.copy()
        out_sig.time_data = new_time_data
    return out_sig


def _lfilter_fir(b: np.ndarray, a: np.ndarray, x: np.ndarray,
                 zi: np.ndarray = None):
    """Variant to the `scipy.signal.lfilter` that uses `scipy.signal.convolve`
    for filtering. The advantage of this is that the convolution will be
    automatically made using fft or direct, depending on the inputs' sizes.
    This is only used for FIR filters.

    """
    assert a == 1, \
        f'{a} is not valid. It has to be 1 in order to be a valid FIR filter'

    # b dimensions handling
    if b.ndim != 1:
        b = np.squeeze(b)
        assert b.ndim == 1, \
            'FIR Filters for audio must be 1D-arrays'

    # Dimensions of zi and x must match
    if zi is not None:
        assert zi.ndim == x.ndim, \
            'Vector to filter and initial values should have the same ' +\
            'number of dimensions!'
    if x.ndim < 2:
        x = x[..., None]
        if zi is not None:
            zi = zi[..., None]
    assert x.ndim == 2, \
        'Filtering only works on 2D-arrays'

    # Convolving
    y = sig.convolve(x, b[..., None], mode='full')

    # Use zi's and take zf's
    if zi is not None:
        y[:zi.shape[0], :] += zi
        zf = y[-zi.shape[0]:, :]

    # Trim output
    y = y[:x.shape[0], :]
    if zi is None:
        return y
    return y, zf
