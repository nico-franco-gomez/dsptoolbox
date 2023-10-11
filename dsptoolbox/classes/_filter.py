"""
Backend for filter class and general filtering functions.
"""
import numpy as np
from warnings import warn
from enum import Enum
import scipy.signal as sig
from .signal_class import Signal
from .multibandsignal import MultiBandSignal
from .._general_helpers import _polyphase_decomposition


def _get_biquad_type(number: int | None = None, name: str | None = None):
    """Helper method that handles string inputs for the biquad filters.

    """
    if name is not None:
        name = name.lower()
        valid_names = ('peaking', 'lowpass', 'highpass', 'bandpass_skirt',
                       'bandpass_peak', 'notch', 'allpass', 'lowshelf',
                       'highshelf')
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


def _biquad_coefficients(eq_type: int | str = 0, fs_hz: int = 48000,
                         frequency_hz: float = 1000, gain_db: float = 1,
                         q: float = 1):
    """Creates the filter coefficients for biquad filters.
    eq_type: 0 PEAKING, 1 LOWPASS, 2 HIGHPASS, 3 BANDPASS_SKIRT,
        4 BANDPASS_PEAK, 5 NOTCH, 6 ALLPASS, 7 LOWSHELF, 8 HIGHSHELF.

    References
    ----------
    - https://www.w3.org/TR/2021/NOTE-audio-eq-cookbook-20210608/

    """
    # Asserts and input safety
    if type(eq_type) is str:
        eq_type = _get_biquad_type(None, eq_type)
    # frequency_hz
    frequency_hz = np.asarray(frequency_hz)
    if frequency_hz.ndim > 0:
        frequency_hz = np.mean(frequency_hz)
        warn('More than one frequency was passed for biquad filter. This is ' +
             'not supported. A mean of passed frequencies was used for the ' +
             'design but this might not give the intended result!')
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


def _impulse(length_samples: int = 512, delay_samples: int = 0):
    """Creates an impulse with the given length

    Parameters
    ----------
    length_samples : int, optional
        Length for the impulse. Default: 512.
    delay_samples : int, optional
        Delay of the impulse. Default: 0.

    Returns
    -------
    imp : `np.ndarray`
        Impulse.

    """
    imp = np.zeros(length_samples)
    imp[delay_samples] = 1
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
    zi : list
        None if passed zi was None.

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

    # Check for complex output
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
                         zi: list | None = None, zero_phase: bool = False,
                         filter_type: str = 'iir',
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
    filter_type : str, optional
        Filter type. When FIR, an own implementation of lfilter is used,
        otherwise scipy.signal.lfilter is used. Default: `'iir'`.
    warning_on_complex_output: bool, optional
        When `True`, there is a warning when the output is complex. Either way,
        only the real part is regarded. Default: `True`.

    Returns
    -------
    new_signal : `Signal`
        New Signal object.
    zi : list
        None if passed zi was None.

    """
    # Take lfilter function, might be a different one depending if filter is
    # FIR or IIR
    if filter_type == 'fir':
        lfilter = _lfilter_fir
    elif filter_type in ('iir', 'biquad'):
        lfilter = sig.lfilter
    else:
        raise ValueError(
            f'{filter_type} is not supported. Use either fir or iir')

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
                zi=zi[:, channels], axis=0)
    else:
        if zero_phase:
            y = sig.filtfilt(
                b=ba[0], a=ba[1], x=signal.time_data[:, channels], axis=0)
        else:
            y = lfilter(
                ba[0], a=ba[1], x=signal.time_data[:, channels], axis=0)

    # Check for complex output
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
                 zi: np.ndarray | None = None, axis: int = 0):
    """Variant to the `scipy.signal.lfilter` that uses `scipy.signal.convolve`
    for filtering. The advantage of this is that the convolution will be
    automatically made using fft or direct, depending on the inputs' sizes.
    This is only used for FIR filters.

    The `axis` parameter is only there for compatibility with
    `scipy.signal.lfilter`, but the first axis is always used.

    """
    assert len(a) == 1, \
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


def _filter_and_downsample(time_data: np.ndarray, down_factor: int,
                           ba_coefficients: list, polyphase: bool) \
        -> np.ndarray:
    """Filters and downsamples time data. If polyphase is `True`, it is
    assumed that the filter is FIR and only b-coefficients are used. In
    that case, an efficient downsampling is done, otherwise standard filtering
    and downsampling is applied.

    Parameters
    ----------
    time_data : `np.ndarray`
        Time data to be filtered and resampled. Shape should be (time samples,
        channels).
    down_factor : int
        Factor by which it will be downsampled.
    ba_coefficients : list
        List containing [b, a] coefficients. If polyphase is set to `True`,
        only b coefficients are regarded.
    polyphase : bool
        Use polyphase representation or not.

    Returns
    -------
    new_time_data : `np.ndarray`
        New time data with downsampling.

    """
    if time_data.ndim == 1:
        time_data = time_data[..., None]
    assert time_data.ndim == 2, \
        'Shape for time data should be (time samples, channels)'

    if polyphase:
        poly, _ = _polyphase_decomposition(time_data, down_factor, flip=False)
        # (time samples, polyphase components, channels)
        # Polyphase representation of filter and filter length
        b = ba_coefficients[0]
        half_length = (len(b) - 1) // 2
        b_poly, _ = _polyphase_decomposition(b, down_factor, flip=True)
        new_time_data = np.zeros(
            (poly.shape[0]+b_poly.shape[0]-1, poly.shape[2]))
        # Accumulator for each channel – it would be better to find a way
        # to do it without loops, but using scipy.signal.convolve since it
        # is advantageous compared to numpy.convolve
        for ch in range(poly.shape[2]):
            temp = np.zeros(new_time_data.shape[0])
            for n in range(poly.shape[1]):
                temp += sig.convolve(
                    poly[:, n, ch], b_poly[:, n, 0], mode='full')
            new_time_data[:, ch] = temp
        # Take correct values from vector
        new_time_data = \
            new_time_data[
                half_length//down_factor:-half_length//down_factor, :]
    else:
        new_time_data = sig.lfilter(
            ba_coefficients[0], ba_coefficients[1], x=time_data, axis=0)
        new_time_data = new_time_data[::down_factor]

    return new_time_data


def _filter_and_upsample(time_data: np.ndarray, up_factor: int,
                         ba_coefficients: list, polyphase: bool):
    """Filters and upsamples time data. If polyphase is `True`, it is
    assumed that the filter is FIR and only b-coefficients are used. In
    that case, an efficient polyphase upsampling is done, otherwise standard
    upsampling and filtering is applied.

    NOTE: The polyphase implementation uses two loops: once for the polyphase
    components and once for the channels. Hence, it might not be much faster
    than usual filtering.

    Parameters
    ----------
    time_data : `np.ndarray`
        Time data to be filtered and resampled. Shape should be (time samples,
        channels).
    up_factor : int
        Factor by which it will be upsampled.
    ba_coefficients : list
        List containing [b, a] coefficients. If polyphase is set to `True`,
        only b coefficients are regarded.
    polyphase : bool
        Use polyphase representation or not.

    Returns
    -------
    new_time_data : `np.ndarray`
        New time data with downsampling.

    """
    if time_data.ndim == 1:
        time_data = time_data[..., None]
    assert time_data.ndim == 2, \
        'Shape for time data should be (time samples, channels)'

    if polyphase:
        b = ba_coefficients[0]
        half_length = (len(b) - 1) // 2

        # Decompose filter
        b_poly, padding = _polyphase_decomposition(b, up_factor)
        b_poly *= up_factor

        # Accumulator – Length is not right!
        new_time_data = np.zeros(
            ((time_data.shape[0] + b_poly.shape[0] - 1)*up_factor,
             time_data.shape[1]))

        # Interpolate per channel and per polyphase component – should be
        # a better way to do it without the loops...
        for ch in range(time_data.shape[1]):
            for ind in range(up_factor):
                new_time_data[ind::up_factor, ch] = sig.convolve(
                        time_data[:, ch], b_poly[:, ind, 0], mode='full')

        # Take right samples from filtered signal
        if padding == up_factor:
            new_time_data = new_time_data[half_length:-half_length, :]
        else:
            new_time_data = \
                new_time_data[half_length+padding:-half_length+padding, :]
    else:
        new_time_data = np.zeros(
            (time_data.shape[0]*up_factor, time_data.shape[1]))
        new_time_data[::up_factor] = time_data
        new_time_data = sig.lfilter(
            ba_coefficients[0], ba_coefficients[1], x=new_time_data, axis=0)
    return new_time_data
