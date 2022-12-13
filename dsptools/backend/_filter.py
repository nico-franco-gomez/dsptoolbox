'''
Backend for filter class and general filtering functions.
'''
import numpy as np
from enum import Enum


def _get_biquad_type(number: int = None, name: str = None):
    if name is not None:
        valid_names = ('peaking', 'lowpass', 'highpass', 'bandpass_skirt',
                       'bandpass_peak', 'notch', 'allpass', 'lowshelf',
                       'highshelf')
        valid_names = (k.casefold() for k in valid_names)
        name = name.casefold()
        assert name in valid_names, f'{name} is not a valid name. Please ' +\
            'select from the _biquad_coefficients'

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
        assert name is not None, 'Either number or name must be given'
        r = eval(f'biquad.{name}')
        r = r.value
    else:
        assert name is None, 'Either number or name must be given, not both'
        r = biquad(number).name
    return r


def _biquad_coefficients(eq_type=0, fs_hz: int = 48000,
                         frequency_hz: float = 1000, gain_db: float = 1,
                         q: float = 1):
    '''
    https://www.musicdsp.org/en/latest/_downloads/3e1dc886e7849251d6747b194d482272/Audio-EQ-Cookbook.txt
    eq_type: 0 PEAKING, 1 LOWPASS, 2 HIGHPASS, 3 BANDPASS_SKIRT,
        4 BANDPASS_PEAK, 5 NOTCH, 6 ALLPASS, 7 LOWSHELF, 8 HIGHSHELF
    '''
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
    '''
    Creates an impulse with the given length

    Parameters
    ----------
    length_samples : int, optional
        Length for the impulse.

    Returns
    -------
    imp : np.ndarray
        Impulse
    '''
    imp = np.zeros(length_samples)
    imp[0] = 1
    return imp


def _group_delay_filter(ba, length_samples: int = 512, fs_hz: int = 48000):
    '''
    Computes group delay using the method in
    https://www.dsprelated.com/freebooks/filters/Phase_Group_Delay.html

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
    f : np.ndarray
        Frequency vector.
    gd : np.ndarray
        Group delay in seconds.
    '''
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
