"""
Here are methods considered as somewhat special or less common.
"""
from numpy import fft, zeros_like, unwrap, angle, abs, log
from .classes.signal_class import Signal


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
    que : ndarray
        Quefrency
    ceps : ndarray
        Cepstrum

    References
    ----------
    https://de.wikipedia.org/wiki/Cepstrum
    """
    mode = mode.lower()
    assert mode in ('power', 'complex', 'real'), \
        f'{mode} is not a supported mode'

    ceps = zeros_like(signal.time_data)
    signal.set_spectrum_parameters(method='standard')
    f, sp = signal.get_spectrum()

    for n in range(signal.number_of_channels):
        if mode in ('power', 'real'):
            cp = abs(fft.irfft((2*log(abs(sp[:, n])))))**2
        else:
            phase = unwrap(angle(sp[:, n]))
            cp = fft.irfft(log(abs(sp[:, n])) + 1j*phase).real
        if mode == 'real':
            cp = (cp**0.5)/2
        ceps[:, n] = cp
    return ceps
