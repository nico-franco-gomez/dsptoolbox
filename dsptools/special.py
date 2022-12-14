'''
Here are methods considered as somewhat special computations
'''
import numpy as np
from .signal_class import Signal


def cepstrum(signal: Signal, mode='power'):
    '''
    Returns the cepstrum of a given signal in the Quefrency domain.
    See: https://de.wikipedia.org/wiki/Cepstrum.

    Parameters
    ----------
    signal : Signal
        Signal to compute the cepstrum from.
    mode : str, optional
        Type of cepstrum. Supported modes are `'power'`, `'real'` and
        `'complex'`. Default: `'power'`.

    Returns
    -------
    que : np.ndarray
        Quefrency
    ceps : np.ndarray
        Cepstrum
    '''
    mode = mode.lower()
    assert mode in ('power', 'complex', 'real'), \
        f'{mode} is not a supported mode'

    ceps = np.zeros_like(signal.time_data)
    signal.set_spectrum_parameters(method='standard')
    f, sp = signal.get_spectrum()

    for n in range(signal.number_of_channels):
        # sp = np.fft.rfft(signal.time_data[:, n] -
        #                  np.mean(signal.time_data[:, n]))
        if mode in ('power', 'real'):
            # cp = np.abs(np.fft.irfft((2*np.log(np.abs(sp)))))**2
            cp = np.abs(np.fft.irfft((2*np.log(np.abs(sp[:, n])))))**2
        else:
            # cp = np.fft.irfft(np.log(sp))
            cp = np.fft.irfft(np.log(sp[:, n]))
        if mode == 'real':
            cp = (cp**0.5)/2
        ceps[:, n] = cp
    return ceps
