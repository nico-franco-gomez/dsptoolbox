"""
Here are functions in experimental phase. This might not work as expected
or not at all.
"""
from scipy.signal import windows
import numpy as np
from scipy.interpolate import interp1d


def _smoothing_log(vector, num_fractions, window_type='hann'):
    """Fractional octave smoothing with efficient logarithmic interpolation
    in the frequency domain.

    References
    ----------
    Tylka, Joseph & Boren, Braxton & Choueiri, Edgar. (2017). A Generalized
    Method for Fractional-Octave Smoothing of Transfer Functions that
    Preserves Log-Frequency Symmetry. Journal of the Audio Engineering
    Society. 65. 239-245. 10.17743/jaes.2016.0053.
    """
    # Parameters
    N = len(vector)
    l1 = np.arange(N)
    k_log = (N)**(l1/(N-1))
    beta = np.log2(k_log[1])
    n_window = int(2 * np.floor(1 / (num_fractions * beta * 2)) + 1)
    # print(n_window)
    # Interpolate
    vec_int = interp1d(np.arange(N)+1, vector, kind='cubic')
    vec_log = vec_int(k_log)
    w_func = eval(f'windows.{window_type}')
    if window_type == 'gaussian':
        window = w_func(n_window, 1, sym=True)
    else:
        window = w_func(n_window, sym=True)
    smoothed = np.convolve(vec_log, window, mode='same')
    smoothed = \
        interp1d(k_log, smoothed, kind='cubic')
    vec_final = smoothed(np.arange(N)+1)
    return vec_final


if __name__ == '__main__':
    windows.__name__
