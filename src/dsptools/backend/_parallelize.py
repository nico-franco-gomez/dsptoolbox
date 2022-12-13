'''
Deeper level functions to be parallelized with numba
'''
import numpy as np
import numba as nb
rfft = np.fft.rfft


# ========= NOTES
# Parallelization with Numba here is not possible because np.fft.rfft is not
# supported. Maybe in the future it will...
# =========
@nb.jit(parallel=True, nopython=True)
def _spectral_density(x: np.ndarray, y: np.ndarray,
                      window, detrend, magnitude, phase, step):
    '''
    This function is the parallelized computation for the density or the
    stft matrix.
    '''
    window_length_samples = len(window)
    start = 0
    n_frames = magnitude.shape[1]
    if y is not None:
        for n in nb.prange(n_frames):
            time_x = x[start:start+window_length_samples].copy()
            time_y = y[start:start+window_length_samples].copy()
            # Windowing
            time_x *= window
            time_y *= window
            # Detrend
            if detrend:
                time_x -= np.mean(time_x)
                time_y -= np.mean(time_y)
            # Spectra
            sp_x = rfft(time_x)
            sp_y = rfft(time_y)
            m = (sp_x.conjugate() * sp_y)
            magnitude[:, n] = np.abs(m)
            phase[:, n] = np.unwrap(np.angle(m))
            start += step
    else:
        for n in nb.prange(n_frames):
            time_x = x[start:start+window_length_samples].copy()
            # Windowing
            time_x *= window
            # Detrend
            if detrend:
                time_x -= np.mean(time_x)
            # Spectra
            magnitude[:, n] = rfft(time_x)
            start += step
        phase = np.array([])
    return magnitude, phase
