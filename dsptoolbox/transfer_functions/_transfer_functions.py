"""
Backend for transfer functions methods
"""
import numpy as np
from dsptoolbox._general_helpers import _find_nearest, _calculate_window


def _spectral_deconvolve(num_fft: np.ndarray, denum_fft: np.ndarray, freqs_hz,
                         mode='regularized', start_stop_hz=None):
    assert num_fft.shape == denum_fft.shape, 'Shapes do not match'
    assert len(freqs_hz) == len(num_fft), 'Frequency vector does not match'

    if mode == 'regularized':
        # Regularized division
        ids = _find_nearest(start_stop_hz, freqs_hz)
        outside = 30
        inside = 10**(-200/20)
        eps = _calculate_window(ids, len(freqs_hz), inverse=True)
        eps += inside
        eps *= outside
        denum_reg = denum_fft.conj() /\
            (denum_fft.conj()*denum_fft + eps)
        new_time_data = \
            np.fft.irfft(num_fft * denum_reg)
    elif mode == 'window':
        ids = _find_nearest(start_stop_hz, freqs_hz)
        window = _calculate_window(ids, len(freqs_hz), inverse=False)
        window += 10**(-200/10)
        num_fft_n = num_fft * window
        new_time_data = np.fft.irfft(
            np.divide(num_fft_n, denum_fft))
    elif mode == 'standard':
        new_time_data = np.fft.irfft(
            np.divide(num_fft, denum_fft))
    else:
        raise ValueError(f'{mode} is not supported. Choose window' +
                         ', regularized or standard')
    return new_time_data


def _window_this_ir(vec, total_length: int, window_type: str = 'hann',
                    exp2_trim: int = 13, constant_percentage: float = 0.75,
                    at_start: bool = True):
    # Trimming
    if exp2_trim is not None:
        # Padding
        if 2**exp2_trim >= len(vec):
            # Padding
            vec = np.hstack([vec, np.zeros(total_length - len(vec))])
            length = np.argmax(abs(vec))
        else:
            # Selecting
            assert constant_percentage > 0 and constant_percentage < 1,\
                'Constant percentage must be between 0 and 1'
            length = int((1-constant_percentage)*2**exp2_trim)//2
            ind_max = np.argmax(abs(vec))
            if ind_max - length < 0:
                length = ind_max
            vec = vec[ind_max-length:ind_max-length+2**exp2_trim]
    else:
        length = np.argmax(abs(vec))
    points = [0, length, total_length-length, total_length]
    window = _calculate_window(points, total_length, window_type,
                               at_start=at_start)
    return vec*window, window
