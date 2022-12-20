"""
General functionality from helper methods
"""
from numpy import (array, ndim, zeros, argmin, abs, ones, concatenate,
                   ndarray, iscomplexobj, sort, log10, max, angle, flip,
                   floor, sum, linspace, zeros_like, sqrt)
from scipy.signal import windows


def _find_nearest(points, vector):
    """Gives back the indexes with the nearest points in vector

    Parameters
    ----------
    points : float or array-like
        Points to look for nearest index in vector.
    vector : array-like
        Vector in which to look for points.

    Returns
    -------
    indexes : int or array
        Indexes of the points
    """
    points = array(points)
    if ndim(points) == 0:
        points = points[..., None]
    indexes = zeros(len(points), dtype=int)
    for ind, p in enumerate(points):
        indexes[ind] = argmin(abs(p - vector))
    return indexes


def _calculate_window(points, window_length: int,
                      window_type='hann', at_start: bool = True,
                      inverse=False):
    """Creates a custom window with given indexes

    Parameters
    ----------
    points: array-like
        Vector containing 4 points for the construction of the custom
        window.
    window_length: int
        Length of the window.
    window_type: str, optional
        Type of window to use. Select from scipy.signal.windows.
        Default: hann.
    at_start: bool, optional
        Creates a half rising window at the start as well. Default: `True`.
    inverse: bool, optional
        When `True`, the window is inversed so that the middle section
        contains 0. Default: False.

    Returns
    -------
    window_full: array
        Custom window
    """
    assert len(points) == 4, 'For the custom window 4 points ' +\
        'are needed'

    idx_start_stop_f = [int(i) for i in points]

    len_low_flank = idx_start_stop_f[1] - idx_start_stop_f[0]
    if at_start:
        low_flank = \
            windows.get_window(
                window_type, len_low_flank*2, fftbins=True)[0:len_low_flank]
    else:
        low_flank = ones(len_low_flank)
    len_high_flank = idx_start_stop_f[3] - idx_start_stop_f[2]
    high_flank = \
        windows.get_window(
            window_type, len_high_flank*2, fftbins=True)[len_high_flank:]
    zeros_low = zeros(idx_start_stop_f[0])
    ones_mid = ones(idx_start_stop_f[2]-idx_start_stop_f[1])
    zeros_high = zeros(window_length-idx_start_stop_f[3])
    window_full = concatenate((zeros_low,
                               low_flank,
                               ones_mid,
                               high_flank,
                               zeros_high))
    if inverse:
        window_full = 1 - window_full
    return window_full


def _get_normalized_spectrum(f, spectra: ndarray, mode='standard',
                             f_range_hz=[20, 20000], normalize: str = None,
                             phase=False, smoothe=0):
    """This function gives a normalized magnitude spectrum with frequency
    vector for a given range. It is also smoothed. Use `None` for the
    spectrum without f_range_hz
    """
    if normalize is not None:
        assert normalize in ('1k', 'max'), \
            f'{normalize} is not a valid normalization mode. Please use ' +\
            '1k or max'
    # Shaping
    if len(spectra.shape) < 2:
        spectra = spectra[..., None]
    # Check for complex spectrum if phase is required
    if phase:
        assert iscomplexobj(spectra), 'Phase computation is not ' +\
            'possible since the spectra are not complex'
    # Factor
    if mode == 'standard':
        factor = 20
    elif mode == 'welch':
        factor = 10
    else:
        raise ValueError(f'{mode} is not supported. Please select standard '
                         'or welch')
    if f_range_hz is not None:
        assert len(f_range_hz) == 2, 'Frequency range must have only ' +\
            'a lower and an upper bound'
        f_range_hz = sort(f_range_hz)
        ids = _find_nearest(f_range_hz, f)
        id1 = ids[0]
        id2 = ids[1]+1  # Contains endpoint
    else:
        id1 = 0
        id2 = -1

    mag_spectra = zeros((id2-id1, spectra.shape[1]))
    phase_spectra = zeros_like(mag_spectra)
    for n in range(spectra.shape[1]):
        sp = spectra[:, n]
        epsilon = 10**(-300/20)
        sp_db = factor*log10(abs(sp) + epsilon)
        if smoothe != 0:
            pass
        if normalize is not None:
            if normalize == '1k':
                id1k = _find_nearest(1e3, f)
                sp_db -= sp_db[id1k]
            else:
                sp_db -= max(sp_db)
        if phase:
            phase_spectra[:, n] = angle(sp[id1:id2])
        mag_spectra[:, n] = sp_db[id1:id2]
    if phase:
        return f[id1:id2], mag_spectra, phase_spectra
    return f[id1:id2], mag_spectra


def _find_frequencies_above_threshold(spec, f, threshold_db, normalize=True):
    """Finds frequencies above a certain threshold in a given spectrum.
    """
    denum_db = 20*log10(abs(spec))
    if normalize:
        denum_db -= max(denum_db)
    freqs = f[denum_db > threshold_db]
    return [freqs[0], freqs[-1]]


def _pad_trim(vector: ndarray, desired_length: int, axis: int = 0,
              in_the_end: bool = True):
    """Pads (with zeros) or trim (depending on size and desired length).
    """
    throw_axis = False
    if len(vector.shape) < 2:
        assert axis == 0, 'You can only pad along the 0 axis'
        vector = vector[..., None]
        throw_axis = True
    type_of_data = vector.dtype
    diff = desired_length - vector.shape[axis]
    if axis == 1:
        vector = vector.T
    if diff > 0:
        if not in_the_end:
            vector = flip(vector, axis=axis)
        new_vec = \
            concatenate(
                [vector,
                    zeros((diff, vector.shape[1]),
                          dtype=type_of_data)])
        if not in_the_end:
            new_vec = flip(new_vec, axis=axis)
    elif diff < 0:
        if not in_the_end:
            vector = flip(vector, axis=axis)
        new_vec = vector[:desired_length, :]
        if not in_the_end:
            new_vec = flip(new_vec, axis=axis)
    else:
        new_vec = vector
    if axis == 1:
        new_vec = new_vec.T
    if throw_axis:
        new_vec = new_vec[:, 0]
    return new_vec


def _compute_number_frames(window_length: int, step: int, signal_length: int):
    """Gives back the number of frames that will be computed.

    Parameters
    ----------
    window_length : int
        Length of the window to be used.
    step : int
        Step size in samples. It is defined as window_length - overlap.
    signal_length : int
        Total signal length.

    Returns
    -------
    n_frames : int
        Number of frames to be observed in the signal.
    padding_samples : int
        Number of samples with which the signal should be padded.
    """
    n_frames = int(floor(signal_length / step)) + 1
    padding_samples = window_length - int(signal_length % step)
    return n_frames, padding_samples


def _normalize(s: ndarray, dbfs: float, mode='peak'):
    """Normalizes a signal.

    Parameters
    ----------
    s: ndarray
        Signal to normalize.
    dbfs: float
        dbfs value to normalize to.
    mode: str, optional
        Mode of normalization, `peak` uses the signal maximum absolute value,
        `rms` uses Root mean square value

    Returns
    -------
    s_out: ndarray
        Normalized signal
    """
    s = s.copy()
    assert mode in ('peak', 'rms'), 'Mode of normalization is not ' +\
        'available. Select either peak or rms'
    if mode == 'peak':
        s /= max(abs(s))
        s *= 10**(dbfs/20)
    if mode == 'rms':
        s *= (10**(dbfs/20) / _rms(s))
    return s


def _rms(x: ndarray):
    """Root mean square computation.
    """
    return sqrt(sum(x**2)/len(x))


def _amplify_db(s: ndarray, db: float):
    """Amplify by dB.
    """
    return s * 10**(db/20)


def _fade(s: ndarray, length_seconds: float = 0.1, mode: str = 'exp',
          sampling_rate_hz: int = 48000, at_start: bool = True):
    """Create a fade in signal.

    Parameters
    ----------
    s : ndarray
        Array to be faded.
    length_seconds : float, optional
        Length of fade in seconds. Default: 0.1.
    mode : str, optional
        Type of fading. Options are `'exp'`, `'lin'`, `'log'`.
        Default: `'lin'`.
    sampling_rate_hz : int, optional
        Sampling rate. Default: 48000.
    at_start : bool, optional
        When `True`, the start is faded. When `False`, the end.
        Default: `True`.

    Returns
    -------
    s : ndarray
        Faded vector.
    """
    mode = mode.lower()
    assert mode in ('exp', 'lin', 'log'), \
        f'{mode} is not supported. Choose from exp, lin, log.'
    assert length_seconds > 0, 'Only positive lengths'
    l_samples = int(length_seconds * sampling_rate_hz)
    assert len(s) > l_samples, \
        'Signal is shorter than the desired fade'
    assert len(s.shape) == 1, 'The fade only takes 1d-arrays'

    if mode == 'exp':
        db = linspace(-100, 0, l_samples)
        fade = 10**(db/20)
    elif mode == 'lin':
        fade = linspace(0, 1, l_samples)
    else:
        db = linspace(-100, 0, l_samples)
        fade = 10**(db/20)
        fade = 1 - flip(fade)
    if not at_start:
        s = flip(s)
    s[:l_samples] *= fade
    if not at_start:
        s = flip(s)
    return s
