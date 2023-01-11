"""
General functionality from helper methods
"""
import numpy as np
from scipy.signal import windows
from scipy.interpolate import interp1d


def _find_nearest(points, vector):
    """Gives back the indexes with the nearest points in vector

    Parameters
    ----------
    points : float or array_like
        Points to look for nearest index in vector.
    vector : array_like
        Vector in which to look for points.

    Returns
    -------
    indexes : int or np.array
        Indexes of the points.

    """
    points = np.array(points)
    if np.ndim(points) == 0:
        points = points[..., None]
    indexes = np.zeros(len(points), dtype=int)
    for ind, p in enumerate(points):
        indexes[ind] = np.argmin(np.abs(p - vector))
    return indexes


def _calculate_window(points, window_length: int,
                      window_type='hann', at_start: bool = True,
                      inverse=False):
    """Creates a custom window with given indexes

    Parameters
    ----------
    points: array_like
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
    window_full: np.array
        Custom window.

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
        low_flank = np.ones(len_low_flank)
    len_high_flank = idx_start_stop_f[3] - idx_start_stop_f[2]
    high_flank = \
        windows.get_window(
            window_type, len_high_flank*2, fftbins=True)[len_high_flank:]
    zeros_low = np.zeros(idx_start_stop_f[0])
    ones_mid = np.ones(idx_start_stop_f[2]-idx_start_stop_f[1])
    zeros_high = np.zeros(window_length-idx_start_stop_f[3])
    window_full = np.concatenate((zeros_low,
                                  low_flank,
                                  ones_mid,
                                  high_flank,
                                  zeros_high))
    if inverse:
        window_full = 1 - window_full
    return window_full


def _get_normalized_spectrum(f, spectra: np.ndarray, mode='standard',
                             f_range_hz=[20, 20000], normalize: str = None,
                             smoothe: int = 0, phase=False):
    """This function gives a normalized magnitude spectrum with frequency
    vector for a given range. It is also smoothed. Use `None` for the
    spectrum without f_range_hz.

    Parameters
    ----------
    f : `np.ndarray`
        Frequency vector.
    spectra : `np.ndarray`
        Spectrum matrix.
    mode : str, optional
        Mode of spectrum, needed for factor in dB respresentation.
        Choose from `'standard'` or `'welch'`. Default: `'standard'`.
    f_range_hz : array-like with length 2
        Range of frequencies to get the normalized spectrum back.
        Default: [20, 20e3].
    normalize : str, optional
        Normalize spectrum (per channel). Choose from `'1k'` (for 1 kHz),
        `'max'` (maximum value) or `None` for no normalization.
        Default: `None`.
    smoothe : int, optional
        1/smoothe-fractional octave band smoothing for magnitude spectra.
        Pass `0` for no smoothing.
        Default: 0.
    phase : bool, optional
        When `True`, phase spectra are also returned. Default: `False`.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    mag_spectra : `np.ndarray`
        Magnitude spectrum matrix.
    phase_spectra : `np.ndarray`
        Phase spectrum matrix, only returned when `phase=True`.

    """
    if normalize is not None:
        normalize = normalize.lower()
        assert normalize in ('1k', 'max'), \
            f'{normalize} is not a valid normalization mode. Please use ' +\
            '1k or max'
    # Shaping
    if spectra.ndim < 2:
        spectra = spectra[..., None]
    # Check for complex spectrum if phase is required
    if phase:
        assert np.iscomplexobj(spectra), 'Phase computation is not ' +\
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
        f_range_hz = np.sort(f_range_hz)
        ids = _find_nearest(f_range_hz, f)
        id1 = ids[0]
        id2 = ids[1]+1  # Contains endpoint
    else:
        id1 = 0
        id2 = len(f)

    mag_spectra = np.zeros((id2-id1, spectra.shape[1]))
    phase_spectra = np.zeros_like(mag_spectra)
    for n in range(spectra.shape[1]):
        sp = np.abs(spectra[:, n])
        if smoothe != 0:
            sp = _fractional_octave_smoothing(sp, smoothe)
        epsilon = 10**(-300/20)
        sp_db = factor*np.log10(sp + epsilon)
        if normalize is not None:
            if normalize == '1k':
                id1k = _find_nearest(1e3, f)
                sp_db -= sp_db[id1k]
            else:
                sp_db -= np.max(sp_db)
        if phase:
            phase_spectra[:, n] = np.angle(sp[id1:id2])
        mag_spectra[:, n] = sp_db[id1:id2]
    if phase:
        return f[id1:id2], mag_spectra, phase_spectra
    return f[id1:id2], mag_spectra


def _find_frequencies_above_threshold(spec, f, threshold_db, normalize=True):
    """Finds frequencies above a certain threshold in a given spectrum.

    """
    denum_db = 20*np.log10(np.abs(spec))
    if normalize:
        denum_db -= np.max(denum_db)
    freqs = f[denum_db > threshold_db]
    return [freqs[0], freqs[-1]]


def _pad_trim(vector: np.ndarray, desired_length: int, axis: int = 0,
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
            vector = np.flip(vector, axis=axis)
        new_vec = \
            np.concatenate(
                [vector,
                    np.zeros((diff, vector.shape[1]),
                             dtype=type_of_data)])
        if not in_the_end:
            new_vec = np.flip(new_vec, axis=axis)
    elif diff < 0:
        if not in_the_end:
            vector = np.flip(vector, axis=axis)
        new_vec = vector[:desired_length, :]
        if not in_the_end:
            new_vec = np.flip(new_vec, axis=axis)
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
        Step size in samples. It is defined as `window_length - overlap`.
    signal_length : int
        Total signal length.

    Returns
    -------
    n_frames : int
        Number of frames to be observed in the signal.
    padding_samples : int
        Number of samples with which the signal should be padded.

    """
    n_frames = int(np.floor(signal_length / step)) + 1
    padding_samples = window_length - int(signal_length % step)
    return n_frames, padding_samples


def _normalize(s: np.ndarray, dbfs: float, mode='peak'):
    """Normalizes a signal.

    Parameters
    ----------
    s: `np.ndarray`
        Signal to normalize.
    dbfs: float
        dbfs value to normalize to.
    mode: str, optional
        Mode of normalization, `peak` uses the signal maximum absolute value,
        `rms` uses Root mean square value

    Returns
    -------
    s_out: `np.ndarray`
        Normalized signal.

    """
    s = s.copy()
    assert mode in ('peak', 'rms'), 'Mode of normalization is not ' +\
        'available. Select either peak or rms'
    if mode == 'peak':
        s /= np.max(np.abs(s))
        s *= 10**(dbfs/20)
    if mode == 'rms':
        s *= (10**(dbfs/20) / _rms(s))
    return s


def _rms(x: np.ndarray):
    """Root mean square computation.

    """
    return np.sqrt(np.sum(x**2)/len(x))


def _amplify_db(s: np.ndarray, db: float):
    """Amplify by dB.

    """
    return s * 10**(db/20)


def _fade(s: np.ndarray, length_seconds: float = 0.1, mode: str = 'exp',
          sampling_rate_hz: int = 48000, at_start: bool = True):
    """Create a fade in signal.

    Parameters
    ----------
    s : `np.ndarray`
        np.array to be faded.
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
    s : `np.ndarray`
        Faded vector.

    """
    mode = mode.lower()
    assert mode in ('exp', 'lin', 'log'), \
        f'{mode} is not supported. Choose from exp, lin, log.'
    assert length_seconds > 0, 'Only positive lengths'
    l_samples = int(length_seconds * sampling_rate_hz)
    assert len(s) > l_samples, \
        'Signal is shorter than the desired fade'
    single_vec = False
    if s.ndim == 1:
        s = s[..., None]
        single_vec = True
    elif s.ndim == 0:
        raise ValueError('Fading can only be applied to vectors, not scalars')
    else:
        assert s.ndim == 2, \
            'Fade only supports 1D and 2D vectors'

    if mode == 'exp':
        db = np.linspace(-100, 0, l_samples)
        fade = 10**(db/20)
    elif mode == 'lin':
        fade = np.linspace(0, 1, l_samples)
    else:
        db = np.linspace(-100, 0, l_samples)
        fade = 10**(db/20)
        fade = 1 - np.flip(fade)
    if not at_start:
        s = np.flip(s, axis=0)
    s[:l_samples, :] *= fade[..., None]
    if not at_start:
        s = np.flip(s, axis=0)
    if single_vec:
        s = s.squeeze()
    return s


def _fractional_octave_smoothing(vector: np.ndarray, num_fractions: int = 3,
                                 window_type='hamming',
                                 extra_parameters: tuple = None,
                                 window_vec: np.ndarray = None):
    """Smoothes a vector using interpolation to a logarithmic scale. Usually
    done for smoothing of frequency data. This implementation is taken from
    the pyfar package, see references.

    Parameters
    ----------
    vector : `np.ndarray`
        Vector to be smoothed.
    num_fractions : int, optional
        Fraction of octave to be smoothed across. Default: 3 (third band).
    window_type : str, optional
        Type of window to be used. See `scipy.signal.windows.get_window` for
        valid types. Default: `'gaussian'`.
    extra_parameters : tuple, optional
        Additional parameters to be passed to the function
        `scipy.signal.windows.get_window`. Default: `None`.
    window_vec : `np.ndarray`, optional
        Window vector to be used as a window. `window_type` should be set to
        `None` if this direct window is going to be used. Default: `None`.

    Returns
    -------
    vec_final : `np.ndarray`
        Vector after smoothing.

    References
    ----------
    - Tylka, Joseph & Boren, Braxton & Choueiri, Edgar. (2017). A Generalized
      Method for Fractional-Octave Smoothing of Transfer Functions that
      Preserves Log-Frequency Symmetry. Journal of the Audio Engineering
      Society. 65. 239-245. 10.17743/jaes.2016.0053.
    - https://github.com/pyfar/pyfar

    """
    if window_type is not None:
        assert window_vec is None, \
            'Set window_vec to None if you wish to create the window ' +\
            'within the function'
    if window_vec is not None:
        assert window_type is None, \
            'Set window_type to None if you wish to pass a vector to use ' +\
            'as window'
    # Linear and logarithmic frequency vector
    N = len(vector)
    l1 = np.arange(N)
    k_log = (N)**(l1/(N-1))
    beta = np.log2(k_log[1])
    n_window = int(2 * np.floor(1 / (num_fractions * beta * 2)) + 1)
    # Generate window
    if window_type is not None:
        if extra_parameters is not None:
            pass_window = []
            pass_window.append(window_type)
            for e in extra_parameters:
                pass_window.append(e)
            pass_window = tuple(pass_window)
        else:
            pass_window = window_type
        window = windows.get_window(pass_window, n_window, fftbins=False)
    else:
        window = window_vec
    # Dimension handling
    one_dim = False
    if vector.ndim == 1:
        one_dim = True
        vector = vector[..., None]

    vec_final = np.zeros_like(vector)
    for n in range(vector.shape[1]):
        # Interpolate to logarithmic scale
        vec_int = interp1d(
            np.arange(N)+1, vector[:, n], kind='cubic',
            copy=False, assume_sorted=True)
        vec_log = vec_int(k_log)
        # Smoothe by convolving with window
        smoothed = np.convolve(vec_log, window/np.sum(window), mode='same')
        # Interpolate back to linear scale
        smoothed = interp1d(
            k_log, smoothed, kind='cubic',
            copy=False, assume_sorted=True)
        vec_final[:, n] = smoothed(np.arange(N)+1)
    if one_dim:
        vec_final = vec_final.squeeze()
    return vec_final


def _frequency_weightning(f: np.ndarray, weightning_mode: str = 'a',
                          db_output: bool = True):
    """Returns the weights for frequency-weightning.

    Parameters
    ----------
    f : `np.ndarray`
        Frequency vector.
    weightning_mode : str, optional
        Type of weightning. Choose from `'a'` or `'c'`. Default: `'a'`.
    db_output : str, optional
        When `True`, output is given in dB. Default: `True`.

    Returns
    -------
    weights : `np.ndarray`
        Weightning values.

    References
    ----------
    - https://en.wikipedia.org/wiki/A-weighting

    """
    f = np.squeeze(f)
    assert f.ndim == 1, \
        'Frequency must be a 1D-array'
    weightning_mode = weightning_mode.lower()
    assert weightning_mode in ('a', 'c'), \
        'weightning_mode must be a or c'

    ind1k = np.argmin(np.abs(f - 1e3))

    if weightning_mode == 'a':
        weights = 12194**2*f**4 / \
            ((f**2 + 20.6**2) *
             np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2)) *
             (f**2 + 12194**2))
    else:
        weights = 12194**2 * f**2 / ((f**2 + 20.6**2) * (f**2 + 12194**2))
    weights /= weights[ind1k]
    if db_output:
        weights = 20*np.log10(weights)
    return weights
