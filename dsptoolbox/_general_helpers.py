"""
General functionality from helper methods
"""
import numpy as np
from scipy.signal import windows, convolve as scipy_convolve
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz as toeplitz_scipy
from os import sep


def _find_nearest(points, vector) -> np.ndarray:
    """Gives back the indexes with the nearest points in vector

    Parameters
    ----------
    points : float or array_like
        Points to look for nearest index in vector.
    vector : array_like
        Vector in which to look for points.

    Returns
    -------
    indexes : `np.ndarray`
        Indexes of the points.

    """
    points = np.array(points)
    if np.ndim(points) == 0:
        points = points[..., None]
    indexes = np.zeros(len(points), dtype=int)
    for ind, p in enumerate(points):
        indexes[ind] = np.argmin(np.abs(p - vector))
    return indexes


def _calculate_window(
    points,
    window_length: int,
    window_type="hann",
    at_start: bool = True,
    inverse=False,
) -> np.ndarray:
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
    window_full: np.ndarray
        Custom window.

    """
    assert len(points) == 4, "For the custom window 4 points " + "are needed"

    idx_start_stop_f = [int(i) for i in points]

    len_low_flank = idx_start_stop_f[1] - idx_start_stop_f[0]
    if at_start:
        low_flank = windows.get_window(
            window_type, len_low_flank * 2, fftbins=True
        )[0:len_low_flank]
    else:
        low_flank = np.ones(len_low_flank)
    len_high_flank = idx_start_stop_f[3] - idx_start_stop_f[2]
    high_flank = windows.get_window(
        window_type, len_high_flank * 2, fftbins=True
    )[len_high_flank:]
    zeros_low = np.zeros(idx_start_stop_f[0])
    ones_mid = np.ones(idx_start_stop_f[2] - idx_start_stop_f[1])
    zeros_high = np.zeros(window_length - idx_start_stop_f[3])
    window_full = np.concatenate(
        (zeros_low, low_flank, ones_mid, high_flank, zeros_high)
    )
    if inverse:
        window_full = 1 - window_full
    return window_full


def _get_normalized_spectrum(
    f,
    spectra: np.ndarray,
    mode="standard",
    f_range_hz=[20, 20000],
    normalize: str | None = None,
    smoothe: int = 0,
    phase=False,
    calibrated_data: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function gives a normalized magnitude spectrum in dB with frequency
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
        `'max'` (maximum value) or `None` for no normalization. The
        normalization for 1 kHz uses a linear interpolation for getting the
        value at 1 kHz regardless of the frequency resolution. Default: `None`.
    smoothe : int, optional
        1/smoothe-fractional octave band smoothing for magnitude spectra.
        Pass `0` for no smoothing.
        Default: 0.
    phase : bool, optional
        When `True`, phase spectra are also returned. Default: `False`.
    calibrated_data : bool, optional
        When `True`, it is assumed that the time data has been calibrated
        to be in Pascal so that it is scaled by p0=20e-6 Pa. Default: `False`.

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
        assert normalize in ("1k", "max"), (
            f"{normalize} is not a valid normalization mode. Please use "
            + "1k or max"
        )
    # Shaping
    one_dimensional = False
    if spectra.ndim < 2:
        spectra = spectra[..., None]
        one_dimensional = True
    # Check for complex spectrum if phase is required
    if phase:
        assert np.iscomplexobj(spectra), (
            "Phase computation is not "
            + "possible since the spectra are not complex"
        )
    # Factor
    if mode == "standard":
        factor = 20
        scale_factor = 20e-6 if calibrated_data and normalize is None else 1
    elif mode == "welch":
        scale_factor = 4e-10 if calibrated_data and normalize is None else 1
        factor = 10
    else:
        raise ValueError(
            f"{mode} is not supported. Please select standard " "or welch"
        )
    if f_range_hz is not None:
        assert len(f_range_hz) == 2, (
            "Frequency range must have only " + "a lower and an upper bound"
        )
        f_range_hz = np.sort(f_range_hz)
        ids = _find_nearest(f_range_hz, f)
        id1 = ids[0]
        id2 = ids[1] + 1  # Contains endpoint
    else:
        id1 = 0
        id2 = len(f)

    mag_spectra = np.zeros((id2 - id1, spectra.shape[1]))
    phase_spectra = np.zeros_like(mag_spectra)
    for n in range(spectra.shape[1]):
        sp = np.abs(spectra[:, n])
        if smoothe != 0:
            sp = _fractional_octave_smoothing(sp, smoothe)
        epsilon = 10 ** (-400 / 10)
        sp_db = factor * np.log10(
            np.clip(sp, a_min=epsilon, a_max=None) / scale_factor
        )
        if normalize is not None:
            if normalize == "1k":
                gain = _get_exact_gain_1khz(f, sp_db)
                sp_db -= gain
            else:
                sp_db -= np.max(sp_db)
        if phase:
            phase_spectra[:, n] = np.angle(sp[id1:id2])
        mag_spectra[:, n] = sp_db[id1:id2]

    if one_dimensional:
        mag_spectra = np.squeeze(mag_spectra)
    if phase:
        return f[id1:id2], mag_spectra, phase_spectra
    return f[id1:id2], mag_spectra


def _find_frequencies_above_threshold(
    spec, f, threshold_db, normalize=True
) -> list:
    """Finds frequencies above a certain threshold in a given spectrum."""
    denum_db = 20 * np.log10(np.abs(spec))
    if normalize:
        denum_db -= np.max(denum_db)
    freqs = f[denum_db > threshold_db]
    return [freqs[0], freqs[-1]]


def _pad_trim(
    vector: np.ndarray,
    desired_length: int,
    axis: int = 0,
    in_the_end: bool = True,
) -> np.ndarray:
    """Pads (with zeros) or trim (depending on size and desired length)."""
    throw_axis = False
    if vector.ndim < 2:
        assert axis == 0, "You can only pad along the 0 axis"
        vector = vector[..., None]
        throw_axis = True
    elif vector.ndim > 2:
        vector = vector.squeeze()
        if vector.ndim > 2:
            raise ValueError(
                "This function is only implemented for 1D and 2D arrays"
            )
    type_of_data = vector.dtype
    diff = desired_length - vector.shape[axis]
    if axis == 1:
        vector = vector.T
    if diff > 0:
        if not in_the_end:
            vector = np.flip(vector, axis=0)
        new_vec = np.concatenate(
            [vector, np.zeros((diff, vector.shape[1]), dtype=type_of_data)]
        )
        if not in_the_end:
            new_vec = np.flip(new_vec, axis=0)
    elif diff < 0:
        if not in_the_end:
            vector = np.flip(vector, axis=0)
        new_vec = vector[:desired_length, :]
        if not in_the_end:
            new_vec = np.flip(new_vec, axis=0)
    else:
        new_vec = vector.copy()
    if axis == 1:
        new_vec = new_vec.T
    if throw_axis:
        new_vec = new_vec[:, 0]
    return new_vec


def _compute_number_frames(
    window_length: int, step: int, signal_length: int
) -> tuple[int, int]:
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


def _normalize(s: np.ndarray, dbfs: float, mode="peak") -> np.ndarray:
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
    assert mode in ("peak", "rms"), (
        "Mode of normalization is not "
        + "available. Select either peak or rms"
    )
    if mode == "peak":
        s /= np.max(np.abs(s))
        s *= 10 ** (dbfs / 20)
    if mode == "rms":
        s *= 10 ** (dbfs / 20) / _rms(s)
    return s


def _rms(x: np.ndarray) -> np.ndarray:
    """Root mean square computation."""
    return np.sqrt(np.sum(x**2) / len(x))


def _amplify_db(s: np.ndarray, db: float) -> np.ndarray:
    """Amplify by dB."""
    return s * 10 ** (db / 20)


def _fade(
    s: np.ndarray,
    length_seconds: float = 0.1,
    mode: str = "exp",
    sampling_rate_hz: int = 48000,
    at_start: bool = True,
) -> np.ndarray:
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
    assert mode in (
        "exp",
        "lin",
        "log",
    ), f"{mode} is not supported. Choose from exp, lin, log."
    assert length_seconds > 0, "Only positive lengths"
    l_samples = int(length_seconds * sampling_rate_hz)
    assert len(s) > l_samples, "Signal is shorter than the desired fade"
    single_vec = False
    if s.ndim == 1:
        s = s[..., None]
        single_vec = True
    elif s.ndim == 0:
        raise ValueError("Fading can only be applied to vectors, not scalars")
    else:
        assert s.ndim == 2, "Fade only supports 1D and 2D vectors"

    if mode == "exp":
        db = np.linspace(-100, 0, l_samples)
        fade = 10 ** (db / 20)
    elif mode == "lin":
        fade = np.linspace(0, 1, l_samples)
    else:
        db = np.linspace(-100, 0, l_samples)
        fade = 10 ** (db / 20)
        fade = 1 - np.flip(fade)
    if not at_start:
        s = np.flip(s, axis=0)
    s[:l_samples, :] *= fade[..., None]
    if not at_start:
        s = np.flip(s, axis=0)
    if single_vec:
        s = s.squeeze()
    return s


def _gaussian_window_sigma(window_length: int, alpha: float = 2.5) -> float:
    """Compute the standard deviation sigma for a gaussian window according to
    its length and `alpha`.

    Parameters
    ----------
    window_length : int
        Total window length.
    alpha : float, optional
        Alpha parameter for defining how wide the shape of the gaussian. The
        greater alpha is, the narrower the window becomes. Default: 2.5.

    Returns
    -------
    float
        Standard deviation.

    """
    return (window_length - 1) / (2 * alpha)


def _fractional_octave_smoothing(
    vector: np.ndarray,
    num_fractions: int = 3,
    window_type="hann",
    window_vec: np.ndarray | None = None,
) -> np.ndarray:
    """Smoothes a vector using interpolation to a logarithmic scale. Usually
    done for smoothing of frequency data. This implementation is taken from
    the pyfar package, see references.

    Parameters
    ----------
    vector : `np.ndarray`
        Vector to be smoothed. It is assumed that the first axis is to
        be smoothed.
    num_fractions : int, optional
        Fraction of octave to be smoothed across. Default: 3 (third band).
    window_type : str, optional
        Type of window to be used. See `scipy.signal.windows.get_window` for
        valid types. If the window is `'gaussian'`, the parameter passed will
        be interpreted as alpha and not sigma. Default: `'hann'`.
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
        assert window_vec is None, (
            "Set window_vec to None if you wish to create the window "
            + "within the function"
        )
    if window_vec is not None:
        assert window_type is None, (
            "Set window_type to None if you wish to pass a vector to use "
            + "as window"
        )
    # Linear and logarithmic frequency vector
    N = len(vector)
    l1 = np.arange(N)
    k_log = (N) ** (l1 / (N - 1))
    beta = np.log2(k_log[1])

    # Window length always odd, so that delay can be easily compensated
    n_window = int(1 / (num_fractions * beta) + 0.5)  # Round
    n_window += 1 - n_window % 2  # Ensure odd length

    # Generate window
    if window_type is not None:
        assert (
            window_vec is None
        ), "When window type is passed, no window vector should be added"
        if "gauss" in window_type[0]:
            window_type = (
                "gaussian",
                _gaussian_window_sigma(n_window, window_type[1]),
            )
        window = windows.get_window(window_type, n_window, fftbins=False)
    else:
        assert (
            window_type is None
        ), "When using a window as a vector, window type should be None"
        window = window_vec
    # Dimension handling
    one_dim = False
    if vector.ndim == 1:
        one_dim = True
        vector = vector[..., None]

    # Normalize window
    window /= window.sum()

    # Interpolate to logarithmic scale
    vec_int = interp1d(
        l1 + 1, vector, kind="cubic", copy=False, assume_sorted=True, axis=0
    )
    vec_log = vec_int(k_log)
    # Smoothe by convolving with window (output is centered)
    smoothed = scipy_convolve(
        vec_log, window[..., None], mode="same", method="auto"
    )
    # Interpolate back to linear scale
    smoothed = interp1d(
        k_log, smoothed, kind="cubic", copy=False, assume_sorted=True, axis=0
    )

    vec_final = smoothed(l1 + 1)
    if one_dim:
        vec_final = vec_final.squeeze()
    return vec_final


def _frequency_weightning(
    f: np.ndarray, weightning_mode: str = "a", db_output: bool = True
) -> np.ndarray:
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
    assert f.ndim == 1, "Frequency must be a 1D-array"
    weightning_mode = weightning_mode.lower()
    assert weightning_mode in ("a", "c"), "weightning_mode must be a or c"

    ind1k = np.argmin(np.abs(f - 1e3))

    if weightning_mode == "a":
        weights = (
            12194**2
            * f**4
            / (
                (f**2 + 20.6**2)
                * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
                * (f**2 + 12194**2)
            )
        )
    else:
        weights = (
            12194**2
            * f**2
            / ((f**2 + 20.6**2) * (f**2 + 12194**2))
        )
    weights /= weights[ind1k]
    if db_output:
        weights = 20 * np.log10(weights)
    return weights


def _polyphase_decomposition(
    in_sig: np.ndarray, number_polyphase_components: int, flip: bool = False
) -> np.ndarray:
    """Converts input signal array with shape (time samples, channels) into
    its polyphase representation with shape (time samples, polyphase
    components, channels).

    Parameters
    ----------
    in_sig : `np.ndarray`
        Input signal array to be rearranged in polyphase representation. It
        should have the shape (time samples, channels).
    number_polyphase_components : int
        Number of polyphase components to be used.
    flip : bool, optional
        When `True`, axis of polyphase components is flipped. Needed for
        downsampling with FIR filter because of the reversing of filter
        coefficients. Default: `False`.

    Returns
    -------
    poly : `np.ndarray`
        Rearranged vector with polyphase representation. New shape is
        (time samples, polyphase components, channels).
    padding : int
        Amount of padded elements in the beginning of array.

    """
    # Dimensions of vector
    if in_sig.ndim == 1:
        in_sig = in_sig[..., None]
    assert (
        in_sig.ndim == 2
    ), "Vector should have exactly two dimensions: (time samples, channels)"
    # Rename for practical purposes
    n = number_polyphase_components
    # Pad zeros in the beginning to avoid remainder
    remainder = in_sig.shape[0] % n
    padding = n - remainder
    if remainder != 0:
        in_sig = _pad_trim(
            in_sig, in_sig.shape[0] + padding, axis=0, in_the_end=False
        )
    # Here (time samples, polyphase, channels)
    poly = np.zeros((in_sig.shape[0] // n, n, in_sig.shape[1]))
    for ind in range(n):
        poly[:, ind, :] = in_sig[ind::n, :]
    if flip:
        poly = np.flip(poly, axis=1)
    return poly, padding


def _polyphase_reconstruction(poly: np.ndarray) -> np.ndarray:
    """Returns the reconstructed input signal array from its polyphase
    representation, possibly with a different length if padded was needed for
    reconstruction. Polyphase representation shape is assumed to be
    (time samples, polyphase components, channels).

    Parameters
    ----------
    poly : `np.ndarray`
        Array with 3 dimensions (time samples, polyphase components, channels)
        as polyphase respresentation of signal.

    Returns
    -------
    in_sig : `np.ndarray`
        Rearranged vector with shape (time samples, channels).

    """
    # If squeezed array with one channel is passed
    if poly.ndim == 2:
        poly = poly[..., None]
    assert poly.ndim == 3, (
        "Invalid shape. The dimensions must be (time samples, polyphase "
        + "components, channels)"
    )
    n = poly.shape[1]
    in_sig = np.zeros((poly.shape[0] * n, poly.shape[2]))
    for ind in range(n):
        in_sig[ind::n, :] = poly[:, ind, :]
    return in_sig


def _hz2mel(f: np.ndarray) -> np.ndarray:
    """Convert frequency in Hz into mel.

    Parameters
    ----------
    f : float or array-like
        Frequency in Hz.

    Returns
    -------
    float or array-like
        Frequency value in mel.

    References
    ----------
    - https://en.wikipedia.org/wiki/Mel_scale

    """
    return 2595 * np.log10(1 + f / 700)


def _mel2hz(mel: np.ndarray) -> np.ndarray:
    """Convert frequency in mel into Hz.

    Parameters
    ----------
    mel : float or array-like
        Frequency in mel.

    Returns
    -------
    float or array-like
        Frequency value in Hz.

    References
    ----------
    - https://en.wikipedia.org/wiki/Mel_scale

    """
    return 700 * (10 ** (mel / 2595) - 1)


def _get_fractional_octave_bandwidth(
    f_c: float, fraction: int = 1
) -> np.ndarray:
    """Returns an array with lower and upper bounds for a given center
    frequency with (1/fraction)-octave width.

    Parameters
    ----------
    f_c : float
        Center frequency.
    fraction : int, optional
        Octave fraction to define bandwidth. Passing 0 just returns the center
        frequency as lower and upper bounds. Default: 1.

    Returns
    -------
    f_bounds : `np.ndarray`
        Array of length 2 with lower and upper bounds.

    """
    if fraction == 0:
        return np.array([f_c, f_c])
    return np.array(
        [f_c * 2 ** (-1 / fraction / 2), f_c * 2 ** (1 / fraction / 2)]
    )


def _toeplitz(h: np.ndarray, length_of_input: int) -> np.ndarray:
    """Creates a toeplitz matrix from a system response given an input length.

    Parameters
    ----------
    h : `np.ndarray`
        System's impulse response.
    length_of_input : int
        Input length needed for the shape of the toeplitz matrix.

    Returns
    -------
    `np.ndarray`
        Toeplitz matrix with shape (len(h)+length_of_input-1, length_of_input).
        Convolution is done by using dot product from the right::

            convolve_result = toeplitz_matrix @ input_vector

    """
    column = np.hstack([h, np.zeros(length_of_input - 1)])
    row = np.zeros((length_of_input))
    row[0] = h[0]
    return toeplitz_scipy(c=column, r=row)


def _check_format_in_path(path: str, desired_format: str) -> str:
    """Checks if a given path already has a format and it matches the desired
    format. If not, an assertion error is raised. If the path does not have
    any format, the desired one is added.

    Parameters
    ----------
    path : str
        Path of file.
    desired_format : str
        Format that the file should have.

    Returns
    -------
    str
        Path with the desired format.

    """
    format = path.split(sep)[-1]
    format = format.split(".")
    if len(format) != 1:
        assert (
            format[-1] == desired_format
        ), f"{format[-1]} is not the desired format"
    else:
        path += f".{desired_format}"
    return path


def _get_next_power_2(number, mode: str = "closest") -> int:
    """This function returns the power of 2 closest to the given number.

    Parameters
    ----------
    number : int, float
        Number for which to find the closest power of 2.
    mode : str {'closest', 'floor', 'ceil'}, optional
        `'closest'` gives the closest value. `'floor'` returns the next smaller
        power of 2 and `'ceil'` the next larger. Default: `'closest'`.

    Returns
    -------
    number_2 : int
        Next power of 2 according to the selected mode.

    """
    assert number > 0, "Only positive numbers are valid"
    mode = mode.lower()
    assert mode in (
        "closest",
        "floor",
        "ceil",
    ), "Mode must be either closest, floor or ceil"

    p = np.log2(number)
    if mode == "closest":
        remainder = p - int(p)
        mode = "floor" if remainder < 0.5 else "ceil"
    if mode == "floor":
        p = np.floor(p).astype(int)
    elif mode == "ceil":
        p = np.ceil(p).astype(int)
    return int(2**p)


def _euclidean_distance_matrix(x: np.ndarray, y: np.ndarray):
    """Compute the euclidean distance matrix between two vectors efficiently.

    Parameters
    ----------
    x : `np.ndarray`
        First vector or matrix with shape (Point x, Dimensions).
    y : `np.ndarray`
        Second vector or matrix with shape (Point y, Dimensions).

    Returns
    -------
    dist : `np.ndarray`
        Euclidean distance matrix with shape (Point x, Point y).

    """
    assert (
        x.ndim == 2 and y.ndim == 2
    ), "Inputs must have exactly two dimensions"
    assert x.shape[1] == y.shape[1], "Dimensions do not match"
    return np.sqrt(
        np.sum(x**2, axis=1, keepdims=True)
        + np.sum(y.T**2, axis=0, keepdims=True)
        - 2 * x @ y.T
    )


def _get_smoothing_factor_ema(
    relaxation_time_s: float, sampling_rate_hz: int, accuracy: float = 0.95
):
    """This computes the smoothing factor needed for a single-pole IIR,
    or exponential moving averager. The returned value (alpha) should be used
    as follows::

        y[n] = alpha * x[n] + (1-alpha)*y[n-1]

    Parameters
    ----------
    relaxation_time_s : float
        Time for the step response to stabilize around the given value
        (with the given accuracy).
    sampling_rate_hz : int
        Sampling rate to be used.
    accuracy : float, optional
        Accuracy with which the value of the step response can differ from
        1 after the relaxation time. This must be between ]0, 1[.
        Default: 0.95.

    Returns
    -------
    alpha : float
        Smoothing value for the exponential smoothing.

    Notes
    -----
    - The formula coincides with the one presented in
      https://en.wikipedia.org/wiki/Exponential_smoothing, but it uses an
      extra factor for accuracy.

    """
    factor = np.log(1 - accuracy)
    return 1 - np.exp(factor / relaxation_time_s / sampling_rate_hz)


def _wrap_phase(phase_vector: np.ndarray) -> np.ndarray:
    """Wraps phase between [-np.pi, np.pi[ after it has been unwrapped.
    This works for 1D and 2D arrays, more dimensions have not been tested.

    Parameters
    ----------
    phase_vector : `np.ndarray`
        Phase vector for which to wrap the phase.

    Returns
    -------
    `np.ndarray`
        Wrapped phase vector.

    """
    return (phase_vector + np.pi) % (2 * np.pi) - np.pi


def _get_exact_gain_1khz(f: np.ndarray, sp_db: np.ndarray) -> float:
    """Uses linear interpolation to get the exact gain value at 1 kHz.

    Parameters
    ----------
    f : `np.ndarray`
        Frequency vector.
    sp : `np.ndarray`
        Spectrum. It can be in dB or not.

    Returns
    -------
    float
        Interpolated value.

    """
    # Get nearest value just before
    ind = _find_nearest(1e3, f)
    if f[ind] > 1e3:
        ind -= 1
    return (sp_db[ind + 1] - sp_db[ind]) / (f[ind + 1] - f[ind]) * (
        1e3 - f[ind]
    ) + sp_db[ind]


def gaussian_window(
    length: int, alpha: float, symmetric: bool, offset: int = 0
):
    """Produces a gaussian window as defined in [1] and [2].

    Parameters
    ----------
    length : int
        Length for the window.
    alpha : float
        Parameter to define window width. It is inversely proportional to the
        standard deviation.
    symmetric : bool
        Define if the window should be symmetric or not.
    offset : int, optional
        The offset moves the middle point of the window to the passed value.
        Default: 0.

    Returns
    -------
    w : `np.ndarray`
        Gaussian window.

    References
    ----------
    - [1]: https://www.mathworks.com/help/signal/ref/gausswin.html.
    - [2]: https://de.wikipedia.org/wiki/Fensterfunktion.

    """
    if not symmetric:
        length += 1

    n = np.arange(length)
    half = (length - 1) / 2
    w = np.exp(-0.5 * (alpha * ((n - offset) - half) / half) ** 2)

    if not symmetric:
        return w[:-1]
    return w
