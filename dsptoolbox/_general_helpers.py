"""
General functionality from helper methods
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import (
    windows,
    oaconvolve,
    hilbert,
    correlate,
    lfilter,
    lfilter_zi,
)
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.linalg import toeplitz as toeplitz_scipy
from scipy.stats import pearsonr
from os import sep
from warnings import warn
from scipy.fft import next_fast_len


def to_db(
    x: NDArray[np.float64],
    amplitude_input: bool,
    dynamic_range_db: float | None = None,
    min_value: float | None = float(np.finfo(np.float64).smallest_normal),
) -> NDArray[np.float64]:
    """Convert to dB from amplitude or power representation. Clipping small
    values can be activated in order to avoid -inf dB outcomes.

    Parameters
    ----------
    x : NDArray[np.float64]
        Array to convert to dB.
    amplitude_input : bool
        Set to True if the values in x are in their linear form. False means
        they have been already squared, i.e., they are in their power form.
    dynamic_range_db : float, None, optional
        If specified, a dynamic range in dB for the vector is applied by
        finding its largest value and clipping to `max - dynamic_range_db`.
        This will always overwrite `min_value` if specified. Pass None to
        ignore. Default: None.
    min_value : float, None, optional
        Minimum value to clip `x` before converting into dB in order to avoid
        `np.nan` or `-np.inf` in the output. Pass None to ignore. Default:
        `np.finfo(np.float64).smallest_normal`.

    Returns
    -------
    NDArray[np.float64]
        New array or float in dB.

    """
    factor = 20.0 if amplitude_input else 10.0

    if min_value is None and dynamic_range_db is None:
        return factor * np.log10(np.abs(x))

    x_abs = np.abs(x)

    if dynamic_range_db is not None:
        min_value = np.max(x_abs) * 10.0 ** (-abs(dynamic_range_db) / factor)

    return factor * np.log10(np.clip(x_abs, a_min=min_value, a_max=None))


def from_db(x: float | NDArray[np.float64], amplitude_output: bool):
    """Get the values in their amplitude or power form from dB.

    Parameters
    ----------
    x : float, NDArray[np.float64]
        Values in dB.
    amplitude_output : bool
        When True, the values are returned in their linear form. Otherwise,
        the squared (power) form is returned.

    Returns
    -------
    float NDArray[np.float64]
        Converted values

    """
    factor = 20.0 if amplitude_output else 10.0
    return 10 ** (x / factor)


def _find_nearest(points, vector) -> NDArray[np.int_]:
    """Gives back the indexes with the nearest points in vector

    Parameters
    ----------
    points : float or array_like
        Points to look for nearest index in vector.
    vector : array_like
        Vector in which to look for points.

    Returns
    -------
    indexes : `NDArray[np.int_]`
        Indexes of the points.

    """
    points = np.array(points)
    if np.ndim(points) == 0:
        points = points[..., None]
    indexes = np.zeros(len(points), dtype=np.int_)
    for ind, p in enumerate(points):
        indexes[ind] = np.argmin(np.abs(p - vector))
    return indexes


def _calculate_window(
    points,
    window_length: int,
    window_type: str | tuple | list = "hann",
    at_start: bool = True,
    inverse=False,
) -> NDArray[np.float64]:
    """Creates a custom window with given indexes

    Parameters
    ----------
    points: array_like
        Vector containing 4 points for the construction of the custom
        window.
    window_length: int
        Length of the window.
    window_type: str, list, tuple, optional
        Type of window to use. Select from scipy.signal.windows. It can be a
        tuple with the window type and extra parameters or a list with two
        window types. Default: `'hann'`.
    at_start: bool, optional
        Creates a half rising window at the start as well. Default: `True`.
    inverse: bool, optional
        When `True`, the window is inversed so that the middle section
        contains 0. Default: False.

    Returns
    -------
    window_full: NDArray[np.float64]
        Custom window.

    """
    assert len(points) == 4, "For the custom window 4 points are needed"
    if type(window_type) in (str, tuple):
        left_window_type = window_type
        right_window_type = window_type
    if type(window_type) is list:
        assert len(window_type) == 2, "There must be exactly two window types"
        left_window_type = window_type[0]
        right_window_type = window_type[1]

    idx_start_stop_f = [int(i) for i in points]

    len_low_flank = idx_start_stop_f[1] - idx_start_stop_f[0]

    if at_start:
        low_flank = windows.get_window(
            left_window_type, len_low_flank * 2, fftbins=True
        )[:len_low_flank]
    else:
        low_flank = np.ones(len_low_flank)

    len_high_flank = idx_start_stop_f[3] - idx_start_stop_f[2]
    high_flank = windows.get_window(
        right_window_type, len_high_flank * 2, fftbins=True
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
    spectra: NDArray[np.float64],
    scaling: str = "amplitude",
    f_range_hz=[20, 20000],
    normalize: str | None = None,
    smoothing: int = 0,
    phase=False,
    calibrated_data: bool = False,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """This function gives a normalized magnitude spectrum in dB with frequency
    vector for a given range. It is also smoothed. Use `None` for the
    spectrum without f_range_hz.

    Parameters
    ----------
    f : NDArray[np.float64]
        Frequency vector.
    spectra : NDArray[np.float64]
        Spectrum matrix.
    scaling : str, optional
        Information about whether the spectrum is scaled as an amplitude or
        power. Choose from `'amplitude'` or `'power'`. Default: `'amplitude'`.
    f_range_hz : array-like with length 2
        Range of frequencies to get the normalized spectrum back.
        Default: [20, 20e3].
    normalize : str, optional
        Normalize spectrum (per channel). Choose from `'1k'` (for 1 kHz),
        `'max'` (maximum value) or `None` for no normalization. The
        normalization for 1 kHz uses a linear interpolation for getting the
        value at 1 kHz regardless of the frequency resolution. Default: `None`.
    smoothing : int, optional
        1/smoothing-fractional octave band smoothing for magnitude spectra.
        Pass `0` for no smoothing. Default: 0.
    phase : bool, optional
        When `True`, phase spectra are also returned. Smoothing is also
        applied to the unwrapped phase. Default: `False`.
    calibrated_data : bool, optional
        When `True`, it is assumed that the time data has been calibrated
        to be in Pascal so that it is scaled by p0=20e-6 Pa. Default: `False`.

    Returns
    -------
    f : NDArray[np.float64]
        Frequency vector.
    mag_spectra : NDArray[np.float64]
        Magnitude spectrum matrix.
    phase_spectra : NDArray[np.float64]
        Phase spectrum matrix, only returned when `phase=True`.

    Notes
    -----
    - The spectrum is clipped at -800 dB by default when standard or -400 dB
      when welch method is used.

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
    if scaling == "amplitude":
        scale_factor = 20e-6 if calibrated_data and normalize is None else 1
        amplitude_scaling = True
    elif scaling == "power":
        scale_factor = 4e-10 if calibrated_data and normalize is None else 1
        amplitude_scaling = False
    else:
        raise ValueError(
            f"{scaling} is not supported. Please select amplitude or "
            + "power scaling"
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

    spectra = spectra[id1:id2]
    f = f[id1:id2]

    mag_spectra = np.abs(spectra)

    if smoothing != 0:
        if scaling == "amplitude":
            mag_spectra = _fractional_octave_smoothing(mag_spectra, smoothing)
        else:  # Smoothing always in amplitude representation
            mag_spectra = (
                _fractional_octave_smoothing(mag_spectra**0.5, smoothing) ** 2
            )

    mag_spectra = to_db(mag_spectra / scale_factor, amplitude_scaling, 500)

    if normalize is not None:
        for i in range(spectra.shape[1]):
            if normalize == "1k":
                mag_spectra[:, i] -= _get_exact_gain_1khz(f, mag_spectra[:, i])
            else:
                mag_spectra[:, i] -= np.max(mag_spectra[:, i])

    if phase:
        phase_spectra = np.angle(spectra)
        if smoothing != 0:
            phase_spectra = _wrap_phase(
                _fractional_octave_smoothing(
                    np.unwrap(phase_spectra, axis=0), smoothing
                )
            )

    if one_dimensional:
        mag_spectra = np.squeeze(mag_spectra)
        if phase:
            phase_spectra = np.squeeze(phase_spectra)

    if phase:
        return f, mag_spectra, phase_spectra

    return f, mag_spectra


def _find_frequencies_above_threshold(
    spec, f, threshold_db, normalize=True
) -> list:
    """Finds frequencies above a certain threshold in a given (amplitude)
    spectrum."""
    denum_db = to_db(spec, True)
    if normalize:
        denum_db -= np.max(denum_db)
    freqs = f[denum_db > threshold_db]
    return [freqs[0], freqs[-1]]


def _pad_trim(
    vector: NDArray[np.float64],
    desired_length: int,
    axis: int = 0,
    in_the_end: bool = True,
) -> NDArray[np.float64]:
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
    window_length: int, step: int, signal_length: int, zero_padding: bool
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
    zero_padding : bool
        When `True`, it is assumed that the signal will be zero padded in the
        end to make use of all time samples. `False` will effectively discard
        the blocks where zero-padding would be needed.

    Returns
    -------
    n_frames : int
        Number of frames to be observed in the signal.
    padding_samples : int
        Number of samples with which the signal should be padded.

    """
    if zero_padding:
        n_frames = int(np.ceil(signal_length / step))
        padding_samples = window_length - int(signal_length % step)
    else:
        padding_samples = 0
        n_frames = int(np.ceil((signal_length - window_length) / step))
    return n_frames, padding_samples


def _normalize(
    s: NDArray[np.float64], dbfs: float, mode: str, per_channel: bool
) -> NDArray[np.float64]:
    """Normalizes a signal.

    Parameters
    ----------
    s: NDArray[np.float64]
        Signal to normalize. It can be 1 or 2D. Time samples are assumed to
        be in the outer axis.
    dbfs: float
        dbfs value to normalize to.
    mode: str, optional
        Mode of normalization, `peak` uses the signal maximum absolute value,
        `rms` uses Root mean square value

    Returns
    -------
    s_out: NDArray[np.float64]
        Normalized signal.

    """
    assert mode in ("peak", "rms"), (
        "Mode of normalization is not "
        + "available. Select either peak or rms"
    )

    onedim = s.ndim == 1
    if onedim:
        s = s[..., None]

    factor = from_db(dbfs, True)
    if mode == "peak":
        factor /= np.max(np.abs(s), axis=0 if per_channel else None)
    elif mode == "rms":
        factor /= _rms(s if per_channel else s.flatten())
    s_norm = s * factor

    return s_norm[..., 0] if onedim else s_norm


def _rms(x: NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Root mean squared value of a discrete time series.

    Parameters
    ----------
    x : NDArray[np.float64]
        Time series.

    Returns
    -------
    rms : float or NDArray[np.float64]
        Root mean squared of a signal. Float or NDArray[np.float64] depending
        on input.

    """
    single_dim = x.ndim == 1
    x = x[..., None] if single_dim else x
    rms_vals = np.std(x, axis=0)
    return rms_vals[..., 0] if single_dim else rms_vals


def _amplify_db(s: NDArray[np.float64], db: float) -> NDArray[np.float64]:
    """Amplify by dB."""
    return s * 10 ** (db / 20)


def _fade(
    s: NDArray[np.float64],
    length_seconds: float = 0.1,
    mode: str = "exp",
    sampling_rate_hz: int = 48000,
    at_start: bool = True,
) -> NDArray[np.float64]:
    """Create a fade in signal.

    Parameters
    ----------
    s : NDArray[np.float64]
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
    s : NDArray[np.float64]
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
        # The constant 50 could be an extra parameter for the user...
        fade = np.log10(np.linspace(1, 50 * 10**0.5, l_samples))
        fade /= fade[-1]
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
    vector: NDArray[np.float64],
    num_fractions: int = 3,
    window_type="hann",
    window_vec: NDArray[np.float64] | None = None,
    clip_values: bool = False,
) -> NDArray[np.float64]:
    """Smoothes a vector using interpolation to a logarithmic scale. Usually
    done for smoothing of frequency data. This implementation is taken from
    the pyfar package, see references.

    Parameters
    ----------
    vector : NDArray[np.float64]
        Vector to be smoothed. It is assumed that the first axis is to
        be smoothed.
    num_fractions : int, optional
        Fraction of octave to be smoothed across. Default: 3 (third band).
    window_type : str, optional
        Type of window to be used. See `scipy.signal.windows.get_window` for
        valid types. If the window is `'gaussian'`, the parameter passed will
        be interpreted as alpha and not sigma. Default: `'hann'`.
    window_vec : NDArray[np.float64], optional
        Window vector to be used as a window. `window_type` should be set to
        `None` if this direct window is going to be used. Default: `None`.
    clip_values : bool, optional
        When `True`, negative values are clipped to 0. Default: `False`.

    Returns
    -------
    vec_final : NDArray[np.float64]
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
    l1 = np.arange(N, dtype=np.float64)
    k_log = (N) ** (l1 / (N - 1))
    l1 += 1.0
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
    vec_log = PchipInterpolator(l1, vector, axis=0)(k_log)

    # Smoothe by convolving with window (output is centered)
    n_window_half = n_window // 2
    smoothed = oaconvolve(
        np.pad(
            vec_log,
            ((n_window_half, n_window_half - (1 - n_window % 2)), (0, 0)),
            mode="edge",
        ),
        window[..., None],
        mode="valid",
        axes=0,
    )

    # Interpolate back to linear scale
    vec_final = interp1d(
        k_log, smoothed, kind="linear", copy=False, assume_sorted=True, axis=0
    )(l1)
    if one_dim:
        vec_final = vec_final.squeeze()

    # Avoid any negative values (numerical errors)
    if clip_values:
        vec_final = np.clip(vec_final, a_min=0, a_max=None)
    return vec_final


def _frequency_weightning(
    f: NDArray[np.float64], weightning_mode: str = "a", db_output: bool = True
) -> NDArray[np.float64]:
    """Returns the weights for frequency-weightning.

    Parameters
    ----------
    f : NDArray[np.float64]
        Frequency vector.
    weightning_mode : str, optional
        Type of weightning. Choose from `'a'` or `'c'`. Default: `'a'`.
    db_output : str, optional
        When `True`, output is given in dB. Default: `True`.

    Returns
    -------
    weights : NDArray[np.float64]
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
        weights = 12194**2 * f**2 / ((f**2 + 20.6**2) * (f**2 + 12194**2))
    weights /= weights[ind1k]
    if db_output:
        weights = to_db(weights, True)
    return weights


def _polyphase_decomposition(
    in_sig: NDArray[np.float64],
    number_polyphase_components: int,
    flip: bool = False,
) -> tuple[NDArray[np.float64], int]:
    """Converts input signal array with shape (time samples, channels) into
    its polyphase representation with shape (time samples, polyphase
    components, channels).

    Parameters
    ----------
    in_sig : NDArray[np.float64]
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
    poly : NDArray[np.float64]
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


def _polyphase_reconstruction(
    poly: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Returns the reconstructed input signal array from its polyphase
    representation, possibly with a different length if padded was needed for
    reconstruction. Polyphase representation shape is assumed to be
    (time samples, polyphase components, channels).

    Parameters
    ----------
    poly : NDArray[np.float64]
        Array with 3 dimensions (time samples, polyphase components, channels)
        as polyphase respresentation of signal.

    Returns
    -------
    in_sig : NDArray[np.float64]
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


def _hz2mel(f: NDArray[np.float64]) -> NDArray[np.float64]:
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


def _mel2hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
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
) -> NDArray[np.float64]:
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
    f_bounds : NDArray[np.float64]
        Array of length 2 with lower and upper bounds.

    """
    if fraction == 0:
        return np.array([f_c, f_c])
    return np.array(
        [f_c * 2 ** (-1 / fraction / 2), f_c * 2 ** (1 / fraction / 2)]
    )


def _toeplitz(
    h: NDArray[np.float64], length_of_input: int
) -> NDArray[np.float64]:
    """Creates a toeplitz matrix from a system response given an input length.

    Parameters
    ----------
    h : NDArray[np.float64]
        System's impulse response.
    length_of_input : int
        Input length needed for the shape of the toeplitz matrix.

    Returns
    -------
    NDArray[np.float64]
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
    format = path.split(sep)[-1].split(".")
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


def _euclidean_distance_matrix(x: NDArray[np.float64], y: NDArray[np.float64]):
    """Compute the euclidean distance matrix between two vectors efficiently.

    Parameters
    ----------
    x : NDArray[np.float64]
        First vector or matrix with shape (Point x, Dimensions).
    y : NDArray[np.float64]
        Second vector or matrix with shape (Point y, Dimensions).

    Returns
    -------
    dist : NDArray[np.float64]
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


def _wrap_phase(phase_vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wraps phase between [-np.pi, np.pi[ after it has been unwrapped.
    This works for 1D and 2D arrays, more dimensions have not been tested.

    Parameters
    ----------
    phase_vector : NDArray[np.float64]
        Phase vector for which to wrap the phase.

    Returns
    -------
    NDArray[np.float64]
        Wrapped phase vector.

    """
    return (phase_vector + np.pi) % (2 * np.pi) - np.pi


def _get_exact_gain_1khz(
    f: NDArray[np.float64], sp_db: NDArray[np.float64]
) -> float:
    """Uses linear interpolation to get the exact gain value at 1 kHz.

    Parameters
    ----------
    f : NDArray[np.float64]
        Frequency vector.
    sp : NDArray[np.float64]
        Spectrum. It can be in dB or not.

    Returns
    -------
    float
        Interpolated value.

    """
    assert np.min(f) < 1e3 and np.max(f) >= 1e3, (
        "No gain at 1 kHz can be obtained because it is outside the "
        + "given frequency vector"
    )
    # Get nearest value just before
    ind = _find_nearest(1e3, f).squeeze()
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
    w : NDArray[np.float64]
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


def _get_chirp_rate(range_hz: list, length_seconds: float) -> float:
    """Compute the chirp rate based on the frequency range of the exponential
    chirp and its duration.

    Parameters
    ----------
    range_hz : list with length 2
        Range of the exponential chirp.
    length_seconds : float
        Chirp's length in seconds.

    Returns
    -------
    float
        The chirp rate in octaves/second.

    """
    range_hz_array = np.atleast_1d(range_hz)
    assert range_hz_array.shape == (
        2,
    ), "Range must contain exactly two elements."
    range_hz_array = np.sort(range_hz_array)
    return np.log2(range_hz_array[1] / range_hz_array[0]) / length_seconds


def _correct_for_real_phase_spectrum(phase_spectrum: NDArray[np.float64]):
    """This function takes in a phase spectrum and corrects it to be for a real
    signal (assuming the last frequency bin corresponds to nyquist, i.e., time
    data had an even length). This effectively adds a small linear phase offset
    so that the phase at nyquist is either 0 or np.pi.

    Parameters
    ----------
    phase_spectrum : NDArray[np.float64]
        Phase to be corrected. It is assumed that its last element
        corresponds to the nyquist frequency.

    Returns
    -------
    NDArray[np.float64]
        Phase spectrum that can correspond to a real signal.

    """
    factor = phase_spectrum[-1] % np.pi
    return (
        phase_spectrum
        - np.linspace(0, 1, len(phase_spectrum), endpoint=True) * factor
    )


def _scale_spectrum(
    spectrum: NDArray[np.float64] | NDArray[np.complex128],
    mode: str | None,
    time_length_samples: int,
    sampling_rate_hz: int,
    window: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Scale the spectrum directly from the unscaled ("backward" normalization)
    (R)FFT. If a window was applied, it is necessary to compute the right
    scaling factor.

    Parameters
    ----------
    spectrum : NDArray[np.float64] | NDArray[np.complex128]
        Spectrum to scale. It is assumed that the frequency bins are along
        the first dimension.
    mode : str, None
        Type of scaling to use. `"power spectral density"`, `"power spectrum"`,
        `"amplitude spectral density"`, `"amplitude spectrum"`. Pass `None`
        to avoid any scaling and return the same spectrum. Using a power
        representation will returned the squared spectrum.
    time_length_samples : int
        Original length of the time data.
    sampling_rate_hz : int
        Sampling rate.
    window : NDArray[np.float64], None, optional
        Applied window when obtaining the spectrum. It is necessary to compute
        the correct scaling factor. In case of None, "boxcar" window is
        assumed. Default: None.

    Returns
    -------
    NDArray[np.float64] | NDArray[np.complex128]
        Scaled spectrum

    Notes
    -----
    - The amplitude spectrum shows the RMS value of each frequency in the
      signal.
    - Integrating the power spectral density over the frequency spectrum
      delivers the total energy contained in the signal (parseval's theorem).

    """
    assert time_length_samples in (
        (spectrum.shape[0] - 1) * 2,
        spectrum.shape[0] * 2 - 1,
    ), "Time length does not match"

    if mode is None:
        return spectrum

    mode = mode.lower()
    assert mode in (
        "amplitude spectral density",
        "amplitude spectrum",
        "power spectral density",
        "power spectrum",
    ), f"{mode} is not a supported mode"

    if "spectral density" in mode:
        if window is None:
            factor = (2 / time_length_samples / sampling_rate_hz) ** 0.5
        else:
            factor = (
                2 / np.sum(window**2, axis=0, keepdims=True) / sampling_rate_hz
            ) ** 0.5
    elif "spectrum" in mode:
        if window is None:
            factor = 2**0.5 / time_length_samples
        else:
            factor = 2**0.5 / np.sum(window, axis=0, keepdims=True)

    spectrum *= factor

    spectrum[0] /= 2**0.5
    if time_length_samples % 2 == 0:
        spectrum[-1] /= 2**0.5

    if "power" in mode:
        spectrum = np.abs(spectrum) ** 2

    return spectrum


def _get_fractional_impulse_peak_index(
    time_data: NDArray[np.float64], polynomial_points: int = 1
):
    """
    Obtain the index for the peak in subsample precision using the root
    of the analytical function.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Time vector with shape (time samples, channels).
    polynomial_points : int, optional
        Number of points to take for the polynomial interpolation and root
        finding of the analytic part of the time series. Default: 1.

    Returns
    -------
    latency_samples : NDArray[np.float64]
        Latency of impulses (in samples). It has shape (channels).

    """
    n_channels = time_data.shape[1]
    delay_samples = np.argmax(np.abs(time_data), axis=0).astype(int)

    # Take only the part of the time vector with the peaks and some safety
    # samples (±200)
    time_data = time_data[: np.max(delay_samples) + 200, :]
    start_offset = max(np.min(delay_samples) - 200, 0)
    time_data = time_data[start_offset:, :]
    delay_samples -= start_offset

    h = hilbert(time_data, axis=0).imag
    x = np.arange(-polynomial_points + 1, polynomial_points + 1)

    latency_samples = np.zeros(n_channels)

    for ch in range(n_channels):
        # ===== Ensure that delay_samples is before the peak in each channel
        selection = h[delay_samples[ch] : delay_samples[ch] + 2, ch]
        move_back_one_sample = selection[0] * selection[1] > 0
        delay_samples[ch] -= int(move_back_one_sample)
        if h[delay_samples[ch], ch] * h[delay_samples[ch] + 1, ch] > 0:
            latency_samples[ch] = delay_samples[ch] + int(move_back_one_sample)
            warn(
                f"Fractional latency detection failed for channel {ch}. "
                + "Integer latency is"
                + " returned"
            )
            continue
        # =====

        # Fit polynomial
        pol = np.polyfit(
            x,
            h[
                delay_samples[ch]
                - polynomial_points
                + 1 : delay_samples[ch]
                + polynomial_points
                + 1,
                ch,
            ],
            deg=2 * polynomial_points - 1,
        )

        # Find roots
        roots = np.roots(pol)
        # Get only root between 0 and 1
        roots = roots[
            (roots == roots.real)  # Real roots
            & (roots <= 1)  # Range
            & (roots >= 0)
        ].real
        try:
            fractional_delay_samples = roots[0]
        except IndexError as e:
            print(e)
            warn(
                f"Fractional latency detection failed for channel {ch}. "
                + "Integer latency is"
                + " returned"
            )
            latency_samples[ch] = delay_samples[ch] + int(move_back_one_sample)
            continue

        latency_samples[ch] = delay_samples[ch] + fractional_delay_samples
    return latency_samples + start_offset


def _remove_ir_latency_from_phase(
    freqs: NDArray[np.float64],
    phase: NDArray[np.float64],
    time_data: NDArray[np.float64],
    sampling_rate_hz: int,
    padding_factor: int,
):
    """
    Remove the impulse delay from a phase response.

    Parameters
    ----------
    freqs : NDArray[np.float64]
        Frequency vector.
    phase : NDArray[np.float64]
        Phase vector.
    time_data : NDArray[np.float64]
        Corresponding time signal.
    sampling_rate_hz : int
        Sample rate.
    padding_factor : int
        Padding factor used to obtain the minimum phase equivalent.

    Returns
    -------
    new_phase : NDArray[np.float64]
        New phase response without impulse delay.

    """
    min_ir = _min_phase_ir_from_real_cepstrum(time_data, padding_factor)
    delays_s = _fractional_latency(time_data, min_ir) / sampling_rate_hz
    return _wrap_phase(phase + 2 * np.pi * freqs[:, None] * delays_s[None, :])


def _min_phase_ir_from_real_cepstrum(
    time_data: NDArray[np.float64], padding_factor: int
):
    """Returns minimum-phase version of a time series using the real cepstrum
    method.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Time series to compute the minimum phase version from. It is assumed
        to have shape (time samples, channels).
    padding_factor : int, optional
        Zero-padding to a length corresponding to
        `current_length * padding_factor` can be done, in order to avoid time
        aliasing errors. Default: 8.

    Returns
    -------
    min_phase_time_data : NDArray[np.float64]
        New time series.

    """
    return np.real(
        np.fft.ifft(
            _get_minimum_phase_spectrum_from_real_cepstrum(
                time_data, padding_factor
            ),
            axis=0,
        )
    )


def _get_minimum_phase_spectrum_from_real_cepstrum(
    time_data: NDArray[np.float64], padding_factor: int
):
    """Returns minimum-phase version of a time series using the real cepstrum
    method.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Time series to compute the minimum phase version from. It is assumed
        to have shape (time samples, channels).
    padding_factor : int, optional
        Zero-padding to a length corresponding to
        `current_length * padding_factor` can be done, in order to avoid time
        aliasing errors. Default: 8.

    Returns
    -------
    NDArray[np.float64]
        New spectrum with minimum phase.

    """
    fft_length = next_fast_len(
        max(time_data.shape[0] * padding_factor, time_data.shape[0])
    )
    # Real cepstrum
    y = np.real(
        ifft(np.log(np.abs(fft(time_data, n=fft_length, axis=0))), axis=0)
    )

    # Window in the cepstral domain, like obtaining hilbert transform
    w = np.zeros(y.shape[0])
    w[1 : len(w) // 2 - 1] = 2
    w[0] = 1
    # If length is even, nyquist is exactly in the middle
    if len(w) % 2 == 0:
        w[len(w) // 2] = 1

    # Windowing in cepstral domain and back to spectral domain
    return np.exp(fft(y * w[..., None], axis=0))


def _fractional_latency(
    td1: NDArray[np.float64],
    td2: NDArray[np.float64] | None = None,
    polynomial_points: int = 1,
):
    """This function computes the sub-sample latency between two signals using
    Zero-Crossing of the analytic (hilbert transformed) correlation function.
    The number of polynomial points taken around the correlation maximum can be
    set, although some polynomial orders might fail to compute the root. In
    that case, integer latency will be returned for the respective channel.

    Parameters
    ----------
    td1 : `np.ndaray`
        Delayed version of the signal.
    td2 : NDArray[np.float64], optional
        Original version of the signal. If `None` is passed, the latencies
        are computed between the first channel of td1 and every other.
        Default: `None`.
    polynomial_points : int, optional
        This corresponds to the number of points taken around the root in order
        to fit a polynomial. Accuracy might improve with higher orders but
        it could also lead to ill-conditioned polynomials. In case root finding
        is not successful, integer latency values are returned. Default: 1.

    Returns
    -------
    lags : NDArray[np.float64]
        Fractional delays. It has shape (channel). In case td2 was `None`, its
        length is `channels - 1`.

    References
    ----------
    - N. S. M. Tamim and F. Ghani, "Hilbert transform of FFT pruned cross
      correlation function for optimization in time delay estimation," 2009
      IEEE 9th Malaysia International Conference on Communications (MICC),
      Kuala Lumpur, Malaysia, 2009, pp. 809-814,
      doi: 10.1109/MICC.2009.5431382.

    """
    if td2 is None:
        td2 = td1[:, 0][..., None]
        td1 = np.atleast_2d(td1[:, 1:])
        xcor = correlate(td2, td1)
    else:
        xcor = np.zeros((td1.shape[0] + td2.shape[0] - 1, td2.shape[1]))
        for i in range(td2.shape[1]):
            xcor[:, i] = correlate(td2[:, i], td1[:, i])
    inds = _get_fractional_impulse_peak_index(xcor, polynomial_points)
    return td1.shape[0] - inds - 1


def _interpolate_fr(
    f_interp: NDArray[np.float64],
    fr_interp: NDArray[np.float64],
    f_target: NDArray[np.float64],
    mode: str | None = None,
    interpolation_scheme: str = "linear",
) -> NDArray[np.float64]:
    """Interpolate one frequency response to a new frequency vector.

    Parameters
    ----------
    f_interp : NDArray[np.float64]
        Frequency vector of the frequency response that should be interpolated.
    fr_interp : NDArray[np.float64]
        Frequency response to be interpolated.
    f_target : NDArray[np.float64]
        Target frequency vector.
    mode : str, None, {"db2amplitude", "amplitude2db", "power2db",\
            "power2amplitude", "amplitude2power"}, optional
        Convert between amplitude, power or dB representation during the
        interpolation step. For instance, using the modes "db2power" means
        input in dB, interpolation in power spectrum, output in dB. Available
        modes are "db2amplitude", "amplitude2db", "power2db",
        "power2amplitude", "amplitude2power". Pass None to avoid any
        conversion. Default: None.
    interpolation_scheme : str, {"linear", "quadratic", "cubic"}, optional
        Type of interpolation to use. See `scipy.interpolation.interp1d` for
        details. Choose from "quadratic" or "cubic" splines, or "linear".
        Default: "linear".

    Returns
    -------
    NDArray[np.float64]
        New interpolated frequency response corresponding to `f_target` vector.

    Notes
    -----
    - The input is always assumed to be already sorted.
    - In case `f_target` has values outside the boundaries of `f_interp`,
      0 is used as the fill value. For interpolation in dB, fill values are
      the vector's edges.
    - The interpolation is always done along the first (outer) axis or the
      vector.
    - When converting to dB, the default clipping value of `to_db` is used.
    - Theoretical thoughts on interpolating an amplitude or power
      frequency response:
        - Using complex and dB values during interpolation are not very precise
          when comparing the results in terms of the amplitude or power
          spectrum.
        - Interpolation can be done with amplitude or power representation with
          similar precision.
        - Changing the frequency resolution in a linear scale means zero-
          padding or trimming the underlying time series. For an amplitude
          representation , i.e. spectrum or spectral density, the values must
          be scaled using the factor `old_length/new_length`. This ensures that
          the RMS values (amplitude spectrum) are still correct, and that
          integrating the new power spectral density still renders the total
          signal's energy truthfully, i.e. parseval's theorem would still hold.
          For the power representation, it also applies with the same squared
          factor.
        - A direct FFT-result which is not in physical units needs rescaling
          depending on the normalization scheme used during the FFT -> IFFT (in
          the complex/amplitude representation):
              - Forward: scaling factor `old_length/new_length`.
              - Backward: no rescaling.
              - Orthogonal: scaling factor `(old_length/new_length)**0.5`
        - Interpolating the (amplitude or power) spectrum to a logarithmic-
          spaced frequency vector can be done without rescaling (the underlying
          transformation in the time domain would be warping). Doing so for the
          (amplitude or power) spectral density only retains its validity if
          the new spectrum is weighted exponentially with increasing frequency
          since each bin contains the energy of a larger “frequency band”
          (this changes the physical units of the spectral density). Doing so
          ensures that integrating the power spectral density over frequency
          still retains the energy of the signal (parseval).
        - Assuming a different time window in each frequency resolution would
          require knowing the specific windows in order to rescale correctly.
          Assuming the same time window while zero-padding in the time domain
          would mean that no rescaling has to be applied.

    """

    fill_value = (0.0, 0.0)
    y = fr_interp.copy()

    # Conversion if necessary
    if mode is not None:
        mode = mode.lower()
        if mode == "power2amplitude":
            y **= 0.5
        elif mode == "amplitude2power":
            y **= 2.0
        elif mode[:3] == "db2":
            y = from_db(y, "amplitude" in mode)
        elif mode[-3:] == "2db":
            y = to_db(y, "amplitude" in mode)
            fill_value = (y[0], y[-1])
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

    interpolated = interp1d(
        f_interp,
        y,
        kind=interpolation_scheme,
        copy=False,
        bounds_error=False,
        assume_sorted=True,
        fill_value=fill_value,
        axis=0,
    )(f_target)

    # Back conversion if activated
    if mode is not None:
        if mode == "power2amplitude":
            interpolated **= 2.0
        elif mode == "amplitude2power":
            interpolated **= 0.5
        elif mode[:3] == "db2":
            interpolated = to_db(interpolated, "amplitude" in mode)
        elif mode[-3:] == "2db":
            interpolated = from_db(interpolated, "amplitude" in mode)

    return interpolated


def _time_smoothing(
    x: NDArray[np.float64],
    sampling_rate_hz: int,
    ascending_time_s: float,
    descending_time_s: float | None = None,
) -> NDArray[np.float64]:
    """Smoothing for a time series with independent ascending and descending
    times using an exponential moving average. It works on 1D and 2D arrays.
    The smoothing is always applied along the longest axis.

    If no descending time is provided, `ascending_time_s` is used for both
    increasing and decreasing values.

    Parameters
    ----------
    x : NDArray[np.float64]
        Vector to apply smoothing to.
    sampling_rate_hz : int
        Sampling rate of the time series `x`.
    ascending_time_s : float
        Corresponds to the needed time for achieving a 95% accuracy of the
        step response when the samples are increasing in value. Pass 0. in
        order to avoid any smoothing for rising values.
    descending_time_s : float, None, optional
        As `ascending_time_s` but for descending values. If None,
        `ascending_time_s` is applied. Default: None.

    Returns
    -------
    NDArray[np.float64]
        Smoothed time series.

    """
    onedim = x.ndim == 1
    x = np.atleast_2d(x)
    if x.shape[0] < x.shape[1]:
        reverse_axis = True
        x = x.T
    else:
        reverse_axis = False

    assert x.ndim < 3, "This function is only available for 2D arrays"
    assert ascending_time_s >= 0.0, "Attack time must be at least 0"
    ascending_factor = (
        _get_smoothing_factor_ema(ascending_time_s, sampling_rate_hz)
        if ascending_time_s > 0.0
        else 1.0
    )

    if descending_time_s is None:
        b, a = [ascending_factor], [1, -(1 - ascending_factor)]
        zi = lfilter_zi(b, a)
        y = lfilter(
            b,
            a,
            x,
            axis=0,
            zi=np.asarray([zi * x[0, ch] for ch in range(x.shape[1])]).T,
        )[0]
        if reverse_axis:
            y = y.T
        if onedim:
            return y.squeeze()
        return y

    assert descending_time_s >= 0.0, "Release time must at least 0"
    assert not (
        ascending_time_s == 0.0 and descending_time_s == ascending_time_s
    ), "These times will not apply any smoothing"

    descending_factor = (
        _get_smoothing_factor_ema(descending_time_s, sampling_rate_hz)
        if descending_time_s > 0.0
        else 1.0
    )

    y = np.zeros_like(x)
    y[0, :] = x[0, :]

    for n in np.arange(1, x.shape[0]):
        for ch in range(x.shape[1]):
            smoothing_factor = (
                ascending_factor
                if x[n, ch] > y[n - 1, ch]
                else descending_factor
            )
            y[n, ch] = (
                smoothing_factor * x[n, ch]
                + (1.0 - smoothing_factor) * y[n - 1, ch]
            )

    if reverse_axis:
        y = y.T
    if onedim:
        return y.squeeze()
    return y


def _get_correlation_of_latencies(
    time_data: NDArray[np.float64],
    other_time_data: NDArray[np.float64],
    latencies: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute the pearson correlation coefficient of each channel between
    `time_data` and `other_time_data` in order to obtain an estimation on the
    quality of the latency computation.

    Parameters
    ----------
    time_data : NDArray[np.float64]
        Original time data. This is the "undelayed" version if the latency
        is positive. It must have either one channel or a matching number
        of channels with `other_time_data`.
    other_time_data : NDArray[np.float64]
        "Delayed" time data, when the latency is positive.
    latencies : NDArray[np.int_]
        Computed latencies for each channel.

    Returns
    -------
    NDArray[np.float64]
        Correlation coefficient for each channel.

    """
    one_channel = time_data.shape[1] == 1

    correlations = np.zeros(len(latencies))

    for ch in range(len(latencies)):
        if latencies[ch] > 0:
            undelayed = time_data[:, 0] if one_channel else time_data[:, ch]
            delayed = other_time_data[:, ch]
        else:
            undelayed = other_time_data[:, ch]
            delayed = time_data[:, 0] if one_channel else time_data[:, ch]

        # Remove delay samples
        delayed = delayed[abs(latencies[ch]) :]

        # Get effective length
        length_to_check = min(len(delayed), len(undelayed))

        delayed = delayed[:length_to_check]
        undelayed = undelayed[:length_to_check]
        correlations[ch] = pearsonr(delayed, undelayed)[0]
    return correlations


def __levison_durbin_recursion(autocorrelation: NDArray[np.float64]):
    """Levinson-Durbin recursion to be applied to the autocorrelation
    estimate.

    """
    signal_variance = autocorrelation[0]
    autocorr_coefficients = autocorrelation[1:]
    num_coefficients = len(autocorr_coefficients)
    ar_parameters = np.zeros(num_coefficients)

    prediction_error = signal_variance

    for order in range(num_coefficients):
        reflection_value = autocorr_coefficients[order]
        if order == 0:
            reflection_coefficient = -reflection_value / prediction_error
        else:
            for lag in range(order):
                reflection_value += (
                    ar_parameters[lag] * autocorr_coefficients[order - lag - 1]
                )
            reflection_coefficient = -reflection_value / prediction_error
        prediction_error *= 1.0 - reflection_coefficient**2.0
        if prediction_error <= 0:
            raise ValueError("Invalid prediction error: Singular Matrix")
        ar_parameters[order] = reflection_coefficient

        if order == 0:
            continue

        half_order = (order + 1) // 2
        for lag in range(half_order):
            reverse_lag = order - lag - 1
            save_value = ar_parameters[lag]
            ar_parameters[lag] = (
                save_value
                + reflection_coefficient * ar_parameters[reverse_lag]
            )
            if lag != reverse_lag:
                ar_parameters[reverse_lag] += (
                    reflection_coefficient * save_value
                )
    # Add first coefficient a0
    return np.hstack([1.0, ar_parameters])
