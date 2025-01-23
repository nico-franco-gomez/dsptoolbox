import numpy as np
from numpy.typing import NDArray

from ..standard.enums import FadeType


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


def _normalize(
    s: NDArray[np.float64],
    dbfs: float,
    peak_normalization: bool,
    per_channel: bool,
) -> NDArray[np.float64]:
    """Normalizes a signal.

    Parameters
    ----------
    s: NDArray[np.float64]
        Signal to normalize. It can be 1 or 2D. Time samples are assumed to
        be in the outer axis.
    dbfs: float
        dbfs value to normalize to.
    peak_normalization: Bool
        Mode of normalization. True -> `peak`, False -> `rms`.

    Returns
    -------
    s_out: NDArray[np.float64]
        Normalized signal.

    """
    onedim = s.ndim == 1
    if onedim:
        s = s[..., None]

    factor = from_db(dbfs, True)
    if peak_normalization:
        factor /= np.max(np.abs(s), axis=0 if per_channel else None)
    else:
        factor /= _rms(s if per_channel else s.flatten())
    s_norm = s * factor

    return s_norm[..., 0] if onedim else s_norm


def _amplify_db(s: NDArray[np.float64], db: float) -> NDArray[np.float64]:
    """Amplify by dB."""
    return s * 10 ** (db / 20)


def _fade(
    s: NDArray[np.float64],
    length_seconds: float,
    mode: FadeType,
    sampling_rate_hz: int,
    at_start: bool,
) -> NDArray[np.float64]:
    """Create a fade in signal.

    Parameters
    ----------
    s : NDArray[np.float64]
        np.array to be faded.
    length_seconds : float
        Length of fade in seconds.
    mode : FadeType
        Type of fading.
    sampling_rate_hz : int
        Sampling rate.
    at_start : bool
        When `True`, the start is faded. When `False`, the end.

    Returns
    -------
    s : NDArray[np.float64]
        Faded vector.

    """
    if mode == FadeType.NoFade:
        return s

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

    if mode == FadeType.Exponential:
        db = np.linspace(-100, 0, l_samples)
        fade = 10 ** (db / 20)
    elif mode == FadeType.Linear:
        fade = np.linspace(0, 1, l_samples)
    else:  # FadeType.Logarithmic
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


def to_db(
    x: NDArray,
    amplitude_input: bool,
    dynamic_range_db: float | None = None,
    min_value: float | None = float(np.finfo(np.float64).smallest_normal),
) -> NDArray[np.float64]:
    """Convert to dB from amplitude or power representation. Clipping small
    values can be activated in order to avoid -inf dB outcomes.

    Parameters
    ----------
    x : NDArray
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
