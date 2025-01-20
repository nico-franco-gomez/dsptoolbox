import numpy as np
from numpy.typing import NDArray

from .gain_and_level import to_db


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
