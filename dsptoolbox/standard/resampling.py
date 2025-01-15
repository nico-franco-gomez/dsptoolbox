import numpy as np
from scipy.signal import resample_poly, bilinear_zpk

from fractions import Fraction
from ..classes import Signal, Filter
from .enums import FilterCoefficientsType


def resample(
    sig: Signal, desired_sampling_rate_hz: int, rescaling: bool = True
) -> Signal:
    """Resamples signal to the desired sampling rate using
    `scipy.signal.resample_poly` with an efficient polyphase representation.

    Parameters
    ----------
    sig : `Signal`
        Signal to be resampled.
    desired_sampling_rate_hz : int
        Sampling rate to convert the signal to.
    rescaling : bool, optional
        When True, the data is rescaled by dividing by the resampling factor.

    Returns
    -------
    new_sig : `Signal`
        Resampled signal. It is rescaled by the resampling factor.

    """
    if sig.sampling_rate_hz == desired_sampling_rate_hz:
        return sig.copy()
    ratio = Fraction(
        numerator=desired_sampling_rate_hz, denominator=sig.sampling_rate_hz
    )
    u, d = ratio.as_integer_ratio()
    new_time_data = resample_poly(sig.time_data, up=u, down=d, axis=0)
    new_sig = sig.copy()
    new_sig.clear_time_window()
    new_sig.time_data = new_time_data * (d / u) if rescaling else new_time_data
    new_sig.sampling_rate_hz = desired_sampling_rate_hz
    return new_sig


def resample_filter(filter: Filter, new_sampling_rate_hz: int) -> Filter:
    """This function resamples a filter by mapping its zpk representation
    to the s-plane and reapplying the bilinear transform to the new sampling
    rate. This approach can deliver satisfactory results for filters whose
    poles and zeros correspond to low normalized frequencies (~0.1), but higher
    frequencies will get significantly distorted due to the bilinear mapping.

    Parameters
    ----------
    filter : Filter
        Filter to resample.
    new_sampling_rate_hz : int
        Target sampling rate in Hz.

    Returns
    -------
    Filter
        Filter with new sampling rate.

    """
    z, p, k = filter.get_coefficients(FilterCoefficientsType.Zpk)
    add_to_poles = max(0, len(z) - len(p))
    add_to_zeros = max(0, len(p) - len(z))

    f = 2 * filter.sampling_rate_hz
    p = f * (p - 1) / (p + 1)
    z = z[z != -1.0]
    z = f * (z - 1) / (z + 1)

    if add_to_poles:
        p = np.hstack([p, [-f] * (len(z) - len(p))])
    if add_to_zeros:
        z = np.hstack([z, [-f] * (len(p) - len(z))])

    k /= np.real(np.prod(f - z) / np.prod(f - p))

    z, p, k = bilinear_zpk(z, p, k, new_sampling_rate_hz)
    return Filter.from_zpk(z, p, k, new_sampling_rate_hz)
