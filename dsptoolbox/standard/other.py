import pickle

import numpy as np
from scipy.signal import (
    hilbert,
    oaconvolve,
)

from ..classes import (
    MultiBandSignal,
    Signal,
)
from ..helpers.other import _check_path_format


def load_pkl_object(path: str):
    """WARNING: This is not secure. Only unpickle data you know!
    Loads an object with all its attributes and methods.

    Parameters
    ----------
    path : str
        Path to the pickle object.

    Returns
    -------
    obj : object
        Unpacked pickle object.

    """
    obj = None
    _check_path_format(path, "pkl")
    with open(path, "rb") as inp:
        obj = pickle.load(inp)
    return obj


def envelope(
    signal: Signal | MultiBandSignal,
    analytic: bool = True,
    window_length_samples: int | None = None,
):
    """This function computes the envelope of a given signal by means of its
    hilbert transformation. It can also compute the RMS value over a certain
    window length (boxcar). The time signal is always detrended with a linear
    polynomial.

    Parameters
    ----------
    signal : `Signal` or `MultiBandSignal`
        Time series for which to find the envelope. If it is a
        `MultiBandSignal`, it must have the same sampling rate for all bands.
    analytic : bool, optional
        When True, the hilbert transform is used to obtain the envelope of the
        analytic signal. Otherwise, a RMS envelope is computed with a boxcar
        window. Default: True.
    window_length_samples : int, optional
        Window length (boxcar) to average the RMS values. Cannot be `None`
        if `mode = 'rms'`. Default: `None`.

    Returns
    -------
    NDArray[np.float64]
        Signal envelope. It has the shape (time sample, channel) or
        (time sample, band, channel) in case of `MultiBandSignal`.

    """
    if isinstance(signal, Signal):
        signal = signal.detrend(1)

        if analytic:
            env = signal.time_data
            env = np.abs(hilbert(env, axis=0))
            return env

        assert window_length_samples is not None, "Some window length must be passed"
        assert window_length_samples > 0, "Window length must be more than 1 sample"
        rms_vec = signal.time_data
        rms_vec = oaconvolve(
            rms_vec**2,
            np.ones(window_length_samples)[..., None] / window_length_samples,
            mode="full",
            axes=0,
        )[: len(rms_vec), ...]
        rms_vec **= 0.5
        return rms_vec
    elif isinstance(signal, MultiBandSignal):
        assert signal.same_sampling_rate, (
            "This is only available for constant sampling rate bands"
        )
        rms_vec = np.zeros(
            (
                len(signal.bands[0]),
                signal.number_of_bands,
                signal.number_of_channels,
            ),
            float,
        )
        for ind, b in enumerate(signal.bands):
            rms_vec[:, ind, :] = envelope(
                b,
                analytic=analytic,
                window_length_samples=window_length_samples,
            )
        return rms_vec
    else:
        raise TypeError("Signal must be type Signal or MultiBandSignal")
