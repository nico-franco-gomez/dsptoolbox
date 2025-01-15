import numpy as np

from ..classes import Signal, MultiBandSignal
from .._general_helpers import _pad_trim


def pad_trim(
    signal: Signal | MultiBandSignal,
    desired_length_samples: int,
    in_the_end: bool = True,
) -> Signal | MultiBandSignal:
    """Returns a copy of the signal with padded or trimmed time data. If signal
    is `MultiBandSignal`, only `same_sampling_rate=True` is valid.

    Parameters
    ----------
    signal : `Signal` or `MultiBandSignal`
        Signal to be padded or trimmed.
    desired_length_samples : int
        Length of resulting signal.
    in_the_end : bool, optional
        Defines if padding or trimming should be done in the beginning or
        in the end of the signal. Default: `True`.

    Returns
    -------
    new_signal : `Signal` or `MultiBandSignal`
        New padded signal.

    """
    if isinstance(signal, Signal):
        new_time_data = np.zeros(
            (desired_length_samples, signal.number_of_channels)
        )
        for n in range(signal.number_of_channels):
            new_time_data[:, n] = _pad_trim(
                signal.time_data[:, n],
                desired_length_samples,
                in_the_end=in_the_end,
            )
        new_sig = signal.copy()
        new_sig.time_data = new_time_data
        return new_sig
    elif isinstance(signal, MultiBandSignal):
        assert (
            signal.same_sampling_rate
        ), "Padding or trimming is not supported for multirate signals"
        new_sig = signal.copy()
        for ind, b in enumerate(signal.bands):
            new_sig.bands[ind] = pad_trim(
                b, desired_length_samples, in_the_end
            )
        return new_sig
    else:
        raise TypeError("Signal must be of type Signal or MultiBandSignal")


def modify_signal_length(
    signal: Signal | MultiBandSignal,
    start_seconds: float | None,
    end_seconds: float | None,
) -> Signal | MultiBandSignal:
    """This function returns a copy of the signal with added silence at the
    beginning or the end of the signal. Time samples can also be trimmed when
    using negative time values.

    Parameters
    ----------
    signal : Signal, MultiBandSignal
        Signal to apply the length change to.
    start_seconds : float, None
        Seconds to add or remove from the start. Positive values append samples
        while negative ones remove them. Pass None to avoid any modification.
    end_seconds : float, None
        Seconds to add or remove from the end. Positive values append samples
        while negative ones remove them. Pass None to avoid any modification.

    Returns
    -------
    Signal or MultiBandSignal
        Copy of the signal with new length.

    """
    if isinstance(signal, Signal):
        assert (
            start_seconds is not None or end_seconds is not None
        ), "At least the start or the end should be modified"
        fs = signal.sampling_rate_hz
        start_samples = (
            0
            if start_seconds is None
            else int(start_seconds * fs + 0.5 * np.sign(start_seconds))
        )
        end_samples = (
            0
            if end_seconds is None
            else int(end_seconds * fs + 0.5 * np.sign(end_seconds))
        )

        # Avoid cutting too many samples
        if start_samples < 0:
            assert len(signal) > -start_samples, "Trimming is too much"
        if end_samples < 0:
            assert len(signal) > -end_samples, "Trimming is too much"
        if start_samples < 0 and end_samples < 0:
            assert len(signal) > -(
                start_samples + end_samples
            ), "Trimming is too much"

        new_sig = signal.copy()
        td = new_sig.time_data
        if start_samples >= 0:
            td = np.pad(td, ((start_samples, 0), (0, 0)))
        else:
            td = td[-start_samples:, ...]

        if end_samples >= 0:
            td = np.pad(td, ((0, end_samples), (0, 0)))
        else:
            td = td[:end_samples, ...]
        new_sig.time_data = td
        new_sig.clear_time_window()
        return new_sig
    elif isinstance(signal, MultiBandSignal):
        bands = []
        for b in signal:
            bands.append(modify_signal_length(b, start_seconds, end_seconds))
        new_mb = signal.copy()
        new_mb.bands = bands
        return new_mb
    else:
        raise TypeError("Unsupported type")
