import numpy as np

from ..classes import Signal, MultiBandSignal
from ..helpers.other import _pad_trim
from ..tools import from_db


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
        return signal.copy_with_new_time_data(new_time_data)
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


def trim_with_level_threshold(
    signal: Signal,
    threshold_db: float,
    at_start: bool = True,
    at_end: bool = True,
) -> tuple[Signal, int, int]:
    """Trim a signal by discarding the edge samples below a certain threshold.

    Parameters
    ----------
    signal : Signal
        Signal to be trimmed. If it is multichannel, it is trimmed to the
        minimum index at start and the maximum at end across all channels.
    threshold_db : float
        (Inclusive) Threshold for trimming. Generally in dBFS, but it can be
        in dBSPL if the signal has been calibrated.
    at_start : bool, optional
        Activate trimming in the beginning. Default: True.
    at_end : bool, optional
        Activate trimming in the end. Default: True.

    Returns
    -------
    Signal
        Copy of input signal with trimmed time series.
    int
        Start index in the original array.
    int
        Stop index in the original array.

    """
    assert at_start or at_end, "Either start or end should be trimmed"

    threshold_linear = from_db(threshold_db, True)
    above_threshold = np.where(np.abs(signal.time_data) >= threshold_linear)
    if at_start:
        indices_along_first_axis = above_threshold[0][
            : signal.number_of_channels
        ]
        start = int(np.min(indices_along_first_axis))
    else:
        start = 0

    if at_end:
        indices_along_first_axis = above_threshold[0][
            -signal.number_of_channels :
        ]
        stop = min(
            signal.length_samples, int(np.max(indices_along_first_axis)) + 1
        )
    else:
        stop = signal.length_samples

    return (
        signal.copy_with_new_time_data(signal.time_data[start:stop]),
        start,
        stop,
    )


def trim_with_time_selection(
    signal: Signal | MultiBandSignal,
    start_time_s: float | None,
    end_time_s: float | None,
    inclusive: bool = True,
):
    """Return a trimmed version of the input signal with a selected time
    window.

    Parameters
    ----------
    signal : Signal
        Input signal.
    start_time_s : float, None
        Start time for the window. Pass None to start the time window
        at the beginning of the signal.
    end_time_s : float, None
        End time for the window. Pass None to place the end of the time window
        at the end of the signal.
    inclusive : bool, optional
        When True, the bounds are inclusive. Default: True.

    Returns
    -------
    Signal
        Trimmed copy.

    """
    if isinstance(signal, Signal):
        assert (
            start_time_s is not None or end_time_s is not None
        ), "At least one bound must be other than None"
        if start_time_s:
            assert start_time_s >= 0.0, "Start time must be at least zero"
            assert (
                start_time_s < signal.length_seconds
            ), "Start time must be less than signal's length"
            start_sample = int(start_time_s * signal.sampling_rate_hz)
            if not inclusive:
                start_sample += 1
        else:
            start_sample = 0

        if end_time_s:
            assert end_time_s > 0.0, "End time must be greater than 0"
            assert (
                end_time_s <= signal.length_seconds
            ), "End time must be less than signal length"
            end_sample = int(end_time_s * signal.sampling_rate_hz)
            if inclusive:
                end_sample += 1
        else:
            end_sample = signal.length_samples

        assert end_sample > start_sample, "Invalid time window"
        selection = slice(start_sample, end_sample)
        return signal.copy_with_new_time_data(signal.time_data[selection, ...])
    elif isinstance(signal, MultiBandSignal):
        output = signal.copy()
        for ind in range(signal.number_of_bands):
            output.bands[ind] = trim_with_time_selection(
                signal.bands[ind], start_time_s, end_time_s, inclusive
            )
        return output

    raise TypeError("No valid type was passed")
