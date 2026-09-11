from ..classes import FilterBank, MultiBandSignal, Signal


def append_signals(
    signals: list[Signal] | list[MultiBandSignal],
    allow_padding_trimming: bool = True,
    at_end: bool = True,
) -> Signal | MultiBandSignal:
    """Append all channels of the signals in the list. If their lengths are not
    the same trimming or padding can be applied to match the length of the
    first signal.

    Parameters
    ----------
    signals : list[Signal] or list[MultiBandSignal]
        First signal.
    allow_padding_trimming : bool, optional
        If the signals do not have the same length, all are trimmed or
        zero-padded to match the first signal's length, when this is True.
        Otherwise, an error will be raised if the lengths do not match.
        Default: `True`.
    at_end : bool, optional
        When `True` and `allow_padding_trimming=True`, padding or trimming is done
        at the end of the signals. Otherwise, it is done in the beginning.
        Default: `True`.

    Returns
    -------
    new_sig : Signal or MultiBandSignal
        Signal with all channels.

    """
    assert len(signals) > 1, "At least two signals should be passed"
    assert isinstance(signals[0], (Signal, MultiBandSignal)), (
        "Signals have to be of type Signal or MultiBandSignal"
    )
    if isinstance(signals[0], Signal):
        signal_list = [signal for signal in signals if isinstance(signal, Signal)]
        assert len(signal_list) == len(signals)
        return signal_list[0].append_signals(
            signal_list[1:],
            allow_padding_trimming=allow_padding_trimming,
            at_end=at_end,
        )
    multiband_list = [
        signal for signal in signals if isinstance(signal, MultiBandSignal)
    ]
    assert len(multiband_list) == len(signals)
    return multiband_list[0].append_signals(
        multiband_list[1:],
        allow_padding_trimming=allow_padding_trimming,
        at_end=at_end,
    )


def append_filterbanks(fbs: list[FilterBank]) -> FilterBank:
    """Merges filterbanks by concatenating all of its filters.

    Parameters
    ----------
    fbs : list[FilterBank]
        List of FilterBanks.

    Returns
    -------
    new_fb : FilterBank
        New filterbank with all filters.

    """
    assert len(fbs) > 1, "At least two filter banks should be passed"
    return fbs[0].append_filterbanks(fbs[1:])
