import numpy as np
from copy import deepcopy

from ..classes import (
    Signal,
    MultiBandSignal,
    FilterBank,
    Spectrum,
)
from ..helpers.other import _pad_trim
from .enums import SpectrumType


def append_signals(
    signals: list[Signal] | list[MultiBandSignal],
    padding_trimming: bool = True,
    at_end: bool = True,
) -> Signal | MultiBandSignal:
    """Append all channels of the signals in the list. If their lengths are not
    the same trimming or padding can be applied to match the length of the
    first signal.

    Parameters
    ----------
    signals : list[Signal] or list[MultiBandSignal]
        First signal.
    padding_trimming : bool, optional
        If the signals do not have the same length, all are trimmed or
        zero-padded to match the first signal's length, when this is True.
        Otherwise, an error will be raised if the lengths do not match.
        Default: `True`.
    at_end : bool, optional
        When `True` and `padding_trimming=True`, padding or trimming is done
        at the end of the signals. Otherwise, it is done in the beginning.
        Default: `True`.

    Returns
    -------
    new_sig : Signal or MultiBandSignal
        Signal with all channels.

    """
    assert len(signals) > 1, "At least two signals should be passed"

    if isinstance(signals[0], Signal):
        complex_data = False
        for s in signals:
            assert isinstance(
                s, Signal
            ), "All signals must be of type Signal or ImpulseResponse"
            assert (
                s.sampling_rate_hz == signals[0].sampling_rate_hz
            ), "Sampling rates do not match"
            if not padding_trimming:
                assert len(s) == len(signals[0]), (
                    "Lengths do not match and padding or trimming "
                    + "is not activated"
                )
            complex_data |= s.is_complex_signal

        total_n_channels = sum([s.number_of_channels for s in signals])
        total_length = len(signals[0])
        td = np.zeros(
            (len(signals[0]), total_n_channels),
            dtype=np.complex128 if complex_data else np.float64,
        )

        current_channel = 0
        for s in signals:
            if complex_data:
                if s.is_complex_signal:
                    td[
                        :,
                        current_channel : current_channel
                        + s.number_of_channels,
                    ] = _pad_trim(
                        s.time_data + 1j * s.time_data_imaginary,
                        total_length,
                        in_the_end=at_end,
                    )
                else:
                    td[
                        :,
                        current_channel : current_channel
                        + s.number_of_channels,
                    ] = _pad_trim(
                        s.time_data.astype(np.complex128),
                        total_length,
                        in_the_end=at_end,
                    )
            else:
                td[
                    :, current_channel : current_channel + s.number_of_channels
                ] = _pad_trim(s.time_data, total_length, in_the_end=at_end)
            current_channel += s.number_of_channels
        new_sig = signals[0].copy()
        new_sig.time_data = td
        return new_sig
    elif isinstance(signals[0], MultiBandSignal):
        for s in signals:
            assert isinstance(
                s, MultiBandSignal
            ), "All signals must be of type MultiBandSignal"
            assert (
                s.same_sampling_rate == signals[0].same_sampling_rate
            ), "Sampling rates do not match"
            assert (
                s.sampling_rate_hz == signals[0].sampling_rate_hz
            ), "Sampling rates do not match"
            if not padding_trimming:
                assert s.length_samples == signals[0].length_samples, (
                    "Lengths do not match and padding or trimming "
                    + "is not activated"
                )
            assert (
                s.number_of_bands == signals[0].number_of_bands
            ), "Number of bands does not match"
        new_bands = []
        signals_without_first = signals.copy()  # Shallow copy
        signals_without_first.pop(0)
        for n in range(signals[0].number_of_bands):
            new_band = signals[0].bands[0].copy()
            for s in signals_without_first:
                new_band = append_signals(
                    [new_band, s.bands[n]], padding_trimming, at_end
                )
            new_bands.append(new_band)
        return MultiBandSignal(
            new_bands, same_sampling_rate=signals[0].same_sampling_rate
        )
    else:
        raise ValueError(
            "Signals have to be type of type Signal or MultiBandSignal"
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
    for f in fbs:
        assert (
            f.same_sampling_rate == fbs[0].same_sampling_rate
        ), "Sampling rates do not match"
        assert (
            f.sampling_rate_hz == fbs[0].sampling_rate_hz
        ), "Sampling rates do not match"

    new_fb = fbs[0].copy()
    for ind in range(1, len(fbs)):
        new_fb.filters += deepcopy(fbs[ind].filters)
    return new_fb


def append_spectra(
    spectra: list[Spectrum], complex_if_available: bool = True
) -> Spectrum:
    """Join all spectra by appending their channels.

    Parameters
    ----------
    spectra : list[Spectrum]
        Spectra to be appended. All spectra will be interpolated to the
        frequency vector of the first one.
    complex_if_available : bool, optional
        If the first spectrum is complex and this is set to True, all other
        spectra are expected to be complex as well and will be appended.
        In any other case, only magnitude spectra will be appended.
        Default: True.

    Returns
    -------
    Spectrum
        New spectrum with all channels.

    """
    assert len(spectra) > 1, "There must be at least two spectra to join"
    complex_append = complex_if_available and not spectra[0].is_magnitude
    if complex_append:
        assert all(
            [not s.is_magnitude for s in spectra]
        ), "At least one spectrum is not complex"

    total_channels = sum([s.number_of_channels for s in spectra])
    freqs = spectra[0].frequency_vector_hz
    spec = np.zeros(
        (len(freqs), total_channels),
        dtype=np.complex128 if complex_append else np.float64,
    )

    ch_ind = 0
    for s in spectra:
        spec[:, ch_ind : ch_ind + s.number_of_channels] = (
            s.get_interpolated_spectrum(
                freqs,
                (
                    SpectrumType.Complex
                    if complex_append
                    else SpectrumType.Magnitude
                ),
            )
        )
        ch_ind += s.number_of_channels

    return Spectrum(freqs, spec)
