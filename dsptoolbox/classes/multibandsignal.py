from copy import deepcopy
from pickle import HIGHEST_PROTOCOL, dump
from warnings import warn

import numpy as np
from numpy import array, atleast_1d, complex128, unique, zeros
from numpy.typing import NDArray

from ..helpers.other import _check_format_in_path
from .signal import Signal


class MultiBandSignal:
    """The `MultiBandSignal` class contains multiple Signal objects which are
    to be interpreted as frequency bands of the same signal. Since every
    signal has also multiple channels, the object resembles somewhat a
    3D-Matrix representation of a signal.

    The `MultiBandSignal` can be multirate system if the attribute
    `same_sampling_rate` is set to `False`. A dictionary called `info` can
    also carry all kinds of metadata that might characterize the signals.

    """

    # ======== Constructor and initializers ===================================
    def __init__(
        self,
        bands: list | None = None,
        same_sampling_rate: bool = True,
        info: dict | None = None,
    ):
        """`MultiBandSignal` contains a composite band list where each index
        is a Signal object with the same number of channels. For multirate
        systems, the parameter `same_sampling_rate` has to be set to `False`.

        Parameters
        ----------
        bands : list, optional
            List or tuple containing different Signal objects. All of them
            should be associated to the same Signal. This means that the
            channel numbers have to match. Set to `None` for initializing the
            object. Default: `None`.
        same_sampling_rate : bool, optional
            When `True`, every Signal should have the same sampling rate.
            Set to `False` for a multirate system. Default: `True`.
        info : dict, optional
            A dictionary with generic information about the `MultiBandSignal`
            can be passed. Default: `None`.

        """
        if info is None:
            info = {}
        self.same_sampling_rate = same_sampling_rate
        self.bands = bands if bands is not None else []
        self.info: dict = info

    # ======== Properties and setters =========================================
    @property
    def sampling_rate_hz(self) -> int:
        """Get the sampling rate(s) in Hz.

        Returns
        -------
        int | list[int]
            Sampling rate(s) in Hz. Returns a single integer if
            `same_sampling_rate` is True, otherwise a list of integers.

        """
        return self.__sampling_rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, new_sampling_rate_hz):
        """Set the sampling rate(s) in Hz.

        Parameters
        ----------
        new_sampling_rate_hz : int or array-like
            Sampling rate(s) in Hz. If `same_sampling_rate` is True, must be
            a single integer. If False, can be a sequence of integers with
            length matching the number of bands.

        Raises
        ------
        AssertionError
            If the number of sampling rates does not match the number of bands
            when `same_sampling_rate` is False, or if a scalar is not provided
            when `same_sampling_rate` is True.

        """
        new_sampling_rate_hz = array(new_sampling_rate_hz)
        if self.same_sampling_rate:
            new_sampling_rate_hz = new_sampling_rate_hz.squeeze()
            assert new_sampling_rate_hz.ndim == 0, (
                "MultiBandSignal has only one sample rate"
            )
            self.__sampling_rate_hz = int(new_sampling_rate_hz)
        else:
            new_sampling_rate_hz = atleast_1d(new_sampling_rate_hz)
            if hasattr(self, "_MultiBandSignal__bands"):
                assert self.number_of_bands == len(new_sampling_rate_hz), (
                    "Number of bands does not match number of sampling rates"
                )
            self.__sampling_rate_hz = [int(s) for s in new_sampling_rate_hz]

    @property
    def bands(self) -> list[Signal]:
        """Get the list of signal bands.

        Returns
        -------
        list[Signal]
            List of Signal objects representing the bands.

        """
        return self.__bands

    @bands.setter
    def bands(self, new_bands: list[Signal]):
        """Set the list of signal bands.

        Parameters
        ----------
        new_bands : list of Signal or None
            List of Signal objects representing the bands. If None or empty,
            the bands list is cleared. All signals must have:
            - The same number of channels
            - Consistent complex/real type
            - If `same_sampling_rate` is True: same sampling rate and duration

        Raises
        ------
        AssertionError
            If bands are not a list, contain non-Signal objects, have
            inconsistent channel counts, mixed complex/real types, or if
            sampling rates/durations don't match when required.

        Notes
        -----
        Setting bands automatically updates the sampling_rate property and
        validates consistency constraints based on the `same_sampling_rate` setting.

        """
        if new_bands is None:
            new_bands = []
        if type(new_bands) is tuple:
            new_bands = list(new_bands)
        assert type(new_bands) is list, "bands has to be a list"
        if new_bands:
            # Validate before mutating: the sampling rate setter cross-checks
            # against the band count, so the bands must be stored first
            number_of_channels = new_bands[0].number_of_channels
            complex_data = new_bands[0].time_data_imaginary is not None
            expected_length_samples = new_bands[0].length_samples
            sr = []
            for s in new_bands:
                assert isinstance(s, Signal), (
                    f"{type(s)} is not a valid " + "band type. Use Signal objects"
                )
                assert s.number_of_channels == number_of_channels, (
                    "Signals have different number of channels. This "
                    + "behavior is not supported"
                )
                assert (s.time_data_imaginary is not None) == complex_data, (
                    "Some bands have imaginary time data and others do "
                    + "not. This behavior is not supported."
                )
                if self.same_sampling_rate:
                    assert s.sampling_rate_hz == new_bands[0].sampling_rate_hz, (
                        "Not all Signals have the same sampling rate. "
                        + "If you wish to create a multirate system, set "
                        + "same_sampling_rate to False"
                    )
                    assert s.time_data.shape[0] == expected_length_samples, (
                        "The length of the bands is not always the same. "
                        + "This behavior is not supported if there is a "
                        + "constant sampling rate"
                    )
                sr.append(s.sampling_rate_hz)

            self.__number_of_channels = number_of_channels
            self.__bands: list[Signal] = new_bands
            self.sampling_rate_hz = (
                new_bands[0].sampling_rate_hz if self.same_sampling_rate else sr
            )
        else:
            self.__bands = new_bands

    @property
    def same_sampling_rate(self) -> bool:
        """Get whether all bands share the same sampling rate.

        Returns
        -------
        bool
            True if all bands have the same sampling rate, False otherwise
            (multirate system).

        """
        return self.__same_sampling_rate

    @same_sampling_rate.setter
    def same_sampling_rate(self, new_same):
        """Set whether all bands share the same sampling rate.

        Parameters
        ----------
        new_same : bool
            When True, all bands in this MultiBandSignal must have the same
            sampling rate. When False, allows bands with different sampling
            rates (multirate system).

        Raises
        ------
        AssertionError
            If new_same is not a boolean.

        Notes
        -----
        When `same_sampling_rate` is True, all bands must also have the same
        duration in samples.

        """
        assert type(new_same) is bool, "Same sampling rate attribute must be a boolean"
        self.__same_sampling_rate = new_same

    @property
    def number_of_bands(self) -> int:
        """Get the number of bands in the MultiBandSignal.

        Returns
        -------
        int
            Number of Signal objects (bands) contained.

        """
        return len(self.bands)

    @property
    def number_of_channels(self) -> int:
        """Get the number of channels in each band.

        Returns
        -------
        int
            Number of channels. All bands must have the same number
            of channels.

        """
        return self.__number_of_channels

    @property
    def length_seconds(self) -> float:
        """Get the duration of the signal in seconds.

        Returns
        -------
        float
            Duration in seconds. Returns 0.0 if there are no bands.
            Only valid when `same_sampling_rate` is True.

        """
        return self.bands[0].length_seconds if self.bands else 0.0

    @property
    def is_complex_signal(self) -> bool:
        """Check if the signal has complex (imaginary) time data.

        Returns
        -------
        bool
            True if the bands contain complex time data, False otherwise.
            Returns False if there are no bands.

        """
        if not self.bands:
            return False
        return self.bands[0].is_complex_signal

    @property
    def length_samples(self) -> list[int] | int:
        """Get the number of samples in each band.

        Returns
        -------
        int | list[int]
            If `same_sampling_rate` is True, returns a single integer
            representing the number of samples in each band. If False,
            returns a list of integers with the number of samples for
            each band respectively. Returns 0 if there are no bands.

        """
        if not self.bands:
            return 0

        return (
            self.bands[0].length_samples
            if self.same_sampling_rate
            else [b.length_samples for b in self.bands]
        )

    def __get_type_of_signal_bands(self):
        """Return type of saved bands (either Signal or ImpulseResponse)."""
        return type(self.bands[0])

    def __len__(self):
        return len(self.bands)

    def __iter__(self):
        return iter(self.bands)

    def __str__(self):
        return self.metadata_str

    @property
    def metadata(self) -> dict:
        """Get a dictionary with metadata about the multibandsignal.

        Returns
        -------
        dict
            Metadata

        """
        info = {}
        info["number_of_bands"] = self.number_of_bands
        if self.bands:
            info["same_sampling_rate"] = self.same_sampling_rate
            if self.same_sampling_rate:
                if hasattr(self, "sampling_rate_hz"):
                    info["sampling_rate_hz"] = self.sampling_rate_hz
                info["length_samples"] = self.length_samples
            info["number_of_channels"] = self.number_of_channels
        return info

    # ======== Add and remove =================================================
    def add_band(self, sig: Signal, index: int = -1):
        """Return a copy of the `MultiBandSignal` with a new band added.

        Parameters
        ----------
        sig : `Signal`
            Signal to be added.
        index : int, optional
            Index at which to insert the new Signal. Default: -1.

        Returns
        -------
        MultiBandSignal
            New multiband signal with the band added.

        """
        new = self.copy()
        bs = new.bands.copy()
        if not new.bands:
            bs.append(sig)
            new.bands = bs
        else:
            if index == -1:
                bs.append(sig)
            else:
                bs.insert(index, sig)
            new.bands = bs
        return new

    def remove_band(self, index: int = -1, return_band: bool = False):
        """Return a copy of the `MultiBandSignal` with a band removed.

        Parameters
        ----------
        index : int, optional
            This is the index from the bands list at which the band
            will be erased. When -1, last band is erased.
            Default: -1.
        return_band : bool, optional
            When `True`, a tuple of the new multiband signal and the erased
            band is returned. Otherwise, only the new multiband signal is
            returned. Default: `False`.

        Returns
        -------
        MultiBandSignal | tuple[MultiBandSignal, Signal]
            New multiband signal with the band removed, and optionally the
            removed band.

        """
        assert self.bands, "There are no filters to remove"
        new = self.copy()
        bs = new.bands.copy()
        f = bs.pop(index)
        new.bands = bs
        if return_band:
            return new, f
        return new

    def swap_bands(self, new_order):
        """Return a copy of the `MultiBandSignal` with the bands rearranged
        in the new given order.

        Parameters
        ----------
        new_order : array-like
            New rearrangement of bands.

        Returns
        -------
        MultiBandSignal
            New multiband signal with the bands rearranged.

        """
        new_order = array(new_order).squeeze()
        assert new_order.ndim == 1, (
            "Too many or too few dimensions are given in the new "
            + "arrangement vector"
        )
        assert self.number_of_bands == len(new_order), (
            "The number of bands does not match"
        )
        assert all(new_order < self.number_of_bands) and all(new_order >= 0), (
            "Indexes of new bands have to be in " + f"[0, {self.number_of_bands - 1}]"
        )
        assert len(unique(new_order)) == len(new_order), (
            "There are repeated indexes in the new order vector"
        )
        new = self.copy()
        new.bands = [self.bands[i] for i in new_order]
        return new

    def collapse(self) -> Signal:
        """Collapses MultiBandSignal by summing all of its bands and returning
        one Signal (possibly multichannel).

        Returns
        -------
        new_sig : `Signal`
            Collapsed Signal.

        """
        assert self.same_sampling_rate, (
            "Collapsing is only available for same sampling rate bands"
        )
        if self.bands[0].time_data_imaginary is None:
            # Copy, otherwise `+=` accumulates into the band's own array
            initial = self.bands[0].time_data.copy()
            for n in range(1, len(self.bands)):
                initial += self.bands[n].time_data
        else:
            initial = zeros(self.bands[0].time_data.shape, dtype=complex128)
            for n in range(len(self.bands)):
                initial += self.bands[n].time_data
                initial += self.bands[n].time_data_imaginary * 1j
        return self.bands[0].copy_with_new_time_data(initial)

    def show_info(self):
        """Show information about the `MultiBandSignal`."""
        print(self.metadata_str)
        return self

    @property
    def metadata_str(self) -> str:
        """Get a formatted string representation of the metadata.

        Returns
        -------
        str
            Formatted metadata string containing information about the
            MultiBandSignal and all its bands.

        """
        header = "Multiband signal:"
        md = self.metadata | self.info
        for k in md:
            header += f""" | {str(k).replace("_", " ").capitalize()}: {md[k]}"""
        txt = header + "\n" + "–" * len(header)
        for ind, band in enumerate(self.bands):
            txt += "\n"
            txt += f"Signal {ind}:"
            band_metadata = band.metadata
            for kf in band_metadata:
                txt += (
                    f""" | {str(kf).replace("_", " ").capitalize()}: """
                    f"""{band_metadata[kf]}"""
                )
        return txt

    # ======== Getters ========================================================
    def get_all_bands(
        self, channel: int = 0
    ) -> Signal | tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        """Broadcasts and returns the `MultiBandSignal` as a `Signal` object
        with all bands as channels in the output. This is done only for a
        single channel of the original signal.

        Parameters
        ----------
        channel : int, optional
            Channel to choose from the band signals.

        Returns
        -------
        sig : `Signal` or list of NDArray[np.float64] and list of int
            Multichannel signal with all the bands. If the `MultiBandSignal`
            does not have the same sampling rate for all signals, a list with
            the time data vectors and a list containing their sampling
            rates are returned.

        """
        if self.same_sampling_rate:
            # Check if there is complex time data
            if self.bands[0].time_data_imaginary is None:
                new_time_data = zeros(
                    (self.bands[0].time_data.shape[0], len(self.bands))
                )
                for n in range(len(self.bands)):
                    new_time_data[:, n] = self.bands[n].time_data[:, channel].copy()
            else:
                new_time_data = zeros(
                    (self.bands[0].time_data.shape[0], len(self.bands)),
                    dtype=complex128,
                )
                for n in range(len(self.bands)):
                    new_time_data[:, n] = (
                        self.bands[n].time_data[:, channel]
                        + self.bands[n].time_data_imaginary[:, channel] * 1j
                    )
            return self.__get_type_of_signal_bands()(
                None, new_time_data, self.sampling_rate_hz
            )

        new_time_data = []
        sr = []
        if self.bands[0].time_data_imaginary is None:
            for n in range(len(self.bands)):
                new_time_data.append(self.bands[n].time_data[:, channel])
                sr.append(self.bands[n].sampling_rate_hz)
        else:
            for n in range(len(self.bands)):
                new_time_data.append(
                    self.bands[n].time_data[:, channel]
                    + self.bands[n].time_data_imaginary[:, channel] * 1j
                )
                sr.append(self.bands[n].sampling_rate_hz)
            warn("Output is complex since signal data had imaginary part", stacklevel=2)
        return new_time_data, sr

    def get_all_time_data(
        self,
    ) -> tuple[NDArray[np.float64], int] | list[tuple[NDArray[np.float64], int]]:
        """
        Get all time data saved in the MultiBandSignal. If it has consistent
        sampling rate, a single array with shape (time samples, band, channel)
        is returned, otherwise a list of bands with arrays with shape (time
        samples, channel) is returned.

        Returns
        -------
        if `self.same_sampling_rate=True` :

            time_data : NDArray[np.float64]
                Time samples.
            int
                Sampling rate in Hz

        else :

            list[tuple[NDArray[np.float64], int]]
                List with each band where time samples and sampling rate are
                contained.

        """
        complex_data = self.bands[0].time_data_imaginary is not None
        if self.same_sampling_rate:
            td = zeros(
                (
                    self.length_samples,
                    self.number_of_bands,
                    self.number_of_channels,
                ),
                dtype=(complex128 if complex_data else "float"),
            )
            for ind, b in enumerate(self.bands):
                td[:, ind, :] = b.time_data + (
                    b.time_data_imaginary * 1j if complex_data else 0.0
                )
            return td, self.sampling_rate_hz
        else:
            td = []
            for b in self.bands:
                td.append(
                    (
                        b.time_data
                        + (b.time_data_imaginary * 1j if complex_data else 0.0),
                        b.sampling_rate_hz,
                    )
                )
            return td

    # ======== Saving and copying =============================================
    def save_signal(self, path: str):
        """Saves the `MultiBandSignal` object as a pickle.

        Parameters
        ----------
        path : str
            Path for the multiband signal to be saved with format `.pkl`.

        """
        path = _check_format_in_path(path, "pkl")
        with open(path, "wb") as data_file:
            dump(self, data_file, HIGHEST_PROTOCOL)
        return self

    def copy(self) -> "MultiBandSignal":
        """Returns a copy of the object.

        Returns
        -------
        new_sig : `MultiBandSignal`
            Copy of Signal.

        """
        return deepcopy(self)

    # ======== Transforms (returning a new MultiBandSignal) ===================
    def pad_trim(
        self, desired_length_samples: int, in_the_end: bool = True
    ) -> "MultiBandSignal":
        """Return a copy of the multiband signal with padded or trimmed time
        data in each band. Only valid for `same_sampling_rate=True`.

        Parameters
        ----------
        desired_length_samples : int
            Length of resulting signal.
        in_the_end : bool, optional
            Defines if padding or trimming should be done in the beginning or
            in the end of the signal. Default: `True`.

        Returns
        -------
        MultiBandSignal
            New padded or trimmed multiband signal.

        """
        assert self.same_sampling_rate, (
            "Padding or trimming is not supported for multirate signals"
        )
        new = self.copy()
        new.bands = [b.pad_trim(desired_length_samples, in_the_end) for b in self.bands]
        return new

    def modify_signal_length(
        self, start_seconds: float | None, end_seconds: float | None
    ) -> "MultiBandSignal":
        """Return a copy of the multiband signal with added silence at the
        beginning or the end of each band. Time samples can also be trimmed
        when using negative time values.

        Parameters
        ----------
        start_seconds : float, None
            Seconds to add or remove from the start. Positive values append
            samples while negative ones remove them. Pass None to avoid any
            modification.
        end_seconds : float, None
            Seconds to add or remove from the end. Positive values append
            samples while negative ones remove them. Pass None to avoid any
            modification.

        Returns
        -------
        MultiBandSignal
            Copy of the multiband signal with new length.

        """
        new = self.copy()
        new.bands = [
            b.modify_signal_length(start_seconds, end_seconds) for b in self.bands
        ]
        return new

    def trim_with_time_selection(
        self,
        start_time_s: float | None,
        end_time_s: float | None,
        inclusive: bool = True,
    ) -> "MultiBandSignal":
        """Return a copy of the multiband signal trimmed to a selected time
        window in each band.

        Parameters
        ----------
        start_time_s : float, None
            Start time for the window. Pass None to start the time window
            at the beginning of the signal.
        end_time_s : float, None
            End time for the window. Pass None to place the end of the time
            window at the end of the signal.
        inclusive : bool, optional
            When True, the bounds are inclusive. Default: True.

        Returns
        -------
        MultiBandSignal
            Trimmed copy.

        """
        new = self.copy()
        new.bands = [
            b.trim_with_time_selection(start_time_s, end_time_s, inclusive)
            for b in self.bands
        ]
        return new

    def normalize(
        self,
        norm_dbfs: float,
        peak_normalization: bool = True,
        each_channel: bool = False,
    ) -> "MultiBandSignal":
        """Return a copy of the multiband signal normalized to a given dBFS
        value in each band. It either normalizes each channel or each band
        as a whole.

        Parameters
        ----------
        norm_dbfs : float
            Value in dBFS to reach after normalization.
        peak_normalization : bool, optional
            When True, signal is normalized at peak. False uses RMS value.
            Default: True.
        each_channel : bool, optional
            When `True`, each channel on its own is normalized. When
            `False`, the peak or rms value across all channels is regarded.
            Default: `False`.

        Returns
        -------
        MultiBandSignal
            Normalized multiband signal.

        """
        new = self.copy()
        new.bands = [
            b.normalize(norm_dbfs, peak_normalization, each_channel) for b in self.bands
        ]
        return new

    def apply_gain(self, gain_db: float | NDArray[np.float64]) -> "MultiBandSignal":
        """Return a copy of the multiband signal with gain applied to each
        band.

        Parameters
        ----------
        gain_db : float, NDArray[np.float64]
            Gain in dB to be applied. If it is an array, it should have as
            many elements as there are channels in the signal.

        Returns
        -------
        MultiBandSignal
            Multiband signal with new gain.

        """
        new = self.copy()
        new.bands = [b.apply_gain(gain_db) for b in self.bands]
        return new

    def detrend(self, polynomial_order: int = 0) -> "MultiBandSignal":
        """Return the detrended multiband signal.

        Parameters
        ----------
        polynomial_order : int, optional
            Polynomial order of the fitted polynomial that will be removed
            from time data. 0 is equal to mean removal. Default: 0.

        Returns
        -------
        MultiBandSignal
            Detrended multiband signal.

        """
        new = self.copy()
        new.bands = [b.detrend(polynomial_order) for b in self.bands]
        return new

    def fractional_delay(
        self,
        delay_seconds: float,
        channels=None,
        keep_length: bool = False,
        order: int = 30,
        side_lobe_suppression_db: float = 60,
    ) -> "MultiBandSignal":
        """Return a copy of the multiband signal with fractional time delay
        applied to each band.

        Parameters
        ----------
        delay_seconds : float
            Delay in seconds.
        channels : int or array-like, optional
            Channels to be delayed. Pass `None` to delay all channels.
            Default: `None`.
        keep_length : bool, optional
            When `True`, the signal retains its original length and loses
            information for the latest samples. Default: `False`.
        order : int, optional
            Order of the sinc filter, higher order yields better results at
            the expense of computation time. Default: 30.
        side_lobe_suppression_db : float, optional
            Side lobe suppression in dB for the Kaiser window. Default: 60.

        Returns
        -------
        MultiBandSignal
            Delayed multiband signal.

        """
        new = self.copy()
        new.bands = [
            b.fractional_delay(
                delay_seconds, channels, keep_length, order, side_lobe_suppression_db
            )
            for b in self.bands
        ]
        return new

    def delay(
        self,
        delay_samples: int,
        channels=None,
        keep_length: bool = False,
    ) -> "MultiBandSignal":
        """Return a copy of the multiband signal with a time delay applied
        to each band. This method is faster than `fractional_delay` because
        it only applies integer delay by zero-padding.

        Parameters
        ----------
        delay_samples : int
            Delay in samples.
        channels : int or array-like, optional
            Channels to be delayed. Pass `None` to delay all channels.
            Default: `None`.
        keep_length : bool, optional
            When `True`, the signal retains its original length and loses
            information for the latest samples. Default: `False`.

        Returns
        -------
        MultiBandSignal
            Delayed multiband signal.

        """
        new = self.copy()
        new.bands = [b.delay(delay_samples, channels, keep_length) for b in self.bands]
        return new

    def append_signals(
        self,
        others: list["MultiBandSignal"],
        allow_padding_trimming: bool = True,
        at_end: bool = True,
    ) -> "MultiBandSignal":
        """Return a copy of the multiband signal with the channels of other
        multiband signals appended to each band. If their lengths are not
        the same, trimming or padding can be applied to match this signal's
        length.

        Parameters
        ----------
        others : list[MultiBandSignal]
            Other multiband signals whose channels should be appended.
        allow_padding_trimming : bool, optional
            If the signals do not have the same length, all are trimmed or
            zero-padded to match this signal's length, when this is True.
            Otherwise, an error will be raised if the lengths do not match.
            Default: `True`.
        at_end : bool, optional
            When `True` and `allow_padding_trimming=True`, padding or
            trimming is done at the end of the signals. Otherwise, it is
            done in the beginning. Default: `True`.

        Returns
        -------
        MultiBandSignal
            Multiband signal with all channels.

        """
        signals = [self] + list(others)
        assert len(signals) > 1, "At least one other signal should be passed"
        for s in signals:
            assert isinstance(s, MultiBandSignal), (
                "All signals must be of type MultiBandSignal"
            )
            assert s.same_sampling_rate == signals[0].same_sampling_rate, (
                "Sampling rates do not match"
            )
            assert s.sampling_rate_hz == signals[0].sampling_rate_hz, (
                "Sampling rates do not match"
            )
            if not allow_padding_trimming:
                assert s.length_samples == signals[0].length_samples, (
                    "Lengths do not match and padding or trimming " + "is not activated"
                )
            assert s.number_of_bands == signals[0].number_of_bands, (
                "Number of bands does not match"
            )
        new_bands = []
        signals_without_first = signals[1:]
        for n in range(signals[0].number_of_bands):
            new_band = signals[0].bands[n].copy()
            for s in signals_without_first:
                new_band = new_band.append_signals(
                    [s.bands[n]], allow_padding_trimming, at_end
                )
            new_bands.append(new_band)
        return MultiBandSignal(
            new_bands, same_sampling_rate=signals[0].same_sampling_rate
        )
