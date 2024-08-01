from numpy import zeros, array, unique, atleast_1d, complex128
import numpy as np
from numpy.typing import NDArray
from copy import deepcopy
from pickle import dump, HIGHEST_PROTOCOL
from warnings import warn

from .signal import Signal
from .._general_helpers import _check_format_in_path


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
        return self.__sampling_rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, new_sampling_rate_hz):
        new_sampling_rate_hz = array(new_sampling_rate_hz)
        if self.same_sampling_rate:
            new_sampling_rate_hz = new_sampling_rate_hz.squeeze()
            assert (
                new_sampling_rate_hz.ndim == 0
            ), "MultiBandSignal has only one sample rate"
            self.__sampling_rate_hz = int(new_sampling_rate_hz)
        else:
            new_sampling_rate_hz = atleast_1d(new_sampling_rate_hz)
            if hasattr(self, "__bands"):
                assert self.number_of_bands == len(
                    new_sampling_rate_hz
                ), "Number of bands does not match number of sampling rates"
            self.__sampling_rate_hz = [int(s) for s in new_sampling_rate_hz]

    @property
    def bands(self) -> list:
        return self.__bands

    @bands.setter
    def bands(self, new_bands):
        if new_bands is None:
            new_bands = []
        if type(new_bands) is tuple:
            new_bands = list(new_bands)
        assert type(new_bands) is list, "bands has to be a list"
        if new_bands:
            # Check length and number of channels
            self.number_of_channels = new_bands[0].number_of_channels
            sr = []
            complex_data = new_bands[0].time_data_imaginary is not None
            for s in new_bands:
                assert type(s) is Signal, (
                    f"{type(s)} is not a valid "
                    + "band type. Use Signal objects"
                )
                assert s.number_of_channels == self.number_of_channels, (
                    "Signals have different number of channels. This "
                    + "behaviour is not supported"
                )
                assert (s.time_data_imaginary is not None) == complex_data, (
                    "Some bands have imaginary time data and others do "
                    + "not. This behavior is not supported."
                )
                sr.append(s.sampling_rate_hz)
            if self.same_sampling_rate:
                self.sampling_rate_hz = new_bands[0].sampling_rate_hz
                self.band_length_samples = new_bands[0].time_data.shape[0]
            else:
                self.sampling_rate_hz = sr
            # Check sampling rate and duration
            if self.same_sampling_rate:
                for s in new_bands:
                    assert s.sampling_rate_hz == self.sampling_rate_hz, (
                        "Not all Signals have the same sampling rate. "
                        + "If you wish to create a multirate system, set "
                        + "same_sampling_rate to False"
                    )
                    assert s.time_data.shape[0] == self.band_length_samples, (
                        "The length of the bands is not always the same. "
                        + "This behaviour is not supported if there is a "
                        + "constant sampling rate"
                    )
        self.__bands = new_bands
        self._generate_metadata()

    @property
    def same_sampling_rate(self) -> bool:
        return self.__same_sampling_rate

    @same_sampling_rate.setter
    def same_sampling_rate(self, new_same):
        assert (
            type(new_same) is bool
        ), "Same sampling rate attribute must be a boolean"
        self.__same_sampling_rate = new_same

    @property
    def number_of_bands(self) -> int:
        return len(self.bands)

    def __len__(self):
        return len(self.bands)

    def __iter__(self):
        return iter(self.bands)

    def __str__(self):
        return self._get_metadata_str()

    def _generate_metadata(self):
        """Generates an information dictionary with metadata about the
        `MultiBandSignal`.

        """
        if not hasattr(self, "info"):
            self.info = {}
        self.info["number_of_bands"] = self.number_of_bands
        if self.bands:
            self.info["same_sampling_rate"] = self.same_sampling_rate
            if self.same_sampling_rate:
                if hasattr(self, "sampling_rate_hz"):
                    self.info["sampling_rate_hz"] = self.sampling_rate_hz
                self.info["band_length_samples"] = self.band_length_samples
            self.info["number_of_channels"] = self.number_of_channels

    # ======== Add and remove =================================================
    def add_band(self, sig: Signal, index: int = -1):
        """Adds a new band to the `MultiBandSignal`.

        Parameters
        ----------
        sig : `Signal`
            Signal to be added.
        index : int, optional
            Index at which to insert the new Signal. Default: -1.

        """
        bs = self.bands.copy()
        if not self.bands:
            bs.append(sig)
            self.bands = bs
        else:
            if index == -1:
                bs.append(sig)
            else:
                bs.insert(index, sig)
            self.bands = bs
        self._generate_metadata()

    def remove_band(self, index: int = -1, return_band: bool = False):
        """Removes a band from the `MultiBandSignal`.

        Parameters
        ----------
        index : int, optional
            This is the index from the bands list at which the band
            will be erased. When -1, last band is erased.
            Default: -1.
        return_band : bool, optional
            When `True`, the erased band is returned. Default: `False`.

        """
        assert self.bands, "There are no filters to remove"
        bs = self.bands.copy()
        f = bs.pop(index)
        self.bands = bs
        self._generate_metadata()
        if return_band:
            return f

    def swap_bands(self, new_order):
        """Rearranges the bands in the new given order.

        Parameters
        ----------
        new_order : array-like
            New rearrangement of bands.

        """
        new_order = array(new_order).squeeze()
        assert new_order.ndim == 1, (
            "Too many or too few dimensions are given in the new "
            + "arrangement vector"
        )
        assert self.number_of_bands == len(
            new_order
        ), "The number of bands does not match"
        assert all(new_order < self.number_of_bands) and all(new_order >= 0), (
            "Indexes of new bands have to be in "
            + f"[0, {self.number_of_bands - 1}]"
        )
        assert len(unique(new_order)) == len(
            new_order
        ), "There are repeated indexes in the new order vector"
        n_b = [self.bands[i] for i in new_order]
        self.bands = n_b

    def collapse(self) -> Signal:
        """Collapses MultiBandSignal by summing all of its bands and returning
        one Signal (possibly multichannel).

        Returns
        -------
        new_sig : `Signal`
            Collapsed Signal.

        """
        assert (
            self.same_sampling_rate
        ), "Collapsing is only available for same sampling rate bands"
        if self.bands[0].time_data_imaginary is None:
            initial = self.bands[0].time_data
            for n in range(1, len(self.bands)):
                initial += self.bands[n].time_data
        else:
            initial = zeros(self.bands[0].time_data.shape, dtype=complex128)
            for n in range(len(self.bands)):
                initial += self.bands[n].time_data
                initial += self.bands[n].time_data_imaginary * 1j
        new_sig = self.bands[0].copy()
        if hasattr(new_sig, "window"):
            del new_sig.window
        new_sig.time_data = initial
        return new_sig

    def show_info(self):
        """Show information about the `MultiBandSignal`."""
        print(self._get_metadata_str())

    def _get_metadata_str(self):
        txt = ""
        for k in self.info:
            txt += f""" | {str(k).replace('_', ' ').
                           capitalize()}: {self.info[k]}"""
        txt = "Multiband signal:" + txt
        txt += "\n"
        txt += "–" * len(txt)
        for ind, f1 in enumerate(self.bands):
            txt += "\n"
            txt += f"Signal {ind}:"
            for kf in f1.info:
                txt += f""" | {str(kf).replace('_', ' ').
                               capitalize()}: {f1.info[kf]}"""
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
                    new_time_data[:, n] = (
                        self.bands[n].time_data[:, channel].copy()
                    )
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
            sig = Signal(None, new_time_data, self.sampling_rate_hz)
            return sig
        else:
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
                warn("Output is complex since signal data had imaginary part")
            return new_time_data, sr

    def get_all_time_data(
        self,
    ) -> (
        tuple[NDArray[np.float64], int] | list[tuple[NDArray[np.float64], int]]
    ):
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
                    self.band_length_samples,
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
                        + (
                            b.time_data_imaginary * 1j if complex_data else 0.0
                        ),
                        b.sampling_rate_hz,
                    )
                )
            return td

    # ======== Saving and copying =============================================
    def save_signal(self, path: str = "multibandsignal"):
        """Saves the `MultiBandSignal` object as a pickle.

        Parameters
        ----------
        path : str, optional
            Path for the signal to be saved. Use only folder/folder/name
            (without format). Default: `'multibandsignal'`
            (local folder, object named multibandsignal).

        """
        path = _check_format_in_path(path, "pkl")
        with open(path, "wb") as data_file:
            dump(self, data_file, HIGHEST_PROTOCOL)

    def copy(self):
        """Returns a copy of the object.

        Returns
        -------
        new_sig : `MultiBandSignal`
            Copy of Signal.

        """
        return deepcopy(self)
