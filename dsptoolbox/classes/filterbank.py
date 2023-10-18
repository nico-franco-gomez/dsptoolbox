from pickle import dump, HIGHEST_PROTOCOL
from copy import deepcopy
import numpy as np
from warnings import warn
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .signal_class import Signal
from .multibandsignal import MultiBandSignal
from .filter_class import Filter
from ._filter import _filterbank_on_signal
from ..generators import dirac
from ..plots import general_plot
from .._general_helpers import _get_normalized_spectrum, _check_format_in_path
from .._standard import _group_delay_direct


class FilterBank:
    """Standard template for filter banks containing filters, filters' initial
    values, metadata and some useful plotting methods.

    """

    # ======== Constructor and initializers ===================================
    def __init__(
        self,
        filters: list | None = None,
        same_sampling_rate: bool = True,
        info: dict | None = None,
    ):
        """FilterBank object saves multiple filters and some metadata.
        It also allows for easy filtering with multiple filters.
        Since the digital filters that are supported are linear systems,
        the order in which they are saved and applied to a signal is not
        relevant.

        Parameters
        ----------
        filters : list, optional
            List containing filters.
        same_sampling_rate : bool, optional
            When `True`, every Filter should have the same sampling rate.
            Set to `False` for a multirate system. Default: `True`.
        info : dict, optional
            Dictionary containing general information about the filter bank.
            Some parameters of the filter bank are automatically read from
            the filters dictionary.

        Methods
        -------
        General
            add_filter, remove_filter, swap_filters, copy, save_filterbank.
        Prints and plots
            plot_magnitude, plot_phase, plot_group_delay, show_info.

        """
        if info is None:
            info = {}
        self.same_sampling_rate = same_sampling_rate
        self.filters = filters
        self.info = self.info | info

    def _generate_metadata(self):
        """Generates the info dictionary with metadata about the FilterBank."""
        self.info = {}
        self.info["number_of_filters"] = self.number_of_filters
        self.info["same_sampling_rate"] = self.same_sampling_rate
        if self.same_sampling_rate:
            if hasattr(self, "sampling_rate_hz"):
                self.info["sampling_rate_hz"] = self.sampling_rate_hz
        self.info["types_of_filters"] = tuple(
            set([f.info["filter_type"] for f in self.filters])
        )

    def initialize_zi(self, number_of_channels: int = 1):
        """Initiates the zi of the filters for the given number of channels.

        Parameters
        ----------
        number_of_channels : int, optional
            Number of channels is needed for the number of filters' zi's.
            Default: 1.

        """
        for f in self.filters:
            f.initialize_zi(number_of_channels)

    @property
    def sampling_rate_hz(self) -> int | np.ndarray:
        return self.__sampling_rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, new_sampling_rate_hz):
        new_sampling_rate_hz = np.asarray(new_sampling_rate_hz).squeeze()
        if self.same_sampling_rate:
            assert (
                new_sampling_rate_hz.ndim == 0
            ), "Sampling rate should be only an integer"
            self.__sampling_rate_hz = int(new_sampling_rate_hz)
        else:
            new_sampling_rate_hz = np.atleast_1d(new_sampling_rate_hz)
            self.__sampling_rate_hz = [int(s) for s in new_sampling_rate_hz]

    @property
    def filters(self) -> list:
        return self.__filters

    @filters.setter
    def filters(self, new_filters):
        if new_filters is None:
            new_filters = []
        if type(new_filters) is tuple:
            new_filters = list(new_filters)
        assert type(new_filters) is list, "Filters have to be passed as a list"
        if new_filters:
            if self.same_sampling_rate:
                self.sampling_rate_hz = new_filters[0].sampling_rate_hz
            else:
                sr = []
                for f in new_filters:
                    sr.append(f.sampling_rate_hz)
                self.sampling_rate_hz = sr
            for ind, f in enumerate(new_filters):
                assert (
                    type(f) is Filter
                ), f"Object at index {ind} is not a supported Filter"
                if self.same_sampling_rate:
                    assert (
                        f.sampling_rate_hz == self.sampling_rate_hz
                    ), "Sampling rates do not match"
        self.__filters = new_filters
        self._generate_metadata()

    @property
    def number_of_filters(self) -> int:
        return len(self.__filters)

    def __len__(self):
        return len(self.__filters)

    def __iter__(self):
        return iter(self.filters)

    def __str__(self):
        return self._get_metadata_str()

    @property
    def same_sampling_rate(self) -> bool:
        return self.__same_sampling_rate

    @same_sampling_rate.setter
    def same_sampling_rate(self, new_same):
        assert type(new_same) is bool, "same_sampling_rate must be a boolean"
        self.__same_sampling_rate = new_same

    # ======== Add and remove =================================================
    def add_filter(self, filt: Filter, index: int = -1):
        """Adds a new filter at the end of the filters dictionary.

        Parameters
        ----------
        filt : `Filter`
            Filter to be added to the FilterBank.
        index : int, optional
            Index at which to insert the new Filter. Default: -1.

        """
        if not self.filters:
            self.sampling_rate_hz = filt.sampling_rate_hz
            self.filters = [filt]
        else:
            fs = self.filters.copy()
            if self.same_sampling_rate:
                assert (
                    self.sampling_rate_hz == filt.sampling_rate_hz
                ), "Sampling rates do not match"
            if index == -1:
                fs.append(filt)
            else:
                fs.insert(index, filt)
            self.filters = fs
        self._generate_metadata()

    def remove_filter(
        self, index: int = -1, return_filter: bool = False
    ) -> None | Filter:
        """Removes a filter from the filter bank.

        Parameters
        ----------
        index : int, optional
            This is the index from the filters list at which the filter
            will be erased. When -1, last filter is erased.
            Default: -1.
        return_filter : bool, optional
            When `True`, the erased filter is returned. Default: `False`.

        """
        assert self.filters, "There are no filters to remove"
        if index == -1:
            index = len(self.filters) - 1
        assert index in range(
            len(self.filters)
        ), f"There is no filter at index {index}."
        n_f = self.filters.copy()
        f = n_f.pop(index)
        self.filters = n_f
        if return_filter:
            return f

    def swap_filters(self, new_order):
        """Rearranges the filters in the new given order.

        Parameters
        ----------
        new_order : array-like
            New rearrangement of filters.

        """
        new_order = np.array(new_order).squeeze()
        assert new_order.ndim == 1, (
            "Too many or too few dimensions are given in the new "
            + "arrangement vector"
        )
        assert self.number_of_filters == len(
            new_order
        ), "The number of filters does not match"
        assert all(new_order < self.number_of_filters) and all(
            new_order >= 0
        ), (
            "Indexes of new filters have to be in "
            + f"[0, {self.number_of_filters - 1}]"
        )
        assert len(np.unique(new_order)) == len(
            new_order
        ), "There are repeated indexes in the new order vector"
        n_f = [self.filters[i] for i in new_order]
        self.filters = n_f

    # ======== Filtering ======================================================
    def filter_signal(
        self,
        signal: Signal,
        mode: str = "parallel",
        activate_zi: bool = False,
        zero_phase: bool = False,
    ) -> Signal | MultiBandSignal:
        """Applies the filter bank to a signal and returns a multiband signal
        or a `Signal` object.
        `'parallel'`: returns a `MultiBandSignal` object where each band is
        the output of each filter.
        `'sequential'`: applies each filter to the given Signal in a sequential
        manner and returns output with same dimension.
        `'summed'`: applies every filter as parallel and then sums the outputs
        returning same dimensional output as input.

        Parameters
        ----------
        signal : `Signal`
            Signal to be filtered.
        mode : str, optional
            Way to apply filter bank to the signal. Supported modes are:
            `'parallel'`, `'sequential'`, `'summed'`. Default: `'parallel'`.
        activate_zi : bool, optional
            Takes in the filter initial values and updates them for
            streaming purposes. Default: `False`.
        zero_phase : bool, optional
            Activates zero_phase filtering for the filter bank. It cannot be
            used at the same time with `zi=True`. Default: `False`.

        Returns
        -------
        new_sig : `'sequential'` or `'summed'` -> `Signal`.
                  `'parallel'` -> `MultiBandSignal`.
            New signal after filtering.

        """
        if type(signal) is MultiBandSignal:
            raise TypeError(
                "This method only supports Signal objects. Use "
                + "filter_multiband_signal() for multirate parallel filtering"
            )
        mode = mode.lower()
        assert mode in (
            "parallel",
            "sequential",
            "summed",
        ), f"{mode} is not a valid mode. Use parallel, sequential or summed"
        if mode in ("sequential", "summed"):
            assert self.same_sampling_rate, (
                "Multirate filtering is not valid for sequential or summed "
                + "filtering"
            )
        assert np.all(
            signal.sampling_rate_hz == self.sampling_rate_hz
        ), "Sampling rates do not match"
        if zero_phase:
            assert not activate_zi, (
                "Zero-phase filtering and zi cannot be used at "
                + "the same time"
            )
        if activate_zi:
            if not hasattr(self.filters[0], "zi"):
                self.initialize_zi(signal.number_of_channels)
            if len(self.filters[0].zi) != signal.number_of_channels:
                self.initialize_zi(signal.number_of_channels)

        new_sig = _filterbank_on_signal(
            signal,
            self.filters,
            mode=mode,
            activate_zi=activate_zi,
            zero_phase=zero_phase,
            same_sampling_rate=self.same_sampling_rate,
        )
        return new_sig

    def filter_multiband_signal(
        self,
        mbsignal: MultiBandSignal,
        activate_zi: bool = False,
        zero_phase: bool = False,
    ) -> MultiBandSignal:
        """Applies the filter bank to a `MultiBandSignal` and returns the
        output as a `MultiBandSignal` as well. Only `'parallel'` mode is
        supported.

        NOTE: all channels contained in the `MultiBandSignal` are filtered.

        Parameters
        ----------
        signal : `Signal`
            Signal to be filtered.
        activate_zi : bool, optional
            Takes in the filter initial values and updates them for
            streaming purposes. Default: `False`.
        zero_phase : bool, optional
            Activates zero_phase filtering for the filter bank. It cannot be
            used at the same time with `zi=True`. Default: `False`.

        Returns
        -------
        new_sig : `MultiBandSignal`.
            New signal after filtering.

        """
        assert np.all(
            mbsignal.sampling_rate_hz == self.sampling_rate_hz
        ), "Sampling rates do not match"
        if zero_phase:
            assert not activate_zi, (
                "Zero-phase filtering and zi cannot be used at "
                + "the same time"
            )
        if activate_zi:
            if not hasattr(self.filters[0], "zi"):
                self.initialize_zi(mbsignal.number_of_channels)
            if len(self.filters[0].zi) != mbsignal.number_of_channels:
                self.initialize_zi(mbsignal.number_of_channels)

        new_sig = mbsignal.copy()

        for n in range(mbsignal.number_of_bands):
            new_sig.bands[n] = self.filters[n].filter_signal(
                mbsignal.bands[n],
                channels=None,
                activate_zi=activate_zi,
                zero_phase=zero_phase,
            )
        return new_sig

    # ======== Get impulse ====================================================
    def get_ir(
        self,
        mode: str = "parallel",
        length_samples: int = 2048,
        test_zi: bool = False,
        zero_phase: bool = False,
    ) -> Signal | MultiBandSignal:
        """Returns impulse response from the filter bank.

        Parameters
        ----------
        mode : str, optional
            Filtering mode. Choose from `'parallel'`, `'sequential'` or
            `'summed'`. Default: `'parallel'`.
        length_samples : int, optional
            Length of the impulse response to be generated. If some filter
            is longer than the given length, then the length is adapted.
            Default: 2048.
        test_zi : bool, optional
            When `True`, filtering is done while updating filters' initial
            values. Default: `False`.
        zero_phase : bool, optional
            When `True`, zero phase filtering is activated. Default: `False`.

        Returns
        -------
        ir : `MultiBandSignal` or `Signal`
            Impulse response of the filter bank.

        """
        # No plotting for multirate system
        if not self.same_sampling_rate:
            warn(
                "Plotting for multirate FilterBank is not supported, "
                + "skipping IR generation"
            )
            return None
        # Obtain biggest filter order from FilterBank
        max_order = 0
        for b in self.filters:
            max_order = max(max_order, b.info["order"])
        if max_order > length_samples:
            warn(
                f"Filter order {max_order} is longer than {length_samples}."
                + "The length will be adapted to be 100 samples longer than"
                + " the longest filter"
            )
            length_samples = max_order + 100

        # Sampling rate
        fs_hz = self.sampling_rate_hz

        # Impulse
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1,
            sampling_rate_hz=fs_hz,
        )

        # Filtering
        ir = self.filter_signal(
            d, mode, activate_zi=test_zi, zero_phase=zero_phase
        )
        return ir

    # ======== Prints and plots ===============================================
    def show_info(self):
        """Show information about the filter bank."""
        print(self._get_metadata_str())

    def _get_metadata_str(self):
        txt = ""
        for k in self.info:
            txt += f""" | {str(k).replace('_', ' ').
                           capitalize()}: {self.info[k]}"""
        txt = "Filter Bank:" + txt
        txt += "\n"
        txt += "–" * len(txt)
        for ind, f1 in enumerate(self.filters):
            txt += "\n"
            txt += f"Filter {ind}:"
            for kf in f1.info:
                if kf == "ba":
                    continue
                txt += f""" | {str(kf).replace('_', ' ').
                               capitalize()}: {f1.info[kf]}"""
        return txt

    def plot_magnitude(
        self,
        mode: str = "parallel",
        range_hz=[20, 20e3],
        length_samples: int = 2048,
        test_zi: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plots the magnitude response of each filter.

        Parameters
        ----------
        mode : str, optional
            Type of plot. `'parallel'` plots every filter's frequency response,
            `'sequential'` plots the frequency response after having filtered
            one impulse with every filter in the FilterBank. `'summed'`
            sums up every frequency response. Default: `'parallel'`.
        range_hz : array-like, optional
            Range of Hz to plot. Default: [20, 20e3].
        length_samples : int, optional
            Length (in samples) of the IR to be generated for the plot.
            Default: 2048.
        test_zi : bool, optional
            Uses the zi's of each filter to test the FilterBank's output.
            Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        # No plotting for multirate system
        if not self.same_sampling_rate:
            warn(
                "Plotting for multirate FilterBank is not supported, "
                + "skipping plots"
            )
            return None
        # Length handling
        max_order = 0
        for b in self.filters:
            max_order = max(max_order, b.info["order"])
        if max_order > length_samples:
            warn(
                f"Filter order {max_order} is longer than {length_samples}."
                + " The length will be adapted to be 100 samples longer than"
                + " the longest filter"
            )
            length_samples = max_order + 100

        # Impulse
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1,
            sampling_rate_hz=self.sampling_rate_hz,
        )

        # Filtering and plot
        if mode == "parallel":
            bs = self.filter_signal(d, mode="parallel", activate_zi=test_zi)
            specs = []
            for b in bs.bands:
                b.set_spectrum_parameters(method="standard")
                f, sp = b.get_spectrum()
                f, sp = _get_normalized_spectrum(
                    f, sp, f_range_hz=range_hz, normalize=None
                )
                specs.append(np.squeeze(sp))
            specs = np.array(specs).T
            if np.min(specs) < np.max(specs) - 50:
                range_y = [np.max(specs) - 50, np.max(specs) + 2]
            else:
                range_y = None
            fig, ax = general_plot(
                f,
                specs,
                range_hz,
                ylabel="Magnitude / dB",
                returns=True,
                labels=[f"Filter {h}" for h in range(bs.number_of_bands)],
                range_y=range_y,
                tight_layout=False,
            )
        elif mode == "sequential":
            bs = self.filter_signal(d, mode="sequential", activate_zi=test_zi)
            bs.set_spectrum_parameters(method="standard")
            f, sp = bs.get_spectrum()
            f, sp = _get_normalized_spectrum(
                f, np.squeeze(sp), f_range_hz=range_hz, normalize=None
            )
            fig, ax = general_plot(
                f,
                sp,
                range_hz,
                ylabel="Magnitude / dB",
                returns=True,
                labels=[
                    f"Sequential - Channel {n}"
                    for n in range(bs.number_of_channels)
                ],
            )
        elif mode == "summed":
            bs = self.filter_signal(d, mode="summed", activate_zi=test_zi)
            bs.set_spectrum_parameters(method="standard")
            f, sp = bs.get_spectrum()
            f, sp = _get_normalized_spectrum(
                f, np.squeeze(sp), f_range_hz=range_hz, normalize=None
            )
            fig, ax = general_plot(
                f,
                sp,
                range_hz,
                ylabel="Magnitude / dB",
                returns=True,
                labels=["Summed"],
            )
        return fig, ax

    def plot_phase(
        self,
        mode: str = "parallel",
        range_hz=[20, 20e3],
        unwrap: bool = False,
        length_samples: int = 2048,
        test_zi: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plots the phase response of each filter.

        Parameters
        ----------
        mode : str, optional
            Type of plot. `'parallel'` plots every filter's frequency response,
            `'sequential'` plots the frequency response after having filtered
            one impulse with every filter in the FilterBank. `'summed'`
            sums up every filter output. Default: `'parallel'`.
        range_hz : array-like, optional
            Range of Hz to plot. Default: [20, 20e3].
        unwrap : bool, optional
            When `True`, unwrapped phase is plotted. Default: `False`.
        length_samples : int, optional
            Length (in samples) of the IR to be generated for the plot.
            Default: 2048.
        test_zi : bool, optional
            Uses the zi's of each filter to test the FilterBank's output.
            Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        # No plotting for multirate system
        if not self.same_sampling_rate:
            warn(
                "Plotting for multirate FilterBank is not supported, "
                + "skipping plots"
            )
            return None
        # Length handling
        max_order = 0
        for b in self.filters:
            max_order = max(max_order, b.info["order"])
        if max_order > length_samples:
            warn(
                f"Filter order {max_order} is longer than {length_samples}."
                + " The length will be adapted to be 100 samples longer than"
                + " the longest filter"
            )
            length_samples = max_order + 100

        # Generate impulse
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1,
            sampling_rate_hz=self.sampling_rate_hz,
        )

        # Plot
        if mode == "parallel":
            bs = self.filter_signal(d, mode="parallel", activate_zi=test_zi)
            phase = []
            f = bs.bands[0].get_spectrum()[0]
            for b in bs.bands:
                phase.append(np.angle(b.get_spectrum()[1]))
            phase = np.squeeze(np.array(phase).T)
            if unwrap:
                phase = np.unwrap(phase, axis=0)
            fig, ax = general_plot(
                f,
                phase,
                range_hz,
                ylabel="Phase / rad",
                returns=True,
                labels=[f"Filter {h}" for h in range(bs.number_of_bands)],
                tight_layout=False,
            )
        elif mode == "sequential":
            bs = self.filter_signal(d, mode="sequential", activate_zi=test_zi)
            f, sp = bs.get_spectrum()
            ph = np.angle(sp)
            if unwrap:
                ph = np.unwrap(ph, axis=0)
            fig, ax = general_plot(
                f,
                ph,
                range_hz,
                ylabel="Phase / rad",
                returns=True,
                labels=[
                    f"Sequential - Channel {n}"
                    for n in range(bs.number_of_channels)
                ],
            )
        elif mode == "summed":
            bs = self.filter_signal(d, mode="summed", activate_zi=test_zi)
            f, sp = bs.get_spectrum()
            ph = np.angle(sp)
            if unwrap:
                ph = np.unwrap(ph, axis=0)
            fig, ax = general_plot(
                f,
                ph,
                range_hz,
                ylabel="Phase / rad",
                returns=True,
                labels=["Summed"],
            )
        return fig, ax

    def plot_group_delay(
        self,
        mode: str = "parallel",
        range_hz=[20, 20e3],
        length_samples: int = 2048,
        test_zi: bool = False,
    ) -> tuple[Figure, Axes]:
        """Plots the phase response of each filter.

        Parameters
        ----------
        mode : str, optional
            Type of plot. `'parallel'` plots every filter's frequency response,
            `'sequential'` plots the frequency response after having filtered
            one impulse with every filter in the FilterBank. `'summed'`
            sums up every filter output. Default: `'parallel'`.
        range_hz : array-like, optional
            Range of Hz to plot. Default: [20, 20e3].
        length_samples : int, optional
            Length (in samples) of the IR to be generated for the plot.
            Default: 2048.
        test_zi : bool, optional
            Uses the zi's of each filter to test the FilterBank's output.
            Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        # No plotting for multirate system
        if not self.same_sampling_rate:
            warn(
                "Plotting for multirate FilterBank is not supported, "
                + "skipping plots"
            )
            return None
        # Length handling
        max_order = 0
        for b in self.filters:
            max_order = max(max_order, b.info["order"])
        if max_order > length_samples:
            warn(
                f"Filter order {max_order} is longer than {length_samples}."
                + " The length will be adapted to be 100 samples longer than"
                + " the longest filter"
            )
            length_samples = max_order + 100

        # Impulse
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1,
            sampling_rate_hz=self.sampling_rate_hz,
        )

        # Plot
        if mode == "parallel":
            bs = self.filter_signal(d, mode="parallel", activate_zi=test_zi)
            gd = []
            f = bs.bands[0].get_spectrum()[0]
            for b in bs.bands:
                gd.append(
                    _group_delay_direct(
                        np.squeeze(b.get_spectrum()[1]), delta_f=f[1] - f[0]
                    )
                )
            gd = np.squeeze(np.array(gd).T) * 1e3
            fig, ax = general_plot(
                f,
                gd,
                range_hz,
                ylabel="Group delay / ms",
                returns=True,
                labels=[f"Filter {h}" for h in range(bs.number_of_bands)],
                tight_layout=False,
            )
        elif mode == "sequential":
            bs = self.filter_signal(d, mode="sequential", activate_zi=test_zi)
            f, sp = bs.get_spectrum()
            gd = _group_delay_direct(sp.squeeze(), f[1] - f[0]) * 1e3
            fig, ax = general_plot(
                f,
                gd[..., None],
                range_hz,
                ylabel="Group delay / ms",
                returns=True,
                labels=[
                    f"Sequential - Channel {n}"
                    for n in range(bs.number_of_channels)
                ],
            )
        elif mode == "summed":
            bs = self.filter_signal(d, mode="summed", activate_zi=test_zi)
            f, sp = bs.get_spectrum()
            gd = _group_delay_direct(sp.squeeze(), f[1] - f[0]) * 1e3
            fig, ax = general_plot(
                f,
                gd[..., None],
                range_hz,
                ylabel="Group delay / ms",
                returns=True,
                labels=["Summed"],
            )
        return fig, ax

    # ======== Saving and export ==============================================
    def save_filterbank(self, path: str = "filterbank"):
        """Saves the FilterBank object as a pickle.

        Parameters
        ----------
        path : str, optional
            Path for the filterbank to be saved. Use only folder1/folder2/name
            (without format). Default: `'filterbank'`
            (local folder, object named filterbank).

        """
        path = _check_format_in_path(path, "pkl")
        with open(path, "wb") as data_file:
            dump(self, data_file, HIGHEST_PROTOCOL)

    def copy(self):
        """Returns a copy of the object.

        Returns
        -------
        new_sig : `FilterBank`
            Copy of filter bank.

        """
        return deepcopy(self)
