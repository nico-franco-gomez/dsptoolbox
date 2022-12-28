from os import sep
from pickle import dump, HIGHEST_PROTOCOL
from copy import deepcopy

from .signal_class import Signal
from .filter_class import Filter
from ._filter import _filterbank_on_signal
from dsptools.generators import dirac
from dsptools.plots import general_plot
from dsptools._general_helpers import _get_normalized_spectrum
from dsptools._standard import _group_delay_direct


class FilterBank():
    """Standard template for filter banks containing filters, filters' initial
    values, metadata and some useful plotting methods.

    """
    # ======== Constructor and initializers ===================================
    def __init__(self, filters=[], same_sampling_rate: bool = True,
                 info: dict = {}):
        """FilterBank object saves multiple filters and some metadata.
        It also allows for easy filtering with multiple filters.
        Since the digital filters that are supported are linear systems,
        the order in which they are saved and applied to a signal is not
        relevant.

        Parameters
        ----------
        filters : list or tuple, optional
            List or tuple containing filters.
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
            add_filter, remove_filter, save_filterbank.
        Prints and plots
            plot_magnitude, plot_phase, plot_group_delay, show_info.

        """
        #
        assert type(filters) in (list, tuple), \
            'Filters should be passed as list or as tuple'
        #
        self.same_sampling_rate = same_sampling_rate
        if filters:
            if self.same_sampling_rate:
                self.sampling_rate_hz = filters[0].sampling_rate_hz
            for ind, f in enumerate(filters):
                assert type(f) == Filter, \
                    f'Object at index {ind} is not a supported Filter'
                if self.same_sampling_rate:
                    assert f.sampling_rate_hz == self.sampling_rate_hz, \
                        'Sampling rates do not match'
        self.filters = filters
        self._generate_metadata()
        self.info = self.info | info

    def _generate_metadata(self):
        """Generates the info dictionary with metadata about the FilterBank.

        """
        self.info = {}
        self.info['number_of_filters'] = len(self.filters)
        self.info['same_sampling_rate'] = self.same_sampling_rate
        if self.same_sampling_rate:
            if hasattr(self, 'sampling_rate_hz'):
                self.info['sampling_rate_hz'] = self.sampling_rate_hz
        self.info['types_of_filters'] = \
            tuple(set([f.info['filter_type']
                       for f in self.filters]))

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
            self.filters.append(filt)
        else:
            if self.same_sampling_rate:
                assert self.sampling_rate_hz == filt.sampling_rate_hz, \
                    'Sampling rates do not match'
            if index == -1:
                self.filters.append(filt)
            else:
                self.filters.insert(index, filt)
        self._generate_metadata()

    def remove_filter(self, index: int = -1, return_filter: bool = False):
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
        assert self.filters, 'There are no filters to remove'
        if index == -1:
            index = len(self.filters) - 1
        assert index in range(len(self.filters)), \
            f'There is no filter at index {index}.'
        f = self.filters.pop(index)
        self._generate_metadata()
        if return_filter:
            return f

    # ======== Filtering ======================================================
    def filter_signal(self, signal: Signal, mode: str = 'parallel',
                      activate_zi: bool = False, zero_phase: bool = False):
        """Applies the filter bank to a signal and returns a multiband signal
        or a `Signal` object.
        `'parallel'`: returns a `MultiBandSignal` object where each band is
        the output of each filter.
        `'sequential'`: applies each filter to the given Signal in a sequential
        manner and returns output with same dimension.
        `'summed'`: applies every filter as parallel and then summs the outputs
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
        mode = mode.lower()
        assert mode in ('parallel', 'sequential', 'summed'), \
            f'{mode} is not a valid mode. Use parallel, sequential or summed'
        if mode in ('sequential', 'summed'):
            assert self.same_sampling_rate, \
                'Multirate filtering is not valid for sequential or summed ' +\
                'filtering'
        if self.same_sampling_rate:
            assert signal.sampling_rate_hz == self.sampling_rate_hz, \
                'Sampling rates do not match'
        if zero_phase:
            assert not activate_zi, \
                'Zero-phase filtering and zi cannot be used at ' +\
                'the same time'
        if activate_zi:
            if len(self.filters[0].zi) != signal.number_of_channels:
                # warn('zi values of the filter have not been correctly ' +
                #      'intialized. They have now been started')
                self.initialize_zi(signal.number_of_channels)

        new_sig = _filterbank_on_signal(
            signal, self.filters,
            mode=mode,
            activate_zi=activate_zi,
            zero_phase=zero_phase,
            same_sampling_rate=self.same_sampling_rate)

        new_sig.signal_type = signal.signal_type
        new_sig.signal_id = signal.signal_id
        return new_sig

    # ======== Prints and plots ===============================================
    def show_info(self, show_filter_info: bool = True):
        """Show information about the filter bank.

        Parameters
        ----------
        show_filters_info : bool, optional
            When `True`, a longer message is printed with all available
            information regarding each filter in the filter bank.
            Default: `True`.

        """
        print()
        txt = ''
        for k in self.info:
            txt += \
                f""" | {str(k).replace('_', ' ').
                        capitalize()}: {self.info[k]}"""
        txt = 'Filter Bank:' + txt
        print(txt)
        if show_filter_info:
            print('-'*len(txt), end='')
            for ind, f1 in enumerate(self.filters):
                print()
                txt = f'Filter {ind}:'
                for kf in f1.info:
                    if kf == 'ba':
                        continue
                    txt += \
                        f""" | {str(kf).replace('_', ' ').
                                capitalize()}: {f1.info[kf]}"""
                print(txt)
        print()

    def plot_magnitude(self, mode: str = 'parallel', range_hz=[20, 20e3],
                       test_zi: bool = False, returns: bool = False):
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
        test_zi : bool, optional
            Uses the zi's of each filter to test the FilterBank's output.
            Default: `False`.
        returns : bool, optional
            When `True`, the figure and axis are returned. Default: `False`.

        Returns
        -------
        fig, ax
            Returned only when `returns=True`.

        """
        import numpy as np
        if hasattr(self.filters[-1], 'ba'):
            length_samples = \
                max(len(self.filters[-1].ba[0]),
                    len(self.filters[-1].ba[1])) + 2
        else:
            length_samples = len(self.filters[-1].sos)*2 + 2
        if length_samples < 1024:
            length_samples = 1024
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1, sampling_rate_hz=48000)
        if mode == 'parallel':
            bs = self.filter_signal(d, mode='parallel', activate_zi=test_zi)
            specs = []
            f = bs.bands[0].get_spectrum()[0]
            for b in bs.bands:
                b.set_spectrum_parameters(method='standard')
                f, sp = \
                    _get_normalized_spectrum(
                        f, np.squeeze(b.get_spectrum()[1]),
                        f_range_hz=range_hz,
                        normalize=None)
                specs.append(np.squeeze(sp))
            specs = np.array(specs).T
            if np.min(specs) < np.max(specs)-50:
                range_y = [np.max(specs)-50, np.max(specs)+2]
            else:
                range_y = None
            fig, ax = general_plot(f, specs, range_hz, ylabel='Magnitude / dB',
                                   returns=True,
                                   labels=[f'Filter {h}'
                                           for h in range(bs.number_of_bands)],
                                   range_y=range_y)
        elif mode == 'sequential':
            bs = self.filter_signal(d, mode='sequential', activate_zi=test_zi)
            bs.set_spectrum_parameters(method='standard')
            f, sp = bs.get_spectrum()
            f, sp = _get_normalized_spectrum(
                f, np.squeeze(sp),
                f_range_hz=range_hz,
                normalize=None
            )
            fig, ax = general_plot(
                f, sp, range_hz, ylabel='Magnitude / dB',
                returns=True,
                labels=[f'Sequential - Channel {n}'
                        for n in range(bs.number_of_channels)])
        elif mode == 'summed':
            bs = self.filter_signal(d, mode='summed', activate_zi=test_zi)
            bs.set_spectrum_parameters(method='standard')
            f, sp = bs.get_spectrum()
            f, sp = _get_normalized_spectrum(
                f, np.squeeze(sp),
                f_range_hz=range_hz,
                normalize=None
            )
            fig, ax = general_plot(
                f, sp, range_hz, ylabel='Magnitude / dB',
                returns=True,
                labels='Summed')
        if returns:
            return fig, ax

    def plot_phase(self, mode: str = 'parallel', range_hz=[20, 20e3],
                   test_zi: bool = False, unwrap: bool = False,
                   returns: bool = False):
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
        test_zi : bool, optional
            Uses the zi's of each filter to test the FilterBank's output.
            Default: `False`.
        unwrap : bool, optional
            When `True`, unwrapped phase is plotted. Default: `False`.
        returns : bool, optional
            When `True`, the figure and axis are returned. Default: `False`.

        Returns
        -------
        fig, ax
            Returned only when `returns=True`.

        """
        import numpy as np
        if hasattr(self.filters[-1], 'ba'):
            length_samples = \
                max(len(self.filters[-1].ba[0]),
                    len(self.filters[-1].ba[1])) + 2
        else:
            length_samples = len(self.filters[-1].sos)*2 + 2
        if length_samples < 1024:
            length_samples = 1024
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1, sampling_rate_hz=48000)
        if mode == 'parallel':
            bs = self.filter_signal(d, mode='parallel', activate_zi=test_zi)
            phase = []
            f = bs.bands[0].get_spectrum()[0]
            for b in bs.bands:
                phase.append(np.angle(b.get_spectrum()[1]))
            phase = np.squeeze(np.array(phase).T)
            if unwrap:
                phase = np.unwrap(phase, axis=0)
            fig, ax = general_plot(f, phase, range_hz, ylabel='Phase / rad',
                                   returns=True,
                                   labels=[f'Filter {h}'
                                           for h in range(bs.number_of_bands)])
        elif mode == 'sequential':
            bs = self.filter_signal(d, mode='sequential', activate_zi=test_zi)
            f, sp = bs.get_spectrum()
            ph = np.angle(sp)
            if unwrap:
                ph = np.unwrap(ph, axis=0)
            fig, ax = general_plot(
                f, ph, range_hz, ylabel='Phase / rad',
                returns=True,
                labels=[f'Sequential - Channel {n}'
                        for n in range(bs.number_of_channels)])
        elif mode == 'summed':
            bs = self.filter_signal(d, mode='summed', activate_zi=test_zi)
            f, sp = bs.get_spectrum()
            ph = np.angle(sp)
            if unwrap:
                ph = np.unwrap(ph, axis=0)
            fig, ax = general_plot(
                f, ph, range_hz, ylabel='Phase / rad',
                returns=True,
                labels='Summed')
        if returns:
            return fig, ax

    def plot_group_delay(self, mode: str = 'parallel', range_hz=[20, 20e3],
                         test_zi: bool = False, returns: bool = False):
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
        test_zi : bool, optional
            Uses the zi's of each filter to test the FilterBank's output.
            Default: `False`.
        returns : bool, optional
            When `True`, the figure and axis are returned. Default: `False`.

        Returns
        -------
        fig, ax
            Returned only when `returns=True`.

        """
        import numpy as np
        if hasattr(self.filters[-1], 'ba'):
            length_samples = \
                max(len(self.filters[-1].ba[0]),
                    len(self.filters[-1].ba[1])) + 2
        else:
            length_samples = len(self.filters[-1].sos)*2 + 2
        if length_samples < 1024:
            length_samples = 1024
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1, sampling_rate_hz=48000)
        if mode == 'parallel':
            bs = self.filter_signal(d, mode='parallel', activate_zi=test_zi)
            gd = []
            f = bs.bands[0].get_spectrum()[0]
            for b in bs.bands:
                gd.append(_group_delay_direct(
                    np.squeeze(b.get_spectrum()[1]), delta_f=f[1]-f[0]))
            gd = np.squeeze(np.array(gd).T)*1e3
            fig, ax = general_plot(f, gd, range_hz, ylabel='Group delay / ms',
                                   returns=True,
                                   labels=[f'Filter {h}'
                                           for h in range(bs.number_of_bands)])
        elif mode == 'sequential':
            bs = self.filter_signal(d, mode='sequential', activate_zi=test_zi)
            f, sp = bs.get_spectrum()
            gd = _group_delay_direct(sp.squeeze(), f[1]-f[0])*1e3
            fig, ax = general_plot(
                f, gd[..., None], range_hz, ylabel='Group delay / ms',
                returns=True,
                labels=[f'Sequential - Channel {n}'
                        for n in range(bs.number_of_channels)])
        elif mode == 'summed':
            bs = self.filter_signal(d, mode='summed', activate_zi=test_zi)
            f, sp = bs.get_spectrum()
            gd = _group_delay_direct(sp.squeeze(), f[1]-f[0])*1e3
            fig, ax = general_plot(
                f, gd[..., None], range_hz, ylabel='Group delay / ms',
                returns=True,
                labels='Summed')
        if returns:
            return fig, ax

    # ======== Saving and export ==============================================
    def save_filterbank(self, path: str = 'filterbank'):
        """Saves the FilterBank object as a pickle.

        Parameters
        ----------
        path : str, optional
            Path for the filterbank to be saved. Use only folder1/folder2/name
            (without format). Default: `'filterbank'`
            (local folder, object named filterbank).

        """
        if '.' in path.split(sep)[-1]:
            raise ValueError('Please introduce the saving path without format')
        path += '.pkl'
        with open(path, 'wb') as data_file:
            dump(self, data_file, HIGHEST_PROTOCOL)

    def copy(self):
        """Returns a copy of the object.
        
        Returns
        -------
        new_sig : `FilterBank`
            Copy of filter bank.

        """
        return deepcopy(self)
