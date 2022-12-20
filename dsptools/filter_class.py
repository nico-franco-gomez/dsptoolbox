"""
Contains Filter classes
"""
import os
import pickle

from scipy import signal as sig
from .signal_class import Signal
from .backend._filter import (_biquad_coefficients, _impulse,
                              _group_delay_filter, _get_biquad_type,
                              _filter_on_signal,
                              _filterbank_on_signal)
from .backend._plots import _zp_plot
from .backend._general_helpers import _get_normalized_spectrum
from .generators import dirac
from .plots import general_plot

__all__ = ['Filter']


class Filter():
    """Class for creating and storing linear digital filters with all their
    metadata.
    """

    # ======== Constructor and initializers ===================================
    def __init__(self, filter_type: str = 'biquad',
                 filter_configuration: dict = None,
                 sampling_rate_hz: int = 48000):
        """The Filter class contains all parameters and metadata needed for
        using a digital filter.

        Constructor
        -----------
        A dictionary containing the filter configuration parameters should
        be passed. It is a wrapper around `scipy.signal.iirfilter`,
        `scipy.signal.firwin` and `_biquad_coefficients`.

        Parameters
        ----------
        filter_type : str, optional
            String defining the filter type. Options are `iir`, `fir`,
            `biquad` or `other`. Default: creates a dummy biquad bell filter
            with no gain.
        filter_configuration : dict, optional
            Dictionary containing configuration for the filter.
            Default: some dummy parameters.
        sampling_rate_hz : int, optional
            Sampling rate in Hz for the digital filter. Default: 48000.

        Keys
        ----
        For `iir`:
            order, freqs, type_of_pass, filter_design_method,
            filter_id (optional).
            freqs (float, array-like): array with len 2 when 'bandpass'
                or 'bandstop'.
            type_of_pass (str): 'bandpass', 'lowpass', 'highpass', 'bandstop'.
            filter_design_method (str): 'butter', 'bessel', 'ellip', 'cheby1',
                'cheby2'.

        For `fir`:
            order, freqs, type_of_pass, filter_design_method (optional),
            width (optional, necessary for 'kaiser'), filter_id (optional).
            filter_design_method (str): Window to be used. Default: 'hamming'.
                Supported types are: 'boxcar', 'triang', 'blackman', 'hamming',
                'hann', 'bartlett', 'flattop', 'parzen', 'bohman',
                'blackmanharris', 'nuttall', 'barthann', 'cosine',
                'exponential', 'tukey', 'taylor'.
            width (float): estimated width of transition region in Hz for
                kaiser window. Default: `None`.
            type_of_pass (str): 'bandpass', 'lowpass', 'highpass', 'bandstop'.

        For `biquad`:
            eq_type, freqs, gain, q, filter_id (optional).
            gain (float): in dB.
            eq_type (int or str): 0 = Bell/Peaking, 1 = Lowpass, 2 = Highpass,
                3 = Bandpass skirt, 4 = Bandpass peak, 5 = Notch, 6 = Allpass,
                7 = Lowshelf, 8 = Highshelf.

        For `other` or `general`:
            ba or sos or zpk, filter_id (optional).

        Methods
        -------
        General: set_filter_parameters, get_filter_parameters, get_ir.
        Plots or prints: show_filter_parameters, plot_magnitude,
            plot_group_delay, plot_phase, plot_zp.
        Filtering: filter_signal.
        """
        self.sampling_rate_hz = sampling_rate_hz
        if filter_configuration is None:
            filter_configuration = \
                {'eq_type': 0, 'freqs': 1000, 'gain': 0, 'q': 1,
                 'filter_id': 'dummy'}
        self.set_filter_parameters(filter_type, filter_configuration)
        self.initialize_zi()

    def initialize_zi(self, number_of_channels: int = 1):
        """Initializes zi for steady-state filtering. The number of parallel
        zi's can be defined externally.
        """
        self.zi = []
        for n in range(number_of_channels):
            self.zi.append(sig.sosfilt_zi(self.sos))

    @property
    def sampling_rate_hz(self):
        return self._sampling_rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, new_sampling_rate_hz):
        assert type(new_sampling_rate_hz) == int, \
            'Sampling rate can only be an integer'
        self._sampling_rate_hz = new_sampling_rate_hz

    # ======== Filtering ======================================================
    def filter_signal(self, signal: Signal, channel=None,
                      activate_zi: bool = False, zero_phase: bool = None):
        """Takes in a Signal object and filters selected channels. Exports a
        new Signal object.

        Parameters
        ----------
        signal : Signal
            Signal to be filtered.
        filt : Filter
            Filter to be used on the signal.
        channel : int or array-like, optional
            Channel or array of channels to be filtered. When `None`, all
            channels are filtered. Default: `None`.
        activate_zi : int, optional
            Gives the zi to update the filter values. Default: `False`.
        zero_phase : bool, optional
            Uses zero-phase filtering on signal. Be aware that the filter
            is doubled in this case. Default: `False`.

        Returns
        -------
        new_signal : Signal
            New Signal object.
        """
        assert not (activate_zi and zero_phase), \
            'Filter initial and final values cannot be updated when ' +\
            'filtering with zero-phase'
        if activate_zi:
            if len(self.zi) != signal.number_of_channels:
                # warn('zi values of the filter have not been correctly ' +
                #      'intialized. They have now been started')
                self.initialize_zi(signal.number_of_channels)
            zi_old = self.zi
        else:
            zi_old = None
        new_signal, zi_new = \
            _filter_on_signal(
                signal=signal,
                sos=self.sos,
                channel=channel,
                zi=zi_old,
                zero_phase=zero_phase)
        if activate_zi:
            self.zi = zi_new
        return new_signal

    # ======== Setters ========================================================
    def set_filter_parameters(self, filter_type: str,
                              filter_configuration: dict):
        if filter_type == 'iir':
            self.sos = \
                sig.iirfilter(N=filter_configuration['order'],
                              Wn=filter_configuration['freqs'],
                              btype=filter_configuration['type_of_pass'],
                              analog=False,
                              fs=self.sampling_rate_hz,
                              ftype=filter_configuration
                              ['filter_design_method'],
                              output='sos')
        elif filter_type == 'fir':
            # Preparing parameters
            if 'filter_design_method' not in filter_configuration.keys():
                filter_configuration['filter_design_method'] = 'hamming'
            if 'width' not in filter_configuration.keys():
                filter_configuration['width'] = None
            # Filter creation
            ba = \
                [sig.firwin(numtaps=filter_configuration['order'],
                            cutoff=filter_configuration['freqs'],
                            window=filter_configuration
                            ['filter_design_method'],
                            width=filter_configuration['width'],
                            pass_zero=filter_configuration['type_of_pass'],
                            fs=self.sampling_rate_hz), [1]]
            self.sos = sig.tf2sos(ba[0], ba[1])
        elif filter_type == 'biquad':
            # Preparing parameters
            if type(filter_configuration['eq_type']) == str:
                filter_configuration['eq_type'] = \
                    _get_biquad_type(None, filter_configuration['eq_type'])
            # Filter creation
            ba = \
                _biquad_coefficients(
                    eq_type=filter_configuration['eq_type'],
                    fs_hz=self.sampling_rate_hz,
                    frequency_hz=filter_configuration['freqs'],
                    gain_db=filter_configuration['gain'],
                    q=filter_configuration['q'])
            # Setting back
            filter_configuration['eq_type'] = \
                _get_biquad_type(filter_configuration['eq_type']).capitalize()
            self.sos = sig.tf2sos(ba[0], ba[1])
        else:
            assert ('ba' in filter_configuration) ^ \
                ('sos' in filter_configuration) ^ \
                ('zpk' in filter_configuration), \
                'Only (and at least) one type of filter coefficients ' +\
                'should be passed to create a filter'
            if ('ba' in filter_configuration):
                ba = filter_configuration['ba']
                self.sos = sig.tf2sos(ba[0], ba[1])
            if ('zpk' in filter_configuration):
                z, p, k = filter_configuration['zpk']
                self.sos = sig.zpk2sos(z, p, k)
            if ('sos' in filter_configuration):
                self.sos = filter_configuration['sos']
        self.info = filter_configuration
        self.info['sampling_rate_hz'] = self.sampling_rate_hz
        self.info['filter_type'] = filter_type
        if 'filter_id' not in self.info.keys():
            self.info['filter_id'] = None

    # ======== Getters ========================================================
    def get_filter_parameters(self):
        """Returns filter parameters.

        Returns
        -------
        info : dict
            Dictionary containing all filter parameters
        """
        return self.info

    def _get_metadata_string(self):
        """Helper for creating a string containing all filter info.
        """
        txt = f"""Filter – ID: {self.info['filter_id']}\n"""
        temp = ''
        for n in range(len(txt)):
            temp += '-'
        txt += (temp+'\n')
        for k in self.info.keys():
            txt += \
                f"""{str(k).replace('_', ' ').
                     capitalize()}: {self.info[k]}\n"""
        return txt

    def get_ir(self, length_samples: int = 512):
        """Gets an impulse response of the filter with given length.

        Parameters
        ----------
        length_samples : int, optional
            Length for the impulse response in samples. Default: 512.

        Returns
        -------
        ir_filt : np.ndarray
            Impulse response of the filter.
        """
        ir_filt = _impulse(length_samples)
        ir_filt = sig.sosfilt(sos=self.sos, x=ir_filt)
        ir_filt = Signal(
            None, ir_filt,
            sampling_rate_hz=self.sampling_rate_hz,
            signal_type='ir')
        return ir_filt

    def get_coefficients(self, mode='sos'):
        """Returns the filter coefficients.

        Parameters
        ----------
        mode : str, optional
            Type of filter coefficients to be returned. Choose from `'sos'`,
            `'ba'` or `'zpk'`. Default: `'sos'`.

        Returns
        -------
        coefficients : array-like
            Array with filter parameters.
        """
        if mode == 'sos':
            coefficients = self.sos
        elif mode == 'ba':
            coefficients = sig.sos2tf(self.sos)
        elif mode == 'zpk':
            coefficients = sig.sos2zpk(self.sos)
        else:
            raise ValueError(f'{mode} is not valid. Use sos, ba or zpk')
        return coefficients

    # ======== Plots and prints ===============================================
    def show_filter_parameters(self):
        """Prints all the filter parameters
        """
        print(self._get_metadata_string())

    def plot_magnitude(self, length_samples: int = 512, range_hz=[20, 20e3],
                       normalize: str = None, show_info_box: bool = True,
                       returns: bool = False):
        """Plots magnitude spectrum.
        Change parameters of spectrum with set_spectrum_parameters.

        Parameters
        ----------
        length_samples : int, optional
            Length of ir for magnitude plot. Default: 512.
        range_hz : array-like with length 2, optional
            Range for which to plot the magnitude response.
            Default: [20, 20000].
        normalize : str, optional
            Mode for normalization, supported are `'1k'` for normalization
            with value at frequency 1 kHz or `'max'` for normalization with
            maximal value. Use `None` for no normalization. Default: `'1k'`.
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `True`.
        returns : bool, optional
            When `True` figure and axis are returned. Default: `False`.

        Returns
        -------
        figure and axis when `returns = True`.
        """
        ir = self.get_ir(length_samples=length_samples)
        fig, ax = ir.plot_magnitude(range_hz, normalize, 0,
                                    show_info_box=False, returns=True)
        if show_info_box:
            txt = self._get_metadata_string()
            ax.text(0.1, 0.5, txt, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='grey', alpha=0.75))
        if returns:
            return fig, ax

    def plot_group_delay(self, length_samples: int = 512,
                         range_hz=[20, 20e3], show_info_box: bool = False,
                         returns: bool = True):
        """
        Plots group delay of the filter. Different methods are used for
        FIR or IIR filters.

        Parameters
        ----------
        length_samples : int, optional
            Length of ir for magnitude plot. Default: 512.
        range_hz : array-like with length 2, optional
            Range for which to plot the magnitude response.
            Default: [20, 20000].
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `False`.
        returns : bool, optional
            When `True` figure and axis are returned. Default: `False`.

        Returns
        -------
        figure and axis when `returns = True`.
        """
        ba = sig.sos2tf(self.sos)
        # import numpy as np
        f, gd = \
            _group_delay_filter(ba, length_samples, self.sampling_rate_hz)
        gd *= 1e3
        ymax = None
        ymin = None
        if any(abs(gd) > 20):
            ymin = -2
            ymax = 20
        fig, ax = general_plot(
            f=f,
            matrix=gd[..., None],
            range_x=range_hz,
            range_y=[ymin, ymax],
            ylabel='Group delay / ms',
            returns=True)
        if show_info_box:
            txt = self._get_metadata_string()
            ax.text(0.1, 0.5, txt, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='grey', alpha=0.75))
        if returns:
            return fig, ax

    def plot_phase(self, length_samples: int = 512, range_hz=[20, 20e3],
                   unwrap: bool = False, show_info_box: bool = False,
                   returns: bool = False):
        """Plots magnitude spectrum.
        Change parameters of spectrum with set_spectrum_parameters.

        Parameters
        ----------
        length_samples : int, optional
            Length of ir for magnitude plot. Default: 512.
        range_hz : array-like with length 2, optional
            Range for which to plot the magnitude response.
            Default: [20, 20000].
        unwrap : bool, optional
            Unwraps the phase to show. Default: `False`.
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `False`.
        returns : bool, optional
            When `True` figure and axis are returned. Default: `False`.

        Returns
        -------
        figure and axis when `returns = True`.
        """
        ir = self.get_ir(length_samples=length_samples)
        fig, ax = ir.plot_phase(range_hz, unwrap, returns=True)
        if show_info_box:
            txt = self._get_metadata_string()
            ax.text(0.1, 0.5, txt, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='grey', alpha=0.75))
        if returns:
            return fig, ax

    def plot_zp(self, show_info_box: bool = False, returns: bool = False):
        """Plots zeros and poles with the unit circle.

        Parameters
        ----------
        returns : bool, optional
            When `True` figure and axis are returned. Default: `False`.
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `False`.

        Returns
        -------
        figure and axis when `returns = True`.
        """
        z, p, k = sig.sos2zpk(self.sos)
        fig, ax = _zp_plot(z, p, returns=True)
        ax.text(0.75, 0.91, rf'$k={k:.1e}$', transform=ax.transAxes,
                verticalalignment='top')
        if show_info_box:
            txt = self._get_metadata_string()
            ax.text(0.1, 0.5, txt, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='grey', alpha=0.75))
        if returns:
            return fig, ax

    # ======== Saving and export ==============================================
    def save_filter(self, path: str = 'filter'):
        """Saves the Filter object as a pickle.

        Parameters
        ----------
        path : str, optional
            Path for the filter to be saved. Use only folder/folder/name
            (without format). Default: `'filter'`
            (local folder, object named filter).
        """
        if '.' in path.split(os.sep)[-1]:
            raise ValueError('Please introduce the saving path without format')
        path += '.pkl'
        with open(path, 'wb') as data_file:
            pickle.dump(self, data_file, pickle.HIGHEST_PROTOCOL)


# == Filter Bank ==============================================================
class FilterBank():
    # ======== Constructor and initializers ===================================
    def __init__(self, filters=None, same_sampling_rate: bool = True,
                 info: dict = None):
        """FilterBank object saves multiple filters and some metadata.
        It also allows for easy filtering with multiple filters.
        Since the digital filters that are supported are linear systems,
        the order in which they are saved and applied to a signal is not
        relevant.

        Parameters
        ----------
        filters : dict
            Dictionary containing filters. Keys are automatically set to
            numbers [0, 1, 2, ... etc.].
        same_sampling_rate : bool, optional
            When `True`, every Filter should have the same sampling rate.
            Set to `False` for a multirate system. Default: `True`.
        info : dict
            Dictionary containing general information about the filter bank.
            Some parameters of the filter bank are automatically read from
            the filters dictionary.

        Methods
        -------
        General: add_filter, remove_filter
        Prints: show_info
        """
        #
        if info is None:
            info = {}
        if filters is None:
            filters = []
        assert type(filters) in (list, tuple), \
            'Filters should be passed a list or as a tuple'
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

    def initialize_zi(self, number_of_channels):
        """Initiates the zi of the filters for the given number of channels.
        """
        for f in self.filters:
            f.initialize_zi(number_of_channels)

    # ======== Add and remove =================================================
    def add_filter(self, filt: Filter, index: int = -1):
        """Adds a new filter at the end of the filters dictionary.

        Parameters
        ----------
        filt : Filter
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
        """Applies the filter bank to a signal and returns a multiband signal.
        `'parallel'`: returns a MultiBandSignal object where each band is
            the output of each filter.
        `'sequential'`: applies each filter to the given Signal in a sequential
            manner and returns output with same dimension.
        `'summed'`: applies every filter as parallel and then summs the outputs
            returning same dimensional output as input.

        Parameters
        ----------
        signal : Signal
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
        new_sig : `'sequential'` or `'summed'` -> Signal.
                  `'parallel'` -> MultiBandSignal
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
        d = dirac(
            length_samples=1024, number_of_channels=1, sampling_rate_hz=48000)
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
            fig, ax = general_plot(f, specs, range_hz, ylabel='Magnitude / dB',
                                   returns=True,
                                   labels=[f'Filter {h}'
                                           for h in range(bs.number_of_bands)])
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
        if returns:
            return fig, ax

    # ======== Saving and export ==============================================
    def save_filterbank(self, path: str = 'filterbank'):
        """Saves the FilterBank object as a pickle.

        Parameters
        ----------
        path : str, optional
            Path for the filterbank to be saved. Use only folder/folder/name
            (without format). Default: `'filterbank'`
            (local folder, object named filterbank).
        """
        if '.' in path.split(os.sep)[-1]:
            raise ValueError('Please introduce the saving path without format')
        path += '.pkl'
        with open(path, 'wb') as data_file:
            pickle.dump(self, data_file, pickle.HIGHEST_PROTOCOL)
