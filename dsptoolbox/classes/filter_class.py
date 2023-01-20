"""
Contains Filter classes
"""
from os import sep
from pickle import dump, HIGHEST_PROTOCOL
from warnings import warn
from copy import deepcopy
import numpy as np
from fractions import Fraction

import scipy.signal as sig
from .signal_class import Signal
from ._filter import (_biquad_coefficients, _impulse,
                      _group_delay_filter, _get_biquad_type,
                      _filter_on_signal, _filter_on_signal_ba,
                      _filter_and_downsample,
                      _filter_and_upsample)
from ._plots import _zp_plot
from dsptoolbox.plots import general_plot


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
        `scipy.signal.firwin` and `_biquad_coefficients`. See down below for
        the parameters needed for creating the filters. Alternatively, you can
        pass directly the filter coefficients while setting
        `filter_type = 'other'`.

        Parameters
        ----------
        filter_type : str, optional
            String defining the filter type. Options are `'iir'`, `'fir'`,
            `'biquad'` or `'other'`. Default: creates a dummy biquad bell
            filter with no gain.
        filter_configuration : dict, optional
            Dictionary containing configuration for the filter.
            Default: some dummy parameters.
        sampling_rate_hz : int, optional
            Sampling rate in Hz for the digital filter. Default: 48000.

        Notes
        -----
        For `iir`:
            Keys: order, freqs, type_of_pass, filter_design_method,
            filter_id (optional).

            - order (int): Filter order
            - freqs (float, array-like): array with len 2 when 'bandpass'
              or 'bandstop'.
            - type_of_pass (str): 'bandpass', 'lowpass', 'highpass',
              'bandstop'.
            - filter_design_method (str): 'butter', 'bessel', 'ellip',
              'cheby1', 'cheby2'.

        For `fir`:
            Keys: order, freqs, type_of_pass, filter_design_method (optional),
            width (optional, necessary for 'kaiser'), filter_id (optional).

            - order (int): Filter order, i.e., number of taps - 1.
            - freqs (float, array-like): array with len 2 when 'bandpass'
              or 'bandstop'.
            - type_of_pass (str): 'bandpass', 'lowpass', 'highpass',
              'bandstop'.
            - filter_design_method (str): Window to be used. Default:
              'hamming'. Supported types are: 'boxcar', 'triang',
              'blackman', 'hamming', 'hann', 'bartlett', 'flattop',
              'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann',
              'cosine', 'exponential', 'tukey', 'taylor'.
            - width (float): estimated width of transition region in Hz for
              kaiser window. Default: `None`.

        For `biquad`:
            Keys: eq_type, freqs, gain, q, filter_id (optional).

            - eq_type (int or str): 0 = Peaking, 1 = Lowpass, 2 = Highpass,
              3 = Bandpass skirt, 4 = Bandpass peak, 5 = Notch, 6 = Allpass,
              7 = Lowshelf, 8 = Highshelf.
            - freqs: float or array-like with length 2 (depending on eq_type).
            - gain (float): in dB.
            - q (float): Q-factor.

        For `other` or `general`:
            ba or sos or zpk, filter_id (optional).

        Methods
        -------
        General
            set_filter_parameters, get_filter_metadata, get_ir.
        Plots or prints
            show_filter_parameters, plot_magnitude, plot_group_delay,
            plot_phase, plot_zp.
        Filtering
            filter_signal.

        """
        self.warning_if_complex = True
        self.sampling_rate_hz = sampling_rate_hz
        if filter_configuration is None:
            filter_configuration = \
                {'eq_type': 0, 'freqs': 1000, 'gain': 0, 'q': 1,
                 'filter_id': 'dummy'}
        self.set_filter_parameters(filter_type.lower(), filter_configuration)
        self.initialize_zi()

    def initialize_zi(self, number_of_channels: int = 1):
        """Initializes zi for steady-state filtering. The number of parallel
        zi's can be defined externally.

        Parameters
        ----------
        number_of_channels : int, optional
            Number of channels is needed for the number of filter's zi's.
            Default: 1.

        """
        self.zi = []
        if hasattr(self, 'sos'):
            for _ in range(number_of_channels):
                self.zi.append(sig.sosfilt_zi(self.sos))
        else:
            for _ in range(number_of_channels):
                self.zi.append(sig.lfilter_zi(self.ba[0], self.ba[1]))

    @property
    def sampling_rate_hz(self):
        return self.__sampling_rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, new_sampling_rate_hz):
        assert type(new_sampling_rate_hz) == int, \
            'Sampling rate can only be an integer'
        self.__sampling_rate_hz = new_sampling_rate_hz

    @property
    def warning_if_complex(self):
        return self.__warning_if_complex

    @warning_if_complex.setter
    def warning_if_complex(self, new_warning):
        assert type(new_warning) == bool, \
            'This attribute must be of boolean type'
        self.__warning_if_complex = new_warning

    @property
    def filter_type(self):
        return self.__filter_type

    @filter_type.setter
    def filter_type(self, new_type: str):
        assert type(new_type) == str, \
            'Filter type must be a string'
        self.__filter_type = new_type.lower()

    # ======== Filtering ======================================================
    def filter_signal(self, signal: Signal, channels=None,
                      activate_zi: bool = False, zero_phase: bool = False) \
            -> Signal:
        """Takes in a `Signal` object and filters selected channels. Exports a
        new `Signal` object.

        Parameters
        ----------
        signal : `Signal`
            Signal to be filtered.
        channels : int or array-like, optional
            Channel or array of channels to be filtered. When `None`, all
            channels are filtered. If only some channels are selected, these
            will be filtered and the others will be bypassed (and returned).
            Default: `None`.
        activate_zi : int, optional
            Gives the zi to update the filter values. Default: `False`.
        zero_phase : bool, optional
            Uses zero-phase filtering on signal. Be aware that the filter
            is applied twice in this case. Default: `False`.

        Returns
        -------
        new_signal : `Signal`
            New Signal object.

        """
        # Zero phase and zi
        assert not (activate_zi and zero_phase), \
            'Filter initial and final values cannot be updated when ' +\
            'filtering with zero-phase'
        # Channels
        if channels is None:
            channels = np.arange(signal.number_of_channels)
        else:
            channels = np.squeeze(channels)
            channels = np.atleast_1d(channels)
            assert channels.ndim == 1, \
                'channels can be only a 1D-array or an int'
            assert all(channels < signal.number_of_channels),\
                f'Selected channels ({channels}) are not valid for the ' +\
                f'signal with {signal.number_of_channels} channels'

        # Zi – create always for all channels and selected channels will get
        # updated while filtering
        if activate_zi:
            if len(self.zi) != signal.number_of_channels:
                warn('zi values of the filter have not been correctly ' +
                     'intialized for the number of channels. They have now' +
                     ' been corrected')
                self.initialize_zi(signal.number_of_channels)
            zi_old = self.zi
        else:
            zi_old = None

        # Check filter length compared to signal
        if self.info['order'] > signal.time_data.shape[0]:
            warn('Filter is longer than signal, results might be ' +
                 'meaningless!')

        # Filter with SOS when possible
        if hasattr(self, 'sos'):
            new_signal, zi_new = \
                _filter_on_signal(
                    signal=signal,
                    sos=self.sos,
                    channels=channels,
                    zi=zi_old,
                    zero_phase=zero_phase,
                    warning_on_complex_output=self.warning_if_complex)
        else:
            # Filter with ba
            new_signal, zi_new = \
                _filter_on_signal_ba(
                    signal=signal,
                    ba=self.ba,
                    channels=channels,
                    zi=zi_old,
                    zero_phase=zero_phase,
                    filter_type=self.filter_type,
                    warning_on_complex_output=self.warning_if_complex)
        if activate_zi:
            self.zi = zi_new
        return new_signal

    def filter_and_resample_signal(self, signal: Signal,
                                   new_sampling_rate_hz: int) -> Signal:
        """Filters and resamples signal. This is only available for all
        channels and sampling rates that are achievable by (only) down- or
        upsampling. This method is for allowing specific filters to be
        decimators/interpolators. If you just want to resample a signal,
        use the function in the standard module.

        If this filter is iir, standard resampling is applied. If it is
        fir, an efficient polyphase representation will be used.

        NOTE: Beware that no additional lowpass filter is used in the
        resampling step which can lead to aliases or other effects if this
        Filter is not adequate!

        Parameters
        ----------
        signal : `Signal`
            Signal to be filtered and resampled.
        new_sampling_rate_hz : int
            New sampling rate to resample to.

        Returns
        -------
        new_sig : `Signal`
            New down- or upsampled signal.

        """
        fraction = Fraction(
            new_sampling_rate_hz, signal.sampling_rate_hz).as_integer_ratio()
        assert fraction[0] == 1 or fraction[1] == 1, \
            f'{new_sampling_rate_hz} is not valid because it needs down- ' +\
            f'AND upsampling (Up/Down: {fraction[0]}/{fraction[1]})'

        # Check if standard or polyphase representation is to be used
        if self.filter_type == 'fir':
            polyphase = True
        elif self.filter_type in ('iir', 'biquad'):
            if not hasattr(self, 'ba'):
                self.ba = sig.sos2tf(self.sos)
            polyphase = False
        else:
            raise ValueError('Wrong filter type for filtering and resampling')

        # Check if down- or upsampling is required
        if fraction[0] == 1:
            assert signal.sampling_rate_hz == self.sampling_rate_hz, \
                'Sampling rates do not match'
            new_time_data = _filter_and_downsample(
                time_data=signal.time_data,
                down_factor=fraction[1],
                ba_coefficients=self.ba,
                polyphase=polyphase)
        elif fraction[1] == 1:
            assert signal.sampling_rate_hz*fraction[0] == \
                self.sampling_rate_hz, \
                'Sampling rates do not match. For the upsampler, the ' +\
                '''sampling rate of the filter should match the output's'''
            new_time_data = _filter_and_upsample(
                time_data=signal.time_data,
                up_factor=fraction[0],
                ba_coefficients=self.ba,
                polyphase=polyphase)

        new_sig = signal.copy()
        new_sig.sampling_rate_hz = new_sampling_rate_hz
        new_sig.time_data = new_time_data
        return new_sig

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
            self.filter_type = filter_type
        elif filter_type == 'fir':
            # Preparing parameters
            if 'filter_design_method' not in filter_configuration.keys():
                filter_configuration['filter_design_method'] = 'hamming'
            if 'width' not in filter_configuration.keys():
                filter_configuration['width'] = None
            # Filter creation
            self.ba = \
                [sig.firwin(numtaps=filter_configuration['order']+1,
                            cutoff=filter_configuration['freqs'],
                            window=filter_configuration
                            ['filter_design_method'],
                            width=filter_configuration['width'],
                            pass_zero=filter_configuration['type_of_pass'],
                            fs=self.sampling_rate_hz), np.asarray([1])]
            self.filter_type = filter_type
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
            filter_configuration['order'] = max(len(ba[0]), len(ba[1])) - 1
            self.filter_type = filter_type
        else:
            assert ('ba' in filter_configuration) ^ \
                ('sos' in filter_configuration) ^ \
                ('zpk' in filter_configuration), \
                'Only (and at least) one type of filter coefficients ' +\
                'should be passed to create a filter'
            if ('ba' in filter_configuration):
                b, a = filter_configuration['ba']
                self.ba = [np.atleast_1d(b), np.atleast_1d(a)]
                filter_configuration['order'] = \
                    max(len(self.ba[0]), len(self.ba[1])) - 1
            if ('zpk' in filter_configuration):
                z, p, k = filter_configuration['zpk']
                self.sos = sig.zpk2sos(z, p, k)
                filter_configuration['order'] = len(self.sos)*2 - 1
            if ('sos' in filter_configuration):
                self.sos = filter_configuration['sos']
                filter_configuration['order'] = len(self.sos)*2 - 1
            # Change filter type to 'fir' or 'iir' depending on coefficients
            self._check_and_update_filter_type()

        # Update Metadata about the Filter
        self.info = filter_configuration
        self.info['sampling_rate_hz'] = self.sampling_rate_hz
        self.info['filter_type'] = self.filter_type
        if hasattr(self, 'ba'):
            self.info['preferred_method_of_filtering'] = 'ba'
        elif hasattr(self, 'sos'):
            self.info['preferred_method_of_filtering'] = 'sos'
        if 'filter_id' not in self.info.keys():
            self.info['filter_id'] = None

    # ======== Check type =====================================================
    def _check_and_update_filter_type(self):
        """Internal method to check filter type (if FIR or IIR) and update
        its filter type.

        """
        # Get filter coefficients
        if hasattr(self, 'ba'):
            b, a = self.ba[0], self.ba[1]
        elif hasattr(self, 'sos'):
            b, a = sig.sos2tf(self.sos)
        # Trim zeros for a
        a = np.atleast_1d(np.trim_zeros(a))
        # Check length of a coefficients and decide filter type
        if len(a) == 1:
            b /= a[0]
            a = a / a[0]
            self.filter_type = 'fir'
        else:
            self.filter_type = 'iir'

    # ======== Getters ========================================================
    def get_filter_metadata(self):
        """Returns filter metadata.

        Returns
        -------
        info : dict
            Dictionary containing all filter metadata.

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
            if k == 'ba':
                continue
            txt += \
                f"""{str(k).replace('_', ' ').
                     capitalize()}: {self.info[k]}\n"""
        return txt

    def get_ir(self, length_samples: int = 512, zero_phase: bool = False):
        """Gets an impulse response of the filter with given length.

        Parameters
        ----------
        length_samples : int, optional
            Length for the impulse response in samples. Default: 512.

        Returns
        -------
        ir_filt : `Signal`
            Impulse response of the filter.

        """
        ir_filt = _impulse(length_samples)
        ir_filt = Signal(None, ir_filt, self.sampling_rate_hz, 'ir')
        return self.filter_signal(ir_filt, zero_phase=zero_phase)


    def get_coefficients(self, mode: str = 'sos'):
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
            if hasattr(self, 'sos'):
                coefficients = self.sos
            else:
                if self.info['order'] > 500:
                    inp = None
                    while inp not in ('y', 'n'):
                        inp = input(
                            'This filter has a large order ' +
                            f'''({self.info['order']}). Are you sure you ''' +
                            'want to get sos? Computation might' +
                            ' take long time. (y/n)')
                        inp = inp.lower()
                        if inp == 'y':
                            break
                        if inp == 'n':
                            return None
                coefficients = sig.tf2sos(self.ba[0], self.ba[1])
        elif mode == 'ba':
            if hasattr(self, 'sos'):
                coefficients = sig.sos2tf(self.sos)
            else:
                coefficients = self.ba
        elif mode == 'zpk':
            if hasattr(self, 'sos'):
                coefficients = sig.sos2zpk(self.sos)
            else:
                # Check if filter is too long
                if self.info['order'] > 500:
                    inp = None
                    while inp not in ('y', 'n'):
                        inp = input(
                            'This filter has a large order ' +
                            f'''({self.info['order']}). Are you sure you ''' +
                            'want to get zeros and poles? Computation might' +
                            ' take long time. (y/n)')
                        inp = inp.lower()
                        if inp == 'y':
                            break
                        if inp == 'n':
                            return None
                coefficients = sig.tf2zpk(self.ba[0], self.ba[1])
        else:
            raise ValueError(f'{mode} is not valid. Use sos, ba or zpk')
        return coefficients

    # ======== Plots and prints ===============================================
    def show_info(self):
        """Prints all the filter parameters to the console.

        """
        print(self._get_metadata_string())

    def plot_magnitude(self, length_samples: int = 512, range_hz=[20, 20e3],
                       normalize: str = None, show_info_box: bool = True,
                       zero_phase: bool = False, returns: bool = False):
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
            maximal value. Use `None` for no normalization. Default: `None`.
        show_info_box : bool, optional
            Shows an information box on the plot. Default: `True`.
        zero_phase : bool, optional
            Plots magnitude for zero phase filtering. Default: `False`.
        returns : bool, optional
            When `True` figure and axis are returned. Default: `False`.

        Returns
        -------
        fig, ax
            Returned only when `returns=True`.

        """
        if self.info['order'] > length_samples:
            length_samples = self.info['order'] + 100
            warn(f'length_samples ({length_samples}) is shorter than the ' +
                 f'''filter order {self.info['order']}. Length will be ''' +
                 'automatically extended.')
        ir = self.get_ir(length_samples=length_samples, zero_phase=zero_phase)
        fig, ax = ir.plot_magnitude(
            range_hz, normalize, show_info_box=False, returns=True)
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
        """Plots group delay of the filter. Different methods are used for
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
        fig, ax
            Returned only when `returns=True`.

        """
        if self.info['order'] > length_samples:
            length_samples = self.info['order'] + 100
            warn(f'length_samples ({length_samples}) is shorter than the ' +
                 f'''filter order {self.info['order']}. Length will be ''' +
                 'automatically extended.')
        if hasattr(self, 'sos'):
            ba = sig.sos2tf(self.sos)
        else:
            ba = self.ba
        f, gd = \
            _group_delay_filter(ba, length_samples, self.sampling_rate_hz)
        gd *= 1e3
        ymax = None
        ymin = None
        if any(abs(gd) > 50):
            ymin = -2
            ymax = 50
        fig, ax = general_plot(
            x=f,
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
        """Plots phase spectrum.

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
        fig, ax
            Returned only when `returns=True`.

        """
        if self.info['order'] > length_samples:
            length_samples = self.info['order'] + 100
            warn(f'length_samples ({length_samples}) is shorter than the ' +
                 f'''filter order {self.info['order']}. Length will be ''' +
                 'automatically extended.')
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
        fig, ax
            Returned only when `returns=True`.

        """
        # Ask explicitely if filter is very long
        if self.info['order'] > 500:
            inp = None
            while inp not in ('y', 'n'):
                inp = input(
                    'This filter has a large order ' +
                    f'''({self.info['order']}). Are you sure you want to''' +
                    ' plot zeros and poles? Computation might take long ' +
                    'time. (y/n)')
                inp = inp.lower()
                if inp == 'y':
                    break
                if inp == 'n':
                    return None
        #
        if hasattr(self, 'sos'):
            z, p, k = sig.sos2zpk(self.sos)
        else:
            z, p, k = sig.tf2zpk(self.ba[0], self.ba[1])
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
            Path for the filter to be saved. Use only folder1/folder2/name
            (without format). Default: `'filter'`
            (local folder, object named filter).

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
        new_sig : `Filter`
            Copy of filter.

        """
        return deepcopy(self)
