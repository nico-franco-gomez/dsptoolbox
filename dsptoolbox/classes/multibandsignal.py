from .signal_class import Signal
from os import sep
from numpy import zeros, array
from copy import deepcopy
from pickle import dump, HIGHEST_PROTOCOL


class MultiBandSignal():
    """The `MultiBandSignal` class contains multiple Signal objects which are
    to be interpreted as frequency bands or the same signal. Since every
    signal has also multiple channels, the object resembles somewhat a
    3D-Matrix representation of a signal.

    The `MultiBandSignal` can contain multirate system if the attribute
    `same_sampling_rate` is set to `False`. A dictionary also can carry
    all kinds of metadata that might characterize the signals.

    """
    # ======== Constructor and initializers ===================================
    def __init__(self, bands: list = [], same_sampling_rate: bool = True,
                 info: dict = {}):
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
        if bands is None:
            bands = []
        self.same_sampling_rate = same_sampling_rate
        self.bands = bands
        self.number_of_bands = len(self.bands)
        if self.bands:
            if self.same_sampling_rate:
                self.sampling_rate_hz = self.bands[0].sampling_rate_hz
            else:
                sr = []
                for b in self.bands:
                    sr.append(b.sampling_rate_hz)
                self.sampling_rate_hz = sr
        self._generate_metadata()
        self.info = self.info | info

    # ======== Properties and setters =========================================
    @property
    def sampling_rate_hz(self):
        return self.__sampling_rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, new_sampling_rate_hz):
        new_sampling_rate_hz = array(new_sampling_rate_hz).squeeze()
        assert new_sampling_rate_hz.dtype == int, \
            'Sampling rate can only be an integer'
        if self.same_sampling_rate:
            assert new_sampling_rate_hz.ndim == 0, \
                'MultiBandSignal has only one sample rate'
            self.__sampling_rate_hz = int(new_sampling_rate_hz)
        else:
            assert self.number_of_bands == len(new_sampling_rate_hz), \
                'Number of bands does not match number of sampling rates'
            self.__sampling_rate_hz = [int(s) for s in new_sampling_rate_hz]

    @property
    def bands(self):
        return self.__bands

    @bands.setter
    def bands(self, new_bands):
        assert type(new_bands) == list, \
            'bands has to be a list'
        if new_bands:
            # Check length and number of channels
            self.number_of_channels = new_bands[0].number_of_channels
            self.signal_type = new_bands[0].signal_type
            sr = []
            for s in new_bands:
                assert type(s) == Signal, f'{type(s)} is not a valid ' +\
                    'band type. Use Signal objects'
                assert s.number_of_channels == self.number_of_channels, \
                    'Signals have different number of channels. This ' +\
                    'behaviour is not supported'
                assert s.signal_type == self.signal_type, \
                    'Signal types do not match'
                sr.append(s.sampling_rate_hz)
            if self.same_sampling_rate:
                self.sampling_rate_hz = new_bands[0].sampling_rate_hz
                self.band_length_samples = new_bands[0].time_data.shape[0]
            else:
                self.same_sampling_rate = sr
            # Check sampling rate and duration
            if self.same_sampling_rate:
                for s in new_bands:
                    assert s.sampling_rate_hz == self.sampling_rate_hz, \
                        'Not all Signals have the same sampling rate. ' +\
                        'If you wish to create a multirate system, set ' +\
                        'same_sampling_rate to False'
                    assert s.time_data.shape[0] == self.band_length_samples,\
                        'The length of the bands is not always the same. ' +\
                        'This behaviour is not supported'
        self.__bands = new_bands

    @property
    def same_sampling_rate(self):
        return self.__same_sampling_rate

    @same_sampling_rate.setter
    def same_sampling_rate(self, new_same):
        assert type(new_same) == bool, \
            'Same sampling rate attribute must be a boolean'
        self.__same_sampling_rate = new_same

    def _generate_metadata(self):
        """Generates an information dictionary with metadata about the
        `MultiBandSignal`.

        """
        if not hasattr(self, 'info'):
            self.info = {}
        self.info['number_of_bands'] = len(self.bands)
        if self.bands:
            self.info['same_sampling_rate'] = self.same_sampling_rate
            self.info['signal_type'] = self.signal_type
            if self.same_sampling_rate:
                if hasattr(self, 'sampling_rate_hz'):
                    self.info['sampling_rate_hz'] = self.sampling_rate_hz
                self.info['band_length_samples'] = self.band_length_samples
            self.info['number_of_channels'] = self.number_of_channels

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
        if not self.bands:
            self.number_of_channels = sig.number_of_channels
            self.sampling_rate_hz = sig.sampling_rate_hz
            self.band_length_samples = sig.time_data.shape[0]
            self.signal_type = sig.signal_type
            self.bands.append(sig)
        else:
            assert sig.number_of_channels == self.number_of_channels, \
                'The number of channels does not match'
            assert sig.signal_type == self.signal_type, \
                'Signal types do not match'
            if self.same_sampling_rate:
                assert sig.sampling_rate_hz == self.sampling_rate_hz, \
                    'Sampling rate of band does not match with the one ' +\
                    'of MultiBandSignal'
                assert sig.time_data.shape[0] == self.band_length_samples, \
                    'The band length does not match'
            if index == -1:
                self.bands.append(sig)
            else:
                self.bands.insert(index, sig)
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
        assert self.bands, 'There are no filters to remove'
        if index == -1:
            index = len(self.bands) - 1
        assert index in range(len(self.bands)), \
            f'There is no band at index {index}.'
        f = self.bands.pop(index)
        self._generate_metadata()
        if return_band:
            return f

    def collapse(self):
        """Collapses MultiBandSignal by summing all of its bands and returning
        one Signal.

        Returns
        -------
        new_sig : `Signal`
            Collapsed Signal.

        """
        assert self.same_sampling_rate, \
            'Collapsing is only available for same sampling rate bands'
        initial = self.bands[0].time_data
        for n in range(1, len(self.bands)):
            initial += self.bands[n].time_data
        new_sig = self.bands[0].copy()
        new_sig.time_data = initial
        return new_sig

    def show_info(self, show_band_info: bool = False):
        """Show information about the `MultiBandSignal`.

        Parameters
        ----------
        show_band_info : bool, optional
            When `True`, a longer message is printed with all available
            information regarding each `Signal` in the `MultiBandSignal`.
            Default: `True`.

        """
        print()
        txt = ''
        for k in self.info:
            txt += \
                f""" | {str(k).replace('_', ' ').
                        capitalize()}: {self.info[k]}"""
        txt = 'Multiband signal:' + txt
        print(txt)
        if show_band_info:
            print('-'*len(txt), end='')
            for ind, f1 in enumerate(self.bands):
                print()
                txt = f'Signal {ind}:'
                for kf in f1.info:
                    txt += \
                        f""" | {str(kf).replace('_', ' ').
                                capitalize()}: {f1.info[kf]}"""
                print(txt)
        print()

    # ======== Getters ========================================================
    def get_all_bands(self, channel: int = 0):
        """Returns a signal with all bands as channels. Done for an specified
        channel.

        Parameters
        ----------
        channel : int, optional
            Channel to choose from the band signals.

        Returns
        -------
        sig : `Signal` or list of `np.ndarray` and dict
            Multichannel signal with all the bands. If the MultiBandSignal
            does not have the same sampling rate for all signals, a list with
            the time data vectors and a dictionary containing their sampling
            rates with the key 'sampling_rates' are returned.

        """
        if self.same_sampling_rate:
            new_time_data = \
                zeros((self.bands[0].time_data.shape[0], len(self.bands)))
            for n in range(len(self.bands)):
                new_time_data[:, n] = \
                    self.bands[n].time_data[:, channel].copy()
            sig = Signal(None, new_time_data, self.same_sampling_rate)
            return sig
        else:
            new_time_data = []
            d = {}
            sr = []
            for n in range(len(self.bands)):
                new_time_data.append(
                    self.bands[n].time_data[:, channel].copy())
                sr.append(self.bands[n].sampling_rate_hz)
            d['sampling_rates'] = sr
            return new_time_data, d

    # ======== Saving and copying =============================================
    def save_signal(self, path: str = 'multibandsignal'):
        """Saves the `MultiBandSignal` object as a pickle.

        Parameters
        ----------
        path : str, optional
            Path for the signal to be saved. Use only folder/folder/name
            (without format). Default: `'multibandsignal'`
            (local folder, object named multibandsignal).

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
        new_sig : `MultiBandSignal`
            Copy of Signal.

        """
        return deepcopy(self)
