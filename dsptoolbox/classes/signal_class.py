"""
Signal class
"""
from warnings import warn
from pickle import dump, HIGHEST_PROTOCOL
from copy import deepcopy
import numpy as np
from soundfile import read, write
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from dsptoolbox.plots import (general_plot,
                              general_subplots_line, general_matrix_plot)
from ._plots import _csm_plot
from dsptoolbox._general_helpers import \
    (_get_normalized_spectrum, _pad_trim, _find_nearest,
     _fractional_octave_smoothing, _check_format_in_path)
from dsptoolbox._standard import (_welch, _group_delay_direct, _stft, _csm)


class Signal():
    """Class for general signals (time series). Most of the methods and
    supported computations are focused on audio signals, but some features
    might be generalizable to all kinds of time series. It is assumed that
    audio is always represented in floating point type.

    """
    # ======== Constructor and State handler ==================================
    def __init__(self, path: str = None, time_data=None,
                 sampling_rate_hz: int = None, signal_type: str = 'general',
                 signal_id: str = '', constrain_amplitude: bool = True):
        """Signal class that saves time data, channel and sampling rate
        information as well as spectrum, cross-spectral matrix and more.

        Parameters
        ----------
        path : str, optional
            A path to audio files. Reading is done with the package soundfile.
            Wave and Flac audio files are accepted. Default: `None`.
        time_data : array-like, `np.ndarray`, optional
            Time data of the signal. It is saved as a matrix with the form
            (time samples, channel number). Default: `None`.
        sampling_rate_hz : int, optional
            Sampling rate of the signal in Hz. Default: `None`.
        signal_type : str, optional
            A generic signal type. Some functionalities are only unlocked for
            signal types `'ir'`, `'h1'`, `'h2'`, `'h3'`, `'rir'`, `'chirp'`,
            `'noise'` or `'dirac'`. Default: `'general'`.
        signal_id : str, optional
            An even more generic signal id that can be set by the user.
            Default: `''`.
        constrain_amplitude : bool, optional
            When `True`, audio is normalized to 0 dBFS peak level in case that
            there are amplitude values greater than 1. Otherwise, there is no
            normalization and the audio data is not constrained to [-1, 1].
            A warning is always shown when audio gets normalized and the used
            normalization factor is saved as `amplitude_scale_factor`.
            Default: `True`.

        Methods
        -------
        Time data:
            add_channel, remove_channel, swap_channels, get_channels.
        Spectrum:
            set_spectrum_parameters, get_spectrum.
        Cross spectral matrix:
            set_csm_parameters, get_csm.
        Spectrogram:
            set_spectrogram_parameters, get_spectrogram.
        Plots:
            plot_magnitude, plot_time, plot_spectrogram, plot_phase, plot_csm.
        General:
            save_signal, get_stream_samples.
        Only for `signal_type in ('rir', 'ir', 'h1', 'h2', 'h3')`:
            set_window, set_coherence, plot_group_delay, plot_coherence.

        """
        self.signal_id = signal_id
        self.signal_type = signal_type
        # Handling amplitude
        self.constrain_amplitude = constrain_amplitude
        self.scale_factor = None
        self.calibrated_signal = False
        # State tracker
        self.__spectrum_state_update = True
        self.__csm_state_update = True
        self.__spectrogram_state_update = True
        self.__time_vector_update = True
        # Import data
        if path is not None:
            assert time_data is None, 'Constructor cannot take a path and ' +\
                'a vector at the same time'
            assert sampling_rate_hz is None, \
                'Constructor cannot take a path and a sampling rate at the' +\
                ' same time'
            time_data, sampling_rate_hz = read(path)
        else:
            assert time_data is not None, \
                'Either a path to an audio file or a time vector has to be ' +\
                'passed'
            assert sampling_rate_hz is not None, \
                'A sampling rate should be passed!'
        self.sampling_rate_hz = sampling_rate_hz
        self.time_data = time_data
        if signal_type in ('rir', 'ir', 'h1', 'h2', 'h3', 'chirp', 'dirac'):
            self.set_spectrum_parameters(method='standard')
        else:
            self.set_spectrum_parameters()
        self.set_csm_parameters()
        self.set_spectrogram_parameters()
        self._generate_metadata()

    def __update_state(self):
        """Internal update of object state. If for instance time data gets
        added, new spectrum, csm or stft has to be computed.

        """
        self.__spectrum_state_update = True
        self.__csm_state_update = True
        self.__spectrogram_state_update = True
        self.__time_vector_update = True
        self._generate_metadata()

    def _generate_metadata(self):
        """Generates an information dictionary with metadata about the signal.

        """
        if not hasattr(self, 'info'):
            self.info = {}
        self.info['sampling_rate_hz'] = self.sampling_rate_hz
        self.info['number_of_channels'] = self.number_of_channels
        self.info['signal_length_samples'] = self.time_data.shape[0]
        self.info['signal_length_seconds'] = \
            self.time_data.shape[0] / self.sampling_rate_hz
        self.info['signal_type'] = self.signal_type
        self.info['signal_id'] = self.signal_id

    def _generate_time_vector(self):
        """Internal method to generate a time vector on demand.

        """
        self.__time_vector_update = False
        self.__time_vector_s = np.linspace(
            0, len(self.time_data)/self.sampling_rate_hz,
            len(self.time_data))

    # ======== Properties and setters =========================================
    @property
    def time_data(self) -> np.ndarray:
        return self.__time_data.copy()

    @time_data.setter
    def time_data(self, new_time_data):
        # Shape of Time Data array
        if not type(new_time_data) == np.ndarray:
            new_time_data = np.asarray(new_time_data)
        if new_time_data.ndim > 2:
            new_time_data = new_time_data.squeeze()
        assert new_time_data.ndim <= 2, \
            f'{new_time_data.ndim} are ' +\
            'too many dimensions for time data. Dimensions should' +\
            ' be [time samples, channels]'
        if len(new_time_data.shape) < 2:
            new_time_data = new_time_data[..., None]

        # Assume always that there are more time samples than channels
        if new_time_data.shape[1] > new_time_data.shape[0]:
            new_time_data = new_time_data.T

        # Handle complex data
        if np.iscomplexobj(new_time_data):
            new_time_data_imag = np.imag(new_time_data)
            new_time_data = np.real(new_time_data)
        else:
            new_time_data_imag = None

        # Normalization for real time data
        if self.constrain_amplitude:
            time_data_max = np.max(np.abs(new_time_data))
            if time_data_max > 1:
                new_time_data /= time_data_max
                warn('Signal was over 0 dBFS, normalizing to 0 dBFS ' +
                     'peak level was triggered')
                # Imaginary part is also scaled by same factor as real part
                if new_time_data_imag is not None:
                    new_time_data_imag /= time_data_max
                self.amplitude_scale_factor = 1/time_data_max
            else:
                self.amplitude_scale_factor = 1

        # Set time data (real and imaginary)
        self.__time_data = new_time_data
        self.time_data_imaginary = new_time_data_imag

        # Set number of channels
        self.number_of_channels = new_time_data.shape[1]
        self.__update_state()

    @property
    def sampling_rate_hz(self) -> int:
        return self.__sampling_rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, new_sampling_rate_hz):
        assert type(new_sampling_rate_hz) == int, \
            'Sampling rate can only be an integer'
        self.__sampling_rate_hz = new_sampling_rate_hz

    @property
    def signal_type(self) -> str:
        return self.__signal_type

    @signal_type.setter
    def signal_type(self, new_signal_type):
        assert type(new_signal_type) == str, \
            'Signal type must be a string'
        self.__signal_type = new_signal_type.lower()

    @property
    def signal_id(self) -> str:
        return self.__signal_id

    @signal_id.setter
    def signal_id(self, new_signal_id: str):
        assert type(new_signal_id) == str, \
            'Signal ID must be a string'
        self.__signal_id = new_signal_id.lower()

    @property
    def number_of_channels(self) -> int:
        return self.__number_of_channels

    @number_of_channels.setter
    def number_of_channels(self, new_number):
        assert type(new_number) == int, \
            'Number of channels must be integer'
        assert new_number > 0, \
            'There has to be at least one channel'
        assert new_number == self.time_data.shape[1], \
            'Number of channels does not match with time data vector'
        self.__number_of_channels = new_number

    @property
    def time_vector_s(self) -> np.ndarray:
        if self.__time_vector_update:
            self._generate_time_vector()
        return self.__time_vector_s

    @property
    def time_data_imaginary(self) -> np.ndarray | None:
        if self.__time_data_imaginary is None:
            # warn('Imaginary part of time data was called, but there is ' +
            #      'None. None is returned.')
            return None
        return self.__time_data_imaginary.copy()

    @time_data_imaginary.setter
    def time_data_imaginary(self, new_imag: np.ndarray):
        if new_imag is not None:
            assert new_imag.shape == self.__time_data.shape, \
                'Shape of imaginary part time data does not match'
        self.__time_data_imaginary = new_imag

    @property
    def constrain_amplitude(self) -> bool | None:
        return self.__constrain_amplitude

    @constrain_amplitude.setter
    def constrain_amplitude(self, nca):
        assert type(nca) == bool, \
            'constrain_amplitude must be of type boolean'
        self.__constrain_amplitude = nca
        # Restart time data setter for triggering normalization if needed
        if hasattr(self, 'time_data'):
            ntd = self.time_data
            self.time_data = ntd

    @property
    def calibrated_signal(self) -> bool:
        return self.__calibrated_signal

    @calibrated_signal.setter
    def calibrated_signal(self, ncs):
        assert type(ncs) == bool, \
            'calibrated_signal must be of type boolean'
        self.__calibrated_signal = ncs

    def __len__(self):
        return self.time_data.shape[0]

    def __str__(self):
        return self._get_metadata_string()

    def set_spectrum_parameters(self, method='welch', smoothe: int = 0,
                                window_length_samples: int = 1024,
                                window_type='hann', overlap_percent=50,
                                detrend=True, average='mean',
                                scaling='power spectral density'):
        """Sets all necessary parameters for the computation of the spectrum.

        Parameters
        ----------
        method : str, optional
            `'welch'` (Welch's method for stochastic signals) or
            `'standard'` (Direct FFT from signal). Default: `'welch'`.
        smoothe : int, optional
            Smoothing across (1/smoothe) octave bands using a hamming
            window. Smoothes magnitude AND phase. For accesing the smoothing
            algorithm, refer to
            `dsptoolbox._general_helpers._fractional_octave_smoothing()`.
            If smoothing is applied here, `Signal.get_spectrum()` returns
            the smoothed spectrum as well and `Signal.plot_magnitude()` plots
            the smoothed version. Default: 0 (no smoothing).
        window_length_samples : int, optional
            Window size. Default: 1024.
        window_type : str, optional
            Choose type of window. `scipy.signal.windows.get_window()` is used.
            Pass a tuple if the window needs extra parameters, e.g.,
            ('chebwin', 50). Default: `'hann'`.
        overlap_percent : float, optional
            Overlap in percent. Default: 50.
        detrend : bool, optional
            Detrending (subtracting mean). Default: True.
        average : str, optional
            Averaging method. Choose from `'mean'` or `'median'`.
            Default: `'mean'`.
        scaling : str, optional
            Scaling for welch's method. Use `'power spectrum'`,
            `'power spectral density'`, `'amplitude spectrum'` or
            `'amplitude spectral density'`. Pass `None` to avoid any scaling.
            See references for details about scaling. When method is set to
            standard, no scaling is applied unless `'amplitude spectrum'` is
            selected. Default: `'power spectral density'`.

        References
        ----------
        - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and
          spectral density estimation by the Discrete Fourier transform (DFT),
          including a comprehensive list of window functions and some new
          at-top windows.

        """
        assert method in ('welch', 'standard'), \
            f'{method} is not a valid method. Use welch or standard'
        if self.signal_type in ('h1', 'h2', 'h3', 'rir', 'ir'):
            if method != 'standard':
                method = 'standard'
                warn(f'For a signal of type {self.signal_type} ' +
                     'the spectrum has to be the standard one and not welch.' +
                     ' This has been automatically changed.')
        _new_spectrum_parameters = \
            dict(
                method=method,
                smoothe=smoothe,
                window_length_samples=window_length_samples,
                window_type=window_type,
                overlap_percent=overlap_percent,
                detrend=detrend,
                average=average,
                scaling=scaling)
        if not hasattr(self, '_spectrum_parameters'):
            self._spectrum_parameters = _new_spectrum_parameters
            self.__spectrum_state_update = True
        else:
            handler = \
                [self._spectrum_parameters[k] == _new_spectrum_parameters[k]
                 for k in self._spectrum_parameters.keys()]
            if not all(handler):
                self._spectrum_parameters = _new_spectrum_parameters
                self.__spectrum_state_update = True

    def set_window(self, window: np.ndarray):
        """Sets the window used for the IR. It only works for
        `signal_type in ('ir', 'h1', 'h2', 'h3', 'rir')`.

        Parameters
        ----------
        window : `np.ndarray`
            Window used for the IR.

        """
        valid_signal_types = ('ir', 'h1', 'h2', 'h3', 'rir')
        assert self.signal_type in valid_signal_types, \
            f'{self.signal_type} is not valid. Please set it to ir or ' +\
            'h1, h2, h3, rir'
        assert window.shape == self.time_data.shape, \
            f'{window.shape} does not match shape {self.time_data.shape}'
        self.window = window

    def set_coherence(self, coherence: np.ndarray):
        """Sets the coherence measurements of the transfer function.
        It only works for `signal_type = ('ir', 'h1', 'h2', 'h3', 'rir')`.

        Parameters
        ----------
        coherence : `np.ndarray`
            Coherence matrix.

        """
        valid_signal_types = ('ir', 'h1', 'h2', 'h3', 'rir')
        assert self.signal_type in valid_signal_types, \
            f'{self.signal_type} is not valid. Please set it to ir or ' +\
            'h1, h2, h3, rir'
        assert coherence.shape[0] == (self.time_data.shape[0]//2 + 1), \
            'Length of signals and given coherence do not match'
        assert coherence.shape[1] == self.number_of_channels, \
            'Number of channels between given coherence and signal ' +\
            'does not match'
        self.coherence = coherence

    def set_csm_parameters(self, window_length_samples: int = 1024,
                           window_type='hann', overlap_percent=75,
                           detrend=True, average='mean',
                           scaling='power spectral density'):
        """Sets all necessary parameters for the computation of the CSM.

        Parameters
        ----------
        window_length_samples : int, optional
            Window size. Default: 1024.
        overlap_percent : float, optional
            Overlap in percent. Default: 75.
        detrend : bool, optional
            Detrending (subtracting mean). Default: True.
        average : str, optional
            Averaging method. Choose from `'mean'` or `'median'`.
            Default: `'mean'`.
        scaling : str, optional
            Scaling. Use `'power spectrum'`, `'power spectral density'`,
            `'amplitude spectrum'` or `'amplitude spectral density'`. Pass
            `None` to avoid any scaling. See references for details about
            scaling. Default: `'power spectral density'`.

        References
        ----------
        - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and
          spectral density estimation by the Discrete Fourier transform (DFT),
          including a comprehensive list of window functions and some new
          at-top windows.

        """
        _new_csm_parameters = \
            dict(
                window_length_samples=window_length_samples,
                window_type=window_type,
                overlap_percent=overlap_percent,
                detrend=detrend,
                average=average,
                scaling=scaling)
        if not hasattr(self, '_csm_parameters'):
            self._csm_parameters = _new_csm_parameters
            self.__csm_state_update = True
        else:
            handler = \
                [self._csm_parameters[k] == _new_csm_parameters[k]
                 for k in self._csm_parameters.keys()]
            if not all(handler):
                self._csm_parameters = _new_csm_parameters
                self.__csm_state_update = True

    def set_spectrogram_parameters(self, channel_number: int = 0,
                                   window_length_samples: int = 1024,
                                   window_type: str = 'hann',
                                   overlap_percent=50,
                                   fft_length_samples: int = None,
                                   detrend: bool = True, padding: bool = True,
                                   scaling: bool = False):
        """Sets all necessary parameters for the computation of the
        spectrogram.

        Parameters
        ----------
        channel_number : int, optional
            Channel for which to compute the spectrogram. Default: 0.
        window_length_samples : int, optional
            Window size. Default: 1024.
        window_type : str, optional
            Type of window to use. Default: `'hann'`.
        overlap_percent : float, optional
            Overlap in percent. Default: 50.
        fft_length_samples : int, optional
            Length of the FFT window for each time window. This affects
            the frequency resolution and can also crop the time window. Pass
            `None` to use the window length. Default: `None`.
        detrend : bool, optional
            Detrending (subtracting mean). Default: `True`.
        padding : bool, optional
            Padding signal in the beginning and end to center it.
            Default: True.
        scaling : bool, optional
            When `True`, the output is scaled as an amplitude spectrum,
            otherwise no scaling is applied. See references for details.
            Default: `False`.

        References
        ----------
        - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and
          spectral density estimation by the Discrete Fourier transform (DFT),
          including a comprehensive list of window functions and some new
          at-top windows.

        """
        _new_spectrogram_parameters = \
            dict(
                channel_number=channel_number,
                window_length_samples=window_length_samples,
                window_type=window_type,
                overlap_percent=overlap_percent,
                fft_length_samples=fft_length_samples,
                detrend=detrend,
                padding=padding,
                scaling=scaling)
        if not hasattr(self, '_spectrogram_parameters'):
            self._spectrogram_parameters = _new_spectrogram_parameters
            self.__spectrogram_state_update = True
        else:
            handler = \
                [self._spectrogram_parameters[k] ==
                 _new_spectrogram_parameters[k]
                 for k in self._spectrogram_parameters.keys()]
            if not all(handler):
                self._spectrogram_parameters = _new_spectrogram_parameters
                self.__spectrogram_state_update = True

    # ======== Add, remove and reorder channels ===============================
    def add_channel(self, path: str = None, new_time_data: np.ndarray = None,
                    sampling_rate_hz: int = None,
                    padding_trimming: bool = True):
        """Adds new channels to this signal object.

        Parameters
        ----------
        path : str, optional
            Path to the file containing new channel information.
        new_time_data : `np.ndarray`, optional
            np.array with new channel data.
        sampling_rate_hz : int, optional
            Sampling rate for the new data
        padding_trimming : bool, optional
            Activates padding or trimming at the end of signal in case the
            new data does not match previous data. Default: `True`.

        """
        if path is not None:
            assert new_time_data is None, 'Only path or new time data is ' +\
                'accepted, not both.'
            new_time_data, sampling_rate_hz = read(path)
        else:
            if new_time_data is not None:
                assert path is None, 'Only path or new time data is ' +\
                    'accepted, not both.'
        assert sampling_rate_hz == self.sampling_rate_hz, \
            f'{sampling_rate_hz} does not match {self.sampling_rate_hz} ' +\
            'as the sampling rate'
        if not type(new_time_data) == np.ndarray:
            new_time_data = np.array(new_time_data)
        if new_time_data.ndim > 2:
            new_time_data = new_time_data.squeeze()
        assert new_time_data.ndim <= 2, \
            f'{new_time_data.ndim} are ' +\
            'too many dimensions for time data. Dimensions should' +\
            ' be (time samples, channels)'
        if new_time_data.ndim < 2:
            new_time_data = new_time_data[..., None]
        if new_time_data.shape[1] > new_time_data.shape[0]:
            new_time_data = new_time_data.T

        diff = new_time_data.shape[0] - self.time_data.shape[0]
        if diff != 0:
            txt = 'Padding' if diff < 0 else 'Trimming'
            if padding_trimming:
                new_time_data = \
                    _pad_trim(new_time_data, self.time_data.shape[0],
                              axis=0, in_the_end=True)
                warn(f'{txt} has been performed ' +
                     'on the end of the new signal to match original one.')
            else:
                raise AttributeError(
                    f'{new_time_data.shape[0]} does not match ' +
                    f'{self.time_data.shape[0]}. Activate padding_trimming ' +
                    'for allowing this channel to be added')
        self.time_data = np.concatenate([self.time_data, new_time_data],
                                        axis=1)
        self.__update_state()

    def remove_channel(self, channel_number: int = -1):
        """Removes a channel.

        Parameters
        ----------
        channel_number : int, optional
            Channel number to be removed. Default: -1 (last).

        """
        if channel_number == -1:
            channel_number = self.time_data.shape[1] - 1
        assert self.time_data.shape[1] > 1, \
            'Cannot not erase only channel'
        assert self.time_data.shape[1]-1 >= channel_number, \
            f'Channel number {channel_number} does not exist. Signal only ' +\
            f'has {self.number_of_channels-1} channels (zero included).'
        self.time_data = np.delete(self.time_data, channel_number, axis=-1)
        self.__update_state()

    def swap_channels(self, new_order):
        """Rearranges the channels in the new given order.

        Parameters
        ----------
        new_order : array-like
            New rearrangement of channels.

        """
        new_order = np.array(new_order).squeeze()
        assert new_order.ndim == 1, \
            'Too many or too few dimensions are given in the new ' +\
            'arrangement vector'
        assert self.number_of_channels == len(new_order), \
            'The number of channels does not match'
        assert all(new_order < self.number_of_channels) \
            and all(new_order >= 0), \
            'Indexes of new channels have to be in ' +\
            f'[0, {self.number_of_channels-1}]'
        assert len(np.unique(new_order)) == len(new_order), \
            'There are repeated indexes in the new order vector'
        self.time_data = self.time_data[:, new_order]

    def get_channels(self, channels):
        """Returns a signal object with the selected channels. Beware that
        first channel index is 0!

        Parameters
        ----------
        channels : array-like or int
            Channels to be returned as a new Signal object.

        Returns
        -------
        new_sig : `Signal`
            New signal object with selected channels.

        """
        channels = np.atleast_1d(np.asarray(channels).squeeze())
        if channels.ndim > 1:
            raise ValueError(
                'Parameter channels must be only an object broadcastable to ' +
                '1D Array')
        assert all(channels < self.number_of_channels), \
            'Indexes of new channels have to be smaller than the number ' +\
            f'of channels {self.number_of_channels}'
        new_sig = self.copy()
        new_sig.time_data = self.time_data[:, channels]
        if hasattr(new_sig, 'window'):
            new_sig.window = new_sig.window[:, channels]
        return new_sig

    # ======== Getters ========================================================
    def get_spectrum(self, force_computation=False) \
            -> tuple[np.ndarray, np.ndarray]:
        """Returns spectrum.

        Parameters
        ----------
        force_computation : bool, optional
            Forces spectrum computation.

        Returns
        -------
        spectrum_freqs : `np.ndarray`
            Frequency vector.
        spectrum : `np.ndarray`
            Spectrum matrix for each channel.

        """
        condition = not hasattr(self, 'spectrum') or \
            self.__spectrum_state_update or force_computation

        if condition:
            if self._spectrum_parameters['method'] == 'welch':
                spectrum = np.zeros((self._spectrum_parameters
                                     ['window_length_samples'] // 2 + 1,
                                     self.number_of_channels), dtype='float')
                for n in range(self.number_of_channels):
                    spectrum[:, n] = _welch(
                        self.time_data[:, n], self.time_data[:, n],
                        self.sampling_rate_hz,
                        self._spectrum_parameters['window_type'],
                        self._spectrum_parameters['window_length_samples'],
                        self._spectrum_parameters['overlap_percent'],
                        self._spectrum_parameters['detrend'],
                        self._spectrum_parameters['average'],
                        self._spectrum_parameters['scaling'])
                time_length = \
                    self._spectrum_parameters['window_length_samples']
            elif self._spectrum_parameters['method'] == 'standard':
                # Get spectrum
                spectrum = np.fft.rfft(self.time_data, axis=0)

                # Smoothing
                if self._spectrum_parameters['smoothe'] != 0:
                    temp_abs = _fractional_octave_smoothing(
                        np.abs(spectrum), self._spectrum_parameters['smoothe'])
                    temp_phase = _fractional_octave_smoothing(
                        np.angle(spectrum),
                        self._spectrum_parameters['smoothe'])
                    spectrum = temp_abs*np.exp(1j*temp_phase)

                # Length of signal for frequency vector and scaling
                time_length = self.time_data.shape[0]
                if self._spectrum_parameters['scaling'] \
                        == 'amplitude spectrum':
                    spectrum /= time_length
            self.spectrum = []
            self.spectrum.append(
                np.fft.rfftfreq(time_length, 1/self.sampling_rate_hz))
            self.spectrum.append(spectrum)
            spectrum_freqs = self.spectrum[0]
            self.__spectrum_state_update = False
        else:
            spectrum_freqs, spectrum = self.spectrum[0], self.spectrum[1]
        return spectrum_freqs.copy(), spectrum.copy()

    def get_csm(self, force_computation=False) -> \
            tuple[np.ndarray, np.ndarray]:
        """Get Cross spectral matrix for all channels with the shape
        (frequencies, channels, channels).

        Returns
        -------
        f_csm : `np.ndarray`
            Frequency vector.
        csm : `np.ndarray`
            Cross spectral matrix with shape (frequency, channels, channels).

        """
        assert self.number_of_channels > 1, \
            'Cross spectral matrix can only be computed when at least two ' +\
            'channels are available'
        condition = not hasattr(self, 'csm') or force_computation or \
            self.__csm_state_update

        if condition:
            self.csm = _csm(self.time_data, self.sampling_rate_hz,
                            **self._csm_parameters)
            self.__csm_state_update = False
        return self.csm[0].copy(), self.csm[1].copy()

    def get_spectrogram(self, channel_number: int = 0,
                        force_computation: bool = False) -> \
            tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns a matrix containing the STFT of a specific channel.

        Parameters
        ----------
        channel_number : int, optional
            Channel number for which to compute the STFT. Default: 0.
        force_computation : bool, optional
            Forces new computation of the STFT. Default: False.

        Returns
        -------
        t_s : `np.ndarray`
            Time vector.
        f_hz : `np.ndarray`
            Frequency vector.
        spectrogram : `np.ndarray`
            Complex spectrogram with shape (frequency, time).

        """
        condition = not hasattr(self, 'spectrogram') or force_computation or \
            self.__spectrogram_state_update or \
            not channel_number == \
            self._spectrogram_parameters['channel_number']

        if condition:
            self._spectrogram_parameters['channel_number'] = channel_number
            self.spectrogram = _stft(
                self.time_data[:, channel_number],
                self.sampling_rate_hz,
                self._spectrogram_parameters['window_length_samples'],
                self._spectrogram_parameters['window_type'],
                self._spectrogram_parameters['overlap_percent'],
                self._spectrogram_parameters['fft_length_samples'],
                self._spectrogram_parameters['detrend'],
                self._spectrogram_parameters['padding'],
                self._spectrogram_parameters['scaling'])
            self.__spectrogram_state_update = False
        t_s, f_hz, spectrogram = \
            self.spectrogram[0], self.spectrogram[1], self.spectrogram[2]
        return t_s, f_hz, spectrogram

    def get_coherence(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the coherence matrix.

        Returns
        -------
        f : `np.ndarray`
            Frequency vector.
        coherence : `np.ndarray`
            Coherence matrix.

        """
        assert hasattr(self, 'coherence'), \
            'There is no coherence data saved in the Signal object'
        f, _ = self.get_spectrum()
        return f, self.coherence

    # ======== Plots ==========================================================
    def plot_magnitude(self, range_hz=[20, 20e3], normalize: str = '1k',
                       range_db=None, smoothe: int = 0,
                       show_info_box: bool = False, scale: bool = True) \
            -> tuple[Figure, Axes]:
        """Plots magnitude spectrum.
        Change parameters of spectrum with set_spectrum_parameters.

        Parameters
        ----------
        range_hz : array-like with length 2, optional
            Range for which to plot the magnitude response.
            Default: [20, 20000].
        normalize : str, optional
            Mode for normalization, supported are `'1k'` for normalization
            with value at frequency 1 kHz or `'max'` for normalization with
            maximal value. Use `None` for no normalization (only by the
            sampling rate if standard method for spectrum is selected).
            Default: `'1k'`.
        range_db : array-like with length 2, optional
            Range in dB for which to plot the magnitude response.
            Default: `None`.
        smoothe : int, optional
            Smoothing across the (1/smoothe) octave band.
            Default: 0 (no smoothing).
        show_info_box : bool, optional
            Plots a info box regarding spectrum parameters and plot parameters.
            If it is str, it overwrites the standard message.
            Default: `False`.
        scale : bool, optional
            When `True`, spectrum gets scaled (divided) by double the length
            of the time signal. This ensures that the energy of the time signal
            is distributed in the whole spectrum. This only applies when
            `method = 'standard'` and no normalization is applied.
            Default: `True`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        Notes
        -----
        - Smoothing is only applied on the plot data.
        - In case the signal has been calibrated and the time data is given in
          Pascal, the plotted values in dB will be scaled by p0=(20e-6 Pa)**2
          when no normalization is active.

        """
        f, sp = self.get_spectrum()
        if self._spectrum_parameters['method'] == 'standard' \
                and normalize is None and scale:
            sp = sp/self.time_data.shape[0]*2
        f, mag_db = _get_normalized_spectrum(
            f=f,
            spectra=sp,
            mode=self._spectrum_parameters['method'],
            f_range_hz=range_hz,
            normalize=normalize,
            smoothe=smoothe,
            calibrated_data=self.calibrated_signal)
        if show_info_box:
            txt = 'Info'
            txt += f"""\nMode: {self._spectrum_parameters['method']}"""
            txt += f'\nRange: [{range_hz[0]}, {range_hz[1]}]'
            txt += f'\nNormalized: {normalize}'
            txt += f"""\nSmoothing: {smoothe}"""
        else:
            txt = None
        if normalize is not None:
            if normalize == '1k':
                y_extra = ' (normalized @ 1 kHz)'
            elif normalize == 'max':
                y_extra = ' (normalized @ peak)'
        else:
            y_extra = '' if self.calibrated_signal else 'FS'
        fig, ax = general_plot(f, mag_db, range_hz,
                               ylabel='Magnitude / dB'+y_extra,
                               info_box=txt, returns=True,
                               labels=[f'Channel {n}' for n in
                                       range(self.number_of_channels)],
                               range_y=range_db)
        return fig, ax

    def plot_time(self) -> tuple[Figure, Axes]:
        """Plots time signals.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        if not hasattr(self, 'time_vector_s'):
            self._generate_time_vector()
        fig, ax = general_subplots_line(
            self.time_vector_s,
            self.time_data,
            sharex=True,
            ylabels=[f'Channel {n}' for n in range(self.number_of_channels)],
            xlabels='Time / s',
            returns=True)
        for n in range(self.number_of_channels):
            mx = np.max(np.abs(self.time_data[:, n])) * 1.1
            if hasattr(self, 'window'):
                ax[n].plot(self.time_vector_s,
                           self.window[:, n] * mx / 1.1, alpha=0.75)
            ax[n].set_ylim([-mx, mx])
        return fig, ax

    def plot_group_delay(self, range_hz=[20, 20000]) -> tuple[Figure, Axes]:
        """Plots group delay of each channel.
        Only works if `signal_type in ('ir', 'h1', 'h2', 'h3', 'rir', 'chirp',
        'noise', 'dirac')`.

        Parameters
        ----------
        range_hz : array-like with length 2, optional
            Range of frequencies for which to show group delay.
            Default: [20, 20e3].

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        valid_signal_types = (
            'rir', 'ir', 'h1', 'h2', 'h3', 'chirp', 'noise', 'dirac')
        assert self.signal_type in valid_signal_types, \
            f'{self.signal_type} is not valid. Please set it to ir or ' +\
            'h1, h2, h3, rir'
        self.set_spectrum_parameters('standard')
        f, sp = self.get_spectrum()
        gd = np.zeros((len(f), self.number_of_channels))
        for n in range(self.number_of_channels):
            gd[:, n] = _group_delay_direct(sp[:, n], f[1]-f[0])
        fig, ax = general_plot(
            f, gd*1e3, range_hz,
            labels=[f'Channel {n}' for n in range(self.number_of_channels)],
            ylabel='Group delay / ms',
            returns=True)
        return fig, ax

    def plot_spectrogram(self, channel_number: int = 0,
                         logfreqs: bool = True,
                         dynamic_range_db: float = 50) -> tuple[Figure, Axes]:
        """Plots STFT matrix of the given channel. The levels in the plot can
        go down until -400 dB.

        Parameters
        ----------
        channel_number : int, optional
            Selected channel to plot spectrogram. Default: 0 (first).
        logfreqs : bool, optional
            When `True`, frequency axis is plotted logarithmically.
            Default: `True`.
        dynamic_range_db : float, optional
            This sets the dynamic range to show for the spectrogram. The
            plotted colormap goes from the maximum down to maximum minus
            dynamic range. For example, dynamic_range_db=50 plots for a peak
            value of 30 dB the colormap of the spectrogram between
            [30, -20] dB. Default: 50.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        t, f, stft = self.get_spectrogram(channel_number)
        ids = _find_nearest([20, 20000], f)
        if ids[0] == 0:
            ids[0] += 1
        f = f[ids[0]:ids[1]]
        stft = stft[ids[0]:ids[1], :]
        stft_db = 20*np.log10(np.clip(np.abs(stft), 1e-20, None))
        stft_db = np.nan_to_num(stft_db, nan=np.min(stft_db))
        if self._spectrogram_parameters['scaling']:
            zlabel = 'dB (Pa$^2$/Hz)'
        else:
            zlabel = 'dB (No Scaling)'
        fig, ax = general_matrix_plot(
            matrix=stft_db, range_x=(t[0], t[-1]),
            range_y=(f[0], f[-1]), range_z=np.abs(dynamic_range_db),
            xlabel='Time / s', ylabel='Frequency / Hz',
            zlabel=zlabel,
            xlog=False, ylog=logfreqs,
            colorbar=True, returns=True)
        return fig, ax

    def plot_coherence(self) -> tuple[Figure, Axes]:
        """Plots coherence measurements if there are any.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        if not hasattr(self, 'coherence'):
            raise AttributeError('There is no coherence data saved in the ' +
                                 'Signal object')
        f, coh = self.get_coherence()
        fig, ax = \
            general_subplots_line(
                x=f,
                matrix=coh,
                column=True,
                sharey=True,
                log=True,
                ylabels=[rf'$\gamma^2$ Coherence {n}'
                         for n in range(self.number_of_channels)],
                xlabels='Frequency / Hz',
                range_y=[-0.1, 1.1],
                returns=True)
        return fig, ax

    def plot_phase(self, range_hz=[20, 20e3], unwrap: bool = False) \
            -> tuple[Figure, Axes]:
        """Plots phase of the frequency response, only available if the method
        for the spectrum parameters is not welch.

        Parameters
        ----------
        range_hz : array-like with length 2, optional
            Range of frequencies for which to show group delay.
            Default: [20, 20e3].
        unwrap : bool, optional
            When `True`, the unwrapped phase is plotted. Default: `False`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        assert self._spectrum_parameters['method'] == 'standard', \
            'Phase cannot be plotted since the spectrum is ' +\
            'welch. Please change spectrum parameters method to ' +\
            'standard'
        f, sp = self.get_spectrum()
        ph = np.angle(sp)
        if unwrap:
            ph = np.unwrap(ph, axis=0)
        fig, ax = general_plot(
            x=f,
            matrix=ph,
            range_x=range_hz,
            labels=[f'Channel {n}'
                    for n in range(self.number_of_channels)],
            ylabel='Phase / rad',
            returns=True)
        return fig, ax

    def plot_csm(self, range_hz=[20, 20e3], logx: bool = True,
                 with_phase: bool = True) -> tuple[Figure, Axes]:
        """Plots the cross spectral matrix of the multichannel signal.

        Parameters
        ----------
        range_hz : array-like with length 2, optional
            Range of Hz to be showed. Default: [20, 20e3].
        logx : bool, optional
            Logarithmic x axis. Default: `True`.
        with_phase : bool, optional
            When `True`, the unwrapped phase is also plotted. Default: `True`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        f, csm = self.get_csm()
        fig, ax = _csm_plot(f, csm, range_hz, logx, with_phase, returns=True)
        return fig, ax

    # ======== Saving and copy ================================================
    def save_signal(self, path: str = 'signal', mode: str = 'wav'):
        """Saves the Signal object as wav, flac or pickle.

        Parameters
        ----------
        path : str, optional
            Path for the signal to be saved. Use only folders/name
            (without format). Default: `'signal'`
            (local folder, object named signal).
        mode : str, optional
            Mode of saving. Available modes are `'wav'`, `'flac'`, `'pkl'`.
            Default: `'wav'`.

        """
        mode = mode.lower()
        path = _check_format_in_path(path, mode)
        if mode in ('wav', 'flac'):
            write(path, self.time_data, self.sampling_rate_hz)
        elif mode == 'pkl':
            with open(path, 'wb') as data_file:
                dump(self, data_file, HIGHEST_PROTOCOL)
        else:
            raise ValueError(
                f'{mode} is not a supported saving mode. Use ' +
                'wav, flac or pkl')

    def copy(self):
        """Returns a copy of the object.

        Returns
        -------
        new_sig : `Signal`
            Copy of Signal.

        """
        return deepcopy(self)

    def _get_metadata_string(self) -> str:
        """Helper for creating a string containing all signal info.

        """
        txt = f"""Signal – ID: {self.info['signal_id']}\n"""
        temp = ''
        for n in range(len(txt)):
            temp += '-'
        txt += (temp+'\n')
        for k in self.info.keys():
            if k == 'signal_id':
                continue
            txt += \
                f"""{str(k).replace('_', ' ').
                     capitalize()}: {self.info[k]}\n"""
        return txt

    def show_info(self):
        """Prints all the signal information to the console.

        """
        print(self._get_metadata_string())

    # ======== Streaming methods ==============================================
    def set_streaming_position(self, position_samples: int = 0) -> bool:
        """Sets the start position for streaming.

        Parameters
        ----------
        position_samples : int, optional
            Position (in samples) for starting the stream. Default: 0.

        Returns
        -------
        stop_flag : bool
            When `True`, all of the available time samples in the signal have
            been used.

        """
        stop_flag = False
        if self.time_data.shape[0] - position_samples < 0:
            stop_flag = True
        assert type(position_samples) == int, \
            'Position must be in samples and thus an integer'
        self.streaming_position = position_samples
        return stop_flag

    def stream_samples(self, blocksize_samples: int,
                       signal_mode: bool = True):
        """Returns block of samples to be reproduced in an audio stream. Use
        `set_streaming_position` to define start of streaming.

        Parameters
        ----------
        blocksize_samples : int
            Blocksize to be returned (in samples).
        signal_mode : bool, optional
            When `True` a `Signal` object is returned containing the desired
            samples. `False` returns a numpy array. Default: `True`.

        Returns
        -------
        sig : `np.ndarray` or `Signal`
            Numpy array with samples used for reproduction with shape
            (time_samples, channels) or `Signal` object.
        stop_flag : bool
            When `True`, all samples of the available signal have been
            reproduced and the stream should be stopped.

        """
        if not hasattr(self, 'streaming_position'):
            stop_flag = self.set_streaming_position()
        sig = self.time_data[
            self.streaming_position:self.streaming_position +
            blocksize_samples, :].copy()
        stop_flag = self.set_streaming_position(
            self.streaming_position + blocksize_samples)
        if signal_mode:
            sig = Signal(
                None, sig, self.sampling_rate_hz,
                self.signal_type, self.signal_id)
            # In an audio stream, welch's method for acquiring a spectrum
            # is not very logical...
            sig.set_spectrum_parameters(method='standard')
        return sig, stop_flag
