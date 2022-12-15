'''
Signal classes
'''
import warnings
import pickle
import os

import numpy as np
import soundfile as sf
from .plots import (general_plot, general_subplots_line, general_matrix_plot)
from .backend._plots import _csm_plot
from .backend._general_helpers import (_get_normalized_spectrum, _pad_trim)
from .backend._standard import (_welch, _group_delay_direct, _stft, _csm)
from .backend._general_helpers import _find_nearest

__all__ = ['Signal', 'MultiBandSignal', ]


class Signal():
    # ======== Constructor and State handler ==================================
    def __init__(self, path=None, time_data=None,
                 sampling_rate_hz: int = 48000, signal_type: str = 'general',
                 signal_id: str = ''):
        '''
        Signal class that saves mainly time data for being used with all the
        methods.

        Parameters
        ----------
        path : str, optional
            A path to audio files. Reading is done with soundfile. Wave and
            Flac audio files are accepted.
            Default: `None`.
        time_data : array-like, np.ndarray, optional
            Time data of the signal. It is saved as a matrix with the form
            (time samples, channel number). Default: `None`.
        sampling_rate_hz : int, optional
            Sampling rate of the signal in Hz. Default: 48000.
        signal_type : str, optional
            A generic signal type. Some functionalities are only unlocked for
            impulse responses with `'ir'` or `'h1'`, `'h2'` or `'h3'`.
            Default: `'general'`.
        signal_id : str, optional
            An even more generic signal id that can be used by the user.
            Default: `''`.

        Methods
        -------
        General: set_signal_type.
        Time data: set_time_data, add_channel, remove_channel.
        Spectrum: set_spectrum_parameters, get_spectrum.
        Cross spectral matrix: set_csm_parameters, get_csm.
        Spectrogram: set_spectrogram_parameters, get_spectrogram.

        Plots: plot_magnitude, plot_time, plot_spectrogram, plot_phase

        Only for `signal_type in ('rir', 'ir', 'h1', 'h2', 'h3')`:
            set_window, set_coherence, plot_group_delay, plot_coherence.
        '''
        self.signal_id = signal_id.lower()
        self.signal_type = signal_type.lower()
        # State tracker
        self.__spectrum_state_update = True
        self.__csm_state_update = True
        self.__spectrogram_state_update = True
        # Import data
        if path is not None:
            assert time_data is None, 'Constructor cannot take a path and ' +\
                'a vector at the same time'
            time_data, sampling_rate_hz = sf.read(path)
            self.set_time_data(time_data, sampling_rate_hz)
        else:
            time_data = np.array(time_data)
            self.set_time_data(time_data, sampling_rate_hz)
        if signal_type in ('rir', 'ir', 'h1', 'h2', 'h3', 'chirp', 'noise'):
            self.set_spectrum_parameters(method='standard')
        else:
            self.set_spectrum_parameters()
        self.set_csm_parameters()
        self.set_spectrogram_parameters()
        self._generate_metadata()

    def __update_state(self):
        '''
        Internal update of object state. If for instance time data gets added,
        new spectrum, csm or stft has to be computed
        '''
        self.__spectrum_state_update = True
        self.__csm_state_update = True
        self.__spectrogram_state_update = True
        self._generate_metadata()

    def _generate_metadata(self):
        '''
        Generates an information dictionary with metadata about the signal
        '''
        self.info = {}
        self.info['sampling_rate_hz'] = self.sampling_rate_hz
        self.info['number_of_channels'] = self.number_of_channels
        self.info['signal_length_samples'] = self.time_data.shape[0]
        self.info['signal_length_seconds'] = \
            self.time_data.shape[0] / self.sampling_rate_hz
        self.info['signal_type'] = self.signal_type

    def _generate_time_vector(self):
        '''
        Internal method to generate a time vector on demand
        '''
        self.time_vector_s = np.linspace(
            0, len(self.time_data)/self.sampling_rate_hz, len(self.time_data))

    # ======== Setters ========================================================
    def set_time_data(self, time_data, sampling_rate_hz: int = None):
        '''
        Sets new time data for object. Called from constructor or outside.

        Parameters
        ----------
        new_time_data : array-like
            New array containing time data. It can be a matrix for multiple
            channel data.
        sampling_rate_hz : int, optional
            Sampling rate for the new data in Hz. If `None`, the previous
            sampling rate is used. Default `None`.
        '''
        if not type(time_data) == np.ndarray:
            time_data = np.array(time_data)
        assert len(time_data.shape) <= 2, f'{len(time_data.shape)} has ' +\
            'too many dimensions for time data. Dimensions should' +\
            ' be [time samples, channels]'
        if len(time_data.shape) < 2:
            time_data = time_data[..., None]
        if time_data.shape[1] > time_data.shape[0]:
            time_data = time_data.T
        time_data_max = np.max(np.abs(time_data))
        if time_data_max > 1:
            time_data /= time_data_max
            warnings.warn('Signal was over 0 dBFS, normalizing to 0 dBFS ' +
                          'peak level was triggered')
        self.time_data = time_data
        if sampling_rate_hz is not None:
            self.sampling_rate_hz = sampling_rate_hz
        self.number_of_channels = time_data.shape[1]
        self.__update_state()

    def set_spectrum_parameters(self, method='welch',
                                window_length_samples: int = 1024,
                                window_type='hann', overlap_percent=50,
                                detrend=True, average='mean',
                                scaling='power'):
        '''
        Sets all necessary parameters for the computation of the spectrum.

        Parameters
        ----------
        method : str, optional
            `'welch'` or `'standard'`. Default: `'welch'`
        window_length_samples : int, optional
            Window size. Default: 1024.
        window_type : str,optional
            Choose type of window from the options in scipy.windows.
            Default: `'hann'`.
        overlap_percent : float, optional
            Overlap in percent. Default: 50.
        detrend : bool, optional
            Detrending (subtracting mean). Default: True.
        average : str, optional
            Averaging method. Choose from `'mean'` or `'median'`.
            Default: `'mean'`.
        scaling : str, optional
            Type of scaling. '`power'` or `'spectrum'`. Default: `'power'`.
        '''
        assert method in ('welch', 'standard'), \
            f'{method} is not a valid method. Use welch or standard'
        if self.signal_type in ('h1', 'h2', 'h3'):
            if method != 'standard':
                method = 'standard'
                warnings.warn(
                    f'For a signal of type {self.signal_type} ' +
                    'the spectrum has to be the standard one and not welch.' +
                    ' This has been automatically changed.')
        _new_spectrum_parameters = \
            dict(
                method=method,
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

    def set_signal_type(self, s_type: str):
        self.signal_type = s_type

    def set_window(self, window):
        '''
        Sets the window used for the IR. It only works for
        `signal_type in ('ir', 'h1', 'h2', 'h3', 'rir')`
        '''
        valid_signal_types = ('ir', 'h1', 'h2', 'h3', 'rir')
        assert self.signal_type in valid_signal_types, \
            f'{self.signal_type} is not valid. Please set it to ir or ' +\
            'h1, h2, h3, rir'
        assert len(window) == self.time_data.shape[0], \
            f'{len(window)} does not match shape {self.time_data.shape}'
        self.window = window

    def set_coherence(self, coherence: np.ndarray):
        '''
        Sets the window used for the IR. It only works for
        `signal_type = ('ir', 'h1', 'h2', 'h3', 'rir')`
        '''
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
                           scaling='power'):
        '''
        Sets all necessary parameters for the computation of the CSM.

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
            Type of scaling. '`power'` or `'spectrum'`. Default: `'power'`.
        '''
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
                                   overlap_percent=75,
                                   detrend: bool = True, padding: bool = True,
                                   scaling: bool = False):
        '''
        Sets all necessary parameters for the computation of the spectrogram.

        Parameters
        ----------
        window_length_samples : int, optional
            Window size. Default: 1024.
        overlap_percent : float, optional
            Overlap in percent. Default: 75.
        detrend : bool, optional
            Detrending (subtracting mean). Default: True.
        padding : bool, optional
            Padding signal in the beginning and end to center it.
            Default: True.
        scaling : bool, optional
            Scaling or not after FFT. Default: False.
        '''
        _new_spectrogram_parameters = \
            dict(
                channel_number=channel_number,
                window_length_samples=window_length_samples,
                window_type=window_type,
                overlap_percent=overlap_percent,
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
        '''
        Adds new channels to this signal object.

        Parameters
        ----------
        path : str, optional
            Path to the file containing new channel information.
        new_time_data : np.ndarray, optional
            Array with new channel data.
        sampling_rate_hz : int, optional
            Sampling rate for the new data
        padding_trimming : bool, optional
            Activates padding or trimming at the end of signal in case the
            new data does not match previous data. Default: `True`.
        '''
        if path is not None:
            assert new_time_data is None, 'Only path or new time data is ' +\
                'accepted, not both.'
            new_time_data, sampling_rate_hz = sf.read(path)
        else:
            if new_time_data is not None:
                assert path is None, 'Only path or new time data is ' +\
                    'accepted, not both.'
        assert sampling_rate_hz == self.sampling_rate_hz, \
            f'{sampling_rate_hz} does not match {self.sampling_rate_hz} as ' +\
            'the sampling rate'
        if not type(new_time_data) == np.ndarray:
            new_time_data = np.array(new_time_data)
        assert len(new_time_data.shape) <= 2, \
            f'{len(new_time_data.shape)} has ' +\
            'too many dimensions for time data. Dimensions should' +\
            ' be (time samples, channels)'
        if len(new_time_data.shape) < 2:
            new_time_data = new_time_data[..., None]
        if new_time_data.shape[1] > new_time_data.shape[0]:
            new_time_data = new_time_data.T

        diff = new_time_data.shape[0] - self.time_data.shape[0]
        if diff != 0:
            if diff < 0:
                txt = 'Padding'
            else:
                txt = 'Trimming'
            if padding_trimming:
                new_time_data = \
                    _pad_trim(
                        new_time_data,
                        self.time_data.shape[0],
                        axis=0, in_the_end=True)
                warnings.warn(
                    f'{txt} has been performed ' +
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
        '''
        Removes a channel

        Parameters
        ----------
        channel_number : int, optional
            Channel number to be removed. Default: -1 (last).
        '''
        if channel_number == -1:
            channel_number = self.time_data.shape[1] - 1
        assert self.time_data.shape[1] > 1, \
            'Cannot not erase only channel'
        assert self.time_data.shape[1]-1 >= channel_number, \
            f'Channel number {channel_number} does not exist. Signal only ' +\
            f'has {self.number_of_channels-1} channels (zero included).'
        self.time_data = np.delete(self.time_data, channel_number, axis=-1)
        self.number_of_channels -= 1
        self.__update_state()

    def swap_channels(self, new_order):
        '''
        Rearranges the channels in the new given order.

        Parameters
        ----------
        new_order : array-like
            New rearrangement of channels.
        '''
        new_order = np.array(new_order).squeeze()
        assert new_order.ndim == 1, \
            'Too many dimensions are given in the new arrangement vector'
        assert self.number_of_channels == len(new_order), \
            'The number of channels does not match'
        assert all(new_order < self.number_of_channels) \
            and all(new_order >= 0), \
            'Indexes of new channels have to be in ' +\
            f'[0 and {self.number_of_channels-1}]'
        self.time_data = self.time_data[:, new_order]
        self.__update_state()

    # ======== Getters ========================================================
    def get_spectrum(self, force_computation=False):
        '''
        Returns spectrum.

        Parameters
        ----------
        force_computation : bool, optional
            Forces spectrum computation.

        Returns
        -------
        spectrum_freqs : np.ndarray
            Frequency vector
        spectrum : np.ndarray
            Spectrum matrix for each channel
        '''
        condition = not hasattr(self, 'spectrum') or \
            self.__spectrum_state_update or force_computation

        if condition:
            if self._spectrum_parameters['method'] == 'welch':
                spectrum = []
                for n in range(self.number_of_channels):
                    spectrum.append(
                        _welch(self.time_data[:, n],
                               self.time_data[:, n],
                               self.sampling_rate_hz,
                               self._spectrum_parameters['window_type'],
                               self.
                               _spectrum_parameters['window_length_samples'],
                               self._spectrum_parameters['overlap_percent'],
                               self._spectrum_parameters['detrend'],
                               self._spectrum_parameters['average'],
                               self._spectrum_parameters['scaling']))
                spectrum = np.array(spectrum).T
            elif self._spectrum_parameters['method'] == 'standard':
                spectrum = np.fft.rfft(self.time_data, axis=0)
            self.spectrum = []
            self.spectrum.append(
                np.fft.rfftfreq(
                    spectrum.shape[0]*2 - 1, 1/self.sampling_rate_hz))
            self.spectrum.append(spectrum)
            spectrum_freqs = self.spectrum[0]
            self.__spectrum_state_update = False
        else:
            spectrum_freqs, spectrum = self.spectrum[0], self.spectrum[1]
        return spectrum_freqs.copy(), spectrum.copy()

    def get_csm(self, force_computation=False):
        '''
        Get Cross spectral matrix for all channels with the shape
        (frequencies, channels, channels)

        Returns
        -------
        f_csm : np.ndarray
            Frequency vector
        csm : np.ndarray
            Cross spectral matrix with shape (frequency, channels, channels).
        '''
        assert self.number_of_channels > 1, \
            'Cross spectral matrix can only be computed when at least two ' +\
            'channels are available'
        condition = not hasattr(self, 'csm') or force_computation or \
            self.__csm_state_update

        if condition:
            self.csm = _csm(self.time_data,
                            self.sampling_rate_hz,
                            window_length_samples=self.
                            _csm_parameters['window_length_samples'],
                            window_type=self._csm_parameters['window_type'],
                            overlap_percent=self.
                            _csm_parameters['overlap_percent'],
                            detrend=self._csm_parameters['detrend'],
                            average=self._csm_parameters['average'],
                            scaling=self._csm_parameters['scaling'])
            self.__csm_state_update = False
        f_csm, csm = self.csm[0], self.csm[1]
        return f_csm, csm

    def get_spectrogram(self, channel_number: int = 0,
                        force_computation: bool = False):
        '''
        Returns a matrix containing the STFT of a specific channel.

        Parameters
        ----------
        channel_number : int, optional
            Channel number for which to compute the STFT. Default: 0.
        force_computation : bool, optional
            Forces new computation of the STFT. Default: False.

        Returns
        -------
        t_s : np.ndarray
            Time vector
        f_hz : np.ndarray
            Frequency vector
        spectrogram : np.ndarray
            Spectrogram
        '''
        condition = not hasattr(self, 'spectrogram') or force_computation or \
            self.__spectrogram_state_update or \
            not channel_number == \
            self._spectrogram_parameters['channel_number']

        if condition:
            self._spectrogram_parameters['channel_number'] = channel_number
            self.spectrogram = \
                _stft(
                    self.time_data[:, channel_number],
                    self.sampling_rate_hz,
                    self._spectrogram_parameters['window_length_samples'],
                    self._spectrogram_parameters['window_type'],
                    self._spectrogram_parameters['overlap_percent'],
                    self._spectrogram_parameters['detrend'],
                    self._spectrogram_parameters['padding'],
                    self._spectrogram_parameters['scaling']
                    )
            self.__spectrogram_state_update = False
        t_s, f_hz, spectrogram = \
            self.spectrogram[0], self.spectrogram[1], self.spectrogram[2]
        return t_s, f_hz, spectrogram

    def get_coherence(self):
        '''
        Returns the coherence matrix
        '''
        assert hasattr(self, 'coherence'), \
            'There is no coherence data saved in the Signal object'
        f, _ = self.get_spectrum()
        return f, self.coherence

    def get_time_vector(self):
        '''
        Returns the time vector associated with the signal
        '''
        if not hasattr(self, 'time_vector_s'):
            self._generate_time_vector()
        return self.time_vector_s

    # ======== Plots ==========================================================
    def plot_magnitude(self, range_hz=[20, 20e3], normalize: str = '1k',
                       smoothe=0, show_info_box: bool = False,
                       returns: bool = False):
        '''
        Plots magnitude spectrum.
        Change parameters of spectrum with set_spectrum_parameters.

        Parameters
        ----------
        range_hz : array-like with length 2, optional
            Range for which to plot the magnitude response.
            Default: [20, 20000].
        normalize : str, optional
            Mode for normalization, supported are `'1k'` for normalization
            with value at frequency 1 kHz or `'max'` for normalization with
            maximal value. Use `None` for no normalization. Default: `'1k'`.
        smoothe : int, optional --------> not yet implemented.
        show_info_box : bool, optional
            Plots a info box regarding spectrum parameters and plot parameters.
            If it is str, it overwrites the standard message.
            Default: `False`.
        returns : bool, optional
            When `True` figure and axis are returned. Default: `False`.

        Returns
        -------
        figure and axis when `returns = True`.
        '''
        f, sp = self.get_spectrum()
        f, mag_db = _get_normalized_spectrum(
            f=f,
            spectra=sp,
            mode=self._spectrum_parameters['method'],
            f_range_hz=range_hz,
            normalize=normalize,
            smoothe=smoothe)
        if show_info_box:
            txt = 'Info'
            txt += f'''\nMode: {self._spectrum_parameters['method']}'''
            txt += f'\nRange: [{range_hz[0]}, {range_hz[1]}]'
            txt += f'\nNormalized: {normalize}'
            txt += f'''\nSmoothing: {smoothe}'''
        else:
            txt = None
        fig, ax = general_plot(f, mag_db, range_hz, ylabel='Magnitude / dB',
                               info_box=txt, returns=True)
        if returns:
            return fig, ax

    def plot_time(self, returns: bool = False):
        '''
        Plots time signals
        '''
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
                           self.window * mx / 1.1, alpha=0.75)
            ax[n].set_ylim([-mx, mx])
        if returns:
            return fig, ax

    def plot_group_delay(self, range_hz=[20, 20000], returns: bool = False):
        '''
        Plots group delay of each channel.
        Only works if `signal_type in ('ir', 'h1', 'h2', 'h3', 'rir')`
        '''
        valid_signal_types = ('rir', 'ir', 'h1', 'h2', 'h3', 'chirp', 'noise')
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
        if returns:
            return fig, ax

    def plot_spectrogram(self, channel_number: int = 0, logfreqs: bool = True,
                         returns: bool = False):
        '''
        Plots STFT matrix of the given channel.
        '''
        t, f, stft = self.get_spectrogram(channel_number)
        epsilon = 10**(-100/10)
        ids = _find_nearest([20, 20000], f)
        if ids[0] == 0:
            ids[0] += 1
        f = f[ids[0]:ids[1]]
        stft = stft[ids[0]:ids[1], :]
        stft_db = 20*np.log10(np.abs(stft)+epsilon)
        stft_db = np.nan_to_num(stft_db, nan=np.min(stft_db))
        fig, ax = general_matrix_plot(
            matrix=stft_db, xrange=(t[0], t[-1]),
            yrange=(f[0], f[-1]), zrange=50,
            xlabel='Time / s', ylabel='Frequency / Hz', zlabel='dB',
            xlog=False, ylog=logfreqs,
            colorbar=True, returns=True)
        if returns:
            return fig, ax

    def plot_coherence(self, returns: bool = False):
        '''
        Plots coherence measurements if there are any
        '''
        assert hasattr(self, 'coherence'), \
            'There is no coherence data saved in the Signal object'

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
                ylims=[-0.1, 1.1],
                returns=True
                )
        if returns:
            return fig, ax

    def plot_phase(self, range_hz=[20, 20e3], unwrap: bool = False,
                   returns: bool = False):
        if self._spectrum_parameters['method'] == 'welch':
            raise AttributeError(
                'Phase cannot be plotted since the spectrum is ' +
                'welch. Please change spectrum parameters method to ' +
                'standard')
        else:
            f, sp = self.get_spectrum()
            ph = np.angle(sp)
            if unwrap:
                ph = np.unwrap(ph, axis=0)
            fig, ax = general_plot(
                f=f,
                matrix=ph,
                range_x=range_hz,
                labels=[f'Channel {n}'
                        for n in range(self.number_of_channels)],
                ylabel='Phase / rad',
                returns=True
            )
            if returns:
                return fig, ax

    def plot_csm(self, range_hz=[20, 20e3], logx: bool = True,
                 with_phase: bool = True, returns: bool = False):
        '''
        Plots the cross spectral matrix of the multichannel signal.

        Parameters
        ----------
        range_hz : array-like with length 2, optional
            Range of Hz to be showed. Default: [20, 20e3].
        logx : bool, optional
            Logarithmic x axis. Default: `True`.
        with_phase : bool, optional
            When `True`, the unwrapped phase is also plotted. Default: `True`.
        returns : bool, optional
            Gives back figure and axis objects. Default: `False`.

        Returns
        -------
        fig, ax if `returns = True`.
        '''
        f, csm = self.get_csm()
        fig, ax = _csm_plot(f, csm, range_hz, logx, with_phase, returns=True)
        if returns:
            return fig, ax

    # ======== Saving and export ==============================================
    def save_signal(self, path: str = 'signal', mode: str = 'wav'):
        '''
        Saves the Signal object

        Parameters
        ----------
        path : str, optional
            Path for the signal to be saved. Use only folder/folder/name
            (without format). Default: `'signal'`
            (local folder, object named signal).
        mode : str, optional
            Mode of saving. Available modes are `'wav'`, `'flac'`, `'pickle'`.
            Default: `'wav'`.
        '''
        if '.' in path.split(os.sep)[-1]:
            raise ValueError('Please introduce the saving path without format')
        if mode == 'wav':
            path += '.wav'
            sf.write(path, self.time_data, self.sampling_rate_hz)
        elif mode == 'flac':
            path += '.flac'
            sf.write(path, self.time_data, self.sampling_rate_hz)
        elif mode == 'pickle':
            path += '.pkl'
            with open(path, 'wb') as data_file:
                pickle.dump(self, data_file, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(
                f'{mode} is not a supported saving mode. Use ' +
                'wav, flac or pkl')


class MultiBandSignal():
    # ======== Constructor and initializers ===================================
    def __init__(self, bands=None, same_sampling_rate: bool = True,
                 info: dict = None):
        '''
        MultiBandSignal contains a composite band list where each index
        is a Signal object with the same number of channels. For multirate
        systems, the parameter `same_sampling_rate` has to be set to `False`.

        Parameters
        ----------
        bands : list or tuple, optional
            List or tuple containing different Signal objects. All of them
            should be associated to the same Signal. This means that the
            channel numbers have to match. Set to `None` for initializing the
            object. Default: `None`.
        same_sampling_rate : bool, optional
            When `True`, every Signal should have the same sampling rate.
            Set to `False` for a multirate system. Default: `True`.
        info : dict, optional
            A dictionary with generic information about the MultiBandSignal
            can be passed. Default: `None`.
        '''
        if bands is None:
            bands = []
        if info is None:
            info = {}
        assert type(bands) in (list, tuple), \
            'bands has to be a list, tuple or None'
        self.same_sampling_rate = same_sampling_rate
        if bands:
            for s in bands:
                assert type(s) == Signal, f'{type(s)} is not a valid ' +\
                    'band type. Use Signal objects'
                assert s.number_of_channels == self.number_of_channels, \
                    'Signals have different number of channels. This ' +\
                    'behaviour is not supported'
                assert s.signal_type == self.signal_type, \
                    'Signal types do not match'
            if self.same_sampling_rate:
                self.sampling_rate_hz = bands[0].sampling_rate_hz
                self.band_length_samples = bands[0].time_data.shape[0]
            # Check length and number of channels
            self.number_of_channels = bands[0].number_of_channels
            self.signal_type = bands[0].signal_type
            # Check sampling rate and duration
            if self.same_sampling_rate:
                for s in bands:
                    assert s.sampling_rate_hz == self.sampling_rate_hz, \
                        'Not all Signals have the same sampling rate. ' +\
                        'If you wish to create a multirate system, set ' +\
                        'same_sampling_rate to False'
                    assert s.time_data.shape[0] == self.band_length_samples,\
                        'The length of the bands is not always the same. ' +\
                        'This behaviour is not supported'
        self.bands = bands
        self._generate_metadata()
        self.info = self.info | info

    def _generate_metadata(self):
        '''
        Generates an information dictionary with metadata about the
        MultiBandSignal.
        '''
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
        '''
        Adds a new band to the MultiBandSignal.

        Parameters
        ----------
        sig : Signal
            Signal to be added.
        index : int, optional
            Index at which to insert the new Signal. Default: -1.
        '''
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
        '''
        Removes a band from the MultiBandSignal.

        Parameters
        ----------
        index : int, optional
            This is the index from the bands list at which the band
            will be erased. When -1, last band is erased.
            Default: -1.
        return_band : bool, optional
            When `True`, the erased band is returned. Default: `False`.
        '''
        assert self.bands, 'There are no filters to remove'
        if index == -1:
            index = len(self.bands) - 1
        assert index in range(len(self.bands)), \
            f'There is no band at index {index}.'
        f = self.bands.pop(index)
        self._generate_metadata()
        if return_band:
            return f

    def show_info(self, show_band_info: bool = False):
        '''
        Show information about the MultiBandSignal.

        Parameters
        ----------
        show_band_info : bool, optional
            When `True`, a longer message is printed with all available
            information regarding each Signal in the MultiBandSignal.
            Default: `True`.
        '''
        print()
        txt = ''
        for k in self.info:
            txt += \
                f''' | {str(k).replace('_', ' ').
                        capitalize()}: {self.info[k]}'''
        txt = 'Multiband band:' + txt
        print(txt)
        if show_band_info:
            print('-'*len(txt), end='')
            for ind, f1 in enumerate(self.bands):
                print()
                txt = f'Signal {ind}:'
                for kf in f1.info:
                    txt += \
                        f''' | {str(kf).replace('_', ' ').
                                capitalize()}: {f1.info[kf]}'''
                print(txt)
        print()

    # ======== Getters ========================================================
    def get_all_bands(self, channel: int = 0):
        '''
        Returns a signal with all bands as channels. Done for an specific
        channel.

        Parameters
        ----------
        channel : int, optional
            Channel to choose from the band signals.

        Returns
        -------
        sig : Signal or list of np.ndarray + dict
            Multichannel signal with all the bands. If the MultiBandSignal
            does not have the same sampling rate for all signals, a list with
            the time vectors and a dictionary containing their sampling rates
            with the key 'sampling_rates' are returned.
        '''
        if self.same_sampling_rate:
            new_time_data = \
                np.zeros((self.bands[0].time_data.shape[0], len(self.bands)))
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

    # ======== Saving and export ==============================================
    def save_signal(self, path: str = 'multibandsignal'):
        '''
        Saves the MultiBandSignal object as a pickle.

        Parameters
        ----------
        path : str, optional
            Path for the signal to be saved. Use only folder/folder/name
            (without format). Default: `'multibandsignal'`
            (local folder, object named multibandsignal).
        '''
        if '.' in path.split(os.sep)[-1]:
            raise ValueError('Please introduce the saving path without format')
        path += '.pkl'
        with open(path, 'wb') as data_file:
            pickle.dump(self, data_file, pickle.HIGHEST_PROTOCOL)
