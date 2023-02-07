"""
Backend for the creation of specific filter banks
"""
import numpy as np
from warnings import warn
from os import sep
from pickle import dump, HIGHEST_PROTOCOL
from copy import deepcopy

from scipy.signal import (sosfilt, sosfilt_zi, butter, sosfiltfilt)
from dsptoolbox.classes import Signal, MultiBandSignal, FilterBank, Filter

from dsptoolbox.generators import dirac
from dsptoolbox.plots import general_plot
from dsptoolbox._general_helpers import _get_normalized_spectrum
from dsptoolbox._standard import _group_delay_direct


# ============== First implementation
class LRFilterBank():
    """This is specially crafted class for a Linkwitz-Riley crossovers filter
    bank since its implementation might be hard to generalize or inherit from
    the FilterBank class.

    It is a cascaded structure that handles every band and its respective
    initial values for the filters to work in streaming applications. Since
    the crossovers need allpasses at every other crossover frequency, the
    structure used for the zi's is very convoluted.

    """
    # ======== Constructor and initiliazers ===================================
    def __init__(self, freqs, order=4, sampling_rate_hz: int = 48000,
                 info: dict = None):
        """Constructor for a linkwitz-riley crossovers filter bank. This is a
        near perfect magnitude reconstruction filter bank.

        Parameters
        ----------
        freqs : array-like
            Frequencies at which to set the crossovers.
        order : array-like or int, optional
            Order for the crossover filters. If one `int` is passed, it is
            used for all the crossovers. Default: 4.
        sampling_rate_hz : int, optional
            Sampling rate for the filter bank. Default: 48000.
        info : dict, optional
            Dictionary containing additional information about the filter
            bank.
            It is empty by default and updated after the filter bank is
            created.

        """
        if info is None:
            info = {}
        if type(order) == int:
            order = np.ones(len(freqs))*order
        freqs = np.atleast_1d(np.asarray(freqs).squeeze())
        order = np.atleast_1d(np.asarray(order).squeeze())
        assert np.max(freqs) <= sampling_rate_hz//2, \
            'Highest frequency is above nyquist frequency for the given ' +\
            'sampling rate'
        assert len(freqs) == len(order), \
            'Number of frequencies and number of order of the crossovers ' +\
            'do not match'
        for o in order:
            assert o % 2 == 0, 'Order of the crossovers has to be an ' +\
                'even number'
        freqs_order = freqs.argsort()
        self.freqs = freqs[freqs_order]
        self.order = order[freqs_order]
        self.number_of_cross = len(freqs)
        self.number_of_bands = self.number_of_cross + 1
        self.sampling_rate_hz = sampling_rate_hz
        # Center Frequencies
        split_freqs = np.insert(self.freqs, 0, 0)
        split_freqs = np.insert(self.freqs, -1, sampling_rate_hz//2)
        self.center_frequencies = [(split_freqs[i+1]+split_freqs[i])/2
                                   for i in range(len(split_freqs)-1)]
        #
        self._create_filters_sos()
        self._generate_metadata()
        self.info = self.info | info

    def _generate_metadata(self):
        """Internal method to update metadata about the filter bank.

        """
        if not hasattr(self, 'info'):
            self.info = {}
        self.info['crossover_frequencies'] = self.freqs
        self.info['crossover_orders'] = self.order
        self.info['number_of_crossovers'] = self.number_of_cross
        self.info['number_of_bands'] = self.number_of_bands
        self.info['sampling_rate_hz'] = self.sampling_rate_hz

    def _create_filters_sos(self):
        """Creates and saves filter's sos representations in a list with
        ascending order.

        """
        self.sos = []
        for i in range(self.number_of_cross):
            lp = butter(self.order[i], self.freqs[i], btype='lowpass',
                        fs=self.sampling_rate_hz, output='sos')
            hp = butter(self.order[i], self.freqs[i], btype='highpass',
                        fs=self.sampling_rate_hz, output='sos')
            self.sos.append([lp, hp])

    def initialize_zi(self, number_of_channels: int = 1):
        """Initiates the zi of the filters for the given number of channels.

        Parameters
        ----------
        number_of_channels : int, optional
            Number of channels is needed for the number of filters' zi's.
            Default: 1.

        """
        # total signal separates low band from the rest (2 zi)
        # all_cross_zi = [[cross0_zi], [cross1_zi], [cross2_zi], ...]
        # cross0_zi = [[low_zi, high_zi], [allpass1_zi], [allpass2_zi], ...]
        # allpass1_zi = [low_zi, high_zi]
        self.channels_zi = []
        for _ in range(number_of_channels):
            cross_zi = []
            allpass_zi = []
            for i in range(self.number_of_cross):
                band_zi_l = sosfilt_zi(self.sos[i][0])  # Low band
                band_zi_h = sosfilt_zi(self.sos[i][1])  # High band
                cross_zi.append([band_zi_l, band_zi_h])
                al = []
                for i2 in range(self.number_of_cross):
                    allp_zi_l = sosfilt_zi(self.sos[i2][0])  # Low band
                    allp_zi_h = sosfilt_zi(self.sos[i2][1])  # High band
                    al.append([allp_zi_l, allp_zi_h])
                    allpass_zi.append(al)
            self.channels_zi.append([cross_zi, allpass_zi])

    # ======== Filtering ======================================================
    def filter_signal(self, s: Signal, mode: str = 'parallel',
                      activate_zi: bool = False, zero_phase: bool = False) \
            -> MultiBandSignal | Signal:
        """Filters a signal regarding the zi's of the filters and returns
        a MultiBandSignal. Only `'parallel'` mode is available for this type
        of filter bank.

        Parameters
        ----------
        s : `Signal`
            Signal to be filtered.
        mode : str, optional
            Way to apply filter bank to the signal. Supported modes are:
            `'parallel'`, `'summed'`. Default: `'parallel'`.
        activate_zi : bool, optional
            When `True`, the zi's are activated for filtering.
            Default: `False`.
        zero_phase : bool, optional
            Activates zero phase filtering. Default: `False`.

        Returns
        -------
        outsig : `MultiBandSignal`
            A MultiBandSignal object containing all bands and all channels.

        """
        mode = mode.lower()
        assert mode in ('parallel', 'sequential', 'summed'), \
            f'{mode} is not a valid mode. Use parallel, sequential or summed'
        if mode == 'sequential':
            warn('sequential mode is not supported for this filter bank. ' +
                 'It is automatically changed to summed')
            mode = 'summed'
        assert s.sampling_rate_hz == self.sampling_rate_hz, \
            'Sampling rates do not match'
        assert not (activate_zi and zero_phase), \
            'Zero phase filtering and activating zi is a valid setting'
        new_time_data = np.zeros((s.time_data.shape[0],
                                  s.number_of_channels,
                                  self.number_of_bands))
        in_sig = s.time_data

        # Filter with zi
        if activate_zi:
            if not hasattr(self, 'channels_zi'):
                self.initialize_zi(s.number_of_channels)
            elif len(self.channels_zi) != s.number_of_channels:
                self.initialize_zi(s.number_of_channels)
            for ch in range(s.number_of_channels):
                for cn in range(self.number_of_cross):
                    band, in_sig[:, ch] = \
                        self._two_way_split_zi(
                            in_sig[:, ch], channel_number=ch,
                            cross_number=cn)
                    # band, in_sig[:, ch] = self._filt(in_sig[:, ch], cn)
                    for ap_n in range(cn+1, self.number_of_cross):
                        band = \
                            self._allpass_zi(
                                band, channel_number=ch,
                                cross_number=cn, ap_number=ap_n)
                        # band = self._filt(band, ap_n, split=False)
                    new_time_data[:, ch, cn] = band
                # Last high frequency component
                new_time_data[:, ch, cn+1] = in_sig[:, ch]
        # Zero phase
        elif zero_phase:
            for cn in range(self.number_of_cross):
                new_time_data[:, :, cn] = \
                    sosfiltfilt(self.sos[cn][0], in_sig, axis=0)
                in_sig = sosfiltfilt(self.sos[cn][1], in_sig, axis=0)
            # Last high frequency component
            new_time_data[:, :, cn+1] = in_sig
        # Standard filtering
        else:
            for cn in range(self.number_of_cross):
                band, in_sig = self._filt(in_sig, cn)
                for ap_n in range(cn+1, self.number_of_cross):
                    band = self._filt(band, ap_n, split=False)
                new_time_data[:, :, cn] = band
            # Last high frequency component
            new_time_data[:, :, cn+1] = in_sig

        b = []
        for n in range(self.number_of_bands):
            b.append(Signal(None, new_time_data[:, :, n], s.sampling_rate_hz,
                            signal_type=s.signal_type))
        d = dict(
            readme='MultiBandSignal made using Linkwitz-Riley filter bank',
            filterbank_freqs=self.freqs,
            filterbank_order=self.order)
        out_sig = MultiBandSignal(bands=b, same_sampling_rate=True, info=d)
        if mode == 'summed':
            out_sig = out_sig.collapse()
        return out_sig

    # ======== Update zi's and backend filtering ============================
    def _allpass_zi(self, s, channel_number, cross_number, ap_number):
        # Unpack zi's
        ap_zi = self.channels_zi[channel_number][1][cross_number][ap_number]
        zi_l = ap_zi[0]
        zi_h = ap_zi[1]
        # Low band
        s_l, zi_l = sosfilt(self.sos[ap_number][0], x=s, zi=zi_l)
        s_l, zi_l = sosfilt(self.sos[ap_number][0], x=s_l, zi=zi_l)
        # High band
        s_h, zi_h = sosfilt(self.sos[ap_number][1], x=s, zi=zi_h)
        s_h, zi_h = sosfilt(self.sos[ap_number][1], x=s_h, zi=zi_h)
        # Pack zi's
        ap_zi[0] = zi_l
        ap_zi[1] = zi_h
        self.channels_zi[channel_number][1][cross_number][ap_number] = ap_zi
        # self.allpass_zi[cross_number][ap_number] = ap_zi
        return s_l + s_h

    def _two_way_split_zi(self, s, channel_number, cross_number):
        # Unpack zi's
        cross_zi = self.channels_zi[channel_number][0][cross_number]
        # cross_zi = self.cross_zi[cross_number]
        zi_l = cross_zi[0]
        zi_h = cross_zi[1]
        # Low band
        s_l, zi_l = sosfilt(self.sos[cross_number][0], x=s, zi=zi_l)
        s_l, zi_l = sosfilt(self.sos[cross_number][0], x=s_l, zi=zi_l)
        # High band
        s_h, zi_h = sosfilt(self.sos[cross_number][1], x=s, zi=zi_h)
        s_h, zi_h = sosfilt(self.sos[cross_number][1], x=s_h, zi=zi_h)
        # Pack zi's
        cross_zi[0] = zi_l
        cross_zi[1] = zi_h
        self.channels_zi[channel_number][0][cross_number] = cross_zi
        # self.cross_zi[cross_number] = cross_zi
        return s_l, s_h

    def _filt(self, s, f_number, split: bool = True):
        """Filters signal with the sos corresponding to f_number.
        `split=True` returns two bands; when `False`, the summed bands are
        returned (allpass).

        """
        # Low band
        s_l = sosfilt(self.sos[f_number][0], x=s, axis=0)
        s_l = sosfilt(self.sos[f_number][0], x=s_l, axis=0)
        # High band
        s_h = sosfilt(self.sos[f_number][1], x=s, axis=0)
        s_h = sosfilt(self.sos[f_number][1], x=s_h, axis=0)
        if split:
            return s_l, s_h
        else:
            return s_l + s_h

    # ======== IR =============================================================
    def get_ir(self, test_zi: bool = False):
        """Returns impulse response from the filter bank. For this filter
        bank only `mode='parallel'` is valid and there is no zero phase
        filtering.

        Parameters
        ----------
        mode : str, optional
            Way to apply filter bank to the signal. Supported modes are:
            `'parallel'`, `'summed'`. Default: `'parallel'`.
        test_zi : bool, optional
            When `True`, filtering is done while updating filters' initial
            values. Default: `False`.

        Returns
        -------
        ir : `MultiBandSignal` or `Signal`
            Impulse response of the filter bank.

        """
        d = dirac(
            length_samples=1024,
            number_of_channels=1)
        ir = self.filter_signal(
            d, activate_zi=test_zi)
        return ir

    # ======== Prints and plots ===============================================
    def plot_magnitude(self, range_hz=[20, 20e3], mode: str = 'parallel',
                       test_zi: bool = False, zero_phase: bool = False,
                       returns: bool = False):
        """Plots the magnitude response of each filter. Only `'parallel'`
        mode is supported, thus no mode parameter can be set.

        Parameters
        ----------
        range_hz : array_like, optional
            Range of Hz to plot. Default: [20, 20e3].
        mode : str, optional
            Way to apply filter bank to the signal. Supported modes are:
            `'parallel'`. Default: `'parallel'`.
        zero_phase : bool, optional
            Activates zero phase filtering. Default: `False`.
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
        mode = mode.lower()
        if mode != 'parallel':
            warn('Plotting for LRFilterBank is only supported with parallel ' +
                 'mode. Setting to parallel')
        d = dirac(
            length_samples=1024, number_of_channels=1,
            sampling_rate_hz=48000)
        bs = self.filter_signal(d, mode='parallel', activate_zi=test_zi,
                                zero_phase=zero_phase)
        specs = []
        f = bs.bands[0].get_spectrum()[0]
        summed = []
        for b in bs.bands:
            b.set_spectrum_parameters(method='standard')
            summed.append(b.time_data[:, 0])
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
                                       for h in range(bs.number_of_bands)],
                               range_y=[-30, 10])
        # Summed signal
        summed = np.sum(np.array(summed).T, axis=1)
        sp_summed = np.fft.rfft(summed)
        f_s, sp_summed = \
            _get_normalized_spectrum(
                f, sp_summed,
                f_range_hz=range_hz,
                normalize=None)
        ax.plot(f_s, sp_summed, alpha=0.7, linestyle='dashed',
                label='Summed signal')
        ax.legend()
        if returns:
            return fig, ax

    def plot_phase(self, range_hz=[20, 20e3], mode: str = 'parallel',
                   test_zi: bool = False, zero_phase: bool = True,
                   unwrap: bool = False, returns: bool = False):
        """Plots the phase response of each filter.

        Parameters
        ----------
        mode : str, optional
            Way to apply filter bank to the signal. Supported modes are:
            `'parallel'`. Default: `'parallel'`.
        range_hz : array-like, optional
            Range of Hz to plot. Default: [20, 20e3].
        test_zi : bool, optional
            Uses the zi's of each filter to test the FilterBank's output.
            Default: `False`.
        zero_phase : bool, optional
            Activates zero phase filtering. Default: `False`.
        unwrap : bool, optional
            When `True`, unwrapped phase is plotted. Default: `False`.
        returns : bool, optional
            When `True`, the figure and axis are returned. Default: `False`.

        Returns
        -------
        fig, ax
            Returned only when `returns=True`.

        """
        mode = mode.lower()
        assert mode in ('parallel', 'summed'), \
            f'{mode} is not supported. Use either parallel or summed'
        length_samples = 1024
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1, sampling_rate_hz=48000)

        if mode == 'parallel':
            bs = self.filter_signal(d, mode='parallel', activate_zi=test_zi,
                                    zero_phase=zero_phase)
            phase = []
            f = bs.bands[0].get_spectrum()[0]
            for b in bs.bands:
                phase.append(np.angle(b.get_spectrum()[1]))
            phase = np.squeeze(np.array(phase).T)
            labels = [f'Filter {h}' for h in range(bs.number_of_bands)]
        elif mode == 'summed':
            bs = self.filter_signal(d, mode='summed', activate_zi=test_zi,
                                    zero_phase=zero_phase)
            f, phase = bs.get_spectrum()
            phase = np.angle(phase)
            labels = ['Summed']

        if unwrap:
            phase = np.unwrap(phase, axis=0)
        fig, ax = general_plot(f, phase, range_hz, ylabel='Phase / rad',
                               returns=True,
                               labels=labels)
        if returns:
            return fig, ax

    def plot_group_delay(self, range_hz=[20, 20e3], mode: str = 'parallel',
                         test_zi: bool = False, zero_phase: bool = False,
                         returns: bool = False):
        """Plots the phase response of each filter.

        Parameters
        ----------
        mode : str, optional
            Way to apply filter bank to the signal. Supported modes are:
            `'parallel'`. Default: `'parallel'`.
        range_hz : array-like, optional
            Range of Hz to plot. Default: [20, 20e3].
        test_zi : bool, optional
            Uses the zi's of each filter to test the FilterBank's output.
            Default: `False`.
        zero_phase : bool, optional
            Activates zero phase filtering. Default: `False`.
        returns : bool, optional
            When `True`, the figure and axis are returned. Default: `False`.

        Returns
        -------
        fig, ax
            Returned only when `returns=True`.

        """
        mode = mode.lower()
        assert mode in ('parallel', 'summed'), \
            f'{mode} is not supported. Use either parallel or summed'
        length_samples = 1024
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1, sampling_rate_hz=48000)
        if mode == 'parallel':
            bs = self.filter_signal(d, mode='parallel', activate_zi=test_zi,
                                    zero_phase=zero_phase)
            gd = []
            f = bs.bands[0].get_spectrum()[0]
            for b in bs.bands:
                gd.append(_group_delay_direct(
                    np.squeeze(b.get_spectrum()[1]), delta_f=f[1]-f[0]))
            gd = np.squeeze(np.array(gd).T)*1e3
            labels = [f'Filter {h}' for h in range(bs.number_of_bands)]
        elif mode == 'summed':
            bs = self.filter_signal(d, mode='summed', activate_zi=test_zi,
                                    zero_phase=zero_phase)
            f, sp = bs.get_spectrum()
            gd = _group_delay_direct(sp.squeeze(), delta_f=f[1]-f[0])
            labels = ['Summed']
        fig, ax = general_plot(f, gd, range_hz, ylabel='Group delay / ms',
                               returns=True,
                               labels=labels)
        if returns:
            return fig, ax

    def show_info(self):
        """Prints out information about the filter bank.

        """
        print()
        for k in self.info.keys():
            print(str(k).replace('_', ' ').capitalize(), end='')
            print(f': {self.info[k]}')

    def save_filterbank(self, path: str = 'filterbank'):
        """Saves the FilterBank object as a pickle.

        Parameters
        ----------
        path : str, optional
            Path for the filterbank to be saved. Use only folder/folder/name
            (without format). Default: `'filterbank'`
            (local folder, object named filterbank).

        """
        if '.' in path.split(sep)[-1]:
            raise ValueError('Please introduce the saving path without ' +
                             'format')
        path += '.pkl'
        with open(path, 'wb') as data_file:
            dump(self, data_file, HIGHEST_PROTOCOL)

    def copy(self):
        """Returns a copy of the object.

        Returns
        -------
        new_sig : `LRFilterBank`
            Copy of filter bank.

        """
        return deepcopy(self)


class GammaToneFilterBank(FilterBank):
    """This class extends the FilterBank to the reconstruction
    possibilites of the Gamma Tone Filter Bank."""
    def __init__(self, filters: list, info: dict, frequencies: np.ndarray,
                 coefficients: np.ndarray, normalizations: np.ndarray):
        """Constructor for the Gamma Tone Filter Bank. It is only available as
        a constant sampling rate filter bank.

        Parameters
        ----------
        filters : list
            List with gamma tone filters.
        info : dict
            Dictionary containing basic information about the filter bank.
        frequencies : `np.ndarray`
            Frequencies used for the filters.
        coefficients : `np.ndarray`
            Filter coefficients.
        normalizations : `np.ndarray`
            Normalizations.

        """
        super().__init__(filters, same_sampling_rate=True, info=info)

        self._frequencies = frequencies
        self._coefficients = coefficients
        self._normalizations = normalizations

        self._delay = 0.004  # Delay in ms
        self._compute_delays_and_phase_factors()
        self._compute_gains()

    # ======== Extra methods for the GammaToneFilterBank ======================
    def _compute_delays_and_phase_factors(self):
        """Section 4 in Hohmann 2002 describes how to derive these values. This
        is a direct Python port of the corresponding function in the AMT
        toolbox `hohmann2002_process.m`.

        """
        # the delay in samples
        delay_samples = int(np.round(self._delay * self.sampling_rate_hz))

        # apply filterbank to impulse to estimate the required values
        d = dirac(delay_samples + 3, sampling_rate_hz=self.sampling_rate_hz)
        d = self.filter_signal(d, mode='parallel')
        d = d.get_all_bands(channel=0)
        real, imag = d.time_data, d.time_data_imaginary

        real = real.T
        imag = imag.T
        ir = real + 1j * imag
        env = np.abs(ir)

        # sample at which the maximum occurs
        # (excluding last sample for a safe calculation of the slope below)
        idx_max = np.argmax(env[:, :delay_samples + 1], axis=-1)
        delays = delay_samples - idx_max

        # calculate the phase factor from the slopes
        slopes = np.array([ir[bb, idx + 1] - ir[bb, idx - 1]
                           for bb, idx in enumerate(idx_max)])

        phase_factors = 1j / (slopes / np.abs(slopes))

        self._delays = delays
        self._phase_factors = phase_factors

    def _compute_gains(self):
        """Section 4 in Hohmann 2002 describes how to derive these values. This
        is a direct Python port of the corresponding function in the AMT
        toolbox `hohmann2002_process.m`.

        """
        # positive and negative center frequencies in the z-plane
        z = np.atleast_2d(
            np.exp(2j * np.pi * self._frequencies / self.sampling_rate_hz)).T
        z_conj = np.conjugate(z)

        # calculate transfer function at all center frequencies for all bands
        # (matrixes contain center frequencies along first dimension and
        h_pos = (1 - np.atleast_2d(self._coefficients) / z)**(-4) * \
            np.atleast_2d(self._normalizations)
        h_neg = (1 - np.atleast_2d(self._coefficients) / z_conj)**(-4) * \
            np.atleast_2d(self._normalizations)

        # apply delay and phase correction
        phase_factors = np.atleast_2d(self._phase_factors)
        delays = np.atleast_2d(self._delays)
        h_pos *= phase_factors * z**(-delays)
        h_neg *= phase_factors * np.conjugate(z)**(-delays)

        # combine positive and negative spectrum
        h = (h_pos + np.conjugate(h_neg)) / 2

        # iteratively find gains
        gains = np.ones((self.number_of_filters, 1))
        for ii in range(100):
            h_fin = np.matmul(h, gains)
            gains /= np.abs(h_fin)

        self._gains = gains.flatten()

    # ======== Reconstruct signal =============================================
    def reconstruct(self, signal: MultiBandSignal) -> Signal:
        """This method reconstructs a signal filtered with the gamma tone
        filter bank. The passed signal should be a MultiBandSignal and have
        imaginary time data.

        The summation process is described in Section 4 of Hohmann 2002 and
        uses the pre-calculated delays, phase factors and gains.

        Parameters
        ----------
        signal : `MultiBandSignal`
            Signal with multiple bands and imaginary time data needed to
            reconstruct the original filtered signal.

        Returns
        -------
        reconstructed_sig : `Signal`
            The summed input.

        """
        condition = all([signal.bands[n].time_data_imaginary is not None
                         for n in range(signal.number_of_bands)])
        assert condition, \
            'Not all bands have imaginary time data. Reconstruction cannot ' +\
            'be done'
        shape = (signal.number_of_bands,
                 signal.bands[0].time_data.shape[0],
                 signal.number_of_channels)
        # (bands, time samples, channels)
        time = np.empty(shape, dtype='cfloat')
        for ind, b in enumerate(signal.bands):
            time[ind, :, :] = b.time_data + b.time_data_imaginary * 1j

        # Reordering axis for later to (bands, channels, time samples) or
        # (bands, time samples) when single channel
        if time.shape[-1] == 1:
            time = time.squeeze()
        else:
            time = np.moveaxis(time, -1, 1)

        reconstructed_sig = signal.bands[0].copy()

        # apply phase shift, delay, and gain
        for bb, (phase_factor, delay, gain) in enumerate(zip(
                self._phase_factors, self._delays, self._gains)):

            time[bb] = \
                np.real(np.roll(time[bb], delay, axis=-1) * phase_factor) * \
                gain

        # sum and squeeze first axis (the signal is already real, but the data
        # type is still complex)
        reconstructed_sig.time_data = \
            np.sum(np.real(time), axis=0)
        return reconstructed_sig


class BaseCrossover(FilterBank):
    """This base class for crossovers is used to hold all maximally decimated
    crossover methods together.

    """
    def __init__(self, analysis_filters: list, synthesis_filters: list,
                 info: dict = None):
        """Constructor for a crossover. Analysis and synthesis filters are
        needed for creating an instance.

        Parameters
        ----------
        analysis_filters : list
            List containing two filters that correspond to lowpass and highpass
            analysis filters.
        synthesis_filters : list
            List containing two filters that correspond to low- and highpass
            synthesis filters
        info : dict, optional
            Dictionary containing generic information about the crossover.

        Attributes and methods
        ----------------------
        - All of attributes and methods of `FilterBank`.
        - `filters_synthesis`: list with synthesis filters.
        - `reconstruct_signal()`: reconstructs a signal by taking in a multi-
          band signal.

        """
        assert len(analysis_filters) == 2, \
            'Exactly two filters are needed for a valid crossover'
        self.filters_synthesis = synthesis_filters
        super().__init__(filters=analysis_filters,
                         same_sampling_rate=True, info=info)

    # ======== Extra properties ===============================================
    @property
    def filters_synthesis(self):
        return self.__filters_synthesis

    @filters_synthesis.setter
    def filters_synthesis(self, new_filters):
        assert len(new_filters) == 2, \
            'Two synthesis filters are needed in a crossover'
        assert all([type(n) == Filter for n in new_filters]), \
            'Filters have to be of type Filter'
        self.__filters_synthesis = new_filters

    # ======== Filtering ======================================================
    def filter_signal(self, signal: Signal, mode: str = 'parallel',
                      activate_zi: bool = False, downsample: bool = True) \
            -> Signal | MultiBandSignal:
        if not downsample:
            return super().filter_signal(
                signal, mode, activate_zi, zero_phase=False)
        # ========== In case of downsampling while filtering ==================
        mode = mode.lower()
        assert mode in ('parallel', 'sequential', 'summed'), \
            f'{mode} is not a valid mode. Use parallel, sequential or summed'
        assert signal.sampling_rate_hz == self.sampling_rate_hz, \
            'Sampling rates do not match'
        if activate_zi:
            if len(self.filters[0].zi) != signal.number_of_channels:
                self.initialize_zi(signal.number_of_channels)
        new_sig = _crossover_downsample(
            signal, self.filters, mode=mode, down_factor=2)
        return new_sig

    # ======== Reconstructing =================================================
    def reconstruct_signal(self, signal: MultiBandSignal,
                           upsample: bool = True):
        """Reconstructs a two band signal using the synthesis filters of the
        crossover.

        Parameters
        ----------
        signal : `MultiBandSignal`
            Multi-band signal containing two bands from which to reconstruct
            the original signal. It is assumed that the first band is the
            low-frequency content.
        upsample : bool, optional
            When `True`, the signal's sampling rate is doubled.
            Default: `True`.

        Returns
        -------
        reconstructed : `Signal`
            Reconstructed signal.

        """
        assert signal.number_of_bands == 2, \
            'There must be exactly two bands in order to reconstruct ' +\
            'signal using a crossover'
        uf = 2 if upsample else 1
        return _reconstruct_from_crossover_upsample(
            signal.bands[0], signal.bands[1], self.filters_synthesis,
            up_factor=uf)

    # ======== Plotting =======================================================
    def plot_magnitude(self, mode: str = 'parallel', range_hz=[20, 20e3],
                       length_samples: int = 2048, test_zi: bool = False,
                       downsample: bool = True):
        if not downsample:
            return super().plot_magnitude(
                mode, range_hz, length_samples, test_zi)

        # If downsampling is activated
        max_order = 0
        for b in self.filters:
            max_order = max(max_order, b.info['order'])
        if max_order > length_samples:
            warn(f'Filter order {max_order} is longer than {length_samples}.' +
                 ' The length will be adapted to be 100 samples longer than' +
                 ' the longest filter')
            length_samples = max_order + 100

        # Impulse
        d = dirac(
            length_samples=length_samples,
            number_of_channels=1, sampling_rate_hz=self.sampling_rate_hz)

        # Filtering and plot
        if mode == 'parallel':
            bs = self.filter_signal(d, mode='parallel',
                                    activate_zi=test_zi, downsample=True)
            specs = []
            f = bs.bands[0].get_spectrum()[0]
            for b in bs.bands:
                b.set_spectrum_parameters(method='standard')
                f, sp = _get_normalized_spectrum(
                    f, np.squeeze(b.get_spectrum()[1]), f_range_hz=range_hz,
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
                                   range_y=range_y,
                                   tight_layout=False)
        elif mode == 'sequential':
            bs = self.filter_signal(d, mode='sequential',
                                    activate_zi=test_zi, downsample=True)
            bs.set_spectrum_parameters(method='standard')
            f, sp = bs.get_spectrum()
            f, sp = _get_normalized_spectrum(
                f, np.squeeze(sp), f_range_hz=range_hz, normalize=None)
            fig, ax = general_plot(
                f, sp, range_hz, ylabel='Magnitude / dB',
                returns=True,
                labels=[f'Sequential - Channel {n}'
                        for n in range(bs.number_of_channels)])
        elif mode == 'summed':
            bs = self.filter_signal(d, mode='summed',
                                    activate_zi=test_zi, downsample=True)
            bs.set_spectrum_parameters(method='standard')
            f, sp = bs.get_spectrum()
            f, sp = _get_normalized_spectrum(
                f, np.squeeze(sp), f_range_hz=range_hz, normalize=None)
            fig, ax = general_plot(
                f, sp, range_hz, ylabel='Magnitude / dB', returns=True,
                labels=['Summed'])
        return fig, ax


class QMFCrossover(BaseCrossover):
    """This class contains methods for the creation of quadrature mirror
    filters, with which near-perfect signal reconstruction can be achieved.

    """
    def __init__(self, lowpass: Filter):
        """Create a quadrature mirror filters crossover based on a lowpass
        filter prototype.

        Parameters
        ----------
        lowpass : `Filter`
            Lowpass filter prototype.

        References
        ----------
        - https://tinyurl.com/2a3frbyv

        """
        if 'freqs' in lowpass.info:
            if lowpass.info['freqs'] != lowpass.sampling_rate_hz//4:
                warn('Cut-off frequency for lowpass filter should be half ' +
                     'the nyquist frequency.')
        # Create FilterBank
        super().__init__(
            analysis_filters=self._get_analysis_filters(lowpass),
            synthesis_filters=self._get_synthesis_filters(lowpass),
            info=dict(Info='Quadrature mirror filters crossover'))

    def _get_analysis_filters(self, lowpass: Filter):
        """Create and return analysis filters based on a lowpass prototype.

        Parameters
        ----------
        lowpass : `Filter`
            Lowpass prototype.

        Returns
        -------
        analysis_filters : list
            List containing exactly two analysis filters. The first one is
            the passed lowpass prototype.

        References
        ----------
        - https://tinyurl.com/2a3frbyv

        """
        if lowpass.filter_type == 'fir':
            # Create highpass filter based on lowpass
            b_base, _ = lowpass.get_coefficients(mode='ba')
            b_high = b_base.copy()
            # H1(z) = H0(-z) <-> odd coefficients are multiplied by -1
            b_high[1::2] *= -1
            # Create filter
            highpass = Filter(
                'other', dict(ba=[b_high, [1]]),
                sampling_rate_hz=lowpass.sampling_rate_hz)
            # Type of filter bank
            self.fir_filterbank = True
        else:
            z_base, p_base, k_base = lowpass.get_coefficients(mode='zpk')
            zpk_new = [z_base*-1, p_base*-1, k_base]
            highpass = Filter(
                'other', dict(zpk=zpk_new),
                sampling_rate_hz=lowpass.sampling_rate_hz)
            # Type of filter bank
            self.fir_filterbank = False
        return [lowpass, highpass]

    def _get_synthesis_filters(self, lowpass: Filter):
        """Create and return synthesis filters based on a lowpass prototype.

        Parameters
        ----------
        lowpass : `Filter`
            Lowpass prototype.

        Returns
        -------
        synthesis_filters : list
            List containing exactly two synthesis filters. The first one is
            the passed lowpass prototype.

        References
        ----------
        - https://tinyurl.com/2a3frbyv

        """
        b, a = lowpass.get_coefficients(mode='ba')
        hp_filter = Filter('other', dict(ba=[-b, a]),
                           sampling_rate_hz=lowpass.sampling_rate_hz)
        return [lowpass, hp_filter]

# Not Working so far
# class CQFCrossover(BaseCrossover):
#     """This class contains methods for the creation of conjugate quadrature
#     filters, with which perfect (magnitude) reconstruction can be achieved.
#     Since these filters do not generally have linear phase, there might be
#     some phase distortion after reconstruction.

#     """
#     def __init__(self, lowpass: Filter):
#         """Create a conjugate quadrature filters crossover based on a lowpass
#         filter prototype.

#         Parameters
#         ----------
#         lowpass : `Filter`
#             Lowpass filter prototype.

#         References
#         ----------
#         - https://tinyurl.com/2cssq2oa

#         """
#         if 'freqs' in lowpass.info:
#             if lowpass.info['freqs'] != lowpass.sampling_rate_hz//4:
#                 warn('Cut-off frequency for lowpass filter should be half ' +
#                      'the nyquist frequency.')
#         # Create FilterBank
#         super().__init__(
#             analysis_filters=self._get_analysis_filters(lowpass),
#             synthesis_filters=self._get_synthesis_filters(lowpass),
#             info=dict(Info='Quadrature mirror filters crossover'))

#     def _get_analysis_filters(self, lowpass: Filter):
#         """Create and return analysis filters based on a lowpass prototype.

#         Parameters
#         ----------
#         lowpass : `Filter`
#             Lowpass prototype.

#         Returns
#         -------
#         analysis_filters : list
#             List containing exactly two analysis filters. The first one is
#             the passed lowpass prototype.

#         References
#         ----------
#         - https://tinyurl.com/2cssq2oa

#         """
#         if lowpass.filter_type == 'fir':
#             # Create highpass filter based on lowpass
#             b_base, _ = lowpass.get_coefficients(mode='ba')
#             b_high = b_base.copy()
#             b_high = np.flip(b_high)
#             b_high[::2] *= -1
#             # Create filter
#             highpass = Filter(
#                 'other', dict(ba=[b_high, [1]]),
#                 sampling_rate_hz=lowpass.sampling_rate_hz)
#             # Type of filter bank
#             self.fir_filterbank = True
#         else:
#             raise ValueError('This type of crossover is only supported ' +
#                              'for FIR filters')
#         return [lowpass, highpass]

#     def _get_synthesis_filters(self, lowpass: Filter):
#         """Create and return synthesis filters based on a lowpass prototype.

#         Parameters
#         ----------
#         lowpass : `Filter`
#             Lowpass prototype.

#         Returns
#         -------
#         synthesis_filters : list
#             List containing exactly two synthesis filters.

#         References
#         ----------
#         - https://tinyurl.com/2cssq2oa

#         """
#         b, a = lowpass.get_coefficients(mode='ba')
#         lp_filter = Filter('other', dict(ba=[np.flip(b), a]),
#                            sampling_rate_hz=lowpass.sampling_rate_hz)
#         b[::2] *= -1
#         hp_filter = Filter('other', dict(ba=[b, a]),
#                            sampling_rate_hz=lowpass.sampling_rate_hz)
#         return [lp_filter, hp_filter]


def _crossover_downsample(signal: Signal, filters: list, mode: str,
                          down_factor: int = 2) -> Signal | MultiBandSignal:
    """Apply crossover and downsample on signal.

    Parameters
    ----------
    signal : `Signal`
        Signal to which to apply the crossover.
    filters : list
        List containing filters to use. Since it is a crossover, it should
        have 2 filters.
    mode : str
        Mode of filtering. Choose from `'parallel'`, `'sequential'` and
        `'summed'`.
    down_factor : int, optional
        Down factor for decimation. Default: 2.

    Returns
    -------
    new_signal : `Signal` or `MultiBandSignal`
        New Signal object.

    """
    n_filt = len(filters)
    assert n_filt == 2, \
        'A crossover should contain exactly 2 filters'
    if mode == 'parallel':
        ss = []
        for n in range(n_filt):
            ss.append(filters[n].filter_and_resample_signal(
                signal,
                new_sampling_rate_hz=signal.sampling_rate_hz//down_factor))
        out_sig = MultiBandSignal(
            ss, same_sampling_rate=True)
    elif mode == 'sequential':
        out_sig = signal.copy()
        for n in range(n_filt):
            out_sig = filters[n].filter_and_resample_signal(
                signal,
                new_sampling_rate_hz=signal.sampling_rate_hz//down_factor)
    else:
        new_time_data = \
            np.zeros((signal.time_data.shape[0]//down_factor,
                      signal.number_of_channels, n_filt))
        for n in range(n_filt):
            s = filters[n].filter_and_resample_signal(
                signal,
                new_sampling_rate_hz=signal.sampling_rate_hz//down_factor)
            new_time_data[:, :, n] = s.time_data
        new_time_data = np.sum(new_time_data, axis=-1)
        out_sig = signal.copy()
        out_sig.sampling_rate_hz = signal.sampling_rate_hz//down_factor
        out_sig.time_data = new_time_data
    return out_sig


def _reconstruct_from_crossover_upsample(sig_low: Signal, sig_high: Signal,
                                         filters: list,
                                         up_factor: int = 2) -> Signal:
    """Reconstructs signal from crossover.

    Parameters
    ----------
    sig_low : `Signal`
        Low-frequency band.
    sig_high : `Signal`
        High-frequency band.
    filters : list
        List containing the two synthesis filters.
    up_factor : int, optional
        Factor by which to upsample. Default: 2.

    Returns
    -------
    rec_sig : `Signal`
        Reconstructed signal.

    """
    n_filt = len(filters)
    assert n_filt == 2, \
        'A crossover should contain exactly 2 filters'
    rec_sig = filters[0].filter_and_resample_signal(
        sig_low, new_sampling_rate_hz=sig_low.sampling_rate_hz*up_factor)
    temp_sig = filters[1].filter_and_resample_signal(
        sig_high, new_sampling_rate_hz=sig_low.sampling_rate_hz*up_factor)
    rec_sig.time_data += temp_sig.time_data
    return rec_sig
