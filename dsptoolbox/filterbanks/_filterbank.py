"""
Backend for the creation of specific filter banks
"""
import numpy as np
from warnings import warn
from os import sep
from pickle import dump, HIGHEST_PROTOCOL
from copy import deepcopy

from scipy.signal import (sosfilt, sosfilt_zi, butter, sosfiltfilt)
from dsptoolbox import Signal, MultiBandSignal, FilterBank

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
                 info: dict = {}):
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
        if type(order) == int:
            order = np.ones(len(freqs))*order
        freqs = np.atleast_1d(np.asarray(freqs).squeeze())
        order = np.atleast_1d(np.asarray(order).squeeze())
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
            The summed input.  ``summed.cshape`` matches the ``cshape`` or the
            original signal before it was filtered.

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
