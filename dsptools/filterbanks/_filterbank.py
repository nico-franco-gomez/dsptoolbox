"""
Backend for the creation of specific filter banks
"""
import numpy as np
from os import sep
from pickle import dump, HIGHEST_PROTOCOL
from scipy.signal import (sosfilt, sosfilt_zi, butter)
from dsptools import Signal, MultiBandSignal
from dsptools.generators import dirac
from dsptools.plots import general_plot
from dsptools._general_helpers import _get_normalized_spectrum


class LRFilterBank():
    """This is special crafted class for a Linkwitz-Riley crossovers filter
    bank since its implementation might be hard to generalize.

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
        freqs : array_like
            Frequencies at which to set the crossovers.
        order : array_like or int, optional
            Order for the crossover filters. If one `int` is passed, it is
            used for all the crossovers. Default: 4.
        sampling_rate_hz : int, optional
            Sampling rate for the filter bank. Default: 48000.
        info : dict, optional
            Dictionary containing additional information about the filter bank.
            It is empty by default and updated after the filter bank is
            created.

        """
        if type(order) == int:
            order = [order]*len(freqs)
        assert len(freqs) == len(order), \
            'Number of frequencies and number of order of the crossovers ' +\
            'do not match'
        for o in order:
            assert o % 2 == 0, 'Order of the crossovers has to be an ' +\
                'even number'
        self.freqs = np.sort(np.array(freqs).squeeze())
        self.order = np.sort(np.array(order).squeeze())
        self.number_of_cross = len(freqs)
        self.number_of_bands = self.number_of_cross + 1
        self.sampling_rate_hz = sampling_rate_hz
        #
        self._create_filters_sos()
        self._update_metadata()
        self.info = self.info | info

    def _update_metadata(self):
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
        for i3 in range(number_of_channels):
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
    def filter_signal(self, s: Signal, activate_zi: bool = False):
        """Filters a signal regarding the zi's of the filters and returns
        a MultiBandSignal. Only `'parallel'` mode is available for this type
        of filter bank.

        Parameters
        ----------
        s : Signal
            Signal to be filtered.
        activate_zi : bool, optional
            When `True`, the zi's are activated for filtering.
            Default: `False`.

        Returns
        -------
        outsig : MultiBandSignal
            A MultiBandSignal object containing all bands and all channels.

        """
        assert s.sampling_rate_hz == self.sampling_rate_hz, \
            'Sampling rates do not match'
        new_time_data = \
            np.zeros((s.time_data.shape[0],
                      s.number_of_channels,
                      self.number_of_bands))
        in_sig = s.time_data

        for ch in range(s.number_of_channels):
            if activate_zi:
                if not hasattr(self, 'channels_zi'):
                    self.initialize_zi(s.number_of_channels)
                elif len(self.channels_zi) != s.number_of_channels:
                    self.initialize_zi(s.number_of_channels)
                for cn in range(self.number_of_cross):
                    band, in_sig[:, ch] = \
                        self._two_way_split_zi(
                            in_sig[:, ch], channel_number=ch, cross_number=cn)
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
            else:
                for cn in range(self.number_of_cross):
                    band, in_sig[:, ch] = self._filt(in_sig[:, ch], cn)
                    for ap_n in range(cn+1, self.number_of_cross):
                        band = self._filt(band, ap_n, split=False)
                    new_time_data[:, ch, cn] = band
                # Last high frequency component
                new_time_data[:, ch, cn+1] = in_sig[:, ch]

        b = []
        for n in range(self.number_of_bands):
            b.append(Signal(None, new_time_data[:, :, n], s.sampling_rate_hz,
                            signal_type=s.signal_type))
        d = dict(
            readme='MultiBandSignal made using Linkwitz-Riley filter bank',
            filterbank_freqs=self.freqs,
            filterbank_order=self.order)
        out_sig = MultiBandSignal(b, True, d)
        return out_sig

    # ======== Update zi's and backend filtering ==============================
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
        s_l = sosfilt(self.sos[f_number][0], x=s)
        s_l = sosfilt(self.sos[f_number][0], x=s_l)
        # High band
        s_h = sosfilt(self.sos[f_number][1], x=s)
        s_h = sosfilt(self.sos[f_number][1], x=s_h)
        if split:
            return s_l, s_h
        else:
            return s_l + s_h

    # ======== Prints and plots ===============================================
    def plot_magnitude(self, range_hz=[20, 20e3], test_zi: bool = False,
                       returns: bool = False):
        """Plots the magnitude response of each filter. Only `'parallel'` mode
        is supported, thus no mode parameter can be set.

        Parameters
        ----------
        range_hz : array_like, optional
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
        d = dirac(
            length_samples=1024, number_of_channels=1, sampling_rate_hz=48000)
        bs = self.filter_signal(d, test_zi)
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
            raise ValueError('Please introduce the saving path without format')
        path += '.pkl'
        with open(path, 'wb') as data_file:
            dump(self, data_file, HIGHEST_PROTOCOL)


def fractional_octave_frequencies(
        num_fractions=1, frequency_range=(20, 20e3), return_cutoff=False):
    """Return the octave center frequencies according to the IEC 61260:1:2014
    standard.

    For numbers of fractions other than ``1`` and ``3``, only the
    exact center frequencies are returned, since nominal frequencies are not
    specified by corresponding standards.

    Parameters
    ----------
    num_fractions : int, optional
        The number of bands an octave is divided into. Eg., ``1`` refers to
        octave bands and ``3`` to third octave bands. The default is ``1``.
    frequency_range : array, tuple
        The lower and upper frequency limits, the default is
        ``frequency_range=(20, 20e3)``.

    Returns
    -------
    nominal : array, float
        The nominal center frequencies in Hz specified in the standard.
        Nominal frequencies are only returned for octave bands and third octave
        bands.
    exact : array, float
        The exact center frequencies in Hz, resulting in a uniform distribution
        of frequency bands over the frequency range.
    cutoff_freq : tuple, array, float
        The lower and upper critical frequencies in Hz of the bandpass filters
        for each band as a tuple corresponding to ``(f_lower, f_upper)``.
    """
    nominal = None

    f_lims = np.asarray(frequency_range)
    if f_lims.size != 2:
        raise ValueError(
            "You need to specify a lower and upper limit frequency.")
    if f_lims[0] > f_lims[1]:
        raise ValueError(
            "The second frequency needs to be higher than the first.")

    if num_fractions in [1, 3]:
        nominal, exact = _center_frequencies_fractional_octaves_iec(
            nominal, num_fractions)

        mask = (nominal >= f_lims[0]) & (nominal <= f_lims[1])
        nominal = nominal[mask]
        exact = exact[mask]

    else:
        exact = _exact_center_frequencies_fractional_octaves(
            num_fractions, f_lims)

    if return_cutoff:
        octave_ratio = 10**(3/10)
        freqs_upper = exact * octave_ratio**(1/2/num_fractions)
        freqs_lower = exact * octave_ratio**(-1/2/num_fractions)
        f_crit = (freqs_lower, freqs_upper)
        return nominal, exact, f_crit
    else:
        return nominal, exact


def _center_frequencies_fractional_octaves_iec(nominal, num_fractions):
    """Returns the exact center frequencies for fractional octave bands
    according to the IEC 61260:1:2014 standard.
    octave ratio
    .. G = 10^{3/10}
    center frequencies
    .. f_m = f_r G^{x/b}
    .. f_m = f_e G^{(2x+1)/(2b)}
    where b is the number of octave fractions, f_r is the reference frequency
    chosen as 1000Hz and x is the index of the frequency band.

    Parameters
    ----------
    num_fractions : 1, 3
        The number of octave fractions. 1 returns octave center frequencies,
        3 returns third octave center frequencies.

    Returns
    -------
    nominal : array, float
        The nominal (rounded) center frequencies specified in the standard.
        Nominal frequencies are only returned for octave bands and third octave
        bands
    exact : array, float
        The exact center frequencies, resulting in a uniform distribution of
        frequency bands over the frequency range.
    """
    if num_fractions == 1:
        nominal = np.array([
            31.5, 63, 125, 250, 500, 1e3,
            2e3, 4e3, 8e3, 16e3], dtype=float)
    elif num_fractions == 3:
        nominal = np.array([
            25, 31.5, 40, 50, 63, 80, 100, 125, 160,
            200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000,
            6300, 8000, 10000, 12500, 16000, 20000], dtype=float)

    reference_freq = 1e3
    octave_ratio = 10**(3/10)

    iseven = np.mod(num_fractions, 2) == 0
    if ~iseven:
        indices = np.around(
            num_fractions * np.log(nominal/reference_freq)
            / np.log(octave_ratio))
        exponent = (indices/num_fractions)
    else:
        indices = np.around(
            2.0*num_fractions *
            np.log(nominal/reference_freq) / np.log(octave_ratio) - 1)/2
        exponent = ((2*indices + 1) / num_fractions / 2)

    exact = reference_freq * octave_ratio**exponent

    return nominal, exact


def _exact_center_frequencies_fractional_octaves(
        num_fractions, frequency_range):
    """Calculate the center frequencies of arbitrary fractional octave bands.

    Parameters
    ----------
    num_fractions : int
        The number of fractions
    frequency_range
        The upper and lower frequency limits

    Returns
    -------
    exact : array, float
        An array containing the center frequencies of the respective fractional
        octave bands

    """
    ref_freq = 1e3
    Nmax = np.around(num_fractions*(np.log2(frequency_range[1]/ref_freq)))
    Nmin = np.around(num_fractions*(np.log2(ref_freq/frequency_range[0])))

    indices = np.arange(-Nmin, Nmax+1)
    exact = ref_freq * 2**(indices / num_fractions)

    return exact
