'''
Backend for the creation of specific filter banks
'''
import numpy as np
import scipy.signal as sig
from ..signal_class import Signal, MultiBandSignal


class LRFilterBank():
    '''
    This is special crafted class for a Linkwitz-Riley crossovers filter
    bank since its implementation might be hard to generalize.

    It is a cascaded structure that handles every band and its respective
    initial values for the filters to work in streaming applications. Since
    the crossovers need allpasses at every other crossover frequency, the
    structure used for the zi's is very convoluted.
    '''
    def __init__(self, freqs, order, sampling_rate_hz: int = 48000):
        if type(order) == int:
            order = [order]*len(freqs)
        assert len(freqs) == len(order), \
            'Number of frequencies and number of order of the crossovers ' +\
            'do not match'
        for o in order:
            assert o % 2 == 0, 'Order of the crossovers has to be an ' +\
                'even number'
        self.freqs = np.array(freqs).squeeze()
        self.order = np.array(order).squeeze()
        self.number_of_cross = len(freqs)
        self.number_of_bands = self.number_of_cross + 1
        self.sampling_rate_hz = sampling_rate_hz
        #
        self._create_filters_sos()
        #
        self.initialize_zi()

    def _create_filters_sos(self):
        '''
        Creates and saves filter's sos representations in a list with
        ascending order
        '''
        self.sos = []
        for i in range(self.number_of_cross):
            lp = sig.butter(self.order[i], self.freqs[i], btype='lowpass',
                            fs=self.sampling_rate_hz, output='sos')
            hp = sig.butter(self.order[i], self.freqs[i], btype='highpass',
                            fs=self.sampling_rate_hz, output='sos')
            self.sos.append([lp, hp])

    def initialize_zi(self):
        '''
        Initializes zi
        '''
        # total signal separates low band from the rest (2 zi)
        # all_cross_zi = [[cross0_zi], [cross1_zi], [cross2_zi], ...]
        # cross0_zi = [low_zi, high_zi, [allpass1_zi], [allpass2_zi], ...]
        # allpass1_zi = [low_zi, high_zi]

        # There's always one allpass less than the number of cross because
        # no band needs it in the first crossover.

        # With ascending cross, the number of allpasses decreases until is zero
        self.all_cross_zi = []
        for i in range(self.number_of_cross):
            band_zi_l = sig.sosfilt_zi(self.sos[i][0])  # Low band
            band_zi_h = sig.sosfilt_zi(self.sos[i][1])  # High band
            allpasses = []
            for i2 in range(i+1, self.number_of_cross):
                allp_zi_l = sig.sosfilt_zi(self.sos[i2][0])  # Low band
                allp_zi_h = sig.sosfilt_zi(self.sos[i2][1])  # High band
                allpasses.append([allp_zi_l, allp_zi_h])
            band_zi = []
            band_zi.append(band_zi_l)
            band_zi.append(band_zi_h)
            for a in allpasses:
                band_zi.append(a)
            self.all_cross_zi.append(band_zi)

    def filter_signal(self, s: Signal, activate_zi: bool = True,
                      multichannel: bool = None):
        '''
        Filters a signal regarding the zi's of the filters and returns
        either a MultiBand- or a multi-channel signal
        '''
        if multichannel is None:
            multichannel = True if s.number_of_channels > 1 else False
        assert s.sampling_rate_hz == self.sampling_rate_hz, \
            'Sampling rates do not match'
        if not multichannel:
            assert s.number_of_channels == 1, \
                'Returning a multichannel signal is only possible if the ' +\
                'original signal has only one band. Choose ' +\
                'multichannel=False to get a MultiBandSignal object!'
            new_time_data = \
                np.zeros((s.time_data.shape[0], self.number_of_bands))
            in_sig = s.time_data.copy().squeeze()
            if activate_zi:
                for cn in range(self.number_of_cross):
                    band, in_sig = self._two_way_split_zi(in_sig, cn)
                    for ap_n in range(cn+1, self.number_of_cross):
                        band = \
                            self._allpass_zi(
                                band, cross_number=cn, ap_number=ap_n)
                    new_time_data[:, cn] = band
                # Last high frequency component
                new_time_data[:, cn+1] = in_sig
            else:
                for cn in range(self.number_of_cross):
                    band, in_sig = self._filt(in_sig, cn)
                    for ap_n in range(cn+1, self.number_of_cross):
                        band = self._filt(band, ap_n, split=False)
                    new_time_data[:, cn] = band
                # Last high frequency component
                new_time_data[:, cn+1] = in_sig
            return Signal(None, new_time_data, s.sampling_rate_hz)
        else:
            new_time_data = \
                np.zeros((s.time_data.shape[0], self.number_of_bands))
            bands = [MultiBandSignal()]
            print(bands)

    def _allpass_zi(self, s, cross_number, ap_number):
        # all_cross_zi = [[cross0_zi], [cross1_zi], [cross2_zi], ...]
        # cross0_zi = [low_zi, high_zi, [allpass1_zi], [allpass2_zi], ...]
        # allpass1_zi = [low_zi, high_zi]

        # Unpack zi's
        ap_zi = self.all_cross_zi[cross_number][ap_number+1]
        # ap_number + 1 so that the indices are right because
        # cross0_zi = [low, high, [allpass]] so it has to start on 2,
        # but the allpass count starts in 1 so only add 1
        zi_l = ap_zi[0]
        zi_h = ap_zi[1]
        # Low band
        s_l, zi_l = sig.sosfilt(self.sos[ap_number][0], x=s, zi=zi_l)
        s_l, zi_l = sig.sosfilt(self.sos[ap_number][0], x=s_l, zi=zi_l)
        # High band
        s_h, zi_h = sig.sosfilt(self.sos[ap_number][1], x=s, zi=zi_h)
        s_h, zi_h = sig.sosfilt(self.sos[ap_number][1], x=s_h, zi=zi_h)
        # Pack zi's
        ap_zi[0] = zi_l
        ap_zi[1] = zi_h
        self.all_cross_zi[cross_number][ap_number+1] = ap_zi
        return s_l + s_h

    def _two_way_split_zi(self, s, cross_number):
        # all_cross_zi = [[cross0_zi], [cross1_zi], [cross2_zi], ...]
        # cross0_zi = [low_zi, high_zi, [allpass1_zi], [allpass2_zi], ...]
        # allpass1_zi = [low_zi, high_zi]

        # Unpack zi's
        cross_zi = self.all_cross_zi[cross_number]
        zi_l = cross_zi[0]
        zi_h = cross_zi[1]
        # Low band
        s_l, zi_l = sig.sosfilt(self.sos[cross_number][0], x=s, zi=zi_l)
        s_l, zi_l = sig.sosfilt(self.sos[cross_number][0], x=s_l, zi=zi_l)
        # High band
        s_h, zi_h = sig.sosfilt(self.sos[cross_number][1], x=s, zi=zi_h)
        s_h, zi_h = sig.sosfilt(self.sos[cross_number][1], x=s_h, zi=zi_h)
        # Pack zi's
        cross_zi[0] = zi_l
        cross_zi[1] = zi_h
        self.all_cross_zi[cross_number] = cross_zi
        return s_l, s_h

    def _filt(self, s, f_number, split: bool = True):
        '''
        Filters signal with the sos corresponding to f_number.
        `split=True` returns two bands; when `False`, the summed bands are
        returned (allpass).
        '''
        # Low band
        s_l = sig.sosfilt(self.sos[f_number][0], x=s)
        s_l = sig.sosfilt(self.sos[f_number][0], x=s_l)
        # High band
        s_h = sig.sosfilt(self.sos[f_number][1], x=s)
        s_h = sig.sosfilt(self.sos[f_number][1], x=s_h)
        if split:
            return s_l, s_h
        else:
            return s_l + s_h
