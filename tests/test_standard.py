import pytest
import dsptoolbox as dsp
import numpy as np


class TestStandardModule():
    fs = 44100
    audio_multi = dsp.generators.noise('white', 2, fs, number_of_channels=3)

    def test_latency(self):
        # Create delayed version of signal
        td = self.audio_multi.time_data
        delay_samples = int(30e-3*self.fs)
        td_del = np.zeros(
            (td.shape[0]+delay_samples, self.audio_multi.number_of_channels))
        td_del[-td.shape[0]:] = td

        # Try latency
        s = dsp.Signal(None, td_del, self.fs)
        vector = dsp.latency(self.audio_multi, s)
        assert np.all(np.abs(vector) == delay_samples)

        # Try latency the other way around
        vector = dsp.latency(s, self.audio_multi)
        assert np.all(np.abs(vector) == delay_samples)

        # Raise assertion when number of channels does not match
        with pytest.raises(AssertionError):
            vector = dsp.latency(s.get_channels(0), self.audio_multi)

        # Single channel
        td = s.time_data[:, :2]
        td[:, 1] = 0
        td[:len(self.audio_multi.time_data[:, 0]), 1] = \
            self.audio_multi.time_data[:, 0]
        s = dsp.Signal(None, td, self.fs)
        value = dsp.latency(s)
        assert np.all(np.abs(value) == delay_samples)

    def test_group_delay(self):
        # Check only that some result is produced, validity should be checked
        # somewhere else
        td = self.audio_multi.time_data
        td = td[:10_000, :]
        s = dsp.Signal(None, td, self.fs)
        dsp.group_delay(s, method='matlab')
        dsp.group_delay(s, method='direct')

        # Single-channel plausibility check
        dsp.group_delay(s.get_channels(0))

    def test_minimum_phase(self):
        # Check only that some result is produced, validity should be checked
        # somewhere else
        td = self.audio_multi.time_data
        td = td[:10_000, :]
        # Only works for some signal types
        s = dsp.Signal(None, td, self.fs, signal_type='rir')
        dsp.minimum_phase(s)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, td, self.fs)
            dsp.minimum_phase(s1)
        # Single-channel plausibility check
        dsp.minimum_phase(s.get_channels(0))

    def test_minimum_group_delay(self):
        # Check only that some result is produced, validity should be checked
        # somewhere else
        td = self.audio_multi.time_data
        td = td[:10_000, :]
        # Only works for some signal types
        s = dsp.Signal(None, td, self.fs, signal_type='rir')
        dsp.minimum_group_delay(s)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, td, self.fs)
            dsp.minimum_group_delay(s1)
        # Single-channel plausibility check
        dsp.minimum_group_delay(s.get_channels(0))

    def test_excess_group_delay(self):
        # Check only that some result is produced, validity should be checked
        # somewhere else
        td = self.audio_multi.time_data
        td = td[:10_000, :]
        # Only works for some signal types
        s = dsp.Signal(None, td, self.fs, signal_type='rir')
        dsp.excess_group_delay(s)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, td, self.fs)
            dsp.excess_group_delay(s1)
        # Single-channel plausibility check
        dsp.excess_group_delay(s.get_channels(0))

    def test_pad_trim(self):
        # Check for signal: Trim at the end
        trim_length = 40_000
        td = self.audio_multi.time_data[:trim_length]
        s = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data ==
                      dsp.pad_trim(self.audio_multi, trim_length).time_data)

        # Check for signal: pad at the end
        pad_length = 10_000
        td = np.concatenate(
            [td, np.zeros((pad_length, self.audio_multi.number_of_channels))],
            axis=0)
        s = dsp.pad_trim(s, s.time_data.shape[0]+pad_length)
        s1 = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data == s1.time_data)

        # Check for signal: trim at start
        trim_length = 30_000
        td = self.audio_multi.time_data[-trim_length:]
        s = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data ==
                      dsp.pad_trim(self.audio_multi, trim_length,
                                   in_the_end=False).time_data)

        # Check for signal: pad at the end
        pad_length = 10_000
        td = np.concatenate(
            [np.zeros((pad_length, self.audio_multi.number_of_channels)), td],
            axis=0)
        s = dsp.pad_trim(s, s.time_data.shape[0]+pad_length, in_the_end=False)
        s1 = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data == s1.time_data)

        # Plausibility for single-channel signal
        s = s.get_channels(0)
        dsp.pad_trim(s, 50_000)

        # MultiBandSignal test
        b = [self.audio_multi.get_channels(0),
             self.audio_multi.get_channels(1)]
        multi = dsp.MultiBandSignal(b)
        dsp.pad_trim(multi, 40_000)

    def test_merge_signal(self):
        # Signal
        s1 = self.audio_multi.get_channels(0)
        s2 = self.audio_multi.get_channels(1)
        s = dsp.merge_signals(s1, s2)
        assert s.number_of_channels == 2
        assert np.all(s.time_data == self.audio_multi.time_data[:, :2])
        # MultiBandSignal
        b = [self.audio_multi.get_channels(0),
             self.audio_multi.get_channels(1)]
        sm = dsp.MultiBandSignal(b)
        sm1 = dsp.MultiBandSignal(b)
        sm_ = dsp.merge_signals(sm, sm1)
        assert sm_.number_of_channels == 2
        assert sm_.number_of_bands == 2

    def test_merge_filterbanks(self):
        fb1 = dsp.filterbanks.auditory_filters_gammatone(
            [600, 800], sampling_rate_hz=self.fs)
        fb2 = dsp.filterbanks.auditory_filters_gammatone(
            [800, 1000], sampling_rate_hz=self.fs)
        dsp.merge_filterbanks(fb1, fb2)

        with pytest.raises(AssertionError):
            fb3 = dsp.filterbanks.auditory_filters_gammatone(
                [800, 1000], sampling_rate_hz=48000)
            dsp.merge_filterbanks(fb1, fb3)

    def test_resample(self):
        # The result itself will not be checked, only that there is an output
        # Since it is a wrapper around scipy's function, it might not be
        # necessary to check...
        dsp.resample(self.audio_multi, desired_sampling_rate_hz=22050)

    def test_fractional_octave_frequencies(self):
        # Only functionality and not result is checked here
        dsp.fractional_octave_frequencies()

    def test_normalize(self):
        td = self.audio_multi.time_data
        n = dsp.normalize(self.audio_multi, peak_dbfs=-20)
        td /= np.max(np.abs(td))
        factor = 10**(-20/20)
        td *= factor
        assert np.isclose(np.max(np.abs(n.time_data)), np.max(np.abs(td)))

    def test_fade(self):
        # Functionality – result only tested for linear fade
        dsp.fade(self.audio_multi, type_fade='lin')
        dsp.fade(self.audio_multi, type_fade='log')
        dsp.fade(self.audio_multi, type_fade='exp')

        f_end = dsp.fade(self.audio_multi, type_fade='lin', at_start=False,
                         at_end=True)
        f_st = dsp.fade(self.audio_multi, type_fade='lin', at_start=True,
                        at_end=False)
        with pytest.raises(AssertionError):
            dsp.fade(self.audio_multi, type_fade='lin',
                     at_start=False, at_end=False)

        # Fade at start
        td = self.audio_multi.time_data
        fade_le = int(td.shape[0]*2.5/100)
        td[:fade_le] *= np.linspace(0, 1, fade_le)[..., None]
        assert np.all(np.isclose(f_st.time_data, td))

        # Fade at end
        td = self.audio_multi.time_data
        td[-fade_le:] *= np.linspace(1, 0, fade_le)[..., None]
        assert np.all(np.isclose(f_end.time_data, td))

    def test_erb_frequencies(self):
        # Only functionality tested here
        dsp.erb_frequencies()

    def test_ir_to_filter(self):
        s = self.audio_multi.time_data[:200, 0]
        s = dsp.Signal(None, s, self.fs, signal_type='rir')
        f = dsp.ir_to_filter(s, channel=0)
        b, _ = f.get_coefficients(mode='ba')
        assert np.all(b == s.time_data[:, 0])
        assert f.sampling_rate_hz == s.sampling_rate_hz

    def test_true_peak_level(self):
        # Only functionality is tested here
        dsp.true_peak_level(self.audio_multi)
        b = [self.audio_multi.get_channels(0),
             self.audio_multi.get_channels(1)]
        mb = dsp.MultiBandSignal(b)
        dsp.true_peak_level(mb)

    def test_fractional_delay(self):
        # Delay in seconds
        delay_s = 150/self.fs
        s = dsp.fractional_delay(self.audio_multi, delay_s)
        lat = dsp.latency(s, self.audio_multi)
        assert np.all(np.isclose(np.abs(lat), 150))

    def test_activity_detector(self):
        # Only functionality tested
        # Create harmonic signal and silence afterwards
        s = dsp.generators.sinus(sampling_rate_hz=self.fs)
        s = dsp.pad_trim(s, s.time_data.shape[0]*2)
        dsp.activity_detector(s)
