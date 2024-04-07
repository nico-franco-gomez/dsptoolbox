import pytest
import dsptoolbox as dsp
import numpy as np
import os


class TestStandardModule:
    fs = 44100
    audio_multi = dsp.generators.noise("white", 2, fs, number_of_channels=3)

    def test_latency(self):
        # Create delayed version of signal
        td = self.audio_multi.time_data
        delay_samples = int(30e-3 * self.fs)
        td_del = np.zeros(
            (td.shape[0] + delay_samples, self.audio_multi.number_of_channels)
        )
        td_del[-td.shape[0] :] = td

        # Try latency
        s = dsp.Signal(None, td_del, self.fs)
        vector = dsp.latency(self.audio_multi, s)
        assert np.all(vector == -delay_samples)

        # Try latency the other way around
        vector = dsp.latency(s, self.audio_multi)
        assert np.all(vector == delay_samples)

        # Raise assertion when number of channels does not match
        with pytest.raises(AssertionError):
            vector = dsp.latency(s.get_channels(0), self.audio_multi)

        # Single channel
        td = s.time_data[:, :2]
        td[:, 1] = 0
        td[: len(self.audio_multi.time_data[:, 0]), 1] = (
            self.audio_multi.time_data[:, 0]
        )
        s = dsp.Signal(None, td, self.fs)
        value = dsp.latency(s)
        assert np.all(-value == delay_samples)

        # ===== Fractional delays
        delay = 0.003301
        noi = dsp.generators.noise(
            "white", length_seconds=1, sampling_rate_hz=10_000
        )
        noi_del = dsp.fractional_delay(noi, delay)
        assert (
            np.abs(
                dsp.latency(noi_del, noi, 2)[0] - delay * noi.sampling_rate_hz
            )
            < 0.9
        )

        noi = dsp.merge_signals(noi_del, noi)
        latencies = dsp.latency(noi, polynomial_points=5)
        assert len(latencies) == noi.number_of_channels - 1
        assert np.abs(latencies[0] + delay * noi.sampling_rate_hz) < 0.5

    def test_pad_trim(self):
        # Check for signal: Trim at the end
        trim_length = 40_000
        td = self.audio_multi.time_data[:trim_length]
        s = dsp.Signal(None, td, self.fs)
        assert np.all(
            s.time_data
            == dsp.pad_trim(self.audio_multi, trim_length).time_data
        )

        # Check for signal: pad at the end
        pad_length = 10_000
        td = np.concatenate(
            [td, np.zeros((pad_length, self.audio_multi.number_of_channels))],
            axis=0,
        )
        s = dsp.pad_trim(s, s.time_data.shape[0] + pad_length)
        s1 = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data == s1.time_data)

        # Check for signal: trim at start
        trim_length = 30_000
        td = self.audio_multi.time_data[-trim_length:]
        s = dsp.Signal(None, td, self.fs)
        assert np.all(
            s.time_data
            == dsp.pad_trim(
                self.audio_multi, trim_length, in_the_end=False
            ).time_data
        )

        # Check for signal: pad at the end
        pad_length = 10_000
        td = np.concatenate(
            [np.zeros((pad_length, self.audio_multi.number_of_channels)), td],
            axis=0,
        )
        s = dsp.pad_trim(
            s, s.time_data.shape[0] + pad_length, in_the_end=False
        )
        s1 = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data == s1.time_data)

        # Plausibility for single-channel signal
        s = s.get_channels(0)
        dsp.pad_trim(s, 50_000)

        # MultiBandSignal test
        b = [
            self.audio_multi.get_channels(0),
            self.audio_multi.get_channels(1),
        ]
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
        b = [
            self.audio_multi.get_channels(0),
            self.audio_multi.get_channels(1),
        ]
        sm = dsp.MultiBandSignal(b)
        sm1 = dsp.MultiBandSignal(b)
        sm_ = dsp.merge_signals(sm, sm1)
        assert sm_.number_of_channels == 2
        assert sm_.number_of_bands == 2

    def test_merge_filterbanks(self):
        fb1 = dsp.filterbanks.auditory_filters_gammatone(
            [600, 800], sampling_rate_hz=self.fs
        )
        fb2 = dsp.filterbanks.auditory_filters_gammatone(
            [800, 1000], sampling_rate_hz=self.fs
        )
        dsp.merge_filterbanks(fb1, fb2)

        with pytest.raises(AssertionError):
            fb3 = dsp.filterbanks.auditory_filters_gammatone(
                [800, 1000], sampling_rate_hz=48000
            )
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
        factor = 10 ** (-20 / 20)
        td *= factor
        assert np.isclose(np.max(np.abs(n.time_data)), np.max(np.abs(td)))

    def test_fade(self):
        # Functionality – result only tested for linear fade
        dsp.fade(self.audio_multi, type_fade="lin")
        dsp.fade(self.audio_multi, type_fade="log")
        dsp.fade(self.audio_multi, type_fade="exp")

        f_end = dsp.fade(
            self.audio_multi, type_fade="lin", at_start=False, at_end=True
        )
        f_st = dsp.fade(
            self.audio_multi, type_fade="lin", at_start=True, at_end=False
        )
        with pytest.raises(AssertionError):
            dsp.fade(
                self.audio_multi, type_fade="lin", at_start=False, at_end=False
            )

        # Fade at start
        td = self.audio_multi.time_data
        fade_le = int(td.shape[0] * 2.5 / 100)
        td[:fade_le] *= np.linspace(0, 1, fade_le)[..., None]
        assert np.all(np.isclose(f_st.time_data, td))

        # Fade at end
        td = self.audio_multi.time_data
        td[-fade_le:] *= np.linspace(1, 0, fade_le)[..., None]
        assert np.all(np.isclose(f_end.time_data, td))

    def test_erb_frequencies(self):
        # Only functionality tested here
        dsp.erb_frequencies()

    def test_true_peak_level(self):
        # Only functionality is tested here
        dsp.true_peak_level(self.audio_multi)
        b = [
            self.audio_multi.get_channels(0),
            self.audio_multi.get_channels(1),
        ]
        mb = dsp.MultiBandSignal(b)
        dsp.true_peak_level(mb)

    def test_fractional_delay(self):
        # Delay in seconds
        delay_s = 150 / self.fs

        # All channels
        s = dsp.fractional_delay(self.audio_multi, delay_s)
        lat = dsp.latency(s, self.audio_multi)
        assert np.all(np.isclose(np.abs(lat), 150))

        # Selected channels only
        s = dsp.fractional_delay(self.audio_multi, delay_s, channels=0)
        lat = dsp.latency(s, self.audio_multi)
        assert np.all(np.isclose(np.abs(lat), [150, 0, 0]))

    def test_activity_detector(self):
        # Only functionality tested
        # Create harmonic signal and silence afterwards
        s = dsp.generators.harmonic(sampling_rate_hz=self.fs)
        s = dsp.pad_trim(s, s.time_data.shape[0] * 2)
        dsp.activity_detector(s)

    def test_detrend(self):
        # Functionality
        s = dsp.generators.harmonic(
            100,
            sampling_rate_hz=700,
            peak_level_dbfs=-20,
            number_of_channels=2,
            uncorrelated=True,
        )
        s.time_data += 0.2
        dsp.detrend(s, polynomial_order=0)

        # One channel
        s = dsp.generators.harmonic(
            100,
            sampling_rate_hz=700,
            peak_level_dbfs=-20,
            number_of_channels=1,
            uncorrelated=True,
        )
        n = 0.3 * np.arange(len(s)) / len(s)
        s.time_data += n[..., None]
        dsp.detrend(s, polynomial_order=1)

        # Large polynomial order
        dsp.detrend(s, polynomial_order=10)

        with pytest.raises(AssertionError):
            dsp.detrend(s, polynomial_order=-10)

    def test_load_pkl_object(self):
        f = dsp.Filter(
            "fir",
            dict(order=216, freqs=1000, type_of_pass="highpass"),
            self.fs,
        )
        f.save_filter(os.path.join("tests", "f"))
        dsp.load_pkl_object(os.path.join("tests", "f"))
        dsp.load_pkl_object(os.path.join("tests", "f.pkl"))
        os.remove(os.path.join("tests", "f.pkl"))

    def test_rms(self):
        td = self.audio_multi.time_data[:, 0]
        rms_vals = dsp.rms(self.audio_multi, in_dbfs=False)
        assert np.isclose(np.sqrt(np.mean(td**2)), rms_vals[0])

    def test_calibration_data(self):
        # Calibration for one channel
        sine = dsp.generators.harmonic(
            sampling_rate_hz=self.audio_multi.sampling_rate_hz,
            peak_level_dbfs=-20,
        )
        calib = dsp.CalibrationData(sine)
        calib.calibrate_signal(self.audio_multi)

        # Wrong number of channels
        with pytest.raises(AssertionError):
            sine = dsp.generators.harmonic(
                sampling_rate_hz=self.audio_multi.sampling_rate_hz,
                peak_level_dbfs=-20,
                number_of_channels=self.audio_multi.number_of_channels - 1,
            )
            calib = dsp.CalibrationData(sine)
            calib.calibrate_signal(self.audio_multi)

        # Calibration for all channels
        sine = dsp.generators.harmonic(
            sampling_rate_hz=self.audio_multi.sampling_rate_hz,
            peak_level_dbfs=-20,
            number_of_channels=self.audio_multi.number_of_channels,
        )
        calib = dsp.CalibrationData(sine)
        calib.calibrate_signal(self.audio_multi)

        # Multiband
        fb = dsp.filterbanks.fractional_octave_bands(
            [125, 1000], sampling_rate_hz=self.audio_multi.sampling_rate_hz
        )
        new_sig = fb.filter_signal(self.audio_multi)
        calib.calibrate_signal(new_sig)

    def test_envelope(self):
        # Only functionality with multi-channel and single-channel data
        s = dsp.generators.oscillator(
            frequency_hz=500,
            mode="triangle",
            sampling_rate_hz=5_000,
            number_of_channels=3,
            uncorrelated=True,
        )
        env = dsp.envelope(s, "rms", 512)
        assert env.shape == s.time_data.shape
        env = dsp.envelope(s, "analytic", None)
        assert env.shape == s.time_data.shape

        s = dsp.generators.oscillator(
            frequency_hz=500,
            mode="sawtooth",
            sampling_rate_hz=5_000,
            number_of_channels=1,
        )
        env = dsp.envelope(s, "rms", 512)
        assert env.shape == s.time_data.shape
        env = dsp.envelope(s, "analytic", None)
        assert env.shape == s.time_data.shape

        fb = dsp.filterbanks.auditory_filters_gammatone(
            [500, 1000], 1, s.sampling_rate_hz
        )
        ss = fb.filter_signal(s)
        dsp.envelope(ss)

    def test_convert_lattice_filter(self):
        fs = 44100
        # Second-order sections
        n = dsp.generators.noise(sampling_rate_hz=fs)
        f = dsp.Filter(
            "iir",
            {
                "filter_design_method": "bessel",
                "order": 9,
                "type_of_pass": "lowpass",
                "freqs": 1000,
            },
            sampling_rate_hz=fs,
        )
        new_f = dsp.convert_into_lattice_filter(f)
        n1 = f.filter_signal(n).time_data.squeeze()
        n2 = new_f.filter_signal(n).time_data.squeeze()
        assert np.all(np.isclose(n1, n2))

        # BA
        b, a = f.get_coefficients("ba")
        f2 = dsp.Filter("other", {"ba": [b, a]}, f.sampling_rate_hz)
        new_f = dsp.convert_into_lattice_filter(f2)
        n1 = f2.filter_signal(n).time_data.squeeze()
        n2 = new_f.filter_signal(n).time_data.squeeze()
        assert np.all(np.isclose(n1, n2))

        # FIR
        n = dsp.generators.noise(sampling_rate_hz=fs)
        f = dsp.Filter(
            "other",
            {"ba": [[1, 13 / 24, 5 / 8, 1 / 3], [1]]},
            sampling_rate_hz=fs,
        )
        new_f = dsp.convert_into_lattice_filter(f)
        n1 = f.filter_signal(n).time_data.squeeze()
        n2 = new_f.filter_signal(n).time_data.squeeze()
        assert np.all(np.isclose(n1, n2))
