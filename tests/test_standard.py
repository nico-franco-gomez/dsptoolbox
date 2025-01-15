import pytest
import dsptoolbox as dsp
import numpy as np
import os


class TestStandardModule:
    fs = 44100
    audio_multi = dsp.generators.noise(2, fs, number_of_channels=3)

    def get_multiband_signal(self) -> dsp.MultiBandSignal:
        fb = dsp.filterbanks.linkwitz_riley_crossovers([1e3], [4], self.fs)
        return fb.filter_signal(self.audio_multi)

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
        vector, corr = dsp.latency(self.audio_multi, s)
        assert np.allclose(corr, 1.0)
        assert np.all(vector == -delay_samples)
        np.testing.assert_array_equal(s.time_data, td_del)

        # Try latency the other way around
        td_previous = s.time_data.copy()
        td_previous2 = self.audio_multi.time_data.copy()
        vector, corr = dsp.latency(s, self.audio_multi)
        assert np.allclose(corr, 1.0)
        assert np.all(vector == delay_samples)
        np.testing.assert_array_equal(s.time_data, td_previous)
        np.testing.assert_array_equal(self.audio_multi.time_data, td_previous2)

        # Raise assertion when number of channels does not match
        with pytest.raises(AssertionError):
            vector, corr = dsp.latency(s.get_channels(0), self.audio_multi)

        # Single channel
        td = s.time_data[:, :2]
        td[:, 1] = 0
        td[: len(self.audio_multi.time_data[:, 0]), 1] = (
            self.audio_multi.time_data[:, 0]
        )
        s = dsp.Signal(None, td, self.fs)
        value, corr = dsp.latency(s)
        assert np.allclose(corr, 1.0)
        assert np.all(-value == delay_samples)

        # Check that data does not change after function
        s = dsp.Signal(None, td, self.fs)
        value, corr = dsp.latency(s)
        np.testing.assert_array_equal(s.time_data, td)

        # ===== Fractional delays
        delay = 0.003301
        noi = dsp.generators.noise(length_seconds=1, sampling_rate_hz=10_000)
        noi_del = dsp.fractional_delay(noi, delay)
        td_previous_noi_del = noi_del.time_data.copy()  # Data does not change
        lat, corr = dsp.latency(noi_del, noi, 2)
        np.testing.assert_array_equal(
            td_previous_noi_del, noi_del.time_data
        )  #
        assert np.allclose(corr, 1.0, atol=1e-2)
        assert np.abs(lat[0] - delay * noi.sampling_rate_hz) < 0.9

        noi = dsp.append_signals([noi_del, noi])
        latencies, corr = dsp.latency(noi, polynomial_points=1)
        assert len(latencies) == noi.number_of_channels - 1
        assert np.allclose(corr, 1.0, atol=1e-2)
        assert np.abs(latencies[0] + delay * noi.sampling_rate_hz) < 0.5
        latencies, corr = dsp.latency(noi, polynomial_points=5)
        assert np.allclose(corr, 1.0, atol=1e-2)
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

    def test_append_signals(self):
        # Signal
        s1 = self.audio_multi.get_channels(0)
        s2 = self.audio_multi.get_channels(1)
        s = dsp.append_signals([s1, s2])
        assert s.number_of_channels == 2
        assert np.all(s.time_data == self.audio_multi.time_data[:, :2])
        # MultiBandSignal
        b = [
            self.audio_multi.get_channels(0),
            self.audio_multi.get_channels(1),
        ]
        sm = dsp.MultiBandSignal(b)
        sm1 = dsp.MultiBandSignal(b)
        sm_ = dsp.append_signals([sm, sm1])
        assert sm_.number_of_channels == 2
        assert sm_.number_of_bands == 2

    def test_append_filterbanks(self):
        fb1 = dsp.filterbanks.auditory_filters_gammatone(
            [600, 800], sampling_rate_hz=self.fs
        )
        fb2 = dsp.filterbanks.auditory_filters_gammatone(
            [800, 1000], sampling_rate_hz=self.fs
        )
        fb_out = dsp.append_filterbanks([fb1, fb2])
        assert len(fb_out) == len(fb1) + len(fb2)

        with pytest.raises(AssertionError):
            fb3 = dsp.filterbanks.auditory_filters_gammatone(
                [800, 1000], sampling_rate_hz=48000
            )
            dsp.append_filterbanks([fb1, fb3])

    def test_resample(self):
        # The result itself will not be checked, only that there is an output
        # Since it is a wrapper around scipy's function, it might not be
        # necessary to check...
        dsp.resample(self.audio_multi, desired_sampling_rate_hz=22050)

    def test_normalize(self):
        # Check peak normalization
        td = self.audio_multi.time_data
        n = dsp.normalize(self.audio_multi, norm_dbfs=-20)
        td /= np.max(np.abs(td))
        factor = 10 ** (-20 / 20)
        td *= factor
        assert np.isclose(np.max(np.abs(n.time_data)), np.max(np.abs(td)))

        # Check rms
        channel = self.audio_multi.get_channels(0)
        rms_previous = dsp.rms(channel)[0]
        n = dsp.normalize(
            channel, norm_dbfs=rms_previous - 10, peak_normalization=False
        )
        rms = dsp.rms(n)[0]
        assert np.isclose(rms_previous - 10, rms)

        # Check rest of api
        dsp.normalize(
            self.audio_multi,
            norm_dbfs=-20,
            peak_normalization=False,
            each_channel=False,
        )
        dsp.normalize(
            self.audio_multi,
            norm_dbfs=-20,
            peak_normalization=False,
            each_channel=True,
        )
        dsp.normalize(
            self.audio_multi,
            norm_dbfs=-20,
            peak_normalization=True,
            each_channel=True,
        )

    def test_fade(self):
        # Functionality – result only tested for linear fade
        dsp.fade(self.audio_multi, fade_type=dsp.FadeType.Linear)
        dsp.fade(self.audio_multi, fade_type=dsp.FadeType.Logarithmic)
        dsp.fade(self.audio_multi, fade_type=dsp.FadeType.Exponential)

        f_end = dsp.fade(
            self.audio_multi,
            fade_type=dsp.FadeType.Linear,
            at_start=False,
            at_end=True,
        )
        f_st = dsp.fade(
            self.audio_multi,
            fade_type=dsp.FadeType.Linear,
            at_start=True,
            at_end=False,
        )
        with pytest.raises(AssertionError):
            dsp.fade(
                self.audio_multi,
                fade_type=dsp.FadeType.Linear,
                at_start=False,
                at_end=False,
            )

        # Fade at start
        td = self.audio_multi.time_data.copy()
        fade_le = int(td.shape[0] * 2.5 / 100)
        td[:fade_le] *= np.linspace(0, 1, fade_le)[..., None]
        assert np.all(np.isclose(f_st.time_data, td))

        # Fade at end
        td = self.audio_multi.time_data.copy()
        td[-fade_le:] *= np.linspace(1, 0, fade_le)[..., None]
        assert np.all(np.isclose(f_end.time_data, td))

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
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), 150))

        # Selected channels only
        s = dsp.fractional_delay(self.audio_multi, delay_s, channels=0)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), [150, 0, 0]))

    def test_delay(self):
        # Delay in seconds
        delay_samp = 150

        # All channels
        s = dsp.delay(self.audio_multi, delay_samp)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), 150))

        # Selected channels only
        s = dsp.delay(self.audio_multi, delay_samp, channels=0)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), [150, 0, 0]))

    def test_activity_detector(self):
        # Only functionality tested
        # Create harmonic signal and silence afterwards
        s = dsp.generators.oscillator(1000.0, sampling_rate_hz=self.fs)
        s = dsp.pad_trim(s, s.time_data.shape[0] * 2)
        dsp.activity_detector(s)

    def test_detrend(self):
        # Functionality
        s = dsp.generators.oscillator(
            100,
            sampling_rate_hz=700,
            peak_level_dbfs=-20,
            number_of_channels=2,
            uncorrelated=True,
        )
        s.time_data += 0.2
        dsp.detrend(s, polynomial_order=0)

        # One channel
        s = dsp.generators.oscillator(
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
        f = dsp.Filter.fir_filter(
            order=216,
            frequency_hz=1000,
            type_of_pass=dsp.FilterPassType.Highpass,
            sampling_rate_hz=self.fs,
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
        sine = dsp.generators.oscillator(
            frequency_hz=100.0,
            sampling_rate_hz=self.audio_multi.sampling_rate_hz,
            peak_level_dbfs=-20,
        )
        calib = dsp.CalibrationData(sine)
        calib.calibrate_signal(self.audio_multi)

        # Wrong number of channels
        with pytest.raises(AssertionError):
            sine = dsp.generators.oscillator(
                frequency_hz=1000.0,
                sampling_rate_hz=self.audio_multi.sampling_rate_hz,
                peak_level_dbfs=-20,
                number_of_channels=self.audio_multi.number_of_channels - 1,
            )
            calib = dsp.CalibrationData(sine)
            calib.calibrate_signal(self.audio_multi)

        # Calibration for all channels
        sine = dsp.generators.oscillator(
            frequency_hz=1000.0,
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
        new_sig = fb.filter_signal(
            self.audio_multi, dsp.FilterBankMode.Parallel
        )
        calib.calibrate_signal(new_sig)

    def test_envelope(self):
        # Only functionality with multi-channel and single-channel data
        s = dsp.generators.oscillator(
            frequency_hz=500,
            mode=dsp.generators.WaveForm.Triangle,
            sampling_rate_hz=5_000,
            number_of_channels=3,
            uncorrelated=True,
        )
        env = dsp.envelope(s, False, 512)
        assert env.shape == s.time_data.shape
        env = dsp.envelope(s, True, None)
        assert env.shape == s.time_data.shape

        s = dsp.generators.oscillator(
            frequency_hz=500,
            mode=dsp.generators.WaveForm.Sawtooth,
            sampling_rate_hz=5_000,
            number_of_channels=1,
        )
        env = dsp.envelope(s, False, 512)
        assert env.shape == s.time_data.shape
        env = dsp.envelope(s, True, None)
        assert env.shape == s.time_data.shape

        fb = dsp.filterbanks.auditory_filters_gammatone(
            [500, 1000], 1, s.sampling_rate_hz
        )
        ss = fb.filter_signal(s, dsp.FilterBankMode.Parallel)
        dsp.envelope(ss)

    def test_dither(self):
        # Functionality
        dsp.dither(self.audio_multi)

        fb = dsp.FilterBank(
            [
                dsp.Filter.biquad(
                    eq_type=dsp.BiquadEqType.Peaking,
                    frequency_hz=500,
                    gain_db=2,
                    q=1,
                    sampling_rate_hz=self.audio_multi.sampling_rate_hz,
                )
            ]
        )
        dsp.dither(self.audio_multi, noise_shaping_filterbank=fb)
        dsp.dither(self.audio_multi, truncate=False)

    def test_apply_gain(self):
        some_signal = self.audio_multi.copy()
        # Signal
        audio_multi = dsp.apply_gain(some_signal, 5)
        np.testing.assert_array_equal(
            audio_multi.time_data,
            some_signal.time_data * dsp.tools.from_db(5, True),
        )

        gains = np.linspace(1, 5, some_signal.number_of_channels)
        audio_multi = dsp.apply_gain(some_signal, gains)
        np.testing.assert_array_equal(
            audio_multi.time_data,
            some_signal.time_data * dsp.tools.from_db(gains, True),
        )

        audio_multi = dsp.apply_gain(some_signal, gains)
        np.testing.assert_array_equal(
            audio_multi.time_data,
            some_signal.time_data * dsp.tools.from_db(gains, True),
        )

        # MultiBandSignal
        audio_multi_mb = self.get_multiband_signal()
        previous = audio_multi_mb.get_all_time_data()[0]
        audio_multi_mb = dsp.apply_gain(audio_multi_mb, 5)
        np.testing.assert_array_equal(
            previous * dsp.tools.from_db(5, True),
            audio_multi_mb.get_all_time_data()[0],
        )

        previous = audio_multi_mb.get_all_time_data()[0]
        gains = np.linspace(1, 5, audio_multi.number_of_channels)
        audio_multi_mb = dsp.apply_gain(audio_multi_mb, gains)
        np.testing.assert_array_equal(
            previous * dsp.tools.from_db(gains, True),
            audio_multi_mb.get_all_time_data()[0],
        )

        # Filter
        iir = dsp.Filter.biquad(
            dsp.BiquadEqType.Peaking, 500.0, 0.0, 0.7, self.fs
        )
        output_level1 = dsp.rms(iir.filter_signal(self.audio_multi))
        gain_db = -5.0
        output_level2 = dsp.rms(
            dsp.apply_gain(iir, gain_db).filter_signal(self.audio_multi)
        )
        np.testing.assert_array_almost_equal(
            output_level1 + gain_db, output_level2
        )
        with pytest.raises(ValueError):
            dsp.apply_gain(iir, [gain_db, 0])

        # FilterBank
        fb = dsp.FilterBank([iir, iir])
        base_level = dsp.rms(
            fb.filter_signal(
                self.audio_multi, mode=dsp.FilterBankMode.Sequential
            )
        )
        fb2 = dsp.apply_gain(fb, gain_db)
        output_level = dsp.rms(
            fb2.filter_signal(
                self.audio_multi, mode=dsp.FilterBankMode.Sequential
            )
        )
        np.testing.assert_array_almost_equal(
            base_level + gain_db * len(fb), output_level
        )

        # multiple gains for a filter bank
        fb2 = dsp.apply_gain(fb, [gain_db] + [0] * (len(fb) - 1))
        output_level = dsp.rms(
            fb2.filter_signal(
                self.audio_multi, mode=dsp.FilterBankMode.Sequential
            )
        )
        np.testing.assert_array_almost_equal(
            base_level + gain_db, output_level
        )
        with pytest.raises(AssertionError):
            dsp.apply_gain(fb, [gain_db] + [0] * (len(fb) + 1))

    def test_resample_filter(self):
        # Functionality
        fs_hz = 48000
        f = dsp.Filter.iir_filter(
            order=8,
            frequency_hz=[500, 2e3],
            type_of_pass=dsp.FilterPassType.Bandpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )
        dsp.resample_filter(f, 24000)
        f = dsp.Filter.iir_filter(
            order=5,
            frequency_hz=500,
            type_of_pass=dsp.FilterPassType.Lowpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )
        dsp.resample_filter(f, 24000)
        f = dsp.Filter.iir_filter(
            order=8,
            frequency_hz=500,
            type_of_pass=dsp.FilterPassType.Highpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )
        dsp.resample_filter(f, 24000)
        f = dsp.Filter.iir_filter(
            order=7,
            frequency_hz=[500, 18e3],
            type_of_pass=dsp.FilterPassType.Bandpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )

    def test_modify_signal_length(self):
        with pytest.raises(AssertionError):
            dsp.modify_signal_length(self.audio_multi, None, None)
        with pytest.raises(AssertionError):
            dsp.modify_signal_length(
                self.audio_multi,
                -(self.audio_multi.length_seconds + 1.0),
                None,
            )
        with pytest.raises(AssertionError):
            dsp.modify_signal_length(
                self.audio_multi,
                None,
                -(self.audio_multi.length_seconds + 1.0),
            )
        with pytest.raises(AssertionError):
            dsp.modify_signal_length(
                self.audio_multi,
                -self.audio_multi.length_seconds / 2.0,
                -self.audio_multi.length_seconds / 1.9,
            )

        original_length = len(self.audio_multi)

        # Add both
        new = dsp.modify_signal_length(self.audio_multi, 1.0, 1.0)
        assert new.length_seconds == self.audio_multi.length_seconds + 2.0
        assert len(self.audio_multi) == original_length

        # Add only start
        new = dsp.modify_signal_length(self.audio_multi, 1.0, None)
        assert new.length_seconds == self.audio_multi.length_seconds + 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data[: new.sampling_rate_hz], 0.0
        )

        # Add only end
        new = dsp.modify_signal_length(self.audio_multi, None, 1.0)
        assert new.length_seconds == self.audio_multi.length_seconds + 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data[-new.sampling_rate_hz :], 0.0
        )

        # Remove both
        new = dsp.modify_signal_length(self.audio_multi, -0.5, -0.5)
        assert new.length_seconds == self.audio_multi.length_seconds - 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data,
            self.audio_multi.time_data[
                new.sampling_rate_hz // 2 : -new.sampling_rate_hz // 2
            ],
        )

        # Remove only start
        new = dsp.modify_signal_length(self.audio_multi, -0.5, None)
        assert new.length_seconds == self.audio_multi.length_seconds - 0.5
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data,
            self.audio_multi.time_data[new.sampling_rate_hz // 2 :],
        )

        # Remove only end
        new = dsp.modify_signal_length(self.audio_multi, None, -0.5)
        assert new.length_seconds == self.audio_multi.length_seconds - 0.5
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data,
            self.audio_multi.time_data[: -new.sampling_rate_hz // 2],
        )

        # Mixed
        new = dsp.modify_signal_length(self.audio_multi, 1.5, -0.5)
        assert new.length_seconds == self.audio_multi.length_seconds + 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data[: 3 * new.sampling_rate_hz // 2], 0.0
        )
        np.testing.assert_array_equal(
            new.time_data[3 * new.sampling_rate_hz // 2 :],
            self.audio_multi.time_data[: -new.sampling_rate_hz // 2],
        )

        # ===== MultiBandSignal: Only functionality
        mb = self.get_multiband_signal()
        dsp.modify_signal_length(mb, 1.5, -0.5)

    def test_merge_fir_filters(self):
        f1 = dsp.Filter.fir_filter(
            50,
            100.0,
            type_of_pass=dsp.FilterPassType.Lowpass,
            window=dsp.Window.Hamming,
            sampling_rate_hz=self.fs,
        )

        # Dirac with some delay
        dirac = np.zeros(30)
        delay = len(dirac) - 1
        dirac[-1] = 1.0
        f2 = dsp.Filter.from_ba(dirac, [1.0], self.fs)

        f3 = dsp.merge_filters([f1, f2])
        np.testing.assert_array_equal(f3.ba[0][delay:], f1.ba[0])

        # With filterbank
        f3 = dsp.merge_filters(dsp.FilterBank([f1, f2]))
        np.testing.assert_array_equal(f3.ba[0][delay:], f1.ba[0])

        with pytest.raises(AssertionError):
            dsp.merge_filters([f1])

        with pytest.raises(AssertionError):
            iir = dsp.Filter.biquad(
                dsp.BiquadEqType.LowpassFirstOrder, 50.0, -3.0, 0.7, self.fs
            )
            dsp.merge_filters([f1, iir])

        with pytest.raises(AssertionError):
            f2 = dsp.Filter.from_ba(dirac, [1.0], self.fs * 2)
            dsp.merge_filters([f1, f2])

    def test_merge_iir_filters(self):
        f1 = dsp.Filter.biquad(
            eq_type=dsp.BiquadEqType.Allpass,
            frequency_hz=500.0,
            gain_db=5.0,
            q=0.7,
            sampling_rate_hz=self.fs,
        )

        f3 = dsp.merge_filters([f1, f1.copy()])
        assert f3.has_sos
        assert f3.sos.shape[0] == 2

    def test_spectral_difference(self):
        filt = dsp.Filter.biquad(
            dsp.BiquadEqType.Peaking, 500.0, 10.0, 1.0, 48000
        )
        spec = dsp.Spectrum.from_filter(
            dsp.tools.log_frequency_vector([20, 20e3], 128), filt, False
        )
        spec_flat = dsp.Spectrum.from_filter(
            dsp.tools.log_frequency_vector([20, 20e3], 128),
            dsp.Filter.biquad(
                dsp.BiquadEqType.Peaking, 500.0, 0.0, 1.0, 48000
            ),
            False,
        )
        sp_out = dsp.spectral_difference(
            spec, spec_flat, energy_normalization=False
        )
        np.testing.assert_almost_equal(
            spec.spectral_data, sp_out.spectral_data
        )

        with pytest.raises(AssertionError):
            sp_out = dsp.spectral_difference(
                spec, spec_flat, energy_normalization=False, complex=True
            )

        # Some different parameters
        dsp.spectral_difference(
            spec,
            spec_flat,
            energy_normalization=True,
            octave_fraction_smoothing=12.0,
            dynamic_range_db=None,
        )
