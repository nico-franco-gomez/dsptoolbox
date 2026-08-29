import os

import numpy as np
import pytest
from scipy.signal import hilbert

import dsptoolbox as dsp


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
        td[: len(self.audio_multi.time_data[:, 0]), 1] = self.audio_multi.time_data[
            :, 0
        ]
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
        noi_del = noi.fractional_delay(delay)
        td_previous_noi_del = noi_del.time_data.copy()  # Data does not change
        lat, corr = dsp.latency(noi_del, noi, 2)
        np.testing.assert_array_equal(td_previous_noi_del, noi_del.time_data)  #
        assert np.allclose(corr, 1.0, atol=1e-2)
        assert np.abs(lat[0] - delay * noi.sampling_rate_hz) < 0.9

        noi = noi_del.append_signals([noi])
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
        assert np.all(s.time_data == self.audio_multi.pad_trim(trim_length).time_data)

        # Check for signal: pad at the end
        pad_length = 10_000
        td = np.concatenate(
            [td, np.zeros((pad_length, self.audio_multi.number_of_channels))],
            axis=0,
        )
        s = s.pad_trim(s.time_data.shape[0] + pad_length)
        s1 = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data == s1.time_data)

        # Check for signal: trim at start
        trim_length = 30_000
        td = self.audio_multi.time_data[-trim_length:]
        s = dsp.Signal(None, td, self.fs)
        assert np.all(
            s.time_data
            == self.audio_multi.pad_trim(trim_length, in_the_end=False).time_data
        )

        # Check for signal: pad at the end
        pad_length = 10_000
        td = np.concatenate(
            [np.zeros((pad_length, self.audio_multi.number_of_channels)), td],
            axis=0,
        )
        s = s.pad_trim(s.time_data.shape[0] + pad_length, in_the_end=False)
        s1 = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data == s1.time_data)

        # Plausibility for single-channel signal
        s = s.get_channels(0)
        s.pad_trim(50_000)

        # MultiBandSignal test
        b = [
            self.audio_multi.get_channels(0),
            self.audio_multi.get_channels(1),
        ]
        multi = dsp.MultiBandSignal(b)
        multi.pad_trim(40_000)

    def test_append_signals(self):
        # Signal
        s1 = self.audio_multi.get_channels(0)
        s2 = self.audio_multi.get_channels(1)
        s = s1.append_signals([s2])
        assert s.number_of_channels == 2
        assert np.all(s.time_data == self.audio_multi.time_data[:, :2])
        # MultiBandSignal
        b = [
            self.audio_multi.get_channels(0),
            self.audio_multi.get_channels(1),
        ]
        sm = dsp.MultiBandSignal(b)
        sm1 = dsp.MultiBandSignal(b)
        sm_ = sm.append_signals([sm1])
        assert sm_.number_of_channels == 2
        assert sm_.number_of_bands == 2

    def test_append_filterbanks(self):
        fb1 = dsp.filterbanks.auditory_filters_gammatone(
            [600, 800], sampling_rate_hz=self.fs
        )
        fb2 = dsp.filterbanks.auditory_filters_gammatone(
            [800, 1000], sampling_rate_hz=self.fs
        )
        fb_out = fb1.append_filterbanks([fb2])
        assert len(fb_out) == len(fb1) + len(fb2)

        with pytest.raises(AssertionError):
            fb3 = dsp.filterbanks.auditory_filters_gammatone(
                [800, 1000], sampling_rate_hz=48000
            )
            fb1.append_filterbanks([fb3])

    def test_resample(self):
        # The result itself will not be checked, only that there is an output
        # Since it is a wrapper around scipy's function, it might not be
        # necessary to check...
        self.audio_multi.resample(desired_sampling_rate_hz=22050)

    def test_resample_preserves_frequency_and_amplitude(self):
        """`resample` wraps `scipy.signal.resample_poly`; a pure tone well
        below both the original and target Nyquist frequencies should keep
        its FFT peak frequency exactly (bin-aligned in both sampling
        rates) and its amplitude close to the original (small ripple from
        the polyphase anti-aliasing filter is expected, not exact).

        """
        fs_in = 8_000
        fs_out = 6_000
        freq = 200.0
        amplitude = 0.6
        n_samples = 4_000
        t = np.arange(n_samples) / fs_in
        x = amplitude * np.sin(2 * np.pi * freq * t)
        sig = dsp.Signal(None, x[:, None], fs_in, constrain_amplitude=False)

        out = sig.resample(fs_out)
        assert out.sampling_rate_hz == fs_out

        X = np.fft.rfft(out.time_data[:, 0])
        f = np.fft.rfftfreq(len(out), 1 / fs_out)
        peak_bin = np.argmax(np.abs(X))

        np.testing.assert_allclose(f[peak_bin], freq, atol=1e-9)
        peak_amplitude = 2 * np.abs(X[peak_bin]) / len(out)
        np.testing.assert_allclose(peak_amplitude, amplitude, rtol=0.02)

    def test_normalize(self):
        # Check peak normalization
        td = self.audio_multi.time_data
        n = self.audio_multi.normalize(norm_dbfs=-20)
        td /= np.max(np.abs(td))
        factor = 10 ** (-20 / 20)
        td *= factor
        assert np.isclose(np.max(np.abs(n.time_data)), np.max(np.abs(td)))

        # Check rms
        channel = self.audio_multi.get_channels(0)
        rms_previous = dsp.rms(channel)[0]
        n = channel.normalize(norm_dbfs=rms_previous - 10, peak_normalization=False)
        rms = dsp.rms(n)[0]
        assert np.isclose(rms_previous - 10, rms)

        # Check rest of api
        self.audio_multi.normalize(
            norm_dbfs=-20,
            peak_normalization=False,
            each_channel=False,
        )
        self.audio_multi.normalize(
            norm_dbfs=-20,
            peak_normalization=False,
            each_channel=True,
        )
        self.audio_multi.normalize(
            norm_dbfs=-20,
            peak_normalization=True,
            each_channel=True,
        )

    def test_fade(self):
        # Functionality - result only tested for linear fade
        self.audio_multi.fade(fade_type=dsp.FadeType.Linear)
        self.audio_multi.fade(fade_type=dsp.FadeType.Logarithmic)
        self.audio_multi.fade(fade_type=dsp.FadeType.Exponential)

        f_end = self.audio_multi.fade(
            fade_type=dsp.FadeType.Linear,
            at_start=False,
            at_end=True,
        )
        f_st = self.audio_multi.fade(
            fade_type=dsp.FadeType.Linear,
            at_start=True,
            at_end=False,
        )
        with pytest.raises(AssertionError):
            self.audio_multi.fade(
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

    def test_true_peak_level_headroom_scaling(self):
        """Per the source, true peak is obtained by gain-reducing by
        12.04 dB, 4x-oversampling (`resample`), then restoring the gain --
        this recovers inter-sample overshoot invisible to the plain sample
        peak. For a sine well below Nyquist (negligible inter-sample
        overshoot when oversampled), true peak (dBTP) should be very close
        to the sample peak (dBFS); true peak must also never be smaller
        than the sample peak, by construction (oversampling can only reveal
        additional peaks, never hide the existing sample peak).

        """
        fs = 8_000
        n_samples = 4_000
        t = np.arange(n_samples) / fs
        amplitude = 0.7
        x = amplitude * np.sin(2 * np.pi * 50.0 * t)
        sig = dsp.Signal(None, x[:, None], fs, constrain_amplitude=False)

        true_peak_db, sample_peak_db = dsp.true_peak_level(sig)
        np.testing.assert_allclose(true_peak_db, sample_peak_db, atol=0.05)
        assert np.all(true_peak_db >= sample_peak_db - 1e-9)

        # A signal with real inter-sample overshoot (a near-Nyquist tone at
        # a phase where the sample grid straddles the true peak, found by
        # an empirical phase search) should show a true peak measurably
        # above its sample peak.
        x_steep = 0.9 * np.sin(2 * np.pi * (fs * 0.495) * t + 1.488)
        sig_steep = dsp.Signal(None, x_steep[:, None], fs, constrain_amplitude=False)
        tp_steep, sp_steep = dsp.true_peak_level(sig_steep)
        assert tp_steep[0] > sp_steep[0] + 1.0

    def test_fractional_delay(self):
        # Delay in seconds
        delay_s = 150 / self.fs

        # All channels
        s = self.audio_multi.fractional_delay(delay_s)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), 150))

        # Selected channels only
        s = self.audio_multi.fractional_delay(delay_s, channels=0)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), [150, 0, 0]))

    def test_delay(self):
        # Delay
        delay_samp = 150

        # All channels
        s = self.audio_multi.delay(delay_samp)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), 150))

        # Selected channels only
        s = self.audio_multi.delay(delay_samp, channels=0)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), [150, 0, 0]))

    def test_activity_detector(self):
        # Only functionality tested
        # Create harmonic signal and silence afterwards
        s = dsp.generators.oscillator(1000.0, sampling_rate_hz=self.fs)
        s = s.pad_trim(s.time_data.shape[0] * 2)
        s.activity_detector()

    def test_activity_detector_synthetic_burst_boundaries(self):
        """A signal built as silence -> tone burst -> silence should have
        its detected `signal_indices` boundaries close to the true burst
        boundaries, lagging by roughly the attack/release smoothing time
        (an EMA envelope follower, not an instantaneous threshold -- exact
        alignment isn't expected, but the lag should be on the order of
        the requested attack/release times, verified empirically: with a
        1 ms attack and 25 ms release at fs=8000 Hz, the onset lags by a
        few samples and the offset by a few hundred).

        """
        fs = 8_000
        silence_len = int(0.5 * fs)
        burst_len = int(0.5 * fs)
        t_burst = np.arange(burst_len) / fs
        tone = 0.8 * np.sin(2 * np.pi * 300 * t_burst)
        x = np.concatenate([np.zeros(silence_len), tone, np.zeros(silence_len)])
        sig = dsp.Signal(None, x[:, None], fs, constrain_amplitude=False)

        _, others = sig.activity_detector(
            threshold_dbfs=-20, attack_time_ms=1, release_time_ms=25
        )
        idx = others["signal_indices"]
        onset = np.argmax(idx)
        offset = len(idx) - 1 - np.argmax(idx[::-1])

        # Onset should lag the true burst start (silence_len) by only a
        # handful of samples (attack is fast); offset should lag the true
        # burst end (silence_len + burst_len) by no more than a few
        # release time constants, and in any case land well before the
        # end of the trailing silence.
        assert 0 <= onset - silence_len < 50
        assert 0 <= offset - (silence_len + burst_len) < 800
        assert offset < len(idx) - 1000

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
        s.detrend(polynomial_order=0)

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
        s.detrend(polynomial_order=1)

        # Large polynomial order
        s.detrend(polynomial_order=10)

        with pytest.raises(AssertionError):
            s.detrend(polynomial_order=-10)

    def test_detrend_removes_known_polynomial_trend(self):
        """Per the source, `detrend` fits and subtracts a `numpy.polyfit`
        polynomial of the requested order (using sample index, not time in
        seconds, as the fit's x-axis). Because least-squares fitting is a
        linear operator, `polyfit(clean + trend) == polyfit(clean) +
        trend` exactly whenever `trend` already lies exactly in the fitted
        polynomial's degree -- so detrending should recover exactly
        `clean - polyval(polyfit(clean), index)`, i.e. `clean` itself minus
        whatever quadratic component was already present by chance in the
        noise (not simply `clean` unchanged, since finite-sample noise is
        never perfectly orthogonal to a quadratic basis).

        """
        fs = 700
        n_samples = 2_000
        index = np.arange(n_samples)
        rng = np.random.default_rng(0)
        clean = rng.normal(0, 0.01, n_samples)
        clean -= clean.mean()

        a2, a1, a0 = 0.5, -1.2, 0.3
        trend = a2 * index**2 + a1 * index + a0
        trended = clean + trend

        sig = dsp.Signal(None, trended[:, None], fs, constrain_amplitude=False)
        detrended = sig.detrend(polynomial_order=2)

        poly_of_clean = np.polyfit(index, clean, deg=2)
        expected = clean - np.polyval(poly_of_clean, index)
        np.testing.assert_allclose(detrended.time_data[:, 0], expected, atol=1e-7)

        # The recovered residual should still be small relative to the
        # trend that was removed (order-of-magnitude sanity check).
        assert np.max(np.abs(detrended.time_data[:, 0])) < 0.1 * np.max(np.abs(trend))

    def test_load_pkl_object(self):
        f = dsp.Filter.fir_filter(
            order=216,
            frequency_hz=1000,
            type_of_pass=dsp.FilterPassType.Highpass,
            sampling_rate_hz=self.fs,
        )
        f.save_filter(os.path.join("tests", "f"))

        # Format is inferred/checked the same way with or without the
        # extension already present in `path`
        reloaded_no_ext = dsp.load_pkl_object(os.path.join("tests", "f"))
        reloaded_with_ext = dsp.load_pkl_object(os.path.join("tests", "f.pkl"))
        for reloaded in (reloaded_no_ext, reloaded_with_ext):
            assert type(reloaded) is dsp.Filter
            np.testing.assert_array_equal(reloaded.ba[0], f.ba[0])
            np.testing.assert_array_equal(reloaded.ba[1], f.ba[1])

        with pytest.raises(AssertionError):
            # Mismatched extension is rejected before even trying to open
            # the file
            dsp.load_pkl_object(os.path.join("tests", "f.txt"))

        os.remove(os.path.join("tests", "f.pkl"))

    def test_rms(self):
        td = self.audio_multi.time_data[:, 0]
        rms_vals = dsp.rms(self.audio_multi, in_dbfs=False)
        assert np.isclose(np.sqrt(np.mean(td**2)), rms_vals[0])

    def test_lufs_integrated(self):
        # Only functionality
        dsp.lufs_integrated(self.audio_multi)
        n = dsp.generators.oscillator(
            997,
            48000,
            length_seconds=2.0,
            peak_level_dbfs=0.0,
            number_of_channels=1,
        )
        np.testing.assert_allclose(dsp.lufs_integrated(n), -3.01, atol=0.07)

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
        )[0]
        new_sig = fb.filter_signal(self.audio_multi, dsp.FilterBankMode.Parallel)
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

    def test_envelope_matches_scipy_hilbert_reference(self):
        """Per the source, `envelope(analytic=True)` first linearly
        detrends the signal (`Signal.detrend(1)`, itself independently
        covered by `test_detrend`), then returns `abs(scipy.signal.hilbert(
        ...))` directly with no further processing -- a genuine exact
        passthrough, not an approximation. For a constant-amplitude sine
        (many full periods, so the linear detrend is a near-no-op), the
        envelope should also be close to the constant amplitude away from
        the Hilbert transform's edge-ringing region.

        """
        fs = 5_000
        freq = 200.0
        n_periods = 100
        n_samples = int(n_periods * fs / freq)
        t = np.arange(n_samples) / fs
        amplitude = 0.6
        x = amplitude * np.sin(2 * np.pi * freq * t)
        sig = dsp.Signal(None, x[:, None], fs, constrain_amplitude=False)

        env = dsp.envelope(sig, analytic=True)

        detrended = sig.detrend(1)
        expected = np.abs(hilbert(detrended.time_data, axis=0))
        np.testing.assert_allclose(env, expected, atol=1e-12)

        # Away from the edges (Hilbert-transform ringing), the envelope of
        # a constant-amplitude tone should stay close to that amplitude.
        interior = env[n_samples // 10 : -n_samples // 10, 0]
        np.testing.assert_allclose(interior, amplitude, atol=0.01)

    def test_dither(self):
        # Functionality
        self.audio_multi.dither()

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
        self.audio_multi.dither(noise_shaping_filterbank=fb)
        self.audio_multi.dither(truncate=False)

    def test_dither_adds_bounded_noise(self):
        """Per the source, triangular-distribution dither (the default) is
        the sum of two `Uniform(-eps/2, eps/2)` draws, bounded exactly by
        `[-eps, eps]`; rectangular dither is a single `Uniform(-eps/2,
        eps/2)` draw, bounded by `[-eps/2, eps/2]`. `epsilon` defaults to
        the float16 smallest subnormal (~5.96e-8).

        """
        fs = 8_000
        sig = dsp.Signal(None, np.zeros((4_000, 2)), fs, constrain_amplitude=False)
        epsilon = float(np.finfo(np.float16).smallest_subnormal)

        dithered_tri = sig.dither(triangular_distribution=True)
        noise_tri = dithered_tri.time_data - sig.time_data
        assert np.all(np.abs(noise_tri) <= epsilon)
        assert np.std(noise_tri) > 0

        dithered_rect = sig.dither(triangular_distribution=False)
        noise_rect = dithered_rect.time_data - sig.time_data
        assert np.all(np.abs(noise_rect) <= epsilon / 2)

    def test_apply_gain(self):
        some_signal = self.audio_multi.copy()
        # Signal
        audio_multi = some_signal.apply_gain(5)
        np.testing.assert_array_equal(
            audio_multi.time_data,
            some_signal.time_data * dsp.tools.from_db(5, True),
        )

        gains = np.linspace(1, 5, some_signal.number_of_channels)
        audio_multi = some_signal.apply_gain(gains)
        np.testing.assert_array_equal(
            audio_multi.time_data,
            some_signal.time_data * dsp.tools.from_db(gains, True),
        )

        audio_multi = some_signal.apply_gain(gains)
        np.testing.assert_array_equal(
            audio_multi.time_data,
            some_signal.time_data * dsp.tools.from_db(gains, True),
        )

        # MultiBandSignal
        audio_multi_mb = self.get_multiband_signal()
        previous = audio_multi_mb.get_all_time_data()[0]
        audio_multi_mb = audio_multi_mb.apply_gain(5)
        np.testing.assert_array_equal(
            previous * dsp.tools.from_db(5, True),
            audio_multi_mb.get_all_time_data()[0],
        )

        previous = audio_multi_mb.get_all_time_data()[0]
        gains = np.linspace(1, 5, audio_multi.number_of_channels)
        audio_multi_mb = audio_multi_mb.apply_gain(gains)
        np.testing.assert_array_equal(
            previous * dsp.tools.from_db(gains, True),
            audio_multi_mb.get_all_time_data()[0],
        )

        # Filter
        iir = dsp.Filter.biquad(dsp.BiquadEqType.Peaking, 500.0, 0.0, 0.7, self.fs)
        output_level1 = dsp.rms(iir.filter_signal(self.audio_multi))
        gain_db = -5.0
        output_level2 = dsp.rms(iir.apply_gain(gain_db).filter_signal(self.audio_multi))
        np.testing.assert_array_almost_equal(output_level1 + gain_db, output_level2)
        with pytest.raises(ValueError):
            iir.apply_gain([gain_db, 0])

        # FilterBank
        fb = dsp.FilterBank([iir, iir])
        base_level = dsp.rms(
            fb.filter_signal(self.audio_multi, mode=dsp.FilterBankMode.Sequential)
        )
        fb2 = fb.apply_gain(gain_db)
        output_level = dsp.rms(
            fb2.filter_signal(self.audio_multi, mode=dsp.FilterBankMode.Sequential)
        )
        np.testing.assert_array_almost_equal(
            base_level + gain_db * len(fb), output_level
        )

        # multiple gains for a filter bank
        fb2 = fb.apply_gain([gain_db] + [0] * (len(fb) - 1))
        output_level = dsp.rms(
            fb2.filter_signal(self.audio_multi, mode=dsp.FilterBankMode.Sequential)
        )
        np.testing.assert_array_almost_equal(base_level + gain_db, output_level)
        with pytest.raises(AssertionError):
            fb.apply_gain([gain_db] + [0] * (len(fb) + 1))

    def test_crest_factor(self):
        # Only functionality
        some_signal = self.audio_multi.copy()
        dsp.crest_factor(some_signal, False)
        cf = dsp.crest_factor(some_signal, True)
        assert np.all(cf > 0.0)
        cf2 = dsp.crest_factor(some_signal, True, True)
        assert np.all(cf > 0.0) and np.all(cf2 >= cf)

        dsp.crest_factor(self.get_multiband_signal(), False)

    def test_crest_factor_matches_closed_form(self):
        """Crest factor is `peak / rms` (linear form, `in_db=False`); for a
        zero-mean sine that is exactly `A/sqrt(2)` (a textbook closed
        form), and for a zero-mean square wave, exactly 1 (peak equals
        rms). A tiny phase offset avoids exact-zero samples in the square
        wave, where `numpy.sign` returns 0 instead of +-1 and would
        introduce a small, uninteresting numerical artifact.

        """
        fs = 8_000
        n_samples = 4_000
        t = np.arange(n_samples) / fs
        freq = 100.0
        phase = 0.001

        sine = np.sin(2 * np.pi * freq * t + phase)
        square = np.sign(np.sin(2 * np.pi * freq * t + phase))

        sig_sine = dsp.Signal(None, sine[:, None], fs, constrain_amplitude=False)
        sig_square = dsp.Signal(None, square[:, None], fs, constrain_amplitude=False)

        cf_sine = dsp.crest_factor(sig_sine, in_db=False)
        cf_square = dsp.crest_factor(sig_square, in_db=False)

        np.testing.assert_allclose(cf_sine, np.sqrt(2), rtol=1e-6)
        np.testing.assert_allclose(cf_square, 1.0, rtol=1e-6)

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
        f.resample_filter(24000)
        f = dsp.Filter.iir_filter(
            order=5,
            frequency_hz=500,
            type_of_pass=dsp.FilterPassType.Lowpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )
        f.resample_filter(24000)
        f = dsp.Filter.iir_filter(
            order=8,
            frequency_hz=500,
            type_of_pass=dsp.FilterPassType.Highpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )
        f.resample_filter(24000)
        f = dsp.Filter.iir_filter(
            order=7,
            frequency_hz=[500, 18e3],
            type_of_pass=dsp.FilterPassType.Bandpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )

    def test_modify_signal_length(self):
        with pytest.raises(AssertionError):
            self.audio_multi.modify_signal_length(None, None)
        with pytest.raises(AssertionError):
            self.audio_multi.modify_signal_length(
                -(self.audio_multi.length_seconds + 1.0),
                None,
            )
        with pytest.raises(AssertionError):
            self.audio_multi.modify_signal_length(
                None,
                -(self.audio_multi.length_seconds + 1.0),
            )
        with pytest.raises(AssertionError):
            self.audio_multi.modify_signal_length(
                -self.audio_multi.length_seconds / 2.0,
                -self.audio_multi.length_seconds / 1.9,
            )

        original_length = len(self.audio_multi)

        # Add both
        new = self.audio_multi.modify_signal_length(1.0, 1.0)
        assert new.length_seconds == self.audio_multi.length_seconds + 2.0
        assert len(self.audio_multi) == original_length

        # Add only start
        new = self.audio_multi.modify_signal_length(1.0, None)
        assert new.length_seconds == self.audio_multi.length_seconds + 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(new.time_data[: new.sampling_rate_hz], 0.0)

        # Add only end
        new = self.audio_multi.modify_signal_length(None, 1.0)
        assert new.length_seconds == self.audio_multi.length_seconds + 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(new.time_data[-new.sampling_rate_hz :], 0.0)

        # Remove both
        new = self.audio_multi.modify_signal_length(-0.5, -0.5)
        assert new.length_seconds == self.audio_multi.length_seconds - 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data,
            self.audio_multi.time_data[
                new.sampling_rate_hz // 2 : -new.sampling_rate_hz // 2
            ],
        )

        # Remove only start
        new = self.audio_multi.modify_signal_length(-0.5, None)
        assert new.length_seconds == self.audio_multi.length_seconds - 0.5
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data,
            self.audio_multi.time_data[new.sampling_rate_hz // 2 :],
        )

        # Remove only end
        new = self.audio_multi.modify_signal_length(None, -0.5)
        assert new.length_seconds == self.audio_multi.length_seconds - 0.5
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data,
            self.audio_multi.time_data[: -new.sampling_rate_hz // 2],
        )

        # Mixed
        new = self.audio_multi.modify_signal_length(1.5, -0.5)
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
        mb.modify_signal_length(1.5, -0.5)

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

        f3 = dsp.FilterBank([f1, f2]).merge_filters()
        np.testing.assert_array_equal(f3.ba[0][delay:], f1.ba[0])

        # With filterbank
        f3 = dsp.FilterBank([f1, f2]).merge_filters()
        np.testing.assert_array_equal(f3.ba[0][delay:], f1.ba[0])

        with pytest.raises(AssertionError):
            dsp.FilterBank([f1]).merge_filters()

        with pytest.raises(AssertionError):
            iir = dsp.Filter.biquad(
                dsp.BiquadEqType.LowpassFirstOrder, 50.0, -3.0, 0.7, self.fs
            )
            dsp.FilterBank([f1, iir]).merge_filters()

        with pytest.raises(AssertionError):
            f2 = dsp.Filter.from_ba(dirac, [1.0], self.fs * 2)
            dsp.FilterBank([f1, f2]).merge_filters()

    def test_merge_iir_filters(self):
        f1 = dsp.Filter.biquad(
            eq_type=dsp.BiquadEqType.Allpass,
            frequency_hz=500.0,
            gain_db=5.0,
            q=0.7,
            sampling_rate_hz=self.fs,
        )

        f3 = dsp.FilterBank([f1, f1.copy()]).merge_filters()
        assert f3.has_sos
        assert f3.sos.shape[0] == 2

    def test_spectral_difference(self):
        filt = dsp.Filter.biquad(dsp.BiquadEqType.Peaking, 500.0, 10.0, 1.0, 48000)
        spec = dsp.Spectrum.from_filter(
            dsp.tools.log_frequency_vector([20, 20e3], 128), filt, False
        )
        spec_flat = dsp.Spectrum.from_filter(
            dsp.tools.log_frequency_vector([20, 20e3], 128),
            dsp.Filter.biquad(dsp.BiquadEqType.Peaking, 500.0, 0.0, 1.0, 48000),
            False,
        )
        sp_out = spec.spectral_difference(spec_flat, energy_normalization=False)
        np.testing.assert_almost_equal(spec.spectral_data, sp_out.spectral_data)

        with pytest.raises(AssertionError):
            sp_out = spec.spectral_difference(
                spec_flat, energy_normalization=False, complex=True
            )

        # Some different parameters
        spec.spectral_difference(
            spec_flat,
            energy_normalization=True,
            octave_fraction_smoothing=12.0,
            dynamic_range_db=None,
        )

    def test_trim_with_level_threshold(self):
        s = np.zeros(1000)

        # ===== Single-channel
        ones_slice = slice(len(s) // 3, len(s) // 2)

        threshold_db = -50.0
        fill_value = dsp.tools.from_db(threshold_db + 1, True)

        s[ones_slice] = fill_value

        # Start and end
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(threshold_db, True, True)[0]
            .time_data.squeeze(),
            s[ones_slice],
        )
        # End
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(
                threshold_db,
                False,
                True,
            )[0]
            .time_data.squeeze(),
            s[: ones_slice.stop],
        )
        # Start
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(
                threshold_db,
                True,
                False,
            )[0]
            .time_data.squeeze(),
            s[ones_slice.start :],
        )
        # None
        with pytest.raises(AssertionError):
            (dsp.Signal.from_time_data(s, self.fs)).trim_with_level_threshold(
                threshold_db,
                False,
                False,
            )

        # ===== Multi-channel
        s = np.zeros((1000, 2))
        ones_slice = slice(len(s) // 3, len(s) // 2)
        s[ones_slice.start + 5 : ones_slice.stop - 5, 0] = fill_value
        s[ones_slice, 1] = fill_value

        threshold_db = -50.0
        fill_value = dsp.tools.from_db(threshold_db + 1, True)

        # Start and end
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(threshold_db, True, True)[0]
            .time_data,
            s[ones_slice],
        )
        # End
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(
                threshold_db,
                False,
                True,
            )[0]
            .time_data,
            s[: ones_slice.stop],
        )
        # Start
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(
                threshold_db,
                True,
                False,
            )[0]
            .time_data,
            s[ones_slice.start :],
        )
        # None
        with pytest.raises(AssertionError):
            (dsp.Signal.from_time_data(s, self.fs)).trim_with_level_threshold(
                threshold_db,
                False,
                False,
            )

    def test_select_time_window(self):
        s = self.audio_multi
        s2 = s.trim_with_time_selection(0.1, 0.3, True)
        assert abs(s2.length_seconds - 0.2) <= 1 / s.sampling_rate_hz
        s.trim_with_time_selection(0.1, 0.3, False)
        s.trim_with_time_selection(None, 0.3, False)
        s.trim_with_time_selection(0.1, None, False)

        mbs = self.get_multiband_signal()
        mbs.trim_with_time_selection(0.1, 0.3, False)
        mbs.trim_with_time_selection(0.1, 0.3, True)

        with pytest.raises(AssertionError):
            s.trim_with_time_selection(0.3, 0.1, False)
        with pytest.raises(AssertionError):
            s.trim_with_time_selection(0.1, s.length_seconds + 1.0, False)
        with pytest.raises(AssertionError):
            s.trim_with_time_selection(None, None, False)
