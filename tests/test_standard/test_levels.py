import numpy as np
import pytest

import dsptoolbox as dsp


class TestStandardModule:
    fs = 44100
    audio_multi = dsp.generators.noise(2, fs, number_of_channels=3, rng=127)

    def get_multiband_signal(self) -> dsp.MultiBandSignal:
        fb = dsp.filterbanks.linkwitz_riley_crossovers([1e3], [4], self.fs)
        return fb.filter_signal(self.audio_multi)

    def test_normalize(self):
        # Peak normalization
        td = self.audio_multi.time_data
        n = self.audio_multi.normalize(norm_dbfs=-20)
        td /= np.max(np.abs(td))
        factor = 10 ** (-20 / 20)
        td *= factor
        assert np.isclose(np.max(np.abs(n.time_data)), np.max(np.abs(td)))

        # RMS normalization
        channel = self.audio_multi.get_channels(0)
        rms_previous = dsp.rms(channel)[0]
        n = channel.normalize(norm_dbfs=rms_previous - 10, peak_normalization=False)
        rms = dsp.rms(n)[0]
        assert np.isclose(rms_previous - 10, rms)

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

    def test_true_peak_level(self):
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

    def test_rms(self):
        td = self.audio_multi.time_data[:, 0]
        rms_vals = dsp.rms(self.audio_multi, in_dbfs=False)
        assert np.isclose(np.sqrt(np.mean(td**2)), rms_vals[0])

    def test_lufs_integrated(self):
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
        sine = dsp.generators.oscillator(
            frequency_hz=100.0,
            sampling_rate_hz=self.audio_multi.sampling_rate_hz,
            peak_level_dbfs=-20,
        )
        calib = dsp.CalibrationData(sine)
        calib.calibrate_signal(self.audio_multi)

        with pytest.raises(AssertionError):
            sine = dsp.generators.oscillator(
                frequency_hz=1000.0,
                sampling_rate_hz=self.audio_multi.sampling_rate_hz,
                peak_level_dbfs=-20,
                number_of_channels=self.audio_multi.number_of_channels - 1,
            )
            calib = dsp.CalibrationData(sine)
            calib.calibrate_signal(self.audio_multi)

        sine = dsp.generators.oscillator(
            frequency_hz=1000.0,
            sampling_rate_hz=self.audio_multi.sampling_rate_hz,
            peak_level_dbfs=-20,
            number_of_channels=self.audio_multi.number_of_channels,
        )
        calib = dsp.CalibrationData(sine)
        calib.calibrate_signal(self.audio_multi)

        fb = dsp.filterbanks.fractional_octave_bands(
            [125, 1000], sampling_rate_hz=self.audio_multi.sampling_rate_hz
        )[0]
        new_sig = fb.filter_signal(self.audio_multi, dsp.FilterBankMode.Parallel)
        calib.calibrate_signal(new_sig)

    def test_dither(self):
        self.audio_multi.dither(rng=128)

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
        self.audio_multi.dither(noise_shaping_filterbank=fb, rng=129)
        self.audio_multi.dither(truncate=False, rng=130)

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

        dithered_tri = sig.dither(triangular_distribution=True, rng=131)
        noise_tri = dithered_tri.time_data - sig.time_data
        assert np.all(np.abs(noise_tri) <= epsilon)
        assert np.std(noise_tri) > 0

        dithered_rect = sig.dither(triangular_distribution=False, rng=132)
        noise_rect = dithered_rect.time_data - sig.time_data
        assert np.all(np.abs(noise_rect) <= epsilon / 2)

    def test_apply_gain(self):
        some_signal = self.audio_multi.copy()
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

        # Multiple gains for a filter bank
        fb2 = fb.apply_gain([gain_db] + [0] * (len(fb) - 1))
        output_level = dsp.rms(
            fb2.filter_signal(self.audio_multi, mode=dsp.FilterBankMode.Sequential)
        )
        np.testing.assert_array_almost_equal(base_level + gain_db, output_level)
        with pytest.raises(AssertionError):
            fb.apply_gain([gain_db] + [0] * (len(fb) + 1))

    def test_crest_factor(self):
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

    def test_rms_includes_dc_offset(self):
        """RMS is sqrt(mean(x**2)), so a constant signal has RMS = |value|."""
        fs = 48000
        constant = dsp.Signal(None, np.ones((1000, 1)) * 2.0, fs)
        np.testing.assert_allclose(dsp.rms(constant, in_dbfs=False), 2.0, rtol=1e-12)

        rng = np.random.default_rng(0)
        ac = rng.normal(0, 1.0, (4000, 1))
        without_offset = dsp.Signal(None, ac, fs)
        with_offset = dsp.Signal(None, ac + 1.0, fs)
        assert (
            dsp.rms(with_offset, in_dbfs=False)[0]
            > dsp.rms(without_offset, in_dbfs=False)[0]
        )

    def test_crest_factor_multibandsignal_not_double_converted(self):
        """The MultiBandSignal path must not apply `to_db` twice."""
        fs = 44100
        rng = np.random.default_rng(0)
        s = dsp.Signal(None, rng.normal(0, 0.1, (fs, 2)), fs)
        fb = dsp.filterbanks.linkwitz_riley_crossovers([1000], [4], fs)
        mb = fb.filter_signal(s, dsp.FilterBankMode.Parallel)

        per_band = dsp.crest_factor(mb)
        for ind, band in enumerate(mb.bands):
            np.testing.assert_allclose(
                per_band[ind, :], dsp.crest_factor(band), rtol=1e-10
            )

    def test_true_peak_level_does_not_modify_input(self):
        fs = 48000
        rng = np.random.default_rng(0)
        s = dsp.Signal(None, rng.normal(0, 0.1, (4096, 2)), fs)
        before = s.time_data.copy()
        dsp.true_peak_level(s)
        np.testing.assert_array_equal(before, s.time_data)

    def test_calibration_data_low_snr_matches_high_snr(self):
        """`high_snr=False` derives the RMS from the 1 kHz spectrum instead of
        the time domain. For a pure 1 kHz tone both routes must agree.

        """
        fs = 48000
        t = np.arange(fs) / fs
        tone = dsp.Signal(
            None, (2.0**0.5 * np.sin(2 * np.pi * 1000.0 * t))[:, None], fs
        )

        high = dsp.CalibrationData(tone, calibration_spl_db=94.0, high_snr=True)
        low = dsp.CalibrationData(tone, calibration_spl_db=94.0, high_snr=False)
        high._compute_calibration_factors()
        low._compute_calibration_factors()

        # 94 dB SPL is 1 Pa RMS and the tone has unit RMS
        np.testing.assert_allclose(high.calibration_factors, 1.0, atol=1e-2)
        np.testing.assert_allclose(
            low.calibration_factors, high.calibration_factors, rtol=1e-3
        )
