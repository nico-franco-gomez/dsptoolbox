"""
Tests regarding functionality of audio fx
"""

import os
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp


class TestEffectsModule:
    speech = (
        dsp.Signal(join(os.path.dirname(__file__), "..", "example_data", "speech.flac"))
    ).resample(
        8_000,
    )
    fs_hz = speech.sampling_rate_hz

    def testSpectralSubtractor(self):
        """Test functionality of the spectral subtractor."""
        # Adaptive
        specSub = dsp.effects.SpectralSubtractor(
            adaptive_mode=True,
            threshold_rms_dbfs=-30,
            block_length_s=0.15,
            spectrum_to_subtract=False,
        )
        specSub.set_advanced_parameters(
            overlap_percent=75,
            window_type=dsp.Window.Hamming,
            noise_forgetting_factor=0.95,
            subtraction_factor=3,
            subtraction_exponent=3,
            ad_attack_time_ms=1.5,
            ad_release_time_ms=30,
        )
        specSub.apply(self.speech)

        # Non-adaptive
        specSub = dsp.effects.SpectralSubtractor(
            adaptive_mode=False,
            threshold_rms_dbfs=-10,
            block_length_s=0.05,
            spectrum_to_subtract=False,
        )
        specSub.set_advanced_parameters(
            overlap_percent=50,
            window_type=dsp.Window.Hamming,
            noise_forgetting_factor=0.9,
            subtraction_factor=1,
            subtraction_exponent=1,
            ad_attack_time_ms=1.5,
            ad_release_time_ms=30,
        )
        specSub.apply(self.speech)

        # With imported spectrum
        spectrum_to_subtract = np.random.uniform(0, 1, specSub.window_length)
        specSub.set_parameters(spectrum_to_subtract=spectrum_to_subtract)
        specSub.apply(self.speech)

    def testSpectralSubtractorReducesNoise(self):
        """A stationary broadband noise floor added to intermittent tone
        bursts should be attenuated more than the tone itself, improving
        the SNR relative to the noisy input. No closed-form reference exists
        for spectral subtraction, so this is a plausibility check.

        """
        fs = 8_000
        n_samples = fs * 3
        t = np.arange(n_samples) / fs
        tone_freq = 500.0

        # Intermittent tone bursts so silent segments exist for the
        # activity detector to learn the noise spectrum from.
        clean = np.zeros(n_samples)
        burst_len = int(0.25 * fs)
        period = int(0.5 * fs)
        for start in range(0, n_samples, period):
            end = min(start + burst_len, n_samples)
            clean[start:end] = 0.5 * np.sin(2 * np.pi * tone_freq * t[start:end])

        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.05, n_samples)
        noisy = clean + noise

        clean_sig = dsp.Signal(None, clean[:, None], fs)
        noisy_sig = dsp.Signal(None, noisy[:, None], fs)

        specSub = dsp.effects.SpectralSubtractor(
            adaptive_mode=True,
            threshold_rms_dbfs=-25,
            block_length_s=0.05,
            spectrum_to_subtract=False,
        )
        denoised = specSub.apply(noisy_sig)

        snr_before = dsp.distances.snr(
            clean_sig, dsp.Signal(None, (noisy - clean)[:, None], fs)
        )[0]
        residual_after = denoised.time_data[:, 0] - clean
        snr_after = dsp.distances.snr(
            clean_sig, dsp.Signal(None, residual_after[:, None], fs)
        )[0]

        assert snr_after > snr_before

    def testSpectralSubtractorPassesCleanSignalThrough(self):
        """A signal with no low-level segments never triggers the activity
        detector, so the noise estimate stays at zero and the output should
        remain very close to the input (plausibility, low added distortion).

        """
        fs = 8_000
        n_samples = fs
        t = np.arange(n_samples) / fs
        clean = 0.5 * np.sin(2 * np.pi * 400 * t)
        sig = dsp.Signal(None, clean[:, None], fs)

        specSub = dsp.effects.SpectralSubtractor(
            adaptive_mode=True,
            threshold_rms_dbfs=-60,
            block_length_s=0.05,
            spectrum_to_subtract=False,
        )
        out = specSub.apply(sig)

        error_power = np.mean((out.time_data[:, 0] - clean) ** 2)
        signal_power = np.mean(clean**2)
        assert error_power / signal_power < 1e-3

    def testSpectralSubtractorInvalidParametersRaise(self):
        specSub = dsp.effects.SpectralSubtractor()
        with pytest.raises(AssertionError):
            specSub.set_parameters(adaptive_mode="yes")
        with pytest.raises(AssertionError):
            specSub.set_parameters(threshold_rms_dbfs="loud")
        with pytest.raises(AssertionError):
            specSub.set_advanced_parameters(overlap_percent=100)
        with pytest.raises(AssertionError):
            specSub.set_advanced_parameters(noise_forgetting_factor=0)
        with pytest.raises(AssertionError):
            specSub.set_advanced_parameters(subtraction_factor=0)
        with pytest.raises(AssertionError):
            specSub.set_advanced_parameters(subtraction_exponent=0)

    def testDistortion(self):
        """Test different distortion parameters."""
        dist = dsp.effects.Distortion(
            distortion_level=25,
            post_gain_db=0,
            type_of_distortion=dsp.effects.DistortionType.Arctan,
        )
        dist.apply(self.speech)

        dist.set_advanced_parameters(
            type_of_distortion=[
                dsp.effects.DistortionType.Arctan,
                dsp.effects.DistortionType.SoftClip,
            ],
            distortion_levels_db=[20, 40],
            mix_percent=[60, 40],
            offset_db=[-3, -np.inf],
            post_gain_db=2,
        )
        dist.apply(self.speech)

    def testDistortionHardClipMatchesNpClip(self):
        """With a single distortion type and offset -inf, the effect chain
        reduces to a closed form: clip the peak-normalized, gain-scaled
        signal with `np.clip`, then rescale to the original peak level
        (peak restoration is part of `_apply_this_effect`).

        """
        fs = 8_000
        t = np.arange(4_000) / fs
        x = 0.7 * np.sin(2 * np.pi * 200 * t)
        sig = dsp.Signal(None, x[:, None], fs)

        distortion_level_db = 15.0
        dist = dsp.effects.Distortion(
            distortion_level=distortion_level_db,
            post_gain_db=0,
            type_of_distortion=dsp.effects.DistortionType.HardClip,
        )
        out = dist.apply(sig)

        peak = np.max(np.abs(x))
        level_linear = 10 ** (distortion_level_db / 20)
        normalized = x / peak
        raw = np.clip(normalized * level_linear, -1, 1)
        expected = peak * raw / np.max(np.abs(raw))

        np.testing.assert_allclose(out.time_data[:, 0], expected, atol=1e-10)

    def testDistortionPeakIsRestored(self):
        """Per the class docstring, the original peak level is always kept
        after distortion regardless of drive level or type; this checks
        that documented invariant directly (plausibility, no closed form
        for the interior samples).

        """
        fs = 8_000
        ramp = np.linspace(-1, 1, 2_000)
        sig = dsp.Signal(None, ramp[:, None], fs)

        for dist_type in (
            dsp.effects.DistortionType.Arctan,
            dsp.effects.DistortionType.HardClip,
            dsp.effects.DistortionType.SoftClip,
        ):
            dist = dsp.effects.Distortion(
                distortion_level=30,
                post_gain_db=0,
                type_of_distortion=dist_type,
            )
            out = dist.apply(sig).time_data[:, 0]
            np.testing.assert_allclose(np.max(np.abs(out)), 1.0, atol=1e-8)

    def testDistortionArctanIsMonotonic(self):
        """Arctan distortion is a strictly increasing nonlinearity at any
        drive level, so it must never reorder a monotonically increasing
        ramp (plausibility, no closed form).

        """
        fs = 8_000
        ramp = np.linspace(-1, 1, 2_000)
        sig = dsp.Signal(None, ramp[:, None], fs)

        dist = dsp.effects.Distortion(
            distortion_level=30,
            post_gain_db=0,
            type_of_distortion=dsp.effects.DistortionType.Arctan,
        )
        out = dist.apply(sig).time_data[:, 0]
        assert np.all(np.diff(out) >= -1e-8)

    def testDistortionSoftClipMonotonicAtLowDrive(self):
        """The cubic soft-clip waveshaper (z - z**3/3) is only monotonic
        for |z| <= 1; at low drive the whole ramp stays in that region, so
        monotonicity is a valid plausibility check there (it folds back at
        high drive by design and is intentionally excluded from this check).

        """
        fs = 8_000
        ramp = np.linspace(-1, 1, 2_000)
        sig = dsp.Signal(None, ramp[:, None], fs)

        dist = dsp.effects.Distortion(
            distortion_level=0,
            post_gain_db=0,
            type_of_distortion=dsp.effects.DistortionType.SoftClip,
        )
        out = dist.apply(sig).time_data[:, 0]
        assert np.all(np.diff(out) >= -1e-8)

    def testDistortionInvalidParametersRaise(self):
        with pytest.raises(ValueError):
            dsp.effects.Distortion(type_of_distortion="not-a-type")
        dist = dsp.effects.Distortion()
        with pytest.raises(AssertionError):
            dist.set_advanced_parameters(mix_percent=150)
        with pytest.raises(AssertionError):
            dist.set_advanced_parameters(
                type_of_distortion=[
                    dsp.effects.DistortionType.Arctan,
                    dsp.effects.DistortionType.HardClip,
                ],
                mix_percent=[30, 30],
            )

    def testCompressor(self):
        comp = dsp.effects.Compressor(
            threshold_dbfs=-10,
            attack_time_ms=2,
            release_time_ms=30,
            ratio=5,
            relative_to_peak_level=True,
        )
        comp.set_advanced_parameters(
            knee_factor_db=5,
            pre_gain_db=1,
            post_gain_db=-2,
            mix_percent=99,
            automatic_make_up_gain=True,
            downward_compression=True,
        )
        comp.apply(self.speech)

        comp.set_parameters(attack_time_ms=1, ratio=3, threshold_dbfs=-10)
        comp.set_advanced_parameters(
            knee_factor_db=2,
            pre_gain_db=0,
            post_gain_db=0,
            mix_percent=99,
            automatic_make_up_gain=False,
            downward_compression=True,
        )
        comp.apply(self.speech)

        comp.show_compression()

    def testCompressorSteadyStateGainMatchesClosedForm(self):
        """For a constant-amplitude input above the threshold (hard knee),
        the compressor's attack/release EMA converges to a fixed gain
        given by the standard downward-compression formula:
        `gain_db = (threshold_db - level_db) * (1 - 1/ratio)`.

        """
        fs = 8_000
        level = 0.5
        threshold_dbfs = -20.0
        ratio = 4.0

        n_samples = 4_000
        x = np.full(n_samples, level)
        sig = dsp.Signal(None, x[:, None], fs)

        comp = dsp.effects.Compressor(
            threshold_dbfs=threshold_dbfs,
            attack_time_ms=1.0,
            release_time_ms=1.0,
            ratio=ratio,
            relative_to_peak_level=False,
        )
        comp.set_advanced_parameters(
            knee_factor_db=0,
            pre_gain_db=0,
            post_gain_db=0,
            mix_percent=100,
            automatic_make_up_gain=False,
            downward_compression=True,
        )
        out = comp.apply(sig)

        level_db = 20 * np.log10(level)
        compressed_db = threshold_dbfs + (level_db - threshold_dbfs) / ratio
        expected_gain_db = compressed_db - level_db
        expected_level = level * 10 ** (expected_gain_db / 20)

        # Attack/release time constants are ~1 ms (~8 samples); well past
        # 20 time constants the EMA has converged to numerical precision.
        steady_state = out.time_data[-500:, 0]
        np.testing.assert_allclose(steady_state, expected_level, rtol=1e-6)

    def testCompressorMixBlendsWithDrySignal(self):
        """`mix_percent` below 100 should blend the compressed signal with
        the untouched dry signal: `out = dry*(1-mix) + compressed*mix`.

        """
        fs = 8_000
        level = 0.5
        threshold_dbfs = -20.0
        ratio = 4.0
        mix_percent = 40.0

        n_samples = 4_000
        x = np.full(n_samples, level)
        sig = dsp.Signal(None, x[:, None], fs)

        def make_compressor(mix):
            comp = dsp.effects.Compressor(
                threshold_dbfs=threshold_dbfs,
                attack_time_ms=1.0,
                release_time_ms=1.0,
                ratio=ratio,
                relative_to_peak_level=False,
            )
            comp.set_advanced_parameters(
                knee_factor_db=0,
                pre_gain_db=0,
                post_gain_db=0,
                mix_percent=mix,
                automatic_make_up_gain=False,
                downward_compression=True,
            )
            return comp

        fully_compressed = make_compressor(100).apply(sig).time_data[-500:, 0]
        mixed = make_compressor(mix_percent).apply(sig).time_data[-500:, 0]

        expected_mixed = level * (1 - mix_percent / 100) + fully_compressed * (
            mix_percent / 100
        )
        np.testing.assert_allclose(mixed, expected_mixed, rtol=1e-6)

    def testCompressorInvalidParametersRaise(self):
        with pytest.raises(AssertionError):
            dsp.effects.Compressor(ratio=0.5)
        comp = dsp.effects.Compressor()
        with pytest.raises(AssertionError):
            comp.set_parameters(attack_time_ms=-1)
        with pytest.raises(AssertionError):
            comp.set_parameters(release_time_ms=-1)
        with pytest.raises(AssertionError):
            comp.set_parameters(ratio=0.9)
        with pytest.raises(AssertionError):
            comp.set_advanced_parameters(knee_factor_db=-1)
        with pytest.raises(AssertionError):
            comp.set_advanced_parameters(mix_percent=0)
        with pytest.raises(AssertionError):
            comp.set_advanced_parameters(mix_percent=150)

    def testLFO(self):
        l_osc = dsp.effects.LFO(
            frequency_hz=100, waveform="triangle", random_phase=True, smooth=5
        )
        l_osc.plot_waveform()
        l_osc.get_waveform(self.fs_hz, 2000)

        l_osc.set_parameters(
            frequency_hz=("dotted quarter", 130), waveform="sawtooth", smooth=0
        )
        l_osc.plot_waveform()
        l_osc.get_waveform(self.fs_hz, 2000)

    def testLFOInvalidParametersRaise(self):
        with pytest.raises(ValueError):
            dsp.effects.LFO(1.0, waveform="not-a-waveform")
        with pytest.raises(TypeError):
            dsp.effects.LFO(frequency_hz="not-a-frequency")
        with pytest.raises(AssertionError):
            dsp.effects.LFO(frequency_hz=(1, 2, 3))
        with pytest.raises(ValueError):
            dsp.effects.get_frequency_from_musical_rhythm("not-a-note", 60)

    def testTremolo(self):
        l_osc = dsp.effects.LFO(
            frequency_hz=("dotted quarter", 130), waveform="sawtooth", smooth=0
        )
        trem = dsp.effects.Tremolo(depth=0.8, modulator=l_osc)
        trem.apply(self.speech)

    def testTremoloMatchesClosedForm(self):
        """The tremolo effect multiplies the carrier by
        `abs(1 + depth * LFO(t))`; for a harmonic LFO with zero phase this
        can be reconstructed exactly with an independently written sine.

        """
        fs = 8_000
        n_samples = 4_000
        t = np.arange(n_samples) / fs
        carrier_freq = 300.0
        lfo_freq = 5.0
        depth = 0.6

        x = 0.5 * np.sin(2 * np.pi * carrier_freq * t)
        sig = dsp.Signal(None, x[:, None], fs)

        lfo = dsp.effects.LFO(lfo_freq, "harmonic", random_phase=False, smooth=0)
        trem = dsp.effects.Tremolo(depth=depth, modulator=lfo)
        out = trem.apply(sig)

        lfo_wave = np.sin(2 * np.pi * lfo_freq * t)
        expected = x * np.abs(1 + depth * lfo_wave)

        np.testing.assert_allclose(out.time_data[:, 0], expected, atol=1e-12)

    def testTremoloInvalidParametersRaise(self):
        with pytest.raises(AssertionError):
            dsp.effects.Tremolo(depth=1.5, modulator=dsp.effects.LFO(1.0))
        with pytest.raises(AssertionError):
            dsp.effects.Tremolo(depth=0, modulator=dsp.effects.LFO(1.0))
        with pytest.raises(AssertionError):
            dsp.effects.Tremolo(depth=0.5, modulator="not-a-modulator")
        with pytest.raises(AssertionError):
            dsp.effects.Tremolo(
                depth=0.5, modulator=np.zeros((10, 2))
            )  # modulator must be 1D

    def testChorus(self):
        l_osc = dsp.effects.LFO(
            frequency_hz=("dotted quarter", 130), waveform="sawtooth", smooth=0
        )
        chor = dsp.effects.Chorus(
            depths_ms=10, base_delays_ms=25, modulators=l_osc, mix_percent=0.95
        )
        chor.apply(self.speech)

        chor.set_parameters(
            depths_ms=[10, 5, 7.5],
            base_delays_ms=25,
            modulators=[l_osc] * 3,
            mix_percent=0.95,
        )
        chor.apply(self.speech)

        chor.set_parameters(
            depths_ms=[10, 5, 7.5],
            base_delays_ms=[25, 20, 23],
            modulators=[l_osc] * 3,
            mix_percent=0.95,
        )
        chor.apply(self.speech)

    def testChorusSingleVoiceMatchesClosedForm(self):
        """With a single voice and zero modulation depth, the chorus reduces
        to a fixed-lag comb filter `y[n] = x[n] + x[n + delay]` (peak values
        restored afterwards). This is verified both as an exact array
        equality and via the cross-correlation peak lag against the dry
        input.

        """
        fs = 8_000
        n_samples = 2_000
        base_delay_ms = 5.0

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, n_samples)
        sig = dsp.Signal(None, x[:, None], fs)

        lfo = dsp.effects.LFO(2.0, "harmonic", random_phase=False)
        chor = dsp.effects.Chorus(
            depths_ms=0, base_delays_ms=base_delay_ms, modulators=lfo, mix_percent=100
        )
        out = chor.apply(sig).time_data[:, 0]

        d = int(round(base_delay_ms * 1e-3 * fs))
        x_padded = np.append(x, np.zeros(d))
        expected = x + x_padded[d : d + n_samples]

        peak_in = np.max(np.abs(x))
        peak_expected = np.max(np.abs(expected))
        expected_scaled = expected * (peak_in / peak_expected)

        np.testing.assert_allclose(out, expected_scaled, atol=1e-10)

        # Independent cross-check: the delayed copy x[n + d] embedded in the
        # output shows up in the cross-correlation between output and dry
        # input as a secondary peak at lag == -d (besides the dominant
        # zero-lag peak from the direct/undelayed path).
        xcorr = np.correlate(out, x, mode="full")
        lags = np.arange(-n_samples + 1, n_samples)
        negative = lags < 0
        peak_lag = lags[negative][np.argmax(xcorr[negative])]
        assert peak_lag == -d

    def testChorusInvalidParametersRaise(self):
        with pytest.raises(AssertionError):
            dsp.effects.Chorus(base_delays_ms=-5)
        with pytest.raises(AssertionError):
            dsp.effects.Chorus(base_delays_ms=0)
        with pytest.raises(AssertionError):
            dsp.effects.Chorus(mix_percent=0)
        with pytest.raises(AssertionError):
            dsp.effects.Chorus(mix_percent=150)
        with pytest.raises(AssertionError):
            dsp.effects.Chorus(
                depths_ms=[1, 2, 3],
                base_delays_ms=[10, 20],
                modulators=dsp.effects.LFO(1.0),
            )  # base_delays length (2) does not match inferred voices (3)

    def testDigitalDelay(self):
        delay = dsp.effects.DigitalDelay(150, feedback=0.15)
        delay.set_advanced_parameters(None)
        delay.apply(self.speech)

        delay.set_advanced_parameters("arctan")
        delay.apply(self.speech)

    def testDigitalDelayEchoRecursionMatchesClosedForm(self):
        """For an impulse input, the delay line without saturation produces
        exact echoes at every multiple of the delay with amplitude
        `feedback ** k` (geometric decay), directly from the recursion
        `y[n] = x[n] + feedback * y[n - delay]`.

        """
        fs = 8_000
        delay_ms = 20.0
        feedback = 0.3
        delay_samples = int(round(delay_ms * 1e-3 * fs))

        n_samples = 5 * delay_samples + 10
        x = np.zeros(n_samples)
        x[0] = 1.0
        sig = dsp.Signal(None, x[:, None], fs)

        delay = dsp.effects.DigitalDelay(delay_ms, feedback=feedback)
        delay.set_advanced_parameters(None)  # linear (no saturation)
        out = delay.apply(sig).time_data[:, 0]

        for k in range(1, 4):
            np.testing.assert_allclose(out[k * delay_samples], feedback**k, atol=1e-10)

    def testDigitalDelayInvalidParametersRaise(self):
        with pytest.raises(AssertionError):
            dsp.effects.DigitalDelay(delay_time_ms=0)
        with pytest.raises(AssertionError):
            dsp.effects.DigitalDelay(delay_time_ms=-10)
        with pytest.raises(AssertionError):
            dsp.effects.DigitalDelay(feedback=0)
        with pytest.raises(AssertionError):
            dsp.effects.DigitalDelay(feedback=-0.1)

    def testOther(self):
        assert 1 == dsp.effects.get_frequency_from_musical_rhythm("quarter", 60)

        assert 2 == dsp.effects.get_frequency_from_musical_rhythm("eighth", 60)

        assert 3 == dsp.effects.get_frequency_from_musical_rhythm("eighth 3", 60)

        assert 2 / 3 == dsp.effects.get_frequency_from_musical_rhythm(
            "dotted quarter", 60
        )
