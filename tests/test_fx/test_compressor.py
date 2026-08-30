"""
Tests for `dsp.effects.Compressor`.
"""

import os
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp


class TestEffectsModule:
    speech = (
        dsp.Signal(
            join(os.path.dirname(__file__), "..", "..", "example_data", "speech.flac")
        )
    ).resample(
        8_000,
    )
    fs_hz = speech.sampling_rate_hz

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
