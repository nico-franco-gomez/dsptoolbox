"""
Tests for `dsp.effects.Distortion`.
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

    def testDistortion(self):
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
