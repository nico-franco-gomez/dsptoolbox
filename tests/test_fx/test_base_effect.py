"""
Tests for the `AudioEffect` base class contract.
"""

import os
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp


class TestAudioEffectBase:
    speech = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "speech.flac")
    ).resample(8_000)

    def get_effect(self):
        return dsp.effects.Distortion(
            distortion_level=25,
            post_gain_db=0,
            type_of_distortion=dsp.effects.DistortionType.Arctan,
        )

    def test_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            dsp.effects.AudioEffect()

    def test_apply_does_not_modify_the_input(self):
        effect = self.get_effect()
        before = self.speech.time_data.copy()
        effect.apply(self.speech)
        np.testing.assert_array_equal(self.speech.time_data, before)

    def test_apply_on_multibandsignal_does_not_modify_the_input(self):
        effect = self.get_effect()
        mbs = dsp.MultiBandSignal(
            [self.speech.copy(), self.speech.copy().apply_gain(-6.0)]
        )
        before = [b.time_data.copy() for b in mbs.bands]

        out = effect.apply(mbs)

        assert out.number_of_bands == mbs.number_of_bands
        for band, original in zip(mbs.bands, before, strict=True):
            np.testing.assert_array_equal(band.time_data, original)

    def test_effect_keeps_no_state_between_applications(self):
        """The peak/RMS restoring used to live on the instance, so applying an
        effect to signals with different channel counts warned about a
        mismatch and skipped the restoring."""
        effect = self.get_effect()

        stereo = self.speech.copy().append_signals([self.speech.copy()])
        effect.apply(stereo)

        mono_first = self.get_effect().apply(self.speech)
        mono_after_stereo = effect.apply(self.speech)
        np.testing.assert_allclose(mono_after_stereo.time_data, mono_first.time_data)
