"""
Tests for `dsp.effects.DigitalDelay` and standalone effects helpers.
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

    def testDigitalDelay(self):
        delay = dsp.effects.DigitalDelay(150, feedback=0.15)
        delay.set_advanced_parameters(None)
        delay.apply(self.speech)

        delay.set_advanced_parameters(dsp.effects.SaturationType.Arctan)
        delay.apply(self.speech)

    def test_set_parameters_leaves_the_other_one_unchanged(self):
        """`set_parameters` documents that None leaves a parameter unchanged,
        but both were asserted on directly, so passing only one raised a
        TypeError.

        """
        delay = dsp.effects.DigitalDelay(150.0, feedback=0.15)

        delay.set_parameters(feedback=0.5)
        assert delay.delay_ms == 150.0 and delay.feedback == 0.5

        delay.set_parameters(delay_time_ms=50.0)
        assert delay.delay_ms == 50.0 and delay.feedback == 0.5

        with pytest.raises(AssertionError):
            delay.set_parameters(feedback=-1.0)
        with pytest.raises(AssertionError):
            delay.set_parameters(delay_time_ms=0.0)

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
