"""
Tests for `dsp.effects.LFO`, `Tremolo`, and `Chorus`.
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

    def testLFO(self):
        l_osc = dsp.effects.LFO(
            frequency_hz=100,
            waveform=dsp.effects.Waveform.Triangle,
            random_phase=True,
            smooth=5,
        )
        l_osc.plot_waveform()
        l_osc.get_waveform(self.fs_hz, 2000)

        l_osc.set_parameters(
            frequency_hz=("dotted quarter", 130),
            waveform=dsp.effects.Waveform.Sawtooth,
            smooth=0,
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
            frequency_hz=("dotted quarter", 130),
            waveform=dsp.effects.Waveform.Sawtooth,
            smooth=0,
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

        lfo = dsp.effects.LFO(
            lfo_freq, dsp.effects.Waveform.Harmonic, random_phase=False, smooth=0
        )
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
            # Modulator must be 1D
            dsp.effects.Tremolo(depth=0.5, modulator=np.zeros((10, 2)))

    def testChorus(self):
        l_osc = dsp.effects.LFO(
            frequency_hz=("dotted quarter", 130),
            waveform=dsp.effects.Waveform.Sawtooth,
            smooth=0,
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

        lfo = dsp.effects.LFO(2.0, dsp.effects.Waveform.Harmonic, random_phase=False)
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
            # base_delays length (2) does not match inferred voices (3)
            dsp.effects.Chorus(
                depths_ms=[1, 2, 3],
                base_delays_ms=[10, 20],
                modulators=dsp.effects.LFO(1.0),
            )
