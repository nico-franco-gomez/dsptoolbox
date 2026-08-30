"""
Tests for `dsp.effects.SpectralSubtractor`.
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

    def testSpectralSubtractor(self):
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

        # Explicit spectrum imported instead of estimated adaptively
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
