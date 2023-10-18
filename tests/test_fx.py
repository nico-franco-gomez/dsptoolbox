"""
Tests regarding functionality of audio fx
"""
import dsptoolbox as dsp
import numpy as np
from os.path import join


class TestEffectsModule:
    speech = dsp.resample(
        dsp.Signal(join("examples", "data", "speech.flac")), 8_000
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
            window_type="hamming",
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
            window_type="hamming",
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

    def testDistortion(self):
        """Test different distortion parameters."""
        dist = dsp.effects.Distortion(
            distortion_level=25, post_gain_db=0, type_of_distortion="arctan"
        )
        dist.apply(self.speech)

        dist.set_advanced_parameters(
            type_of_distortion=["arctan", "soft clip"],
            distortion_levels_db=[20, 40],
            mix_percent=[60, 40],
            offset_db=[-3, -np.inf],
            post_gain_db=2,
        )
        dist.apply(self.speech)

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

    def testTremolo(self):
        l_osc = dsp.effects.LFO(
            frequency_hz=("dotted quarter", 130), waveform="sawtooth", smooth=0
        )
        trem = dsp.effects.Tremolo(depth=0.8, modulator=l_osc)
        trem.apply(self.speech)

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

    def testDigitalDelay(self):
        delay = dsp.effects.DigitalDelay(150, feedback=0.15)
        delay.set_advanced_parameters(None)
        delay.apply(self.speech)

        delay.set_advanced_parameters("arctan")
        delay.apply(self.speech)

    def testOther(self):
        assert 1 == dsp.effects.get_frequency_from_musical_rhythm(
            "quarter", 60
        )

        assert 2 == dsp.effects.get_frequency_from_musical_rhythm("eighth", 60)

        assert 3 == dsp.effects.get_frequency_from_musical_rhythm(
            "eighth 3", 60
        )

        assert 2 / 3 == dsp.effects.get_frequency_from_musical_rhythm(
            "dotted quarter", 60
        )
