import dsptoolbox as dsp
import pytest


class TestGeneratorsModule():
    def test_noise(self):
        # Only functionality
        dsp.generators.noise(
            type_of_noise='white', length_seconds=2, sampling_rate_hz=5_000,
            peak_level_dbfs=-20, number_of_channels=3, fade='log',
            padding_end_seconds=1)
        dsp.generators.noise(
            type_of_noise='pink', length_seconds=2, sampling_rate_hz=5_000,
            peak_level_dbfs=-20, number_of_channels=3, fade='log',
            padding_end_seconds=1)
        dsp.generators.noise(
            type_of_noise='red', length_seconds=2, sampling_rate_hz=5_000,
            peak_level_dbfs=-20, number_of_channels=3, fade='log',
            padding_end_seconds=1)
        dsp.generators.noise(
            type_of_noise='blue', length_seconds=2, sampling_rate_hz=5_000,
            peak_level_dbfs=-20, number_of_channels=3, fade='log',
            padding_end_seconds=1)
        dsp.generators.noise(
            type_of_noise='violet', length_seconds=2, sampling_rate_hz=5_000,
            peak_level_dbfs=-20, number_of_channels=3, fade='log',
            padding_end_seconds=1)
        dsp.generators.noise(
            type_of_noise='grey', length_seconds=2, sampling_rate_hz=5_000,
            peak_level_dbfs=-20, number_of_channels=3, fade='log',
            padding_end_seconds=1)

        # Other parameters
        dsp.generators.noise(
            type_of_noise='white', length_seconds=2, sampling_rate_hz=5_000,
            peak_level_dbfs=-20, number_of_channels=1, fade='log',
            padding_end_seconds=None)

        # Peak level over 0 dBFS
        with pytest.raises(AssertionError):
            dsp.generators.noise(
                type_of_noise='white', length_seconds=2,
                sampling_rate_hz=5_000, peak_level_dbfs=20,
                number_of_channels=1, fade='log', padding_end_seconds=0)

    def test_chirp(self):
        # Only functionality
        dsp.generators.chirp(
            type_of_chirp='log', range_hz=None, length_seconds=2,
            sampling_rate_hz=10_000, peak_level_dbfs=-10, number_of_channels=2,
            fade='log', padding_end_seconds=1)
        dsp.generators.chirp(
            type_of_chirp='lin', range_hz=None, length_seconds=2,
            sampling_rate_hz=10_000, peak_level_dbfs=-10, number_of_channels=2,
            fade='log', padding_end_seconds=1)

        dsp.generators.chirp(
            type_of_chirp='lin', range_hz=[100, 4000], length_seconds=2,
            sampling_rate_hz=10_000, peak_level_dbfs=-10, number_of_channels=1,
            fade=None, padding_end_seconds=0)
        dsp.generators.chirp(
            type_of_chirp='lin', range_hz=[100, 4000], length_seconds=1,
            sampling_rate_hz=10_000, peak_level_dbfs=-10, number_of_channels=1,
            fade='lin', padding_end_seconds=None)

        with pytest.raises(AssertionError):
            dsp.generators.chirp(
                type_of_chirp='lin', range_hz=[100, 7000], length_seconds=1,
                sampling_rate_hz=10_000, peak_level_dbfs=-10,
                number_of_channels=1, fade='lin', padding_end_seconds=None)

    def test_dirac(self):
        # Only functionality
        dsp.generators.dirac(
            1024, delay_samples=0, number_of_channels=1,
            sampling_rate_hz=5_000)
        dsp.generators.dirac(
            1024, delay_samples=100, number_of_channels=2,
            sampling_rate_hz=5_000)

    def test_harmonic(self):
        # Only functionality
        dsp.generators.harmonic(
            frequency_hz=1000, length_seconds=2, sampling_rate_hz=5_000,
            peak_level_dbfs=-5, number_of_channels=3, uncorrelated=False,
            fade='log', padding_end_seconds=1)
        dsp.generators.harmonic(
            frequency_hz=1000, length_seconds=2, sampling_rate_hz=5_000,
            peak_level_dbfs=-5, number_of_channels=1, uncorrelated=False,
            fade='lin', padding_end_seconds=None)
        dsp.generators.harmonic(
            frequency_hz=1000, length_seconds=1, sampling_rate_hz=5_000,
            peak_level_dbfs=-5, number_of_channels=3, uncorrelated=True,
            fade=None, padding_end_seconds=None)

        with pytest.raises(AssertionError):
            dsp.generators.harmonic(
                frequency_hz=4000, length_seconds=1, sampling_rate_hz=5_000,
                peak_level_dbfs=-5, number_of_channels=3, uncorrelated=True,
                fade=None, padding_end_seconds=None)

    def test_oscillator(self):
        # Only functionality
        dsp.generators.oscillator(frequency_hz=150,
                                  sampling_rate_hz=5_000,
                                  mode='triangle',
                                  number_of_channels=2, uncorrelated=False)
        dsp.generators.oscillator(frequency_hz=150,
                                  sampling_rate_hz=3_000,
                                  harmonic_cutoff_hz=1_000,
                                  mode='sawtooth',
                                  number_of_channels=2, uncorrelated=True)
        dsp.generators.oscillator(frequency_hz=1000,
                                  sampling_rate_hz=10_000,
                                  mode='square',
                                  number_of_channels=1, uncorrelated=False)
