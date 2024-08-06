import dsptoolbox as dsp
import numpy as np
from scipy.signal import chirp
import pytest


class TestGeneratorsModule:
    def test_noise(self):
        # Only functionality
        dsp.generators.noise(
            type_of_noise="white",
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade="log",
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise="pink",
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade="log",
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise="red",
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade="log",
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise="blue",
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade="log",
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise="violet",
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade="log",
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise="grey",
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade="log",
            padding_end_seconds=1,
        )

        # Other parameters
        dsp.generators.noise(
            type_of_noise="white",
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=1,
            fade="log",
            padding_end_seconds=0,
        )

        dsp.generators.noise(
            type_of_noise=-0.5,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=1,
            fade="log",
            padding_end_seconds=0,
        )

        # Peak level over 0 dBFS
        with pytest.raises(AssertionError):
            dsp.generators.noise(
                type_of_noise="white",
                length_seconds=2,
                sampling_rate_hz=5_000,
                peak_level_dbfs=20,
                number_of_channels=1,
                fade="log",
                padding_end_seconds=0,
            )

    def test_chirp(self):
        # Only functionality
        dsp.generators.chirp(
            type_of_chirp="log",
            range_hz=None,
            length_seconds=2,
            sampling_rate_hz=10_000,
            peak_level_dbfs=-10,
            number_of_channels=2,
            fade="log",
            padding_end_seconds=1,
        )
        dsp.generators.chirp(
            type_of_chirp="lin",
            range_hz=None,
            length_seconds=2,
            sampling_rate_hz=10_000,
            peak_level_dbfs=-10,
            number_of_channels=2,
            fade="log",
            padding_end_seconds=1,
        )

        dsp.generators.chirp(
            type_of_chirp="lin",
            range_hz=[100, 4000],
            length_seconds=2,
            sampling_rate_hz=10_000,
            peak_level_dbfs=-10,
            number_of_channels=1,
            fade=None,
            padding_end_seconds=0,
        )
        dsp.generators.chirp(
            type_of_chirp="lin",
            range_hz=[100, 4000],
            length_seconds=1,
            sampling_rate_hz=10_000,
            peak_level_dbfs=-10,
            number_of_channels=1,
            fade="lin",
            padding_end_seconds=0,
        )

        # Same as with scipy's chirp
        fs = 44_100
        duration = 1
        t = np.linspace(0, duration, duration * fs)
        s = chirp(t=t, f0=20, t1=1, f1=20e3, method="logarithmic")
        s2 = dsp.generators.chirp(
            length_seconds=1,
            sampling_rate_hz=fs,
            type_of_chirp="log",
            fade=None,
            peak_level_dbfs=0,
            phase_offset=np.pi / 2,  # Offset because scipy uses cosine
            range_hz=[20, 20e3],
        )
        s2 = s2.time_data[:, 0]
        assert np.all(np.isclose(s, s2))

        # Same as with scipy's chirp
        fs = 44_100
        fs = 44100
        duration = 1
        t = np.linspace(0, duration, duration * fs)
        s = chirp(t=t, f0=20, t1=1, f1=20e3, method="logarithmic")
        s2 = dsp.generators.chirp(
            length_seconds=1,
            sampling_rate_hz=fs,
            type_of_chirp="log",
            fade=None,
            peak_level_dbfs=0,
            phase_offset=np.pi / 2,  # Offset because scipy uses cosine
            range_hz=[20, 20e3],
        )
        s2 = s2.time_data[:, 0]
        assert np.all(np.isclose(s, s2))

        with pytest.raises(AssertionError):
            dsp.generators.chirp(
                type_of_chirp="lin",
                range_hz=[100, 7000],
                length_seconds=1,
                sampling_rate_hz=10_000,
                peak_level_dbfs=-10,
                number_of_channels=1,
                fade="lin",
                padding_end_seconds=0,
            )

    def test_dirac(self):
        # Only functionality
        dsp.generators.dirac(
            1024, delay_samples=0, number_of_channels=1, sampling_rate_hz=5_000
        )
        dsp.generators.dirac(
            1024,
            delay_samples=100,
            number_of_channels=2,
            sampling_rate_hz=5_000,
        )

    def test_harmonic(self):
        # Only functionality
        dsp.generators.harmonic(
            frequency_hz=1000,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-5,
            number_of_channels=3,
            uncorrelated=False,
            fade="log",
            padding_end_seconds=1,
        )
        dsp.generators.harmonic(
            frequency_hz=1000,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-5,
            number_of_channels=1,
            uncorrelated=False,
            fade="lin",
            padding_end_seconds=0,
        )
        dsp.generators.harmonic(
            frequency_hz=1000,
            length_seconds=1,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-5,
            number_of_channels=3,
            uncorrelated=True,
            fade=None,
            padding_end_seconds=0,
        )

        with pytest.raises(AssertionError):
            dsp.generators.harmonic(
                frequency_hz=4000,
                length_seconds=1,
                sampling_rate_hz=5_000,
                peak_level_dbfs=-5,
                number_of_channels=3,
                uncorrelated=True,
                fade=None,
                padding_end_seconds=0,
            )

    def test_oscillator(self):
        # Only functionality
        dsp.generators.oscillator(
            frequency_hz=150,
            sampling_rate_hz=5_000,
            mode="triangle",
            number_of_channels=2,
            uncorrelated=False,
        )
        dsp.generators.oscillator(
            frequency_hz=150,
            sampling_rate_hz=3_000,
            harmonic_cutoff_hz=1_000,
            mode="sawtooth",
            number_of_channels=2,
            uncorrelated=True,
        )
        dsp.generators.oscillator(
            frequency_hz=1000,
            sampling_rate_hz=10_000,
            mode="square",
            number_of_channels=1,
            uncorrelated=False,
        )
