import dsptoolbox as dsp
import numpy as np
from scipy.signal import chirp
import pytest


class TestGeneratorsModule:
    def test_noise(self):
        # Only functionality
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.White,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Pink,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Red,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Blue,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Violet,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Grey,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
        )

        # Other parameters
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.White,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=1,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=0,
        )

        dsp.generators.noise(
            type_of_noise=-0.5,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=1,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=0,
        )

        # Peak level over 0 dBFS
        with pytest.raises(AssertionError):
            dsp.generators.noise(
                type_of_noise=dsp.generators.NoiseType.White,
                length_seconds=2,
                sampling_rate_hz=5_000,
                peak_level_dbfs=20,
                number_of_channels=1,
                fade=dsp.FadeType.Logarithmic,
                padding_end_seconds=0,
            )

    def test_chirp(self):
        # Only functionality
        dsp.generators.chirp(
            type_of_chirp=dsp.generators.ChirpType.Logarithmic,
            range_hz=None,
            length_seconds=2,
            sampling_rate_hz=10_000,
            peak_level_dbfs=-10,
            number_of_channels=2,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
        )
        dsp.generators.chirp(
            type_of_chirp=dsp.generators.ChirpType.Linear,
            range_hz=None,
            length_seconds=2,
            sampling_rate_hz=10_000,
            peak_level_dbfs=-10,
            number_of_channels=2,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
        )

        dsp.generators.chirp(
            type_of_chirp=dsp.generators.ChirpType.Linear,
            range_hz=[100, 4000],
            length_seconds=2,
            sampling_rate_hz=10_000,
            peak_level_dbfs=-10,
            number_of_channels=1,
            fade=None,
            padding_end_seconds=0,
        )
        dsp.generators.chirp(
            type_of_chirp=dsp.generators.ChirpType.Linear,
            range_hz=[100, 4000],
            length_seconds=1,
            sampling_rate_hz=10_000,
            peak_level_dbfs=-10,
            number_of_channels=1,
            fade=dsp.FadeType.Linear,
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
            type_of_chirp=dsp.generators.ChirpType.Logarithmic,
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
            type_of_chirp=dsp.generators.ChirpType.Logarithmic,
            fade=None,
            peak_level_dbfs=0,
            phase_offset=np.pi / 2,  # Offset because scipy uses cosine
            range_hz=[20, 20e3],
        )
        s2 = s2.time_data[:, 0]
        assert np.all(np.isclose(s, s2))

        with pytest.raises(AssertionError):
            dsp.generators.chirp(
                type_of_chirp=dsp.generators.ChirpType.Linear,
                range_hz=[100, 7000],
                length_seconds=1,
                sampling_rate_hz=10_000,
                peak_level_dbfs=-10,
                number_of_channels=1,
                fade=dsp.FadeType.Linear,
                padding_end_seconds=0,
            )

        nominal_duration_seconds = 1.0
        _, sync_duration_seconds = dsp.generators.chirp(
            type_of_chirp=dsp.generators.ChirpType.SyncLog,
            range_hz=[20, 4e3],
            length_seconds=nominal_duration_seconds,
            sampling_rate_hz=10_000,
            peak_level_dbfs=-10,
            number_of_channels=2,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
        )
        assert abs(nominal_duration_seconds - sync_duration_seconds) < 0.1

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

    def test_oscillator(self):
        # Only functionality
        dsp.generators.oscillator(
            frequency_hz=150,
            sampling_rate_hz=5_000,
            mode=dsp.generators.WaveForm.Harmonic,
            number_of_channels=2,
            uncorrelated=False,
        )
        dsp.generators.oscillator(
            frequency_hz=150,
            sampling_rate_hz=5_000,
            mode=dsp.generators.WaveForm.Triangle,
            number_of_channels=2,
            uncorrelated=False,
        )
        dsp.generators.oscillator(
            frequency_hz=150,
            sampling_rate_hz=3_000,
            harmonic_cutoff_hz=1_000,
            mode=dsp.generators.WaveForm.Sawtooth,
            number_of_channels=2,
            uncorrelated=True,
        )
        dsp.generators.oscillator(
            frequency_hz=1000,
            sampling_rate_hz=10_000,
            mode=dsp.generators.WaveForm.Square,
            number_of_channels=1,
            uncorrelated=False,
        )
