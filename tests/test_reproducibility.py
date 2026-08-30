"""
Every stochastic entry point must accept an `rng` argument and produce
bit-identical output for the same seed.
"""

import numpy as np

import dsptoolbox as dsp


class TestReproducibility:
    fs = 8_000

    def test_noise_is_reproducible(self):
        def make(rng):
            return dsp.generators.noise(
                length_seconds=0.5,
                sampling_rate_hz=self.fs,
                number_of_channels=2,
                fade=None,
                rng=rng,
            ).time_data

        np.testing.assert_array_equal(make(1234), make(1234))
        assert not np.array_equal(make(1234), make(4321))

    def test_noise_accepts_a_generator_and_advances_it(self):
        generator = np.random.default_rng(7)
        first = dsp.generators.noise(
            length_seconds=0.2, sampling_rate_hz=self.fs, fade=None, rng=generator
        ).time_data
        second = dsp.generators.noise(
            length_seconds=0.2, sampling_rate_hz=self.fs, fade=None, rng=generator
        ).time_data
        assert not np.array_equal(first, second)

        np.testing.assert_array_equal(
            first,
            dsp.generators.noise(
                length_seconds=0.2,
                sampling_rate_hz=self.fs,
                fade=None,
                rng=np.random.default_rng(7),
            ).time_data,
        )

    def test_uncorrelated_oscillator_is_reproducible(self):
        def make(rng):
            return dsp.generators.oscillator(
                frequency_hz=440.0,
                sampling_rate_hz=self.fs,
                length_seconds=0.2,
                number_of_channels=3,
                uncorrelated=True,
                fade=None,
                rng=rng,
            ).time_data

        np.testing.assert_array_equal(make(11), make(11))
        assert not np.array_equal(make(11), make(12))

    def test_dither_is_reproducible(self):
        base = dsp.Signal(
            None, np.random.default_rng(0).normal(0, 0.1, (512, 2)), self.fs
        )
        for triangular in (True, False):
            first = base.dither(triangular_distribution=triangular, rng=99)
            second = base.dither(triangular_distribution=triangular, rng=99)
            other = base.dither(triangular_distribution=triangular, rng=100)
            np.testing.assert_array_equal(first.time_data, second.time_data)
            assert not np.array_equal(first.time_data, other.time_data)

    def test_lfo_random_phase_is_reproducible(self):
        def make(rng):
            return dsp.effects.LFO(
                5.0, dsp.effects.Waveform.Harmonic, random_phase=True, rng=rng
            ).get_waveform(self.fs, 400)

        np.testing.assert_array_equal(make(3), make(3))
        assert not np.array_equal(make(3), make(4))

    def test_lpc_synthesis_is_reproducible(self):
        sig = dsp.generators.noise(
            length_seconds=0.3, sampling_rate_hz=self.fs, fade=None, rng=0
        )

        def make(rng):
            return dsp.transforms.lpc(
                sig,
                order=8,
                window_length_samples=256,
                synthesize_encoded_signal=True,
                rng=rng,
            )[2].time_data

        np.testing.assert_array_equal(make(5), make(5))
        assert not np.array_equal(make(5), make(6))

    def test_synthetic_rir_noise_tail_is_reproducible(self):
        room = dsp.room_acoustics.ShoeboxRoom([3.0, 4.0, 2.5], t60_s=0.15)

        def make(rng):
            return dsp.room_acoustics.generate_synthetic_rir(
                room=room,
                source_position=[1.0, 1.0, 1.0],
                receiver_position=[2.0, 3.0, 1.5],
                sampling_rate_hz=self.fs,
                total_length_seconds=0.2,
                add_noise_reverberant_tail=True,
                max_order=6,
                rng=rng,
            ).time_data

        np.testing.assert_array_equal(make(21), make(21))
        assert not np.array_equal(make(21), make(22))
