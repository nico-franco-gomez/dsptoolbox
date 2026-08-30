import numpy as np
import pytest

import dsptoolbox as dsp


class TestStandardModule:
    fs = 44100
    audio_multi = dsp.generators.noise(2, fs, number_of_channels=3, rng=137)

    def get_multiband_signal(self) -> dsp.MultiBandSignal:
        fb = dsp.filterbanks.linkwitz_riley_crossovers([1e3], [4], self.fs)
        return fb.filter_signal(self.audio_multi)

    def test_latency(self):
        td = self.audio_multi.time_data
        delay_samples = int(30e-3 * self.fs)
        td_del = np.zeros(
            (td.shape[0] + delay_samples, self.audio_multi.number_of_channels)
        )
        td_del[-td.shape[0] :] = td

        s = dsp.Signal(None, td_del, self.fs)
        vector, corr = dsp.latency(self.audio_multi, s)
        assert np.allclose(corr, 1.0)
        assert np.all(vector == -delay_samples)
        np.testing.assert_array_equal(s.time_data, td_del)

        # Latency the other way around
        td_previous = s.time_data.copy()
        td_previous2 = self.audio_multi.time_data.copy()
        vector, corr = dsp.latency(s, self.audio_multi)
        assert np.allclose(corr, 1.0)
        assert np.all(vector == delay_samples)
        np.testing.assert_array_equal(s.time_data, td_previous)
        np.testing.assert_array_equal(self.audio_multi.time_data, td_previous2)

        with pytest.raises(AssertionError):
            vector, corr = dsp.latency(s.get_channels(0), self.audio_multi)

        # Single channel
        td = s.time_data[:, :2]
        td[:, 1] = 0
        td[: len(self.audio_multi.time_data[:, 0]), 1] = self.audio_multi.time_data[
            :, 0
        ]
        s = dsp.Signal(None, td, self.fs)
        value, corr = dsp.latency(s)
        assert np.allclose(corr, 1.0)
        assert np.all(-value == delay_samples)

        # Data must not change after the call
        s = dsp.Signal(None, td, self.fs)
        value, corr = dsp.latency(s)
        np.testing.assert_array_equal(s.time_data, td)

        # Fractional delays
        delay = 0.003301
        noi = dsp.generators.noise(length_seconds=1, sampling_rate_hz=10_000, rng=138)
        noi_del = noi.fractional_delay(delay)
        td_previous_noi_del = noi_del.time_data.copy()
        lat, corr = dsp.latency(noi_del, noi, 2)
        np.testing.assert_array_equal(td_previous_noi_del, noi_del.time_data)
        assert np.allclose(corr, 1.0, atol=1e-2)
        assert np.abs(lat[0] - delay * noi.sampling_rate_hz) < 0.9

        noi = noi_del.append_signals([noi])
        latencies, corr = dsp.latency(noi, polynomial_points=1)
        assert len(latencies) == noi.number_of_channels - 1
        assert np.allclose(corr, 1.0, atol=1e-2)
        assert np.abs(latencies[0] + delay * noi.sampling_rate_hz) < 0.5
        latencies, corr = dsp.latency(noi, polynomial_points=5)
        assert np.allclose(corr, 1.0, atol=1e-2)
        assert np.abs(latencies[0] + delay * noi.sampling_rate_hz) < 0.5

    def test_fractional_delay(self):
        delay_s = 150 / self.fs

        s = self.audio_multi.fractional_delay(delay_s)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), 150))

        s = self.audio_multi.fractional_delay(delay_s, channels=0)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), [150, 0, 0]))

    def test_delay(self):
        delay_samp = 150

        s = self.audio_multi.delay(delay_samp)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), 150))

        s = self.audio_multi.delay(delay_samp, channels=0)
        lat = dsp.latency(s, self.audio_multi)[0]
        assert np.all(np.isclose(np.abs(lat), [150, 0, 0]))

    def test_activity_detector(self):
        # Harmonic signal followed by silence
        s = dsp.generators.oscillator(1000.0, sampling_rate_hz=self.fs)
        s = s.pad_trim(s.time_data.shape[0] * 2)
        s.activity_detector()

    def test_activity_detector_synthetic_burst_boundaries(self):
        """A signal built as silence -> tone burst -> silence should have
        its detected `signal_indices` boundaries close to the true burst
        boundaries, lagging by roughly the attack/release smoothing time
        (an EMA envelope follower, not an instantaneous threshold -- exact
        alignment isn't expected, but the lag should be on the order of
        the requested attack/release times, verified empirically: with a
        1 ms attack and 25 ms release at fs=8000 Hz, the onset lags by a
        few samples and the offset by a few hundred).

        """
        fs = 8_000
        silence_len = int(0.5 * fs)
        burst_len = int(0.5 * fs)
        t_burst = np.arange(burst_len) / fs
        tone = 0.8 * np.sin(2 * np.pi * 300 * t_burst)
        x = np.concatenate([np.zeros(silence_len), tone, np.zeros(silence_len)])
        sig = dsp.Signal(None, x[:, None], fs, constrain_amplitude=False)

        _, others = sig.activity_detector(
            threshold_dbfs=-20, attack_time_ms=1, release_time_ms=25
        )
        idx = others["signal_indices"]
        onset = np.argmax(idx)
        offset = len(idx) - 1 - np.argmax(idx[::-1])

        # Onset should lag the true burst start (silence_len) by only a
        # handful of samples (attack is fast); offset should lag the true
        # burst end (silence_len + burst_len) by no more than a few
        # release time constants, and in any case land well before the
        # end of the trailing silence.
        assert 0 <= onset - silence_len < 50
        assert 0 <= offset - (silence_len + burst_len) < 800
        assert offset < len(idx) - 1000

    def test_trim_with_level_threshold(self):
        s = np.zeros(1000)

        # Single-channel
        ones_slice = slice(len(s) // 3, len(s) // 2)

        threshold_db = -50.0
        fill_value = dsp.tools.from_db(threshold_db + 1, True)

        s[ones_slice] = fill_value

        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(threshold_db, True, True)[0]
            .time_data.squeeze(),
            s[ones_slice],
        )
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(
                threshold_db,
                False,
                True,
            )[0]
            .time_data.squeeze(),
            s[: ones_slice.stop],
        )
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(
                threshold_db,
                True,
                False,
            )[0]
            .time_data.squeeze(),
            s[ones_slice.start :],
        )
        with pytest.raises(AssertionError):
            (dsp.Signal.from_time_data(s, self.fs)).trim_with_level_threshold(
                threshold_db,
                False,
                False,
            )

        # Multi-channel
        s = np.zeros((1000, 2))
        ones_slice = slice(len(s) // 3, len(s) // 2)
        s[ones_slice.start + 5 : ones_slice.stop - 5, 0] = fill_value
        s[ones_slice, 1] = fill_value

        threshold_db = -50.0
        fill_value = dsp.tools.from_db(threshold_db + 1, True)

        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(threshold_db, True, True)[0]
            .time_data,
            s[ones_slice],
        )
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(
                threshold_db,
                False,
                True,
            )[0]
            .time_data,
            s[: ones_slice.stop],
        )
        np.testing.assert_array_equal(
            (dsp.Signal.from_time_data(s, self.fs))
            .trim_with_level_threshold(
                threshold_db,
                True,
                False,
            )[0]
            .time_data,
            s[ones_slice.start :],
        )
        with pytest.raises(AssertionError):
            (dsp.Signal.from_time_data(s, self.fs)).trim_with_level_threshold(
                threshold_db,
                False,
                False,
            )

    def test_select_time_window(self):
        s = self.audio_multi
        s2 = s.trim_with_time_selection(0.1, 0.3, True)
        assert abs(s2.length_seconds - 0.2) <= 1 / s.sampling_rate_hz
        s.trim_with_time_selection(0.1, 0.3, False)
        s.trim_with_time_selection(None, 0.3, False)
        s.trim_with_time_selection(0.1, None, False)

        mbs = self.get_multiband_signal()
        mbs.trim_with_time_selection(0.1, 0.3, False)
        mbs.trim_with_time_selection(0.1, 0.3, True)

        with pytest.raises(AssertionError):
            s.trim_with_time_selection(0.3, 0.1, False)
        with pytest.raises(AssertionError):
            s.trim_with_time_selection(0.1, s.length_seconds + 1.0, False)
        with pytest.raises(AssertionError):
            s.trim_with_time_selection(None, None, False)

    def test_latency_rejects_wrong_type_before_use(self):
        """The type check must run before any attribute of `in2` is read."""
        fs = 48000
        s = dsp.Signal(None, np.zeros((100, 2)), fs)
        with pytest.raises(AssertionError, match="type Signal"):
            dsp.latency(s, "not a signal")
