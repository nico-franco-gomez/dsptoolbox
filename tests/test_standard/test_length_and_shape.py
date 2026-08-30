import numpy as np
import pytest

import dsptoolbox as dsp


class TestStandardModule:
    fs = 44100
    audio_multi = dsp.generators.noise(2, fs, number_of_channels=3, rng=126)

    def get_multiband_signal(self) -> dsp.MultiBandSignal:
        fb = dsp.filterbanks.linkwitz_riley_crossovers([1e3], [4], self.fs)
        return fb.filter_signal(self.audio_multi)

    def test_pad_trim(self):
        # Trim at the end
        trim_length = 40_000
        td = self.audio_multi.time_data[:trim_length]
        s = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data == self.audio_multi.pad_trim(trim_length).time_data)

        # Pad at the end
        pad_length = 10_000
        td = np.concatenate(
            [td, np.zeros((pad_length, self.audio_multi.number_of_channels))],
            axis=0,
        )
        s = s.pad_trim(s.time_data.shape[0] + pad_length)
        s1 = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data == s1.time_data)

        # Trim at start
        trim_length = 30_000
        td = self.audio_multi.time_data[-trim_length:]
        s = dsp.Signal(None, td, self.fs)
        assert np.all(
            s.time_data
            == self.audio_multi.pad_trim(trim_length, in_the_end=False).time_data
        )

        # Pad at start
        pad_length = 10_000
        td = np.concatenate(
            [np.zeros((pad_length, self.audio_multi.number_of_channels)), td],
            axis=0,
        )
        s = s.pad_trim(s.time_data.shape[0] + pad_length, in_the_end=False)
        s1 = dsp.Signal(None, td, self.fs)
        assert np.all(s.time_data == s1.time_data)

        # Single-channel signal
        s = s.get_channels(0)
        s.pad_trim(50_000)

        # MultiBandSignal
        b = [
            self.audio_multi.get_channels(0),
            self.audio_multi.get_channels(1),
        ]
        multi = dsp.MultiBandSignal(b)
        multi.pad_trim(40_000)

    def test_append_signals(self):
        s1 = self.audio_multi.get_channels(0)
        s2 = self.audio_multi.get_channels(1)
        s = s1.append_signals([s2])
        assert s.number_of_channels == 2
        assert np.all(s.time_data == self.audio_multi.time_data[:, :2])

        b = [
            self.audio_multi.get_channels(0),
            self.audio_multi.get_channels(1),
        ]
        sm = dsp.MultiBandSignal(b)
        sm1 = dsp.MultiBandSignal(b)
        sm_ = sm.append_signals([sm1])
        assert sm_.number_of_channels == 2
        assert sm_.number_of_bands == 2

    def test_append_filterbanks(self):
        fb1 = dsp.filterbanks.auditory_filters_gammatone(
            [600, 800], sampling_rate_hz=self.fs
        )
        fb2 = dsp.filterbanks.auditory_filters_gammatone(
            [800, 1000], sampling_rate_hz=self.fs
        )
        fb_out = fb1.append_filterbanks([fb2])
        assert len(fb_out) == len(fb1) + len(fb2)

        with pytest.raises(AssertionError):
            fb3 = dsp.filterbanks.auditory_filters_gammatone(
                [800, 1000], sampling_rate_hz=48000
            )
            fb1.append_filterbanks([fb3])

    def test_resample(self):
        # A thin wrapper around scipy's resample_poly; only checked for
        # producing output, see test_resample_preserves_frequency_and_amplitude
        # for a numerical check.
        self.audio_multi.resample(desired_sampling_rate_hz=22050)

    def test_resample_preserves_frequency_and_amplitude(self):
        """`resample` wraps `scipy.signal.resample_poly`; a pure tone well
        below both the original and target Nyquist frequencies should keep
        its FFT peak frequency exactly (bin-aligned in both sampling
        rates) and its amplitude close to the original (small ripple from
        the polyphase anti-aliasing filter is expected, not exact).

        """
        fs_in = 8_000
        fs_out = 6_000
        freq = 200.0
        amplitude = 0.6
        n_samples = 4_000
        t = np.arange(n_samples) / fs_in
        x = amplitude * np.sin(2 * np.pi * freq * t)
        sig = dsp.Signal(None, x[:, None], fs_in, constrain_amplitude=False)

        out = sig.resample(fs_out)
        assert out.sampling_rate_hz == fs_out

        X = np.fft.rfft(out.time_data[:, 0])
        f = np.fft.rfftfreq(len(out), 1 / fs_out)
        peak_bin = np.argmax(np.abs(X))

        np.testing.assert_allclose(f[peak_bin], freq, atol=1e-9)
        peak_amplitude = 2 * np.abs(X[peak_bin]) / len(out)
        np.testing.assert_allclose(peak_amplitude, amplitude, rtol=0.02)

    def test_modify_signal_length(self):
        with pytest.raises(AssertionError):
            self.audio_multi.modify_signal_length(None, None)
        with pytest.raises(AssertionError):
            self.audio_multi.modify_signal_length(
                -(self.audio_multi.length_seconds + 1.0),
                None,
            )
        with pytest.raises(AssertionError):
            self.audio_multi.modify_signal_length(
                None,
                -(self.audio_multi.length_seconds + 1.0),
            )
        with pytest.raises(AssertionError):
            self.audio_multi.modify_signal_length(
                -self.audio_multi.length_seconds / 2.0,
                -self.audio_multi.length_seconds / 1.9,
            )

        original_length = len(self.audio_multi)

        # Add both
        new = self.audio_multi.modify_signal_length(1.0, 1.0)
        assert new.length_seconds == self.audio_multi.length_seconds + 2.0
        assert len(self.audio_multi) == original_length

        # Add only start
        new = self.audio_multi.modify_signal_length(1.0, None)
        assert new.length_seconds == self.audio_multi.length_seconds + 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(new.time_data[: new.sampling_rate_hz], 0.0)

        # Add only end
        new = self.audio_multi.modify_signal_length(None, 1.0)
        assert new.length_seconds == self.audio_multi.length_seconds + 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(new.time_data[-new.sampling_rate_hz :], 0.0)

        # Remove both
        new = self.audio_multi.modify_signal_length(-0.5, -0.5)
        assert new.length_seconds == self.audio_multi.length_seconds - 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data,
            self.audio_multi.time_data[
                new.sampling_rate_hz // 2 : -new.sampling_rate_hz // 2
            ],
        )

        # Remove only start
        new = self.audio_multi.modify_signal_length(-0.5, None)
        assert new.length_seconds == self.audio_multi.length_seconds - 0.5
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data,
            self.audio_multi.time_data[new.sampling_rate_hz // 2 :],
        )

        # Remove only end
        new = self.audio_multi.modify_signal_length(None, -0.5)
        assert new.length_seconds == self.audio_multi.length_seconds - 0.5
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data,
            self.audio_multi.time_data[: -new.sampling_rate_hz // 2],
        )

        # Mixed: add at start, remove at end
        new = self.audio_multi.modify_signal_length(1.5, -0.5)
        assert new.length_seconds == self.audio_multi.length_seconds + 1.0
        assert len(self.audio_multi) == original_length
        np.testing.assert_array_equal(
            new.time_data[: 3 * new.sampling_rate_hz // 2], 0.0
        )
        np.testing.assert_array_equal(
            new.time_data[3 * new.sampling_rate_hz // 2 :],
            self.audio_multi.time_data[: -new.sampling_rate_hz // 2],
        )

        # MultiBandSignal: only functionality
        mb = self.get_multiband_signal()
        mb.modify_signal_length(1.5, -0.5)
