import numpy as np
import pytest
from scipy.signal import chirp, welch

import dsptoolbox as dsp


class TestGeneratorsModule:
    def test_noise(self):
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.White,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
            rng=113,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Pink,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
            rng=114,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Red,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
            rng=115,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Blue,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
            rng=116,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Violet,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
            rng=117,
        )
        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.Grey,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=3,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=1,
            rng=118,
        )

        dsp.generators.noise(
            type_of_noise=dsp.generators.NoiseType.White,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=1,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=0,
            rng=119,
        )

        dsp.generators.noise(
            type_of_noise=-0.5,
            length_seconds=2,
            sampling_rate_hz=5_000,
            peak_level_dbfs=-20,
            number_of_channels=1,
            fade=dsp.FadeType.Logarithmic,
            padding_end_seconds=0,
            rng=120,
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
                rng=121,
            )

    def test_chirp(self):
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
        t = np.arange(duration * fs) / fs
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

    def test_noise_psd_slope_matches_beta(self):
        """Per the docstring, `psd * frequency**(-beta)`, i.e. the PSD's
        log-log slope should equal `-beta` (beta=1 -> pink, beta=-1 ->
        blue). Fitted here via Welch + linear regression on log-log data,
        independent of the generator's own internal shaping.

        """
        fs = 20_000

        for beta, tol in ((1.0, 0.15), (-1.0, 0.15), (2.0, 0.3)):
            sig = dsp.generators.noise(
                length_seconds=10.0,
                sampling_rate_hz=fs,
                type_of_noise=beta,
                peak_level_dbfs=-3,
                fade=None,
                rng=122,
            )
            f, pxx = welch(sig.time_data[:, 0], fs=fs, nperseg=4096)
            mask = (f > 20) & (f < fs / 2 * 0.8)
            slope, _ = np.polyfit(np.log(f[mask]), np.log(pxx[mask]), 1)
            np.testing.assert_allclose(slope, -beta, atol=tol)

    def test_noise_named_colors_match_beta_convention(self):
        """Pink (`beta=1`) and blue (`beta=-1`) noise, generated via the
        named `NoiseType` enum, should show the same PSD slope sign as
        their `beta`-parametrized equivalents (per the docstring's
        equivalence note).

        """
        fs = 20_000

        def fitted_slope(sig):
            f, pxx = welch(sig.time_data[:, 0], fs=fs, nperseg=4096)
            mask = (f > 20) & (f < fs / 2 * 0.8)
            slope, _ = np.polyfit(np.log(f[mask]), np.log(pxx[mask]), 1)
            return slope

        pink = dsp.generators.noise(
            length_seconds=10.0,
            sampling_rate_hz=fs,
            type_of_noise=dsp.generators.NoiseType.Pink,
            peak_level_dbfs=-3,
            fade=None,
            rng=123,
        )
        blue = dsp.generators.noise(
            length_seconds=10.0,
            sampling_rate_hz=fs,
            type_of_noise=dsp.generators.NoiseType.Blue,
            peak_level_dbfs=-3,
            fade=None,
            rng=124,
        )
        assert fitted_slope(pink) < -0.5
        assert fitted_slope(blue) > 0.5

    def test_dirac(self):
        dsp.generators.dirac(
            1024, delay_samples=0, number_of_channels=1, sampling_rate_hz=5_000
        )
        dsp.generators.dirac(
            1024,
            delay_samples=100,
            number_of_channels=2,
            sampling_rate_hz=5_000,
        )

    def test_dirac_is_exact_unit_impulse(self):
        length = 1024
        delay = 137
        d = dsp.generators.dirac(
            length, delay_samples=delay, number_of_channels=2, sampling_rate_hz=5_000
        )
        expected = np.zeros((length, 2))
        expected[delay, :] = 1.0
        np.testing.assert_array_equal(d.time_data, expected)

    def test_dirac_invalid_parameters_raise(self):
        with pytest.raises(AssertionError):
            dsp.generators.dirac(0, sampling_rate_hz=5_000)
        with pytest.raises(AssertionError):
            dsp.generators.dirac(10, delay_samples=-1, sampling_rate_hz=5_000)
        with pytest.raises(AssertionError):
            dsp.generators.dirac(10, delay_samples=10, sampling_rate_hz=5_000)
        with pytest.raises(AssertionError):
            dsp.generators.dirac(10, number_of_channels=0, sampling_rate_hz=5_000)

    def test_oscillator(self):
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
            rng=125,
        )
        dsp.generators.oscillator(
            frequency_hz=1000,
            sampling_rate_hz=10_000,
            mode=dsp.generators.WaveForm.Square,
            number_of_channels=1,
            uncorrelated=False,
        )

    def test_oscillator_harmonic_matches_fft_peak_frequency_and_amplitude(self):
        """A pure `Harmonic` tone at a frequency landing exactly on an FFT
        bin (no fade, no leakage) should have its `rfft` peak at exactly
        that bin, with magnitude `N/2 * A` (the standard real-FFT scaling
        for a sinusoid of amplitude A), where A is the peak-normalized
        linear amplitude implied by `peak_level_dbfs`.

        """
        fs = 8_000
        n_samples = 4_000  # bin width = fs/n_samples = 2 Hz
        freq = 100.0  # exact bin (50th)
        peak_level_dbfs = -6.0

        sig = dsp.generators.oscillator(
            frequency_hz=freq,
            sampling_rate_hz=fs,
            length_seconds=n_samples / fs,
            mode=dsp.generators.WaveForm.Harmonic,
            peak_level_dbfs=peak_level_dbfs,
            fade=None,
        )
        td = sig.time_data[:, 0]
        X = np.fft.rfft(td)
        f = np.fft.rfftfreq(len(td), 1 / fs)
        peak_bin = np.argmax(np.abs(X))

        expected_amplitude = 10 ** (peak_level_dbfs / 20)
        np.testing.assert_allclose(f[peak_bin], freq, atol=1e-9)
        np.testing.assert_allclose(
            np.abs(X[peak_bin]), len(td) / 2 * expected_amplitude, rtol=1e-9
        )
        np.testing.assert_allclose(np.max(np.abs(td)), expected_amplitude, rtol=1e-9)

    def test_oscillator_square_harmonics_follow_fourier_series(self):
        """A square wave's Fourier series has odd harmonics at amplitude
        `(4/pi)/m` relative to the fundamental (all in phase, m=1,3,5,...);
        even harmonics are absent. Because the generator peak-normalizes
        the *summed* waveform (an unknown overall scale factor), this is
        checked as a ratio to the fundamental's FFT magnitude rather than
        an absolute value -- the ratio is scale-invariant.

        """
        fs = 8_000
        n_samples = 4_000  # bin width = 2 Hz
        freq = 50.0  # harmonics at 50, 150, 250, ... all exact bins

        sig = dsp.generators.oscillator(
            frequency_hz=freq,
            sampling_rate_hz=fs,
            length_seconds=n_samples / fs,
            mode=dsp.generators.WaveForm.Square,
            harmonic_cutoff_hz=1_000,
            fade=None,
        )
        td = sig.time_data[:, 0]
        X = np.fft.rfft(td)
        f = np.fft.rfftfreq(len(td), 1 / fs)

        def mag_at(target_f):
            return np.abs(X[np.argmin(np.abs(f - target_f))])

        fundamental = mag_at(freq)
        for m in (3, 5, 7):
            ratio = mag_at(freq * m) / fundamental
            np.testing.assert_allclose(ratio, 1.0 / m, rtol=1e-6)
        # Even harmonics should be at the FFT noise floor, not a real peak.
        assert mag_at(freq * 2) / fundamental < 1e-6

    def test_oscillator_sawtooth_harmonics_follow_fourier_series(self):
        """A sawtooth's Fourier series has every harmonic at amplitude
        `(2/pi)/m`, but alternating sign every step in m (per this
        generator's `(-1)**k` convention, verified empirically -- not the
        naive "all-positive 1/m" textbook phase convention). The sign
        alternation is checked via a pi-radians phase flip between
        consecutive harmonics.

        """
        fs = 8_000
        n_samples = 4_000
        freq = 50.0

        sig = dsp.generators.oscillator(
            frequency_hz=freq,
            sampling_rate_hz=fs,
            length_seconds=n_samples / fs,
            mode=dsp.generators.WaveForm.Sawtooth,
            harmonic_cutoff_hz=1_000,
            fade=None,
        )
        td = sig.time_data[:, 0]
        X = np.fft.rfft(td)
        f = np.fft.rfftfreq(len(td), 1 / fs)

        def bin_at(target_f):
            return X[np.argmin(np.abs(f - target_f))]

        fundamental = bin_at(freq)
        for m in (2, 3, 4, 5):
            c = bin_at(freq * m)
            ratio = np.abs(c) / np.abs(fundamental)
            np.testing.assert_allclose(ratio, 1.0 / m, rtol=1e-6)
            expected_sign = -1 if m % 2 == 0 else 1
            phase_diff = np.angle(c / fundamental)
            actual_sign = 1 if np.cos(phase_diff) > 0 else -1
            assert actual_sign == expected_sign

    def test_oscillator_triangle_harmonics_follow_fourier_series(self):
        """A triangle wave's Fourier series has odd harmonics at amplitude
        `(8/pi**2)/m**2` (m=1,3,5,...), with sign alternating every step in
        k where m=2k-1 (verified empirically); even harmonics are absent.

        """
        fs = 8_000
        n_samples = 4_000
        freq = 50.0

        sig = dsp.generators.oscillator(
            frequency_hz=freq,
            sampling_rate_hz=fs,
            length_seconds=n_samples / fs,
            mode=dsp.generators.WaveForm.Triangle,
            harmonic_cutoff_hz=1_000,
            fade=None,
        )
        td = sig.time_data[:, 0]
        X = np.fft.rfft(td)
        f = np.fft.rfftfreq(len(td), 1 / fs)

        def bin_at(target_f):
            return X[np.argmin(np.abs(f - target_f))]

        fundamental = bin_at(freq)
        expected_signs = {3: -1, 5: 1, 7: -1}
        for m, expected_sign in expected_signs.items():
            c = bin_at(freq * m)
            ratio = np.abs(c) / np.abs(fundamental)
            np.testing.assert_allclose(ratio, 1.0 / m**2, rtol=1e-6)
            phase_diff = np.angle(c / fundamental)
            actual_sign = 1 if np.cos(phase_diff) > 0 else -1
            assert actual_sign == expected_sign
        # Even harmonics should be at the FFT noise floor, not a real peak.
        assert np.abs(bin_at(freq * 2)) / np.abs(fundamental) < 1e-6

    def test_oscillator_invalid_parameters_raise(self):
        with pytest.raises(AssertionError):
            dsp.generators.oscillator(frequency_hz=0, sampling_rate_hz=5_000)
        with pytest.raises(AssertionError):
            dsp.generators.oscillator(frequency_hz=-100, sampling_rate_hz=5_000)
        with pytest.raises(AssertionError):
            dsp.generators.oscillator(frequency_hz=3_000, sampling_rate_hz=5_000)
        with pytest.raises(AssertionError):
            dsp.generators.oscillator(
                frequency_hz=100, sampling_rate_hz=5_000, harmonic_cutoff_hz=0
            )


class TestTimeVectorSpacing:
    def test_signal_time_vector_uses_sampling_period(self):
        fs = 48000
        s = dsp.Signal(None, np.zeros((fs, 2)), fs)
        np.testing.assert_allclose(s.time_vector_s, np.arange(fs) / fs, atol=1e-15)

    def test_chirp_instantaneous_frequency_hits_the_range_edges(self):
        """The linear sweep's time base must be `n / fs`. A `T / (N - 1)`
        spacing skews the sweep rate, shifting the end frequency.

        """
        fs = 48000
        length_seconds = 2.0
        range_hz = [1000.0, 2000.0]
        c = dsp.generators.chirp(
            sampling_rate_hz=fs,
            type_of_chirp=dsp.generators.ChirpType.Linear,
            range_hz=range_hz,
            length_seconds=length_seconds,
            fade=None,
            padding_end_seconds=0.0,
        )
        td = c.time_data[:, 0]
        # Instantaneous frequency from the zero crossings of the last period
        crossings = np.where(np.diff(np.signbit(td)))[0]
        final_period_samples = np.diff(crossings)[-1] * 2
        np.testing.assert_allclose(fs / final_period_samples, range_hz[1], rtol=2e-3)
