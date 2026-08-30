import numpy as np
import pytest
from scipy.signal import hilbert

import dsptoolbox as dsp


class TestStandardModule:
    fs = 44100
    audio_multi = dsp.generators.noise(2, fs, number_of_channels=3, rng=133)

    def test_fade(self):
        # Result is only checked for the linear fade
        self.audio_multi.fade(fade_type=dsp.FadeType.Linear)
        self.audio_multi.fade(fade_type=dsp.FadeType.Logarithmic)
        self.audio_multi.fade(fade_type=dsp.FadeType.Exponential)

        f_end = self.audio_multi.fade(
            fade_type=dsp.FadeType.Linear,
            at_start=False,
            at_end=True,
        )
        f_st = self.audio_multi.fade(
            fade_type=dsp.FadeType.Linear,
            at_start=True,
            at_end=False,
        )
        with pytest.raises(AssertionError):
            self.audio_multi.fade(
                fade_type=dsp.FadeType.Linear,
                at_start=False,
                at_end=False,
            )

        td = self.audio_multi.time_data.copy()
        fade_le = int(td.shape[0] * 2.5 / 100)
        td[:fade_le] *= np.linspace(0, 1, fade_le)[..., None]
        assert np.all(np.isclose(f_st.time_data, td))

        td = self.audio_multi.time_data.copy()
        td[-fade_le:] *= np.linspace(1, 0, fade_le)[..., None]
        assert np.all(np.isclose(f_end.time_data, td))

    def test_detrend(self):
        s = dsp.generators.oscillator(
            100,
            sampling_rate_hz=700,
            peak_level_dbfs=-20,
            number_of_channels=2,
            uncorrelated=True,
            rng=134,
        )
        s.time_data += 0.2
        s.detrend(polynomial_order=0)

        s = dsp.generators.oscillator(
            100,
            sampling_rate_hz=700,
            peak_level_dbfs=-20,
            number_of_channels=1,
            uncorrelated=True,
            rng=135,
        )
        n = 0.3 * np.arange(len(s)) / len(s)
        s.time_data += n[..., None]
        s.detrend(polynomial_order=1)

        s.detrend(polynomial_order=10)

        with pytest.raises(AssertionError):
            s.detrend(polynomial_order=-10)

    def test_detrend_removes_known_polynomial_trend(self):
        """Per the source, `detrend` fits and subtracts a `numpy.polyfit`
        polynomial of the requested order (using sample index, not time in
        seconds, as the fit's x-axis). Because least-squares fitting is a
        linear operator, `polyfit(clean + trend) == polyfit(clean) +
        trend` exactly whenever `trend` already lies exactly in the fitted
        polynomial's degree -- so detrending should recover exactly
        `clean - polyval(polyfit(clean), index)`, i.e. `clean` itself minus
        whatever quadratic component was already present by chance in the
        noise (not simply `clean` unchanged, since finite-sample noise is
        never perfectly orthogonal to a quadratic basis).

        """
        fs = 700
        n_samples = 2_000
        index = np.arange(n_samples)
        rng = np.random.default_rng(0)
        clean = rng.normal(0, 0.01, n_samples)
        clean -= clean.mean()

        a2, a1, a0 = 0.5, -1.2, 0.3
        trend = a2 * index**2 + a1 * index + a0
        trended = clean + trend

        sig = dsp.Signal(None, trended[:, None], fs, constrain_amplitude=False)
        detrended = sig.detrend(polynomial_order=2)

        poly_of_clean = np.polyfit(index, clean, deg=2)
        expected = clean - np.polyval(poly_of_clean, index)
        np.testing.assert_allclose(detrended.time_data[:, 0], expected, atol=1e-7)

        # The recovered residual should still be small relative to the
        # trend that was removed (order-of-magnitude sanity check).
        assert np.max(np.abs(detrended.time_data[:, 0])) < 0.1 * np.max(np.abs(trend))

    def test_envelope(self):
        s = dsp.generators.oscillator(
            frequency_hz=500,
            mode=dsp.generators.WaveForm.Triangle,
            sampling_rate_hz=5_000,
            number_of_channels=3,
            uncorrelated=True,
            rng=136,
        )
        env = dsp.envelope(s, False, 512)
        assert env.shape == s.time_data.shape
        env = dsp.envelope(s, True, None)
        assert env.shape == s.time_data.shape

        s = dsp.generators.oscillator(
            frequency_hz=500,
            mode=dsp.generators.WaveForm.Sawtooth,
            sampling_rate_hz=5_000,
            number_of_channels=1,
        )
        env = dsp.envelope(s, False, 512)
        assert env.shape == s.time_data.shape
        env = dsp.envelope(s, True, None)
        assert env.shape == s.time_data.shape

        fb = dsp.filterbanks.auditory_filters_gammatone(
            [500, 1000], 1, s.sampling_rate_hz
        )
        ss = fb.filter_signal(s, dsp.FilterBankMode.Parallel)
        dsp.envelope(ss)

    def test_envelope_matches_scipy_hilbert_reference(self):
        """Per the source, `envelope(analytic=True)` first linearly
        detrends the signal (`Signal.detrend(1)`, itself independently
        covered by `test_detrend`), then returns `abs(scipy.signal.hilbert(
        ...))` directly with no further processing -- a genuine exact
        passthrough, not an approximation. For a constant-amplitude sine
        (many full periods, so the linear detrend is a near-no-op), the
        envelope should also be close to the constant amplitude away from
        the Hilbert transform's edge-ringing region.

        """
        fs = 5_000
        freq = 200.0
        n_periods = 100
        n_samples = int(n_periods * fs / freq)
        t = np.arange(n_samples) / fs
        amplitude = 0.6
        x = amplitude * np.sin(2 * np.pi * freq * t)
        sig = dsp.Signal(None, x[:, None], fs, constrain_amplitude=False)

        env = dsp.envelope(sig, analytic=True)

        detrended = sig.detrend(1)
        expected = np.abs(hilbert(detrended.time_data, axis=0))
        np.testing.assert_allclose(env, expected, atol=1e-12)

        # Away from the edges (Hilbert-transform ringing), the envelope of
        # a constant-amplitude tone should stay close to that amplitude.
        interior = env[n_samples // 10 : -n_samples // 10, 0]
        np.testing.assert_allclose(interior, amplitude, atol=0.01)

    def test_spectral_difference(self):
        filt = dsp.Filter.biquad(dsp.BiquadEqType.Peaking, 500.0, 10.0, 1.0, 48000)
        spec = dsp.Spectrum.from_filter(
            dsp.tools.log_frequency_vector([20, 20e3], 128), filt, False
        )
        spec_flat = dsp.Spectrum.from_filter(
            dsp.tools.log_frequency_vector([20, 20e3], 128),
            dsp.Filter.biquad(dsp.BiquadEqType.Peaking, 500.0, 0.0, 1.0, 48000),
            False,
        )
        sp_out = spec.spectral_difference(spec_flat, energy_normalization=False)
        np.testing.assert_almost_equal(spec.spectral_data, sp_out.spectral_data)

        with pytest.raises(AssertionError):
            sp_out = spec.spectral_difference(
                spec_flat, energy_normalization=False, complex=True
            )

        spec.spectral_difference(
            spec_flat,
            energy_normalization=True,
            octave_fraction_smoothing=12.0,
            dynamic_range_db=None,
        )
