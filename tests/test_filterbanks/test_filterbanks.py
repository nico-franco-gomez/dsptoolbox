"""
Tests for the filterbanks module (crossovers, gammatone, QMF, octave
filter bank, weighting filters, phase linearizer, group delay designer,
etc.).
"""

import os
import pickle
import tempfile

import numpy as np
import pytest
import scipy.signal as sig
from matplotlib.pyplot import close, subplots

import dsptoolbox as dsp


class TestFilterbanksModule:
    fs = 5000

    def get_noise(self):
        """Seeded on purpose: some of these tests compare against a scipy
        reference or a reconstruction-error bound with a tight tolerance,
        which the specific noise realization can otherwise push past.

        """
        return dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=self.fs, rng=0)

    def test_linkwitz(self):
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [500, 1000], order=4, sampling_rate_hz=self.fs
        )
        with pytest.raises(AssertionError):
            dsp.filterbanks.linkwitz_riley_crossovers(
                [500, 1000], order=[2, 4, 4], sampling_rate_hz=self.fs
            )
        with pytest.raises(AssertionError):
            dsp.filterbanks.linkwitz_riley_crossovers(
                [500, 5000], order=4, sampling_rate_hz=self.fs
            )

        fb.plot_group_delay(length_samples=512)
        fb.plot_phase(length_samples=512)
        fb.plot_magnitude(length_samples=512)

        s = self.get_noise()
        fb.filter_signal(s, mode=dsp.FilterBankMode.Parallel)

    def test_lr_filterbank_save_round_trip_and_format_checking(self):
        """`LRFilterBank.save_filterbank` follows the same path convention as
        every other `save_*` method: the extension is required and checked."""
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [500, 1000], order=4, sampling_rate_hz=self.fs
        )
        with tempfile.TemporaryDirectory() as d:
            fb.save_filterbank(os.path.join(d, "lr_fb.pkl"))
            with open(os.path.join(d, "lr_fb.pkl"), "rb") as fh:
                reloaded = pickle.load(fh)
            assert reloaded.number_of_bands == fb.number_of_bands
            assert reloaded.sampling_rate_hz == fb.sampling_rate_hz

            with pytest.raises(ValueError):
                fb.save_filterbank(os.path.join(d, "lr_fb"))
            with pytest.raises(ValueError):
                fb.save_filterbank(os.path.join(d, "lr_fb.txt"))

    def test_linkwitz_riley_summed_magnitude_is_flat(self):
        """Per the docstring, LR crossovers are a "near perfect magnitude
        reconstruction filter bank": summing all bands reconstructs a flat
        0 dB magnitude response, but NOT a time-domain identity -- the
        combined system is allpass-like (flat magnitude, non-trivial phase,
        confirmed empirically: only ~84% of a reconstructed impulse's
        energy lands in its main lobe, though the peak stays exactly
        aligned with the input impulse). So this checks magnitude only, via
        the FFT of a summed impulse response, not `np.isclose` on samples.

        """
        fs = 48_000
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [500, 2000], order=4, sampling_rate_hz=fs
        )
        n_samples = 2**13
        d = dsp.generators.dirac(
            n_samples, delay_samples=n_samples // 2, sampling_rate_hz=fs
        )
        mb = fb.filter_signal(d, mode=dsp.FilterBankMode.Parallel)
        summed = np.sum(mb.collapse().time_data, axis=1)

        # Peak stays exactly at the input impulse's position.
        assert np.argmax(np.abs(summed)) == n_samples // 2

        mag_db = 20 * np.log10(np.abs(np.fft.rfft(summed)) + 1e-300)
        freqs = np.fft.rfftfreq(n_samples, 1 / fs)
        band = (freqs > 20) & (freqs < fs / 2 - 100)
        np.testing.assert_allclose(mag_db[band], 0.0, atol=1e-6)

    def test_reconstructing_fractional_octave_bands(self):
        n = self.get_noise()
        fb = dsp.filterbanks.reconstructing_fractional_octave_bands(
            octave_fraction=1,
            frequency_range_hz=[63, 1024],
            overlap=0.5,
            slope=1,
            n_samples=2**10,
            sampling_rate_hz=self.fs,
        )
        fb.filter_signal(n, dsp.FilterBankMode.Parallel)
        fb.filter_signal(n, dsp.FilterBankMode.Summed)

    def test_reconstructing_fractional_octave_bands_perfect_reconstruction(self):
        """Unlike LR crossovers, these are linear-phase FIR filters (per
        the source, group delay is exactly `n_samples/2` samples), so
        summing all bands reconstructs the original signal in the time
        domain almost exactly, once shifted by that known group delay.

        """
        fs = 5_000
        n_filt = 2**10
        fb = dsp.filterbanks.reconstructing_fractional_octave_bands(
            octave_fraction=1,
            frequency_range_hz=[63, 1024],
            overlap=0.5,
            slope=1,
            n_samples=n_filt,
            sampling_rate_hz=fs,
        )
        n_samples = 4_000
        delay_in = 100
        d = dsp.generators.dirac(n_samples, delay_samples=delay_in, sampling_rate_hz=fs)
        mb = fb.filter_signal(d, dsp.FilterBankMode.Parallel)
        summed = np.sum(mb.collapse().time_data, axis=1)

        expected = np.zeros(n_samples)
        expected[delay_in + n_filt // 2] = 1.0
        np.testing.assert_allclose(summed, expected, atol=1e-5)

    def test_auditory_filters_gammatone(self):
        fb = dsp.filterbanks.auditory_filters_gammatone(
            frequency_range_hz=[500, 1000], sampling_rate_hz=self.fs
        )
        with pytest.raises(AssertionError):
            dsp.filterbanks.auditory_filters_gammatone(
                frequency_range_hz=[500, 3000], sampling_rate_hz=self.fs
            )

        n = self.get_noise()
        mb = fb.filter_signal(n, dsp.FilterBankMode.Parallel)
        fb.reconstruct(mb)

    def test_auditory_filters_gammatone_center_frequencies_match_erb_spacing(self):
        """The filter bank's center frequencies are (per the source of
        `auditory_filters_gammatone`) computed directly by
        `dsptoolbox.tools.erb_frequencies`; this cross-checks that the two
        public APIs agree (the bank stores them as the private
        `_frequencies` attribute -- there is no other public accessor).

        """
        freq_range = [200.0, 2000.0]
        resolution = 0.5
        fb = dsp.filterbanks.auditory_filters_gammatone(
            frequency_range_hz=freq_range,
            resolution=resolution,
            sampling_rate_hz=self.fs,
        )
        expected = dsp.tools.erb_frequencies(freq_range, resolution)
        np.testing.assert_allclose(fb._frequencies, expected, rtol=1e-12)

    def test_qmf_crossover(self):
        # Factors around half band frequency were manually extracted for satisfactory
        # reconstruction precision
        lp_iir = dsp.Filter.iir_filter(
            12, (self.fs / 2) * 0.5095, dsp.FilterPassType.Lowpass, self.fs
        )
        lp_fir = dsp.Filter.fir_filter(
            order=11,
            frequency_hz=(self.fs / 2) * 0.572,
            type_of_pass=dsp.FilterPassType.Lowpass,
            sampling_rate_hz=self.fs,
        )
        s = self.get_noise()

        for lp in [lp_fir, lp_iir]:
            fb = dsp.filterbanks.qmf_crossover(lp)
            fb.filter_signal(
                s,
                mode=dsp.FilterBankMode.Parallel,
                activate_zi=False,
                downsample=False,
            )
            fb.filter_signal(
                s,
                mode=dsp.FilterBankMode.Parallel,
                activate_zi=True,
                downsample=False,
            )
            mb_ = fb.filter_signal(
                s,
                mode=dsp.FilterBankMode.Parallel,
                activate_zi=False,
                downsample=True,
            )

            # Reconstruction
            round_trip = fb.reconstruct_signal(mb_, upsample=True)
            spec = s.spectral_difference(round_trip, energy_normalization=False)
            spec.spectral_data[:2] = 1.0  # Remove DC, dominated by rounding noise
            np.testing.assert_allclose(
                dsp.tools.to_db(spec.spectral_data, True), 0.0, atol=1
            )

        fb.plot_magnitude(
            length_samples=512, mode=dsp.FilterBankMode.Parallel, downsample=True
        )
        fb.plot_magnitude(
            length_samples=512, mode=dsp.FilterBankMode.Parallel, downsample=False
        )
        fb.plot_group_delay(length_samples=512, mode=dsp.FilterBankMode.Parallel)
        fb.plot_phase(length_samples=512, mode=dsp.FilterBankMode.Parallel)

    def test_octave_filter_bank(self):
        fs_hz = 10_000
        dsp.filterbanks.fractional_octave_bands(
            frequency_range_hz=[31, 2000],
            octave_fraction=1,
            filter_order=6,
            sampling_rate_hz=fs_hz,
        )
        _, center, (low, up) = dsp.filterbanks.fractional_octave_bands(
            frequency_range_hz=[31, 4500],
            octave_fraction=12,
            filter_order=6,
            sampling_rate_hz=fs_hz,
        )
        assert len(center) == len(low) and len(low) == len(up)
        assert np.all(np.ediff1d(center) > 0)

        with pytest.raises(AssertionError):
            dsp.filterbanks.fractional_octave_bands(
                frequency_range_hz=[31, 8000],
                octave_fraction=1,
                filter_order=6,
                sampling_rate_hz=fs_hz,
            )

    def test_weighting_filter(self):
        fs_hz = 5_000
        dsp.filterbanks.weighting_filter(True, fs_hz)
        dsp.filterbanks.weighting_filter(False, fs_hz)

    def test_a_weighting_matches_iec_61672_reference_values(self):
        """Published IEC 61672-1:2013 Table 2 A-weighting values (dB) at
        standard nominal frequencies: 31.5->-39.4, 63->-26.2, 125->-16.1,
        250->-8.6, 500->-3.2, 1000->0.0, 2000->+1.2, 4000->+1.0. A higher
        sampling rate (48 kHz) is used here (rather than this class's
        default 5 kHz fixture) to keep all reference points well below
        Nyquist -- `weighting_filter` designs via a plain `bilinear_zpk`
        transform with no analog pre-warping, so its error grows quickly
        as a reference frequency approaches Nyquist (empirically confirmed:
        at fs=48 kHz the deviation is <0.15 dB up to 4 kHz, but already
        ~0.6 dB at 8 kHz and several dB by 16 kHz -- an inherent bilinear-
        transform limitation, not a bug, so those higher points are
        excluded here rather than loosening the tolerance for everyone).

        """
        fs_hz = 48_000
        f = dsp.filterbanks.weighting_filter(True, fs_hz)
        freqs = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000])
        expected_db = np.array([-39.4, -26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0])

        h = f.get_transfer_function(freqs)
        mag_db = 20 * np.log10(np.abs(h))
        np.testing.assert_allclose(mag_db, expected_db, atol=0.3)

    def test_complementary_filter_fir(self):
        fs_hz = 5000
        f = dsp.Filter.fir_filter(
            type_of_pass=dsp.FilterPassType.Highpass,
            order=120,
            frequency_hz=400,
            sampling_rate_hz=fs_hz,
        )
        f2 = dsp.filterbanks.complementary_fir_filter(f)
        coefficients = f.get_coefficients(dsp.FilterCoefficientsType.Ba)[0]

        h = np.zeros(len(coefficients))
        h[len(coefficients) // 2] = 1

        # Both filters summed should give a perfect impulse
        assert np.all(
            np.isclose(
                h,
                f2.get_coefficients(dsp.FilterCoefficientsType.Ba)[0] + coefficients,
            )
        )

        # Even filter length
        f = dsp.Filter.fir_filter(
            type_of_pass=dsp.FilterPassType.Lowpass,
            order=121,
            frequency_hz=400,
            sampling_rate_hz=fs_hz,
        )
        dsp.filterbanks.complementary_fir_filter(f)

    def test_phase_linearizer(self):
        fs_hz = 48_000
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [570, 2000], order=[2, 2], sampling_rate_hz=fs_hz
        )
        ir = fb.get_ir(length_samples=2**14).collapse()
        ir.spectrum_method = dsp.SpectrumMethod.FFT
        _, sp = ir.get_spectrum()

        # Phase vector length must match the IR length
        with pytest.raises(AssertionError):
            dsp.filterbanks.PhaseLinearizer(np.angle(sp[:, 0]), len(ir) // 2, fs_hz)

        # Without interpolating
        pl = dsp.filterbanks.PhaseLinearizer(np.angle(sp[:, 0]), len(ir), fs_hz)
        with pytest.raises(AssertionError):
            pl.set_parameters(-10)
        pl.get_filter_as_ir()
        pl.get_filter()
        pl.set_parameters()

        # With interpolation
        ir = fb.get_ir(length_samples=2**9).collapse()
        ir.spectrum_method = dsp.SpectrumMethod.FFT
        _, sp = ir.get_spectrum()
        pl = dsp.filterbanks.PhaseLinearizer(np.angle(sp[:, 0]), len(ir), fs_hz)
        pl.get_filter_as_ir()
        pl.get_filter()

    def test_group_delay_designer(self):
        fs_hz = 48_000
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [570, 2000], order=[2, 2], sampling_rate_hz=fs_hz
        )
        ir = fb.get_ir(length_samples=2**14).collapse()
        _, gd = dsp.transfer_functions.group_delay(ir)
        gd = np.max(gd) * 2 - gd
        pl = dsp.filterbanks.GroupDelayDesigner(gd.squeeze(), len(ir), fs_hz)
        pl.set_parameters(1.0)
        min_length_filt = pl.get_filter()

        new_filt = (
            dsp.filterbanks.GroupDelayDesigner(gd.squeeze(), len(ir), fs_hz)
            .set_parameters(1.0, 10)
            .get_filter()
        )
        assert len(new_filt) - 10 == len(min_length_filt)

        new_filt = (
            dsp.filterbanks.GroupDelayDesigner(gd.squeeze(), len(ir), fs_hz)
            .set_parameters(1.0, 0, False)
            .get_filter()
        )

    def test_group_delay_designer_flattens_group_delay(self):
        """Designing a correction filter with `target_group_delay = 2*max(gd)
        - gd` (mirroring the original response's group delay around its
        max) and cascading it with the original system should flatten the
        combined group delay much closer to a constant than the original
        alone (monotonic-improvement plausibility, no exact target value --
        `smoothing` and a passband well away from DC/Nyquist are needed
        here since raw numerical group-delay estimates are noisy at the
        edges, empirically confirmed to dominate the spread otherwise).

        """
        fs_hz = 48_000
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [570, 2000], order=[2, 2], sampling_rate_hz=fs_hz
        )
        ir = fb.get_ir(length_samples=2**14).collapse()
        f1, gd1 = dsp.transfer_functions.group_delay(
            ir, analytic_computation=True, smoothing=6
        )
        gd1 = gd1.squeeze()
        target = np.max(gd1) * 2 - gd1

        pl = dsp.filterbanks.GroupDelayDesigner(target, len(ir), fs_hz)
        pl.set_parameters(1.0)
        corr_filt = pl.get_filter()
        corr_ir = dsp.transfer_functions.filter_to_ir(corr_filt)

        combined_td = np.convolve(ir.time_data[:, 0], corr_ir.time_data[:, 0])
        combined = dsp.ImpulseResponse(None, combined_td[:, None], fs_hz)
        f2, gd2 = dsp.transfer_functions.group_delay(
            combined, analytic_computation=True, smoothing=6
        )
        gd2 = gd2.squeeze()

        mask1 = (f1 > 100) & (f1 < 10_000)
        mask2 = (f2 > 100) & (f2 < 10_000)
        assert np.std(gd2[mask2]) < np.std(gd1[mask1]) / 5

    def test_phase_linearizer_flattens_phase_nonlinearity(self):
        """Same monotonic-improvement idea as `GroupDelayDesigner`, but
        starting from a phase response directly (as `PhaseLinearizer` is
        meant to be used): the phase's deviation from a straight-line
        (linear-phase) fit should shrink substantially once corrected.

        """
        fs_hz = 48_000
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [570, 2000], order=[2, 2], sampling_rate_hz=fs_hz
        )
        ir = fb.get_ir(length_samples=2**14).collapse()
        ir.spectrum_method = dsp.SpectrumMethod.FFT
        _, sp = ir.get_spectrum()
        phase = np.angle(sp[:, 0])

        pl = dsp.filterbanks.PhaseLinearizer(phase, len(ir), fs_hz)
        corr_filt = pl.get_filter()
        corr_ir = dsp.transfer_functions.filter_to_ir(corr_filt)

        combined_td = np.convolve(ir.time_data[:, 0], corr_ir.time_data[:, 0])
        combined = dsp.ImpulseResponse(None, combined_td[:, None], fs_hz)
        combined.spectrum_method = dsp.SpectrumMethod.FFT
        f2, sp2 = combined.get_spectrum()
        phase2 = np.unwrap(np.angle(sp2[:, 0]))

        f1 = np.fft.rfftfreq(len(ir), 1 / fs_hz)
        phase1 = np.unwrap(phase)

        def linear_fit_residual_std(f, p):
            a = np.vstack([f, np.ones_like(f)]).T
            coeffs = np.linalg.lstsq(a, p, rcond=None)[0]
            return np.std(p - a @ coeffs)

        mask1 = (f1 > 200) & (f1 < 10_000)
        mask2 = (f2 > 200) & (f2 < 10_000)
        residual_before = linear_fit_residual_std(f1[mask1], phase1[mask1])
        residual_after = linear_fit_residual_std(f2[mask2], phase2[mask2])
        assert residual_after < residual_before / 5

    def test_pinking_filter(self):
        fs_hz = 44100
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=fs_hz, rng=109)
        n = n.set_spectrum_parameters(window_length_samples=1024)
        f = dsp.filterbanks.pinking_filter(3000, fs_hz)
        n2 = f.filter_signal(n)
        n2 = n2.append_signals(
            [
                dsp.generators.noise(
                    length_seconds=1.0,
                    type_of_noise=dsp.generators.NoiseType.Pink,
                    sampling_rate_hz=fs_hz,
                    rng=110,
                ),
            ]
        )
        n2 = n2.append_signals([n])

    def test_matched_biquads(self):
        fs_hz = 48000
        freq = 10e3
        gain_db = -20
        q = 2**0.5 / 2

        for eq_type in [
            dsp.BiquadEqType.Peaking,
            dsp.BiquadEqType.Lowpass,
            dsp.BiquadEqType.Highpass,
            dsp.BiquadEqType.Lowshelf,
            dsp.BiquadEqType.Highshelf,
            dsp.BiquadEqType.BandpassPeak,
            dsp.BiquadEqType.BandpassSkirt,
        ]:
            dsp.filterbanks.matched_biquad(eq_type, freq, gain_db, q, fs_hz)

    def test_gaussian_kernel(self):
        fs_hz = 44100
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=fs_hz, rng=111)

        f = dsp.filterbanks.gaussian_kernel(0.02, sampling_rate_hz=fs_hz)
        n1 = f.filter_signal(n, zero_phase=True)

        # Compare to a plain scipy Gaussian window filter
        length = int(0.02 * fs_hz + 0.5)
        sigma = length / (2.0 * np.log(1 / 1e-2)) ** 0.5
        w = sig.windows.gaussian(length, sigma, True)
        w /= w.sum()
        f = dsp.Filter.from_ba(w, [1.0], fs_hz)
        n1 = n1.append_signals([f.filter_signal(n, zero_phase=False)])

    def test_gaussian_kernel_matches_scipy_window_shape(self):
        """`gaussian_kernel` is documented as a first-order IIR
        *approximation* of a true Gaussian FIR window, not an exact match,
        so this compares its zero-phase-filtered impulse response against
        `scipy.signal.windows.gaussian` (built with the same sigma
        derivation used internally, per the source) via correlation and a
        bounded per-sample error rather than requiring bit-identical
        values (empirically verified: correlation > 0.998, max abs error
        well under 1e-3 for this configuration).

        """
        fs_hz = 44100
        kernel_length_s = 0.02
        length = int(kernel_length_s * fs_hz + 0.5)
        sigma = length / (2.0 * np.log(1 / 1e-2)) ** 0.5
        w = sig.windows.gaussian(length, sigma, True)
        w /= w.sum()

        f = dsp.filterbanks.gaussian_kernel(kernel_length_s, sampling_rate_hz=fs_hz)
        n_samples = 2_000
        imp = np.zeros(n_samples)
        imp[n_samples // 2] = 1.0
        imp_sig = dsp.Signal(None, imp[:, None], fs_hz)
        out = f.filter_signal(imp_sig, zero_phase=True)
        ir = out.time_data[:, 0]

        peak = np.argmax(np.abs(ir))
        half = length // 2
        center = ir[peak - half : peak - half + length]

        np.testing.assert_allclose(center, w, atol=5e-4)
        assert np.corrcoef(center, w)[0, 1] > 0.998

    def test_arma(self):
        rir = dsp.ImpulseResponse(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "example_data", "rir.wav"
            )
        )
        dsp.filterbanks.arma(rir, 10, 0)
        dsp.filterbanks.arma(rir, 10, 1)
        dsp.filterbanks.arma(rir, 10, 11)
        dsp.filterbanks.arma(rir.pad_trim(len(rir) - 1), 10, 11)

        for m in [
            dsp.filterbanks.ArmaMethod.SteiglitzMcBride,
            dsp.filterbanks.ArmaMethod.Prony,
            dsp.filterbanks.ArmaMethod.Burg,
        ]:
            dsp.filterbanks.arma(rir, 10, 0, method=m)
            dsp.filterbanks.arma(rir, 10, 1, method=m)
            dsp.filterbanks.arma(rir, 10, 11, method=m)

    def test_arma_recovers_spectrum_of_a_simple_resonator(self):
        """Easy case for the default `ArmaMethod.YuleWalker`: a two-pole,
        all-pole resonator (no zeros) is itself an exact AR(2) process, which
        is exactly the model `ArmaMethod.YuleWalker` assumes. Fitting
        `arma()` at the matching order to a sufficiently long, (numerically)
        fully decayed impulse response of that resonator should therefore
        recover a filter whose magnitude spectrum matches the original one
        closely.

        Note this is deliberately *not* done with a general biquad (e.g. a
        peaking EQ): a biquad has both poles and zeros, i.e. it is an
        ARMA(2, 2) process, and naively solving the plain (non-extended)
        Yule-Walker equations for an AR(2) model of such a process is a
        known-biased estimator of the poles (the presence of zeros corrupts
        the low-lag autocorrelation the AR fit relies on) -- that would not
        be an "easy case".

        Comparing magnitude spectra (rather than the raw `a` coefficients)
        avoids any sign/scaling ambiguity in the fitted parametrization.

        """
        pole_radius = 0.9
        pole_angle_rad = 2 * np.pi * 500.0 / self.fs
        a_true = np.array(
            [1.0, -2 * pole_radius * np.cos(pole_angle_rad), pole_radius**2.0]
        )
        original = dsp.Filter.from_ba(np.array([1.0]), a_true, self.fs)
        ir = original.get_ir(length_samples=2000)

        fitted = dsp.filterbanks.arma(
            ir, order_a=2, order_b=0, method=dsp.filterbanks.ArmaMethod.YuleWalker
        )

        freqs = np.linspace(20.0, self.fs / 2 * 0.95, 500)
        original_mag_db = 20 * np.log10(np.abs(original.get_transfer_function(freqs)))
        fitted_mag_db = 20 * np.log10(np.abs(fitted.get_transfer_function(freqs)))

        np.testing.assert_allclose(fitted_mag_db, original_mag_db, atol=1e-6)

    def test_fractional_delay(self):
        noise = dsp.Filter.iir_filter(
            8, self.fs / 4, dsp.FilterPassType.Lowpass, self.fs
        ).filter_signal(
            dsp.generators.noise(0.5, self.fs, padding_end_seconds=0.5, rng=112)
        )

        fractional = 0.5
        order = 30
        delay = dsp.filterbanks.fractional_delay(fractional, order, self.fs)
        noise_delayed = delay.filter_signal(noise)
        latency = dsp.latency(noise_delayed, noise, polynomial_points=3)[0][0]
        assert abs(latency - (fractional + order)) < 1e-3

    def test_crossover_plot_magnitude_modes_and_ax(self):
        """The crossover override used to drop `zero_phase` and `ax`, and its
        Summed branch still called the pre-enum `_get_normalized_spectrum`.

        """
        lp = dsp.Filter.from_ba(sig.firwin(31, 0.5), [1.0], self.fs)
        fb = dsp.filterbanks.qmf_crossover(lp)

        for mode in (dsp.FilterBankMode.Parallel, dsp.FilterBankMode.Summed):
            for downsample in (True, False):
                fig, _ = fb.plot_magnitude(512, mode, downsample=downsample)
                close(fig)

        # Sequential cannot downsample: the second filter would no longer
        # match the rate it is handed
        with pytest.raises(ValueError):
            fb.plot_magnitude(512, dsp.FilterBankMode.Sequential, downsample=True)

        # The base class arguments must still get through (A10)
        _, ax = subplots(1, 1)
        fb.plot_magnitude(512, dsp.FilterBankMode.Parallel, ax=ax)
        for downsample in (True, False):
            fb.plot_magnitude(
                512,
                dsp.FilterBankMode.Parallel,
                zero_phase=True,
                downsample=downsample,
            )
        close("all")

    def test_crossover_zero_phase_downsampling(self):
        """Forward-backward filtering cannot use the polyphase decimation, so
        each band is filtered at the original rate and decimated afterwards.
        That must be exactly `filtfilt` followed by decimation, and it must
        leave the band symmetric around the impulse.

        """
        rng = np.random.default_rng(0)
        td = rng.normal(0, 0.1, (2048, 2))
        s = dsp.Signal(None, td, self.fs, constrain_amplitude=False)

        lp = dsp.Filter.from_ba(sig.firwin(31, 0.5), [1.0], self.fs)
        fb = dsp.filterbanks.qmf_crossover(lp)

        bands = fb.filter_signal(
            s, dsp.FilterBankMode.Parallel, zero_phase=True, downsample=True
        )
        summed = fb.filter_signal(
            s, dsp.FilterBankMode.Summed, zero_phase=True, downsample=True
        )
        assert all(b.sampling_rate_hz == self.fs // 2 for b in bands.bands)
        assert summed.sampling_rate_hz == self.fs // 2

        reference = []
        for filt in fb.filters:
            b, a = filt.get_coefficients(dsp.FilterCoefficientsType.Ba)
            reference.append(sig.filtfilt(b, a, td, axis=0)[::2])
        for band, expected in zip(bands.bands, reference, strict=True):
            np.testing.assert_allclose(band.time_data, expected, atol=1e-12)
        np.testing.assert_allclose(summed.time_data, sum(reference), atol=1e-12)

        # A centered impulse stays centered, unlike with causal filtering
        delay = 256
        d = dsp.ImpulseResponse.from_time_data(
            np.eye(2 * delay + 1, 1, -delay), self.fs
        )
        low_band = fb.filter_signal(
            d, dsp.FilterBankMode.Parallel, zero_phase=True, downsample=True
        ).bands[0]
        td_low = low_band.time_data[:, 0]
        np.testing.assert_allclose(td_low, td_low[::-1], atol=1e-12)

        # Filter states cannot be carried through forward-backward filtering
        with pytest.raises(AssertionError):
            fb.filter_signal(
                s,
                dsp.FilterBankMode.Parallel,
                activate_zi=True,
                zero_phase=True,
                downsample=True,
            )
