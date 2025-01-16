import dsptoolbox as dsp
import pytest
import numpy as np
import scipy.signal as sig
import os


class TestFilterbanksModule:

    fs = 5000

    def get_noise(self):
        return dsp.generators.noise(
            length_seconds=1.0, sampling_rate_hz=self.fs
        )

    def test_linkwitz(self):
        # Only functionality
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

        # Test filtering
        s = self.get_noise()
        fb.filter_signal(s, mode=dsp.FilterBankMode.Parallel)

    def test_reconstructing_fractional_octave_bands(self):
        # Only functionality
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

    def test_auditory_filters_gammatone(self):
        # Only functionality
        fb = dsp.filterbanks.auditory_filters_gammatone(
            frequency_range_hz=[500, 1000], sampling_rate_hz=self.fs
        )
        with pytest.raises(AssertionError):
            dsp.filterbanks.auditory_filters_gammatone(
                frequency_range_hz=[500, 3000], sampling_rate_hz=self.fs
            )

        # Reconstruct signal
        n = self.get_noise()
        mb = fb.filter_signal(n, dsp.FilterBankMode.Parallel)
        fb.reconstruct(mb)

    def test_qmf_crossover(self):
        # Only functionality
        ny_hz = self.fs // 2
        lp = dsp.Filter.fir_filter(
            order=10,
            frequency_hz=ny_hz // 2,
            type_of_pass=dsp.FilterPassType.Lowpass,
            sampling_rate_hz=self.fs,
        )
        fb = dsp.filterbanks.qmf_crossover(lp)
        s = self.get_noise()
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
        fb.reconstruct_signal(mb_, upsample=True)

    def test_octave_filter_bank(self):
        fs_hz = 10_000
        dsp.filterbanks.fractional_octave_bands(
            frequency_range_hz=[31, 2000],
            octave_fraction=1,
            filter_order=6,
            sampling_rate_hz=fs_hz,
        )
        dsp.filterbanks.fractional_octave_bands(
            frequency_range_hz=[31, 4500],
            octave_fraction=12,
            filter_order=6,
            sampling_rate_hz=fs_hz,
        )

        with pytest.raises(AssertionError):
            dsp.filterbanks.fractional_octave_bands(
                frequency_range_hz=[31, 8000],
                octave_fraction=1,
                filter_order=6,
                sampling_rate_hz=fs_hz,
            )

    def test_weightning_filter(self):
        fs_hz = 5_000
        dsp.filterbanks.weighting_filter(True, fs_hz)
        dsp.filterbanks.weighting_filter(False, fs_hz)

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

        # Get perfect impulse
        h = np.zeros(len(coefficients))
        h[len(coefficients) // 2] = 1

        # Assert that both filters summed give a perfect impulse
        assert np.all(
            np.isclose(
                h,
                f2.get_coefficients(dsp.FilterCoefficientsType.Ba)[0]
                + coefficients,
            )
        )

        # Check functionality for even length
        f = dsp.Filter.fir_filter(
            type_of_pass=dsp.FilterPassType.Lowpass,
            order=121,
            frequency_hz=400,
            sampling_rate_hz=fs_hz,
        )
        dsp.filterbanks.complementary_fir_filter(f)

    def test_phase_linearizer(self):
        # Get some phase response
        fs_hz = 48_000
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [570, 2000], order=[2, 2], sampling_rate_hz=fs_hz
        )
        ir = fb.get_ir(length_samples=2**14).collapse()
        ir.spectrum_method = dsp.SpectrumMethod.FFT
        _, sp = ir.get_spectrum()

        # Initialize with wrong length
        with pytest.raises(AssertionError):
            dsp.filterbanks.PhaseLinearizer(
                np.angle(sp[:, 0]), len(ir) // 2, fs_hz
            )

        # Phase linearizer - Without interpolating
        pl = dsp.filterbanks.PhaseLinearizer(
            np.angle(sp[:, 0]), len(ir), fs_hz
        )
        with pytest.raises(AssertionError):
            pl.set_parameters(-10)
        pl.get_filter_as_ir()
        pl.get_filter()
        pl.set_parameters()

        # Phase linearizer – with interpolation
        ir = fb.get_ir(length_samples=2**9).collapse()
        ir.spectrum_method = dsp.SpectrumMethod.FFT
        _, sp = ir.get_spectrum()
        pl = dsp.filterbanks.PhaseLinearizer(
            np.angle(sp[:, 0]), len(ir), fs_hz
        )
        pl.get_filter_as_ir()
        pl.get_filter()

    def test_group_delay_designer(self):
        fs_hz = 48_000
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [570, 2000], order=[2, 2], sampling_rate_hz=fs_hz
        )
        ir = fb.get_ir(length_samples=2**14).collapse()

        # Group delay-based correction
        ir = fb.get_ir(length_samples=2**14).collapse()
        _, gd = dsp.transfer_functions.group_delay(ir)
        gd = np.max(gd) * 2 - gd
        pl = dsp.filterbanks.GroupDelayDesigner(gd.squeeze(), len(ir), fs_hz)
        pl.set_parameters(1.0)
        min_length_filt = pl.get_filter()

        # ir = dsp.pad_trim(pl.get_filter_as_ir(), 2**15)
        # ir.plot_time()
        # ir.plot_magnitude()
        # dsp.plots.show()

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

    def test_pinking_filter(self):
        # Only functionality
        fs_hz = 44100
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=fs_hz)
        n.set_spectrum_parameters(window_length_samples=1024)
        f = dsp.filterbanks.pinking_filter(3000, fs_hz)
        n2 = f.filter_signal(n)
        n2 = dsp.append_signals(
            [
                n2,
                dsp.generators.noise(
                    length_seconds=1.0,
                    type_of_noise=dsp.generators.NoiseType.Pink,
                    sampling_rate_hz=fs_hz,
                ),
            ]
        )
        n2 = dsp.append_signals([n2, n])

    def test_matched_biquads(self):
        # Only functionality and plausibility
        # Parameters
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
        # Only functionality
        fs_hz = 44100
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=fs_hz)

        # Get kernel and apply filtering
        f = dsp.filterbanks.gaussian_kernel(0.02, sampling_rate_hz=fs_hz)
        n1 = f.filter_signal(n, zero_phase=True)

        # Compare to normal gaussian window
        length = int(0.02 * fs_hz + 0.5)
        sigma = length / (2.0 * np.log(1 / 1e-2)) ** 0.5
        w = sig.windows.gaussian(length, sigma, True)
        w /= w.sum()
        f = dsp.Filter.from_ba(w, [1.0], fs_hz)
        n1 = dsp.append_signals([n1, f.filter_signal(n, zero_phase=False)])

        # n1.plot_time()
        # dsp.plots.show()

    def test_arma(self):
        # Only functionality
        rir = dsp.ImpulseResponse(
            os.path.join(
                os.path.dirname(__file__), "..", "example_data", "rir.wav"
            )
        )
        dsp.filterbanks.arma(rir, 10, 0)
        dsp.filterbanks.arma(rir, 10, 1)
        dsp.filterbanks.arma(rir, 10, 11)
        dsp.filterbanks.arma(dsp.pad_trim(rir, len(rir) - 1), 10, 11)

        dsp.filterbanks.arma(rir, 10, 0, method_ar="burg")
        dsp.filterbanks.arma(rir, 10, 1, method_ar="burg")
        dsp.filterbanks.arma(rir, 10, 11, method_ar="burg")


class TestLatticeLadderFilter:
    b = np.array([1, 3, 3, 1])
    a = np.array([1, -0.9, 0.64, -0.576])

    def test_lattice_filter_coefficients(self):
        # Example values taken from Oppenheim, A. V., Schafer, R. W.,,
        # Buck, J. R. (1999). Discrete-Time Signal Processing.
        # Prentice-hall Englewood Cliffs.
        from dsptoolbox.classes.lattice_ladder_filter import (
            _get_lattice_ladder_coefficients_iir,
        )

        k, c = _get_lattice_ladder_coefficients_iir(self.b, self.a)

        k_expected = np.array([0.6728, -0.182, 0.576])
        c_expected = np.array([4.5404, 5.4612, 3.9, 1])

        assert np.all(np.isclose(k, k_expected, rtol=5))
        assert np.all(np.isclose(c, c_expected, rtol=5))

    def test_lattice_filter_filtering(self):
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=200)
        expected = sig.lfilter(self.b / 10, self.a, n.time_data.squeeze())

        from dsptoolbox.classes.lattice_ladder_filter import (
            _get_lattice_ladder_coefficients_iir,
        )

        k, c = _get_lattice_ladder_coefficients_iir(self.b / 10, self.a)

        f = dsp.filterbanks.LatticeLadderFilter(k, c, sampling_rate_hz=200)
        out = f.filter_signal(n)
        out = out.time_data.squeeze()
        assert np.all(np.isclose(expected, out))

    def test_convert_lattice_filter(self):
        fs = 44100
        # Second-order sections
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=fs)
        f = dsp.Filter.iir_filter(
            filter_design_method=dsp.IirDesignMethod.Bessel,
            order=9,
            type_of_pass=dsp.FilterPassType.Lowpass,
            frequency_hz=1000,
            sampling_rate_hz=fs,
        )
        new_f = dsp.filterbanks.convert_into_lattice_filter(f)
        n1 = f.filter_signal(n).time_data.squeeze()
        n2 = new_f.filter_signal(n).time_data.squeeze()
        assert np.all(np.isclose(n1, n2))

        # BA
        b, a = f.get_coefficients(dsp.FilterCoefficientsType.Ba)
        f2 = dsp.Filter(
            {dsp.FilterCoefficientsType.Ba: [b, a]},
            f.sampling_rate_hz,
        )
        new_f = dsp.filterbanks.convert_into_lattice_filter(f2)
        n1 = f2.filter_signal(n).time_data.squeeze()
        n2 = new_f.filter_signal(n).time_data.squeeze()
        assert np.all(np.isclose(n1, n2))

        # FIR
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=fs)
        f = dsp.Filter(
            {dsp.FilterCoefficientsType.Ba: [[1, 13 / 24, 5 / 8, 1 / 3], [1]]},
            sampling_rate_hz=fs,
        )
        new_f = dsp.filterbanks.convert_into_lattice_filter(f)
        n1 = f.filter_signal(n).time_data.squeeze()
        n2 = new_f.filter_signal(n).time_data.squeeze()
        assert np.all(np.isclose(n1, n2))
