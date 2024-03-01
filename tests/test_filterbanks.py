import dsptoolbox as dsp
import pytest
import numpy as np
import scipy.signal as sig


class TestFilterbanksModule:
    def test_linkwitz(self):
        # Only functionality
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [500, 1000], order=4, sampling_rate_hz=5_000
        )
        with pytest.raises(AssertionError):
            dsp.filterbanks.linkwitz_riley_crossovers(
                [500, 1000], order=[2, 4, 4], sampling_rate_hz=5_000
            )
        with pytest.raises(AssertionError):
            dsp.filterbanks.linkwitz_riley_crossovers(
                [500, 5000], order=4, sampling_rate_hz=5_000
            )

        # Test filtering
        s = dsp.generators.noise("white", sampling_rate_hz=5_000)
        fb.filter_signal(s, mode="parallel")

    def test_reconstructing_fractional_octave_bands(self):
        # Only functionality
        dsp.filterbanks.reconstructing_fractional_octave_bands(
            octave_fraction=1,
            frequency_range_hz=[63, 1024],
            overlap=0.5,
            slope=1,
            n_samples=2**10,
            sampling_rate_hz=5_000,
        )

    def test_auditory_filters_gammatone(self):
        # Only functionality
        fb = dsp.filterbanks.auditory_filters_gammatone(
            frequency_range_hz=[500, 1000], sampling_rate_hz=4_000
        )
        with pytest.raises(AssertionError):
            dsp.filterbanks.auditory_filters_gammatone(
                frequency_range_hz=[500, 3000], sampling_rate_hz=4_000
            )

        # Reconstruct signal
        s = dsp.generators.noise(type_of_noise="pink", sampling_rate_hz=4_000)
        mb = fb.filter_signal(s)
        fb.reconstruct(mb)

    def test_qmf_crossover(self):
        # Only functionality
        fs_hz = 4_000
        ny_hz = fs_hz // 2
        lp = dsp.Filter(
            "fir",
            {"order": 10, "freqs": ny_hz // 2, "type_of_pass": "lowpass"},
            sampling_rate_hz=fs_hz,
        )
        fb = dsp.filterbanks.qmf_crossover(lp)
        s = dsp.generators.noise("white", sampling_rate_hz=fs_hz)
        fb.filter_signal(
            s, mode="parallel", activate_zi=False, downsample=False
        )
        fb.filter_signal(
            s, mode="parallel", activate_zi=True, downsample=False
        )
        mb_ = fb.filter_signal(
            s, mode="parallel", activate_zi=False, downsample=True
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
        dsp.filterbanks.weightning_filter("a", fs_hz)
        dsp.filterbanks.weightning_filter("c", fs_hz)

    def test_complementary_filter_fir(self):
        fs_hz = 5000
        f = dsp.Filter(
            "fir",
            {"type_of_pass": "highpass", "order": 120, "freqs": 400},
            fs_hz,
        )
        f2 = dsp.filterbanks.complementary_fir_filter(f)
        coefficients = f.get_coefficients("ba")[0]

        # Get perfect impulse
        h = np.zeros(len(coefficients))
        h[len(coefficients) // 2] = 1

        # Assert that both filters summed give a perfect impulse
        assert np.all(
            np.isclose(h, f2.get_coefficients("ba")[0] + coefficients)
        )

        # Check functionality for even length
        f = dsp.Filter(
            "fir",
            {"type_of_pass": "lowpass", "order": 121, "freqs": 400},
            fs_hz,
        )
        dsp.filterbanks.complementary_fir_filter(f)

    def test_phase_linearizer(self):
        # Get some phase response
        fs_hz = 48_000
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [570, 2000], order=[2, 2], sampling_rate_hz=fs_hz
        )
        ir = fb.get_ir(length_samples=2**14).collapse()
        ir.set_spectrum_parameters(method="standard")
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
            pl.set_parameters(-10, 2)
        pl.get_filter_as_ir()
        pl.get_filter()

        # Parameters
        pl.set_parameters(80, 0.8)
        pl.set_parameters()

        # Phase linearizer – with interpolation
        ir = fb.get_ir(length_samples=2**9).collapse()
        ir.set_spectrum_parameters(method="standard")
        _, sp = ir.get_spectrum()
        pl = dsp.filterbanks.PhaseLinearizer(
            np.angle(sp[:, 0]), len(ir), fs_hz
        )
        pl.get_filter_as_ir()
        pl.get_filter()

    def test_TPTFilter(self):
        fs_hz = 10_000
        f = dsp.filterbanks.TPTFilter(500, np.sqrt(2), fs_hz)

        n = dsp.generators.noise(sampling_rate_hz=fs_hz)
        f.filter_signal(n)

        n = dsp.generators.noise(number_of_channels=3, sampling_rate_hz=fs_hz)
        f.filter_signal(n)

        f.get_ir()

        f.plot_magnitude(4096)
        f.plot_group_delay(4096)
        f.plot_phase(4096)


class TestLatticeLadderFilter:
    b = np.array([1, 3, 3, 1])
    a = np.array([1, -0.9, 0.64, -0.576])

    def test_lattice_filter_coefficients(self):
        # Example values taken from Oppenheim, A. V., Schafer, R. W.,,
        # Buck, J. R. (1999). Discrete-Time Signal Processing.
        # Prentice-hall Englewood Cliffs.
        from dsptoolbox.classes._lattice_ladder_filter import (
            _get_lattice_ladder_coefficients_iir,
        )

        k, c = _get_lattice_ladder_coefficients_iir(self.b, self.a)

        k_expected = np.array([0.6728, -0.182, 0.576])
        c_expected = np.array([4.5404, 5.4612, 3.9, 1])

        assert np.all(np.isclose(k, k_expected, rtol=5))
        assert np.all(np.isclose(c, c_expected, rtol=5))

    def test_lattice_filter_filtering(self):
        n = dsp.generators.noise(sampling_rate_hz=200)
        expected = sig.lfilter(self.b / 10, self.a, n.time_data.squeeze())

        from dsptoolbox.classes._lattice_ladder_filter import (
            _get_lattice_ladder_coefficients_iir,
        )

        k, c = _get_lattice_ladder_coefficients_iir(self.b / 10, self.a)

        f = dsp.filterbanks.LatticeLadderFilter(k, c, sampling_rate_hz=200)
        out = f.filter_signal(n)
        out = out.time_data.squeeze()
        assert np.all(np.isclose(expected, out))
