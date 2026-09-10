"""
Tests for the lattice-ladder filter conversion/topology.
"""

import numpy as np
import pytest
import scipy.signal as sig

import dsptoolbox as dsp


class TestLatticeLadderFilter:
    b = np.array([1, 3, 3, 1])
    a = np.array([1, -0.9, 0.64, -0.576])

    def test_lattice_filter_coefficients(self):
        # Example values taken from Oppenheim, A. V., Schafer, R. W.,,
        # Buck, J. R. (1999). Discrete-Time Signal Processing.
        # Prentice-hall Englewood Cliffs.
        from dsptoolbox.realtime.lattice_ladder_filter import (
            _get_lattice_ladder_coefficients_iir,
        )

        k, c = _get_lattice_ladder_coefficients_iir(self.b, self.a)

        k_expected = np.array([0.6728, -0.182, 0.576])
        c_expected = np.array([4.5404, 5.4612, 3.9, 1])

        assert np.all(np.isclose(k, k_expected, rtol=5))
        assert np.all(np.isclose(c, c_expected, rtol=5))

    def test_lattice_filter_filtering(self):
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=200, rng=0)
        expected = sig.lfilter(self.b / 10, self.a, n.time_data.squeeze())

        from dsptoolbox.realtime.lattice_ladder_filter import (
            _get_lattice_ladder_coefficients_iir,
        )

        k, c = _get_lattice_ladder_coefficients_iir(self.b / 10, self.a)

        f = dsp.realtime.LatticeLadderFilter(k, c, sampling_rate_hz=200)
        out = f.filter_signal(n)
        out = out.time_data.squeeze()
        assert np.all(np.isclose(expected, out))

    def test_lattice_fir_rust_backend_parity(self):
        from dsptoolbox.realtime.lattice_ladder_filter import (
            _lattice_filtering_fir,
            _lattice_filtering_fir_python,
            _lattice_filtering_fir_rust,
        )

        if _lattice_filtering_fir_rust is None:
            pytest.skip("Rust extension is not available")

        rng = np.random.default_rng(0)
        k = rng.uniform(-0.5, 0.5, 16)
        td = rng.normal(size=(257, 3))
        state = rng.normal(size=(16, 3))
        expected_td, expected_state = _lattice_filtering_fir_python(
            k, td.copy(), state.copy()
        )
        actual_td, actual_state = _lattice_filtering_fir(k, td.copy(), state.copy())

        np.testing.assert_array_equal(actual_td, expected_td)
        np.testing.assert_array_equal(actual_state, expected_state)

    def test_convert_lattice_filter(self):
        fs = 44100
        # Second-order sections
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=fs, rng=0)
        f = dsp.Filter.iir_filter(
            filter_design_method=dsp.IirDesignMethod.Bessel,
            order=9,
            type_of_pass=dsp.FilterPassType.Lowpass,
            frequency_hz=1000,
            sampling_rate_hz=fs,
        )
        new_f = dsp.realtime.LatticeLadderFilter.from_filter(f)
        n1 = f.filter_signal(n).time_data.squeeze()
        n2 = new_f.filter_signal(n).time_data.squeeze()
        assert np.all(np.isclose(n1, n2))

        # BA
        b, a = f.get_coefficients(dsp.FilterCoefficientsType.Ba)
        f2 = dsp.Filter.from_ba(b, a, f.sampling_rate_hz)
        new_f = dsp.realtime.LatticeLadderFilter.from_filter(f2)
        n1 = f2.filter_signal(n).time_data.squeeze()
        n2 = new_f.filter_signal(n).time_data.squeeze()
        assert np.all(np.isclose(n1, n2))

        # FIR
        n = dsp.generators.noise(length_seconds=1.0, sampling_rate_hz=fs, rng=1)
        f = dsp.Filter.from_ba([1, 13 / 24, 5 / 8, 1 / 3], [1], fs)
        new_f = dsp.realtime.LatticeLadderFilter.from_filter(f)
        n1 = f.filter_signal(n).time_data.squeeze()
        n2 = new_f.filter_signal(n).time_data.squeeze()
        assert np.all(np.isclose(n1, n2))
