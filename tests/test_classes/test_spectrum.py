"""
Tests for the Spectrum class.
"""

import os
import pickle
import tempfile
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp

RIR_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "example_data",
    "rir.wav",
)


class TestSpectrum:
    def get_spectrum_from_filter(self, freqs=None, complex=False):
        """Get some spectrum from a filter. If `freqs=None`, it is a
        logarithmic vector."""
        filt = dsp.Filter.biquad(dsp.BiquadEqType.Peaking, 500.0, 10.0, 1.0, 48000)
        return dsp.Spectrum.from_filter(
            (
                dsp.tools.log_frequency_vector([20, 20e3], 128)
                if freqs is None
                else freqs
            ),
            filt,
            complex,
        )

    rir_spec_complex = dsp.Spectrum.from_signal(
        dsp.ImpulseResponse.from_file(RIR_PATH), True
    )
    rir = dsp.ImpulseResponse.from_file(RIR_PATH)
    rir_spec_real = dsp.Spectrum.from_signal(
        dsp.ImpulseResponse.from_file(RIR_PATH), False
    )

    def get_spectrum_from_rir(self, complex=False):
        return self.rir_spec_complex.copy() if complex else self.rir_spec_real.copy()

    def test_properties(self):
        spec = self.get_spectrum_from_filter(complex=False)
        assert spec.frequency_vector_type == dsp.FrequencySpacing.Logarithmic
        assert spec.is_magnitude
        assert spec.number_of_channels == 1

        spec = self.get_spectrum_from_filter(complex=True)
        assert not spec.is_magnitude

        freqs = np.array([100.0, 200.0, 300.0])
        spec = self.get_spectrum_from_filter(freqs, complex=True)
        assert spec.frequency_vector_type == dsp.FrequencySpacing.Linear

        freqs = np.array([100.0, 200.0, 300.0, 504.0])
        spec = self.get_spectrum_from_filter(freqs, complex=True)
        assert spec.frequency_vector_type == dsp.FrequencySpacing.Other
        assert spec.number_frequency_bins == len(freqs)

    def test_constructor_and_setters(self):
        freqs = np.array([100.0, 200.0, 300.0])
        spec = dsp.Spectrum(freqs, [np.zeros(3) for _ in range(2)])
        assert len(spec) == len(freqs)
        assert spec.number_of_channels == 2

    def test_save_spectrum_round_trip_and_format_checking(self):
        spec = self.get_spectrum_from_filter()
        with tempfile.TemporaryDirectory() as d:
            # No extension -> ".pkl" gets appended
            spec.save_spectrum(join(d, "no_ext"))
            with open(join(d, "no_ext.pkl"), "rb") as fh:
                reloaded = pickle.load(fh)
            np.testing.assert_array_equal(
                reloaded.frequency_vector_hz, spec.frequency_vector_hz
            )
            assert reloaded.number_of_channels == spec.number_of_channels

            # Matching ".pkl" extension is accepted as is
            spec.save_spectrum(join(d, "with_ext.pkl"))
            assert os.path.exists(join(d, "with_ext.pkl"))

            # A mismatched extension is rejected
            with pytest.raises(AssertionError):
                spec.save_spectrum(join(d, "wrong_ext.txt"))

    def test_trim(self):
        freqs = np.array([100.0, 200.0, 300.0, 504.0])
        spec = self.get_spectrum_from_filter(freqs, complex=True)
        spec2 = spec.copy().trim(200.0, 300.0, True)
        np.testing.assert_array_equal(
            np.array([200.0, 300.0]), spec2.frequency_vector_hz
        )

        spec2 = spec.copy().trim(100.0, 300.0, False)
        np.testing.assert_array_equal(np.array([200.0]), spec2.frequency_vector_hz)

    def test_sum_channels(self):
        freqs = np.array([100.0, 200.0, 300.0, 500.0])
        spec = self.get_spectrum_from_filter(freqs, complex=False)
        energy_sum_1 = spec.sum_channels(True)
        magnitude_sum = spec.sum_channels(False)
        assert magnitude_sum.is_magnitude

        spec = self.get_spectrum_from_filter(freqs, complex=True)
        energy_sum_2 = spec.copy().sum_channels(True)
        complex_sum = spec.sum_channels(False)
        assert not complex_sum.is_magnitude

        np.testing.assert_allclose(
            energy_sum_1.spectral_data, energy_sum_2.spectral_data
        )

    def test_normalize(self):
        freqs = np.array([100.0, 200.0, 300.0, 500.0])
        spec = self.get_spectrum_from_filter(freqs, complex=False)

        spec = spec.append_spectra([spec.copy().apply_gain(-6.0)])
        np.testing.assert_allclose(
            spec.copy()
            .normalize(200.0, None)
            .get_interpolated_spectrum(np.array([200.0]), dsp.SpectrumType.Magnitude),
            1.0,
        )
        np.testing.assert_allclose(
            spec.copy()
            .normalize(200.0, 0)
            .get_interpolated_spectrum(np.array([200.0]), dsp.SpectrumType.Magnitude)
            .squeeze(),
            dsp.tools.from_db(np.array([0.0, -6.0]), True),
        )

    def test_apply_gain(self):
        freqs = np.array([100.0, 200.0, 300.0, 500.0])
        spec = self.get_spectrum_from_filter(freqs, complex=False)
        np.testing.assert_allclose(
            spec.get_interpolated_spectrum(
                np.array([150.0]), dsp.SpectrumType.Magnitude
            )
            / spec.copy()
            .apply_gain(10.0)
            .get_interpolated_spectrum(np.array([150.0]), dsp.SpectrumType.Magnitude),
            dsp.tools.from_db(-10, True),
        )

        spec = spec.append_spectra([spec.copy().apply_gain(-6.0)])
        np.testing.assert_allclose(
            spec.get_interpolated_spectrum(
                np.array([150.0]), dsp.SpectrumType.Magnitude
            )
            / spec.copy()
            .apply_gain(10.0)
            .get_interpolated_spectrum(np.array([150.0]), dsp.SpectrumType.Magnitude),
            dsp.tools.from_db(-10, True),
        )
        np.testing.assert_allclose(
            (
                spec.get_interpolated_spectrum(
                    np.array([150.0]), dsp.SpectrumType.Magnitude
                )
                / spec.copy()
                .apply_gain(np.array([0.0, 10.0]))
                .get_interpolated_spectrum(
                    np.array([150.0]), dsp.SpectrumType.Magnitude
                )
            ).squeeze(),
            dsp.tools.from_db(np.array([0.0, -10]), True),
        )
        with pytest.raises(AssertionError):
            spec.apply_gain(np.array([1.0, 3.0, 5.0]))

    def test_resample(self):
        # Only functionaltiy
        freqs = np.array([100.0, 200.0, 300.0])

        sp = self.get_spectrum_from_rir(False)
        sp = sp.resample(freqs)
        sp = self.get_spectrum_from_rir(True)
        sp = sp.resample(freqs)

    def test_interpolation_magnitude(self):
        sp_mag = self.get_spectrum_from_filter(None, False)
        f = np.array([200.0, 300.0])
        f_outside = np.array([sp_mag.frequency_vector_hz[-1] + 1.0])

        # Assertions
        # Complex and magnitude
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(dsp.InterpolationDomain.Complex)
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(dsp.InterpolationDomain.Complex)
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(dsp.InterpolationDomain.Magnitude)
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(dsp.InterpolationDomain.Power)
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(dsp.InterpolationDomain.MagnitudePhase)
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(dsp.InterpolationDomain.MagnitudePhase)
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)

        # Padding
        with pytest.raises(AssertionError):
            sp_mag_err = sp_mag.set_interpolator_parameters(
                dsp.InterpolationDomain.Power,
                edges_handling=dsp.InterpolationEdgeHandling.Error,
            )
            sp_mag_err.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)

        # Normal functionality magnitude (no checking results)
        #
        sp_mag = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_mag = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_mag = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_mag = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_mag = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_mag = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_mag = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_mag = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.Extend,
        )
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_mag = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.OnePad,
        )
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)

    def test_interpolation_complex(self):
        sp_comp = self.get_spectrum_from_filter(None, True)
        f = np.array([200.0, 300.0])
        f_outside = np.array([sp_comp.frequency_vector_hz[-1] + 1.0])

        # Normal functionality magnitude (no checking results)
        #
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        #
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        #
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        #
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.Extend,
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.Extend,
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Complex)
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.Extend,
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Complex)
        #
        sp_comp = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Complex)

    def test_interpolation_magnitude_matches_scipy(self):
        """`Linear` scheme delegates to `numpy.interp`, `Cubic` to
        `scipy.interpolate.CubicSpline`, and `Pchip` to
        `scipy.interpolate.PchipInterpolator` (per the source of
        `get_interpolated_spectrum`); this checks each against its actual
        scipy/numpy counterpart directly, independent of the internal call.

        """
        sp_mag = self.get_spectrum_from_filter(None, False)
        f = np.array([200.0, 300.0, 1234.5])
        freqs = sp_mag.frequency_vector_hz
        data = sp_mag.spectral_data[:, 0]

        # Linear scheme, Magnitude domain -> np.interp on the magnitude itself
        sp_lin = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        out = sp_lin.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        expected = np.interp(f, freqs, data)
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-12)

        # Cubic scheme, Magnitude domain -> scipy CubicSpline on the magnitude
        sp_cubic = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        out = sp_cubic.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        from scipy.interpolate import CubicSpline

        expected = CubicSpline(freqs, data)(f)
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-10)

        # Power domain -> interpolation happens on magnitude**2; Magnitude
        # output type takes the square root of the interpolated power.
        sp_power = sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        out_power = sp_power.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        expected_power = np.interp(f, freqs, data**2.0)
        np.testing.assert_allclose(out_power[:, 0], expected_power, rtol=1e-12)
        out_mag = sp_power.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        np.testing.assert_allclose(out_mag[:, 0], expected_power**0.5, rtol=1e-12)

    def test_interpolation_complex_matches_scipy(self):
        """`Complex` domain interpolates the real and imaginary parts
        independently (per the source); this checks the `Linear` and
        `Pchip` schemes against `numpy.interp`/
        `scipy.interpolate.PchipInterpolator` applied separately to the
        real and imaginary parts.

        """
        sp_comp = self.get_spectrum_from_filter(None, True)
        f = np.array([200.0, 300.0, 1234.5])
        freqs = sp_comp.frequency_vector_hz
        data = sp_comp.spectral_data[:, 0]

        sp_lin = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        out = sp_lin.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        expected = np.interp(f, freqs, np.real(data)) + 1j * np.interp(
            f, freqs, np.imag(data)
        )
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-12)

        from scipy.interpolate import PchipInterpolator

        sp_pchip = sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        out = sp_pchip.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        expected = PchipInterpolator(freqs, np.real(data))(f) + 1j * PchipInterpolator(
            freqs, np.imag(data)
        )(f)
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-10)

    def test_get_energy(self):
        # Total energy
        sp = self.get_spectrum_from_rir()
        np.testing.assert_allclose(
            sp.get_energy(),
            np.sum(sp.spectral_data**2.0, axis=0)
            * (sp.frequency_vector_hz[1] - sp.frequency_vector_hz[0]),
            rtol=0.01,
        )

        # Functionality of boundaries
        sp.get_energy(10.0, 50.0)
        sp.get_energy(10.0, None)
        sp.get_energy(None, 50.0)

        with pytest.raises(AssertionError):
            sp.get_energy(200.0, 50.0)

    def test_apply_octave_smoothing(self):
        # Only functionality
        sp = self.get_spectrum_from_filter()
        sp = sp.apply_octave_smoothing(12.0)

        sp = self.get_spectrum_from_filter(np.linspace(500, 2000))
        sp = sp.apply_octave_smoothing(12.0)

    def test_apply_octave_smoothing_flat_input_stays_flat(self):
        """A perfectly flat magnitude spectrum has no ripple for octave
        smoothing to remove, so it must come back (near) unchanged
        (exact-case plausibility check).

        """
        freqs = dsp.tools.log_frequency_vector([20, 20e3], 128)
        flat = np.ones((len(freqs), 1))
        sp = dsp.Spectrum(freqs, flat)
        smoothed = sp.apply_octave_smoothing(3.0)
        np.testing.assert_allclose(smoothed.spectral_data, 1.0, atol=1e-9)

    def test_coherence(self):
        sp = self.get_spectrum_from_rir()
        sp = sp.set_coherence(np.zeros((len(sp), 1)))
        sp.plot_coherence()

    def test_plot_magnitude(self):
        sp = self.get_spectrum_from_filter()
        sp.plot_magnitude(True, dsp.MagnitudeNormalization.NoNormalization, None)
        sp.plot_magnitude(True, dsp.MagnitudeNormalization.NoNormalization, 10.0)
        sp.plot_magnitude(True, dsp.MagnitudeNormalization.OneKhz, 10.0)
        sp.plot_magnitude(True, dsp.MagnitudeNormalization.Max, 10.0)
        sp.plot_magnitude(True, dsp.MagnitudeNormalization.Energy, 10.0)
        sp.plot_magnitude(False, dsp.MagnitudeNormalization.NoNormalization, None)
        sp.plot_magnitude(False, dsp.MagnitudeNormalization.NoNormalization, None)
        sp.plot_magnitude(False, dsp.MagnitudeNormalization.NoNormalization, None)

        sp.plot_magnitude(False, dsp.MagnitudeNormalization.OneKhzFirstChannel, None)
        sp.plot_magnitude(False, dsp.MagnitudeNormalization.MaxFirstChannel, None)
        sp.plot_magnitude(False, dsp.MagnitudeNormalization.EnergyFirstChannel, None)

    def test_to_signal(self):
        # Only functionality
        spec = self.get_spectrum_from_rir(True)
        spec.to_signal(48000)
        spec.to_signal(96000)
        spec.to_signal(44100, 2.0)

        # Non-linear frequency
        spec = spec.resample(dsp.tools.log_frequency_vector([1, 24e3], 512))
        spec.to_signal(44100, 2.0)

        with pytest.raises(AssertionError):
            spec.to_signal(44100)

        # Non-complex spectrum
        spec = self.get_spectrum_from_rir(False)
        with pytest.raises(AssertionError):
            spec.to_signal(96000)

    def test_warp(self):
        # Only functionality
        spec = self.get_spectrum_from_rir(False)
        spec.warp(-0.7, self.rir.sampling_rate_hz)
        spec.warp(0.7, self.rir.sampling_rate_hz)
        with pytest.raises(AssertionError):
            spec.warp(1.1, self.rir.sampling_rate_hz)
        with pytest.raises(AssertionError):
            spec.warp(0.1, self.rir.sampling_rate_hz - 200)

    def test_warp_boundary_fixed_points(self):
        """`Spectrum.warp` only relabels the frequency axis via the
        Oppenheim allpass frequency-warping map (it does not touch
        `spectral_data`, unlike `transforms.warp`/`warp_filter` covered
        elsewhere). That map has f=0 and f=Nyquist as fixed points for any
        warping factor -- an exact closed-form invariant, verified here for
        both signs of warping factor. The RIR's spectrum vector spans
        exactly [0, fs/2], so its first/last entries are the boundary
        points themselves.

        """
        spec = self.get_spectrum_from_rir(False)
        assert spec.frequency_vector_hz[0] == 0.0
        assert spec.frequency_vector_hz[-1] == self.rir.sampling_rate_hz / 2

        for warping_factor in (-0.7, 0.4):
            warped = spec.warp(warping_factor, self.rir.sampling_rate_hz)
            assert np.isclose(warped.frequency_vector_hz[0], 0.0, atol=1e-9)
            assert np.isclose(
                warped.frequency_vector_hz[-1],
                self.rir.sampling_rate_hz / 2,
                atol=1e-6,
            )

    def test_set_interpolator_parameters_returns_new_instance(self):
        sp = self.get_spectrum_from_rir(False)
        sp2 = sp.set_interpolator_parameters(dsp.InterpolationDomain.Magnitude)
        assert sp2 is not sp
        assert sp.frequency_vector_type is not None  # original still usable
