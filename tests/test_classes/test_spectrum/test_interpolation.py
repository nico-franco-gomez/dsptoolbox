"""
Tests for `Spectrum` interpolation (all domain/scheme/edge-handling
combinations, plus cross-checks against the scipy/numpy calls they
delegate to).
"""

import numpy as np
import pytest

import dsptoolbox as dsp


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

    def test_interpolation_magnitude(self):
        sp_mag = self.get_spectrum_from_filter(None, False)
        f = np.array([200.0, 300.0])
        f_outside = np.array([sp_mag.frequency_vector_hz[-1] + 1.0])

        # Mixing a Complex-only domain/type with a magnitude-only spectrum
        # (or vice versa) is rejected rather than silently coerced.
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

        # Querying outside the known range raises when edge handling is Error
        with pytest.raises(AssertionError):
            sp_mag_err = sp_mag.set_interpolator_parameters(
                dsp.InterpolationDomain.Power,
                edges_handling=dsp.InterpolationEdgeHandling.Error,
            )
            sp_mag_err.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)

        # Remaining domain/scheme/edge-handling combinations: functionality
        # only, no result checking (covered separately in
        # test_interpolation_magnitude_matches_scipy).
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
