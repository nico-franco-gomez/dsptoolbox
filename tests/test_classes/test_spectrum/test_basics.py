"""
Tests for the Spectrum class: construction, basic transforms, and I/O.
"""

import os
import pickle
import tempfile
from os.path import join

import numpy as np
import pytest
from matplotlib.pyplot import close

import dsptoolbox as dsp

RIR_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
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
        freqs = np.array([100.0, 200.0, 300.0])

        sp = self.get_spectrum_from_rir(False)
        sp = sp.resample(freqs)
        sp = self.get_spectrum_from_rir(True)
        sp = sp.resample(freqs)

    def test_get_energy(self):
        sp = self.get_spectrum_from_rir()
        np.testing.assert_allclose(
            sp.get_energy(),
            np.sum(sp.spectral_data**2.0, axis=0)
            * (sp.frequency_vector_hz[1] - sp.frequency_vector_hz[0]),
            rtol=0.01,
        )

        sp.get_energy(10.0, 50.0)
        sp.get_energy(10.0, None)
        sp.get_energy(None, 50.0)

        with pytest.raises(AssertionError):
            sp.get_energy(200.0, 50.0)

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
        spec = self.get_spectrum_from_rir(True)
        spec.to_signal(48000)
        spec.to_signal(96000)
        spec.to_signal(44100, 2.0)

        # Non-linear frequency vector
        spec = spec.resample(dsp.tools.log_frequency_vector([1, 24e3], 512))
        spec.to_signal(44100, 2.0)

        with pytest.raises(AssertionError):
            spec.to_signal(44100)

        # Only defined for a complex (not magnitude-only) spectrum
        spec = self.get_spectrum_from_rir(False)
        with pytest.raises(AssertionError):
            spec.to_signal(96000)

    def test_set_interpolator_parameters_returns_new_instance(self):
        sp = self.get_spectrum_from_rir(False)
        sp2 = sp.set_interpolator_parameters(dsp.InterpolationDomain.Magnitude)
        assert sp2 is not sp
        assert sp.frequency_vector_type is not None  # original still usable

    def test_one_khz_first_channel_uses_a_single_reference(self):
        """`OneKhzFirstChannel` must normalize every channel by channel 0."""
        fs = 48000
        rng = np.random.default_rng(0)
        s = dsp.Signal(None, rng.normal(0, 0.1, (4096, 3)), fs)
        s = s.set_spectrum_parameters(dsp.SpectrumMethod.FFT)
        sp = dsp.Spectrum.from_signal(s)

        fig, ax = sp.plot_magnitude(
            normalization=dsp.MagnitudeNormalization.OneKhzFirstChannel
        )
        curves = np.column_stack([line.get_ydata() for line in ax.get_lines()])
        raw_db = 20 * np.log10(np.abs(sp.spectral_data))
        offsets = curves - raw_db
        np.testing.assert_allclose(
            offsets, np.repeat(offsets[:, :1], offsets.shape[1], axis=1), atol=1e-9
        )
        close(fig)

    def test_gain_and_normalize_do_not_modify_original(self):
        rng = np.random.default_rng(0)
        freqs = np.linspace(0, 20000, 500)
        sp = dsp.Spectrum(freqs, rng.uniform(0.1, 1.0, (500, 2)))
        before = sp.spectral_data.copy()
        sp.apply_gain(6.0)
        sp.normalize(1000.0)
        np.testing.assert_array_equal(before, sp.spectral_data)

    def test_magnitude_normalization_agrees_across_plot_apis(self):
        """`Signal.plot_magnitude`, `ImpulseResponse.plot_bode` and
        `Spectrum.plot_magnitude` share one normalization helper and must
        therefore produce the same normalized curve.

        The energy normalizations are excluded: `Spectrum` derives them from
        `get_energy()`, which integrates `|X|**2 df`, so dividing by the
        number of bins leaves a factor `sqrt(df)` that the other two do not
        have.

        """
        fs = 48_000
        rng = np.random.default_rng(0)
        td = np.zeros((4096, 2))
        td[100, :] = [1.0, 0.5]
        td += rng.normal(0, 1e-3, (4096, 2))
        ir = dsp.ImpulseResponse(None, td, fs)
        spectrum = dsp.Spectrum.from_signal(ir)

        for normalization in (
            dsp.MagnitudeNormalization.NoNormalization,
            dsp.MagnitudeNormalization.OneKhz,
            dsp.MagnitudeNormalization.OneKhzFirstChannel,
            dsp.MagnitudeNormalization.Max,
            dsp.MagnitudeNormalization.MaxFirstChannel,
        ):
            _, ax_signal = ir.plot_magnitude(
                normalize=normalization, range_hz=None, smoothing=0
            )
            _, ax_bode = ir.plot_bode(normalize=normalization)
            _, ax_spectrum = spectrum.plot_magnitude(normalization=normalization)

            from_signal = ax_signal.get_lines()[0].get_ydata()
            from_bode = ax_bode[0].get_lines()[0].get_ydata()
            from_spectrum = ax_spectrum.get_lines()[0].get_ydata()
            close("all")

            np.testing.assert_allclose(from_bode, from_signal, atol=1e-10)
            np.testing.assert_allclose(from_spectrum, from_signal, atol=1e-10)

    def test_one_khz_normalization_lands_exactly_on_zero_db(self):
        fs = 48_000
        rng = np.random.default_rng(1)
        ir = dsp.ImpulseResponse(None, rng.normal(0, 0.1, (2048, 1)), fs)

        _, ax = ir.plot_magnitude(
            normalize=dsp.MagnitudeNormalization.OneKhz, range_hz=None, smoothing=0
        )
        f = ax.get_lines()[0].get_xdata()
        magnitude_db = ax.get_lines()[0].get_ydata()
        close("all")

        np.testing.assert_allclose(np.interp(1000.0, f, magnitude_db), 0.0, atol=1e-10)
