"""
Tests for minimum/linear-phase reconstruction and group-delay functions.
"""

import os
from os.path import join

import numpy as np
import pytest
import scipy.signal

import dsptoolbox as dsp


class TestTransferFunctionsModule:
    y_st = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "chirp_stereo.wav")
    )
    x = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "chirp.wav")
    )
    fs = 5_000

    def test_min_phase_from_mag(self):
        self.y_st = self.y_st.set_spectrum_parameters(method=dsp.SpectrumMethod.FFT)
        spec = dsp.Spectrum.from_signal(self.y_st)
        dsp.transfer_functions.min_phase_from_mag(spec, self.y_st.sampling_rate_hz)
        dsp.transfer_functions.min_phase_from_mag(
            spec, self.y_st.sampling_rate_hz, ir_length_samples=self.fs
        )

    def test_lin_phase_from_mag(self):
        self.y_st = self.y_st.set_spectrum_parameters(method=dsp.SpectrumMethod.FFT)
        spec = dsp.Spectrum.from_signal(self.y_st)
        dsp.transfer_functions.lin_phase_from_mag(
            spec,
            self.y_st.sampling_rate_hz,
            group_delay_ms=None,
        )
        dsp.transfer_functions.lin_phase_from_mag(
            spec,
            self.y_st.sampling_rate_hz,
            group_delay_ms=500.0,
            check_causality=False,
        )

        with pytest.raises(AssertionError):
            dsp.transfer_functions.lin_phase_from_mag(
                spec,
                self.y_st.sampling_rate_hz,
                group_delay_ms=1.0,
                check_causality=True,
            )
        dsp.transfer_functions.lin_phase_from_mag(
            spec,
            self.y_st.sampling_rate_hz,
            group_delay_ms=None,
            minimum_group_delay_factor=10.0,
        )

    def test_group_delay(self):
        ir = dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            apply_regularization=True,
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        ir, _ = dsp.transfer_functions.window_ir(
            ir, total_length_samples=2**12, at_start=True
        )
        dsp.transfer_functions.group_delay(ir, analytic_computation=True)
        dsp.transfer_functions.group_delay(ir, analytic_computation=False)

        dsp.transfer_functions.group_delay(ir, analytic_computation=True, smoothing=4)
        dsp.transfer_functions.group_delay(
            ir,
            analytic_computation=False,
            smoothing=4,
            remove_ir_latency=True,
        )

        # Single-channel plausibility check
        dsp.transfer_functions.group_delay(ir.get_channels(0))

    def test_minimum_phase(self):
        ir = dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            apply_regularization=True,
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        ir, _ = dsp.transfer_functions.window_ir(ir, 2**12, at_start=True)
        f, min_phases = dsp.transfer_functions.minimum_phase(ir)
        assert len(f) == len(min_phases)

        f, min_phases = dsp.transfer_functions.minimum_phase(ir.pad_trim(len(ir) + 1))
        assert len(f) == len(min_phases)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.minimum_phase(s1)
        dsp.transfer_functions.minimum_phase(ir.get_channels(0))

    def test_minimum_group_delay(self):
        ir = dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            apply_regularization=True,
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        ir, _ = dsp.transfer_functions.window_ir(ir, 2**12, at_start=True)
        dsp.transfer_functions.minimum_group_delay(ir)
        dsp.transfer_functions.minimum_group_delay(ir, smoothing=3)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.minimum_group_delay(s1)
        dsp.transfer_functions.minimum_group_delay(ir.get_channels(0))

    def test_excess_group_delay(self):
        ir = dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            apply_regularization=True,
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        ir, _ = dsp.transfer_functions.window_ir(ir, 2**12, at_start=True)
        dsp.transfer_functions.excess_group_delay(ir)
        dsp.transfer_functions.excess_group_delay(
            ir, smoothing=3, remove_ir_latency=True
        )
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.excess_group_delay(s1)
        dsp.transfer_functions.excess_group_delay(ir.get_channels(0))

    def test_group_delay_of_delayed_dirac_matches_delay(self):
        """A pure integer-sample-delayed dirac has exactly linear phase, so
        its group delay is constant and equal to the delay itself (in
        seconds), for both the analytic and numerical computation methods.

        """
        fs = 8_000
        delay_samples = 37
        ir = dsp.generators.dirac(
            length_samples=2_048, delay_samples=delay_samples, sampling_rate_hz=fs
        )

        _, gd_analytic = dsp.transfer_functions.group_delay(
            ir, analytic_computation=True
        )
        _, gd_numeric = dsp.transfer_functions.group_delay(
            ir, analytic_computation=False
        )

        expected_s = delay_samples / fs
        np.testing.assert_allclose(gd_analytic[:, 0], expected_s, atol=1e-12)
        np.testing.assert_allclose(gd_numeric[:, 0], expected_s, atol=1e-9)

    def test_excess_group_delay_is_small_for_minimum_phase_system(self):
        """A true type-I linear-phase FIR (odd length, symmetric taps, built
        directly with `scipy.signal.firwin` as an independent reference) has
        an exactly constant group delay of `(N-1)/2` samples in its
        passband -- this part is an exact check via `group_delay`.
        Converting the same magnitude response to its minimum-phase
        equivalent (`min_phase_ir`) should collapse nearly all of that delay
        away: its excess group delay in the same passband should be a small
        fraction of the original linear-phase delay (plausibility; the
        cepstral minimum-phase estimate is not exact, so this checks orders
        of magnitude, not near-zero).

        """
        fs = 8_000
        order = 200
        taps = scipy.signal.firwin(order + 1, 1000, fs=fs)
        linear_phase_ir = dsp.ImpulseResponse(None, taps[:, None], fs)

        f, gd = dsp.transfer_functions.group_delay(linear_phase_ir)
        passband = (f > 500) & (f < 1500)
        expected_delay_samples = order / 2
        np.testing.assert_allclose(
            gd[passband, 0] * fs, expected_delay_samples, atol=1e-6
        )

        min_phase_ir_sig = dsp.transfer_functions.min_phase_ir(
            linear_phase_ir, padding_factor=16
        )
        f2, ex_gd = dsp.transfer_functions.excess_group_delay(min_phase_ir_sig)
        passband2 = (f2 > 500) & (f2 < 1500)
        min_phase_excess_samples = np.median(np.abs(ex_gd[passband2, 0])) * fs

        assert min_phase_excess_samples < expected_delay_samples / 10

    def test_lin_phase_from_mag_produces_delayed_dirac_for_flat_magnitude(self):
        """A flat magnitude spectrum (that of a dirac impulse) combined with
        an exactly linear phase must reconstruct to a pure delayed dirac at
        `round(group_delay_ms/1000 * fs)` samples (round-trip identity;
        direct phase-slope fitting is unreliable here because this
        function's output length is always exactly twice the delay, so the
        phase step between adjacent bins sits exactly at pi radians -- a
        degenerate case for `numpy.unwrap`).

        """
        fs = 8_000
        n = 512
        td = np.zeros((n, 1))
        td[0, 0] = 1.0
        ir = dsp.ImpulseResponse(None, td, fs)
        spec = dsp.Spectrum.from_signal(ir)

        for group_delay_ms in (3.0, 7.5, 12.25):
            lp_ir = dsp.transfer_functions.lin_phase_from_mag(
                spec, fs, group_delay_ms=group_delay_ms, check_causality=False
            )
            expected_delay = round(group_delay_ms / 1000 * fs)
            peak = np.argmax(np.abs(lp_ir.time_data[:, 0]))
            assert peak == expected_delay
            np.testing.assert_allclose(lp_ir.time_data[peak, 0], 1.0, atol=1e-10)
            other = np.delete(lp_ir.time_data[:, 0], peak)
            np.testing.assert_allclose(other, 0.0, atol=1e-10)

    def test_min_phase_from_mag_preserves_magnitude_shape(self):
        """The defining property of `min_phase_from_mag` is that it returns
        a signal with the same magnitude spectrum as the input. Because the
        resulting `ImpulseResponse` may get auto-normalized to 0 dBFS
        (`constrain_amplitude=True` default), the comparison is made up to
        a constant scale factor rather than requiring bit-identical values.

        """
        fs = 8_000
        n = 512
        rng = np.random.default_rng(1)
        td = rng.normal(0, 0.1, (n, 1))
        ir = dsp.ImpulseResponse(None, td, fs)
        spec = dsp.Spectrum.from_signal(ir)

        mp_ir = dsp.transfer_functions.min_phase_from_mag(spec, fs, ir_length_samples=n)
        mag_orig = np.abs(np.fft.rfft(td[:, 0]))
        mag_new = np.abs(np.fft.rfft(mp_ir.time_data[:, 0], n=len(mp_ir)))

        ratio = mag_new / mag_orig
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-8)
