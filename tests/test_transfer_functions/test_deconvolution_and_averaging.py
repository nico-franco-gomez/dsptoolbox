"""
Tests for `spectral_deconvolve`, `compute_transfer_function`, and
`average_irs`.
"""

import os
from os.path import join

import numpy as np

import dsptoolbox as dsp


class TestTransferFunctionsModule:
    y_m = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "chirp_mono.wav")
    )
    y_st = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "chirp_stereo.wav")
    )
    x = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "chirp.wav")
    )

    def test_deconvolve(self):
        # Regularized
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            apply_regularization=True,
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            apply_regularization=True,
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            apply_regularization=True,
            start_stop_hz=[30, 15e3],
            threshold_db=None,
            padding=False,
            keep_original_length=False,
        )
        # Standard (no regularization)
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            apply_regularization=False,
            start_stop_hz=None,
            threshold_db=None,
            padding=False,
            keep_original_length=False,
        )
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            apply_regularization=False,
            start_stop_hz=None,
            threshold_db=None,
            padding=True,
            keep_original_length=False,
        )
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            apply_regularization=False,
            start_stop_hz=None,
            threshold_db=None,
            padding=True,
            keep_original_length=True,
        )

    def test_spectral_deconvolve_self_deconvolution_is_unit_impulse(self):
        """Deconvolving a signal with itself (`H = X/X`) should produce an
        exact unit impulse at sample 0 when regularization is disabled.

        """
        fs = 8_000
        n = 4_096
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, n)
        sig = dsp.Signal(None, x[:, None], fs)

        h = dsp.transfer_functions.spectral_deconvolve(
            sig,
            sig,
            apply_regularization=False,
            start_stop_hz=None,
            threshold_db=None,
            padding=False,
            keep_original_length=False,
        )
        expected = np.zeros(n)
        expected[0] = 1.0
        np.testing.assert_allclose(h.time_data[:, 0], expected, atol=1e-9)

    def test_compute_transfer_function(self):
        # Multi-channel
        dsp.transfer_functions.compute_transfer_function(
            self.y_st,
            self.x,
            window_length_samples=1024,
            mode=dsp.transfer_functions.TransferFunctionType.H1,
        )
        dsp.transfer_functions.compute_transfer_function(
            self.y_st,
            self.x,
            window_length_samples=1024,
            mode=dsp.transfer_functions.TransferFunctionType.H3,
        )
        # Single-channel with other windows
        h = dsp.transfer_functions.compute_transfer_function(
            self.y_m,
            self.x,
            window_length_samples=1024,
            mode=dsp.transfer_functions.TransferFunctionType.H2,
        )
        # Coherence should have been saved alongside the transfer function
        h.plot_coherence()

    def test_average_irs(self):
        h = dsp.transfer_functions.spectral_deconvolve(self.y_st, self.x)
        dsp.transfer_functions.average_irs(h, normalize_energy=True)
        dsp.transfer_functions.average_irs(h, normalize_energy=False)
        dsp.transfer_functions.average_irs(h, time_average=False)

    def test_average_irs_identical_channels_returns_same_ir(self):
        """Averaging N identical channels (no noise, `time_average=True`)
        must return exactly that same IR (round-trip identity).

        """
        fs = 8_000
        n = 512
        clean = np.zeros((n, 3))
        clean[50, :] = 1.0
        clean[51, :] = 0.5

        ir = dsp.ImpulseResponse(None, clean, fs)
        avg = dsp.transfer_functions.average_irs(
            ir, time_average=True, normalize_energy=False
        )
        np.testing.assert_allclose(avg.time_data[:, 0], clean[:, 0], atol=1e-12)

    def test_average_irs_noise_floor_shrinks_with_sqrt_n(self):
        """Averaging N repeated measurements of the same clean IR, each with
        an independent noise realization, should shrink the noise floor's
        standard deviation roughly like `noise_std / sqrt(N)` (plausibility;
        checked as an order-of-magnitude/ratio comparison, not exact).

        """
        fs = 8_000
        n = 512
        clean = np.zeros(n)
        clean[50] = 1.0
        clean[51] = 0.5
        noise_std = 0.01
        rng = np.random.default_rng(5)

        stds = {}
        for num_repetitions in (4, 64):
            td = np.zeros((n, num_repetitions))
            for i in range(num_repetitions):
                td[:, i] = clean + rng.normal(0, noise_std, n)
            ir = dsp.ImpulseResponse(None, td, fs)
            avg = dsp.transfer_functions.average_irs(
                ir, time_average=True, normalize_energy=False
            )
            # Region with no clean-signal content: pure averaged noise floor.
            stds[num_repetitions] = np.std(avg.time_data[200:, 0])

        expected_ratio = np.sqrt(64 / 4)  # 4x more repetitions -> sqrt(16)=4x
        measured_ratio = stds[4] / stds[64]
        np.testing.assert_allclose(measured_ratio, expected_ratio, rtol=0.3)

    def test_spectral_deconvolve_regularizes_each_channel(self):
        """The automatic regularization band must be found per channel."""
        fs = 44100
        low = dsp.generators.chirp(fs, range_hz=[100, 2000], length_seconds=0.5)
        high = dsp.generators.chirp(fs, range_hz=[3000, 15000], length_seconds=0.5)
        excitation = low.append_signals([high])

        both = dsp.transfer_functions.spectral_deconvolve(excitation, excitation)
        for channel in range(2):
            single = dsp.transfer_functions.spectral_deconvolve(
                excitation.get_channels(channel), excitation.get_channels(channel)
            )
            np.testing.assert_allclose(
                both.time_data[:, channel], single.time_data[:, 0], atol=1e-9
            )

    def test_average_irs_does_not_modify_input(self):
        rng = np.random.default_rng(0)
        ir = dsp.ImpulseResponse.from_time_data(
            np.stack([rng.normal(0, 0.1, 512), rng.normal(0, 0.5, 512)], axis=1),
            48000,
        )
        before = ir.time_data.copy()
        dsp.transfer_functions.average_irs(ir)
        np.testing.assert_array_equal(before, ir.time_data)

    def test_average_irs_energy_normalization_equalizes_channel_energy(self):
        """`normalize_energy` must apply the amplitude factor
        `sqrt(E_0 / E_i)`, not the energy ratio `E_i / E_0`.

        """
        fs = 8_000
        n = 512
        td = np.zeros((n, 3))
        td[50, :] = [1.0, 0.2, 3.0]
        ir = dsp.ImpulseResponse(None, td, fs, constrain_amplitude=False)

        energies = np.sum(td**2.0, axis=0)
        avg = dsp.transfer_functions.average_irs(
            ir, time_average=True, normalize_energy=True
        )
        np.testing.assert_allclose(np.sum(avg.time_data**2.0), energies[0], rtol=1e-9)

    def test_average_irs_energy_normalization_applies_in_both_branches(self):
        fs = 8_000
        n = 512
        rng = np.random.default_rng(3)
        td = rng.normal(0, 0.1, (n, 2)) * np.array([1.0, 4.0])
        ir = dsp.ImpulseResponse(None, td, fs, constrain_amplitude=False)

        for time_average in (True, False):
            on = dsp.transfer_functions.average_irs(
                ir, time_average=time_average, normalize_energy=True
            )
            off = dsp.transfer_functions.average_irs(
                ir, time_average=time_average, normalize_energy=False
            )
            assert not np.allclose(on.time_data, off.time_data)
