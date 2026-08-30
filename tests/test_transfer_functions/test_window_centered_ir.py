"""
Tests for `window_centered_ir`.
"""

import os
from os.path import join

import numpy as np

import dsptoolbox as dsp


class TestTransferFunctionsModule:
    y_m = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "chirp_mono.wav")
    )
    x = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "chirp.wav")
    )
    fs = 5_000

    def test_window_centered_ir(self):
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)

        # Even length: shorter, then longer
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) - 10, window_type=dsp.Window.Hann
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) + 10, window_type=dsp.Window.Hann
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # Window with extra parameters
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h,
            len(h),
            window_type=dsp.Window.Gaussian.with_extra_parameter(5000),
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

        # Odd length: shorter, then longer
        h.time_data = h.time_data[:-1]
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) - 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) + 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        h_, _ = dsp.transfer_functions.window_centered_ir(
            h,
            len(h),
            window_type=dsp.Window.Gaussian.with_extra_parameter(5000),
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

        # Impulse on the second half, odd length
        h.time_data = h.time_data[::-1]
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) - 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) + 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        h_, _ = dsp.transfer_functions.window_centered_ir(
            h,
            len(h),
            window_type=dsp.Window.Gaussian.with_extra_parameter(5000),
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

        # Impulse on the second half, even length
        h.time_data = h.time_data[:-1]
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) - 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) + 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        h_, _ = dsp.transfer_functions.window_centered_ir(
            h,
            len(h),
            window_type=dsp.Window.Gaussian.with_extra_parameter(5000),
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

        # Impulse in the middle, no changing lengths, even length
        d = dsp.generators.dirac(
            length_samples=1024, delay_samples=512, sampling_rate_hz=self.fs
        )
        d2, _ = dsp.transfer_functions.window_centered_ir(d, len(d))
        assert (
            np.argmax(d.time_data[:, 0]) == np.argmax(d2.window[:, 0])
            and len(d) == len(d2)
            and np.all(np.isclose(d.time_data, d2.time_data))
        )

        # Impulse in the middle, no changing lengths, odd length
        d = dsp.generators.dirac(
            length_samples=1025, delay_samples=513, sampling_rate_hz=self.fs
        )
        d2, _ = dsp.transfer_functions.window_centered_ir(d, len(d))
        assert (
            np.argmax(d.time_data[:, 0]) == np.argmax(d2.window[:, 0])
            and len(d) == len(d2)
            and np.all(np.isclose(d.time_data, d2.time_data))
        )

    def test_window_centered_ir_matches_elementwise_product(self):
        """Same elementwise-product invariant as `window_ir`, but for
        `window_centered_ir`: `out[n] == original[start + n] * window[n]`,
        checked here for a peak in the first half (no internal flip branch).

        """
        fs = 8_000
        n = 300
        td = np.zeros((n, 1))
        td[100, 0] = 1.0
        ir = dsp.ImpulseResponse(None, td, fs)

        total_length = 200  # peak_ind (100) <= half_length (100): no flip
        result, start_pos = dsp.transfer_functions.window_centered_ir(
            ir, total_length, window_type=dsp.Window.Hann
        )
        sp = int(start_pos[0])
        placed = td[sp : sp + total_length, 0]

        np.testing.assert_allclose(
            result.time_data[:, 0], placed * result.window[:, 0], atol=1e-12
        )
