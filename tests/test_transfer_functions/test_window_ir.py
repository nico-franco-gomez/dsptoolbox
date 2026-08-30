"""
Tests for `window_ir_tukey` and `window_ir`.
"""

import os
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp


class TestTransferFunctionsModule:
    y_m = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "chirp_mono.wav")
    )
    x = dsp.Signal(
        join(os.path.dirname(__file__), "..", "..", "example_data", "chirp.wav")
    )
    fs = 5_000

    def test_window_ir_tukey(self):
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)
        h.time_data = np.roll(h.time_data, 256 - np.argmax(np.abs(h.time_data)), axis=0)
        h.time_data = np.repeat(h.time_data, 2, axis=1)

        delay_second_channel = 10
        h = h.delay(delay_second_channel, [1], True)
        hh = dsp.transfer_functions.window_ir_tukey(
            h, 210 / h.sampling_rate_hz, 10 / h.sampling_rate_hz
        )
        assert (
            np.ediff1d(np.argmax(np.abs(hh.time_data), axis=0))[0]
            == delay_second_channel
        )
        assert hasattr(hh, "window")

        dsp.transfer_functions.window_ir_tukey(h, 210 / h.sampling_rate_hz, None)
        dsp.transfer_functions.window_ir_tukey(h, None, 10 / h.sampling_rate_hz)
        with pytest.raises(AssertionError):
            dsp.transfer_functions.window_ir_tukey(h, None, None)
        with pytest.raises(AssertionError):
            dsp.transfer_functions.window_ir_tukey(
                h, h.length_seconds / 2, h.length_seconds * 3 / 2
            )
        with pytest.raises(AssertionError):
            dsp.transfer_functions.window_ir_tukey(
                h, h.length_seconds / 10, None, dsp.Window.Tukey
            )

    def test_window_ir_tukey_extended(self):
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)
        h.time_data = np.roll(h.time_data, 256 - np.argmax(np.abs(h.time_data)), axis=0)

        # Different window types
        window_types = [
            dsp.Window.Hann,
            dsp.Window.Hamming,
            dsp.Window.Blackman,
        ]
        for window_type in window_types:
            result = dsp.transfer_functions.window_ir_tukey(
                h, 50 / h.sampling_rate_hz, 50 / h.sampling_rate_hz, window_type
            )
            assert isinstance(result, dsp.ImpulseResponse)
            assert result.length_samples == h.length_samples
            assert hasattr(result, "window")
            assert np.all(result.window >= -1e-10) and np.all(
                result.window <= 1 + 1e-10
            )

        # Kaiser window with an extra shape parameter
        result = dsp.transfer_functions.window_ir_tukey(
            h,
            50 / h.sampling_rate_hz,
            50 / h.sampling_rate_hz,
            dsp.Window.Kaiser.with_extra_parameter(10),
        )
        assert isinstance(result, dsp.ImpulseResponse)
        assert np.all(result.window >= -1e-10) and np.all(result.window <= 1 + 1e-10)

        # Mono channel IR
        h_mono = h.get_channels(0)
        result_mono = dsp.transfer_functions.window_ir_tukey(
            h_mono, 50 / h.sampling_rate_hz, 50 / h.sampling_rate_hz
        )
        assert result_mono.number_of_channels == 1
        assert result_mono.length_samples == h_mono.length_samples
        assert np.all(result_mono.window >= -1e-10) and np.all(
            result_mono.window <= 1 + 1e-10
        )

        # Multi-channel IR
        h_multi = h.copy()
        h_multi.time_data = np.repeat(h_multi.time_data, 4, axis=1)
        result_multi = dsp.transfer_functions.window_ir_tukey(
            h_multi, 100 / h.sampling_rate_hz, 100 / h.sampling_rate_hz
        )
        assert result_multi.number_of_channels == 4
        assert result_multi.length_samples == h_multi.length_samples
        for ch in range(result_multi.number_of_channels):
            assert np.all(result_multi.window[:, ch] >= -1e-10)
            assert np.all(result_multi.window[:, ch] <= 1 + 1e-10)

        # Constant region should stay close to 1
        result = dsp.transfer_functions.window_ir_tukey(
            h, 100 / h.sampling_rate_hz, 100 / h.sampling_rate_hz
        )
        left_flank_samples = int(100 / h.sampling_rate_hz * h.sampling_rate_hz)
        right_flank_samples = int(100 / h.sampling_rate_hz * h.sampling_rate_hz)
        constant_region = result.window[left_flank_samples:-right_flank_samples, :]
        if len(constant_region) > 0:
            assert np.all(np.isclose(constant_region, 1.0))

        # Asymmetric flanks
        left_flank_s = 100 / h.sampling_rate_hz
        right_flank_s = 200 / h.sampling_rate_hz
        result = dsp.transfer_functions.window_ir_tukey(h, left_flank_s, right_flank_s)
        assert result.length_samples == h.length_samples
        assert hasattr(result, "window")

        # Very small flanks
        result = dsp.transfer_functions.window_ir_tukey(
            h, 1 / h.sampling_rate_hz, 1 / h.sampling_rate_hz
        )
        assert result.length_samples == h.length_samples
        assert isinstance(result, dsp.ImpulseResponse)

        # Only left flank: right side (constant region) stays unchanged
        result = dsp.transfer_functions.window_ir_tukey(
            h,
            100 / h.sampling_rate_hz,
            None,
            dsp.Window.Hamming,
        )
        assert result.length_samples == h.length_samples
        assert np.all(result.window[-100:, 0] == 1.0)

        # Only right flank: left side (constant region) stays unchanged
        result = dsp.transfer_functions.window_ir_tukey(
            h,
            None,
            100 / h.sampling_rate_hz,
            dsp.Window.Blackman,
        )
        assert result.length_samples == h.length_samples
        assert np.all(result.window[:100, 0] == 1.0)

        # Windowing must never add energy
        result = dsp.transfer_functions.window_ir_tukey(
            h, 100 / h.sampling_rate_hz, 100 / h.sampling_rate_hz
        )
        original_energy = np.sum(h.time_data**2)
        windowed_energy = np.sum(result.time_data**2)
        assert windowed_energy <= original_energy

        # Symmetric flank durations should produce (anti-)symmetric slopes
        flank_duration = 75 / h.sampling_rate_hz
        result = dsp.transfer_functions.window_ir_tukey(
            h, flank_duration, flank_duration, dsp.Window.Hann
        )
        left_flank_samples = int(flank_duration * h.sampling_rate_hz)
        right_flank_samples = int(flank_duration * h.sampling_rate_hz)
        assert left_flank_samples == right_flank_samples
        left_flank = result.window[:left_flank_samples, 0]
        right_flank = result.window[-right_flank_samples:, 0]
        left_slope = np.diff(left_flank)
        right_slope = np.diff(right_flank)[::-1]
        correlation = np.corrcoef(left_slope, right_slope)[0, 1]
        assert abs(correlation) > 0.95

    def test_window_ir(self):
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)
        h.time_data = np.roll(h.time_data, 256 - np.argmax(np.abs(h.time_data)), axis=0)
        h = h.pad_trim(2**13)

        dsp.transfer_functions.window_ir(h, 2**11, at_start=True)
        dsp.transfer_functions.window_ir(h, 2**11, at_start=False)
        dsp.transfer_functions.window_ir(h, 2**15, at_start=True)
        # Window with extra parameters
        dsp.transfer_functions.window_ir(
            h,
            2**12,
            window_type=dsp.Window.Kaiser.with_extra_parameter(10),
            at_start=True,
        )
        dsp.transfer_functions.window_ir(
            h,
            2**12,
            adaptive=False,
            window_type=dsp.Window.Kaiser.with_extra_parameter(10),
            at_start=True,
        )
        dsp.transfer_functions.window_ir(
            h,
            2**12,
            adaptive=False,
            window_type=dsp.Window.Kaiser.with_extra_parameter(10),
            at_start=True,
            offset_samples=200,
        )
        dsp.transfer_functions.window_ir(
            h,
            2**12,
            adaptive=False,
            window_type=[dsp.Window.Hann, dsp.Window.Hamming],
            at_start=False,
            offset_samples=200,
        )
        dsp.transfer_functions.window_ir(
            h,
            2**12,
            adaptive=True,
            window_type=[
                dsp.Window.Hann,
                dsp.Window.Kaiser.with_extra_parameter(10),
            ],
            at_start=False,
            offset_samples=200,
            left_to_right_flank_length_ratio=0.5,
        )
        dsp.transfer_functions.window_ir(
            h,
            2**15,
            adaptive=True,
            window_type=[
                dsp.Window.Hann,
                dsp.Window.Kaiser.with_extra_parameter(10),
            ],
            at_start=False,
            offset_samples=200,
            left_to_right_flank_length_ratio=0.5,
        )

    def test_window_ir_logic_paths(self):
        total_length_samples = 1024
        constant_percentage = 0.75

        # Short IR, impulse early: requires left padding
        ir_short_early = dsp.ImpulseResponse(None, np.zeros((512, 2)), self.fs)
        ir_short_early.time_data[50, :] = 1.0
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_short_early,
            total_length_samples,
            adaptive=False,
            constant_percentage=constant_percentage,
            window_type=dsp.Window.Hann,
            at_start=True,
            offset_samples=0,
            left_to_right_flank_length_ratio=1.0,
        )
        assert len(result) == total_length_samples
        assert result.number_of_channels == 2
        assert hasattr(result, "window")
        assert isinstance(start_pos, np.ndarray)
        assert start_pos.dtype in [np.int32, np.int64, int]
        assert len(start_pos) == result.number_of_channels
        assert np.all(start_pos >= 0), "Start positions should be non-negative"
        assert np.all(start_pos < total_length_samples), (
            "Start positions should be within bounds"
        )
        for ch in range(result.number_of_channels):
            impulse_pos_in_result = np.argmax(np.abs(result.time_data[:, ch]))
            assert impulse_pos_in_result > 0, (
                "Impulse should have been placed with some padding"
            )
            assert result.time_data[impulse_pos_in_result, ch] > 0, (
                "Peak should be positive"
            )

        # IR with impulse in the middle, non-adaptive
        ir_mid = dsp.ImpulseResponse(None, np.zeros((512, 1)), self.fs)
        ir_mid.time_data[256, 0] = 1.0
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=False,
            offset_samples=50,
        )
        assert len(result) == total_length_samples
        assert isinstance(start_pos, np.ndarray)
        assert len(start_pos) == result.number_of_channels
        assert np.all(start_pos >= 0)
        impulse_pos = np.argmax(np.abs(result.time_data[:, 0]))
        assert impulse_pos > 0

        # Very short IR: requires right padding
        ir_very_short = dsp.ImpulseResponse(None, np.zeros((256, 1)), self.fs)
        ir_very_short.time_data[128, 0] = 1.0
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_very_short,
            total_length_samples,
            adaptive=False,
        )
        assert len(result) == total_length_samples
        assert len(start_pos) == 1
        assert start_pos[0] >= 0
        impulse_pos = np.argmax(np.abs(result.time_data[:, 0]))
        assert impulse_pos > 0

        # Long IR: requires trimming
        ir_long = dsp.ImpulseResponse(None, np.zeros((2048, 1)), self.fs)
        ir_long.time_data[512, 0] = 1.0
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_long,
            total_length_samples,
            adaptive=False,
        )
        assert len(result) == total_length_samples
        assert len(start_pos) == 1
        assert start_pos[0] >= 0
        assert start_pos[0] < total_length_samples
        impulse_pos = np.argmax(np.abs(result.time_data[:, 0]))
        assert impulse_pos > 0

        # Non-adaptive with a different flank ratio
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=False,
            left_to_right_flank_length_ratio=0.5,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)

        # Non-adaptive with at_start=False
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=False,
            at_start=False,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)

        # Non-adaptive with a different constant_percentage
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=False,
            constant_percentage=0.5,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)

        # Adaptive, short IR: left flank adaptation
        ir_early_adapt = dsp.ImpulseResponse(None, np.zeros((512, 1)), self.fs)
        ir_early_adapt.time_data[20, 0] = 1.0
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_early_adapt,
            total_length_samples,
            adaptive=True,
            offset_samples=50,
        )
        assert len(result) == total_length_samples
        assert len(start_pos) == 1
        assert start_pos[0] >= 0
        assert start_pos[0] < total_length_samples
        impulse_pos = np.argmax(np.abs(result.time_data[:, 0]))
        assert impulse_pos > 0

        # Adaptive, signal longer than target: trim branch
        ir_long_adapt = dsp.ImpulseResponse(None, np.zeros((2048, 1)), self.fs)
        ir_long_adapt.time_data[512, 0] = 1.0
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_long_adapt,
            total_length_samples,
            adaptive=True,
        )
        assert len(result) == total_length_samples
        assert len(start_pos) == 1
        assert start_pos[0] >= 0
        assert start_pos[0] < total_length_samples
        impulse_pos = np.argmax(np.abs(result.time_data[:, 0]))
        assert impulse_pos > 0

        # Adaptive, signal shorter than target: padding branch
        ir_short_adapt = dsp.ImpulseResponse(None, np.zeros((256, 1)), self.fs)
        ir_short_adapt.time_data[128, 0] = 1.0
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_short_adapt,
            total_length_samples,
            adaptive=True,
        )
        assert len(result) == total_length_samples
        assert len(start_pos) == 1
        assert start_pos[0] >= 0
        assert start_pos[0] < total_length_samples
        impulse_pos = np.argmax(np.abs(result.time_data[:, 0]))
        assert impulse_pos > 0

        # Adaptive with a large offset: right flank adjustment
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=True,
            offset_samples=500,
        )
        assert len(result) == total_length_samples
        assert len(start_pos) == 1
        assert start_pos[0] >= 0
        assert start_pos[0] < total_length_samples

        # Adaptive with a different flank ratio
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=True,
            left_to_right_flank_length_ratio=2.0,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)

        # Adaptive with at_start=False
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=True,
            at_start=False,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)

        # Window property and multichannel checks
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_short_early,
            total_length_samples,
            adaptive=False,
        )
        assert np.all(result.window >= 0)
        assert np.all(result.window <= 1)
        assert len(start_pos) == result.number_of_channels
        assert isinstance(start_pos, np.ndarray)
        assert start_pos.dtype in [np.int32, np.int64, int]
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)
        for ch in range(result.number_of_channels):
            impulse_peak = np.argmax(np.abs(result.time_data[:, ch]))
            assert impulse_peak > 0, f"Impulse should be detected in channel {ch}"

        # List of window types
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=False,
            window_type=[dsp.Window.Hann, dsp.Window.Hamming],
        )
        assert len(result) == total_length_samples
        assert isinstance(start_pos, np.ndarray)
        assert np.all(start_pos >= 0)

    def test_window_ir_matches_elementwise_product(self):
        """`window_ir` returns a `.window` array and `start_positions_samples`
        such that the windowed output equals the elementwise product of the
        window with the original IR sliced (and zero-padded as needed) at
        that start offset: `out[n] == original[start + n] * window[n]`.

        """
        fs = 8_000
        n = 300
        td = np.zeros((n, 1))
        td[100, 0] = 1.0
        ir = dsp.ImpulseResponse(None, td, fs)

        total_length = 512
        result, start_pos = dsp.transfer_functions.window_ir(
            ir, total_length, adaptive=False
        )

        sp = int(start_pos[0])
        original = td[:, 0]
        if sp < 0:
            extended = np.concatenate([np.zeros(-sp), original])
            sp = 0
        else:
            extended = original
        if sp + total_length > len(extended):
            extended = np.concatenate(
                [extended, np.zeros(sp + total_length - len(extended))]
            )
        placed = extended[sp : sp + total_length]

        np.testing.assert_allclose(
            result.time_data[:, 0], placed * result.window[:, 0], atol=1e-12
        )
