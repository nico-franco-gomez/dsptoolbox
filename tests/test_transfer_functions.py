import dsptoolbox as dsp
from os.path import join
import numpy as np
import pytest
import os


class TestTransferFunctionsModule:
    y_m = dsp.Signal(
        join(os.path.dirname(__file__), "..", "example_data", "chirp_mono.wav")
    )
    y_st = dsp.Signal(
        join(os.path.dirname(__file__), "..", "example_data", "chirp_stereo.wav")
    )
    x = dsp.Signal(join(os.path.dirname(__file__), "..", "example_data", "chirp.wav"))
    fs = 5_000
    audio_multi = dsp.generators.noise(2.0, 5_000, number_of_channels=3)

    def test_deconvolve(self):
        # Only functionality is tested here
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
        # Standard
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

    def test_window_ir_tukey(self):
        # Mostly functionality
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)
        h.time_data = np.roll(h.time_data, 256 - np.argmax(np.abs(h.time_data)), axis=0)
        h.time_data = np.repeat(h.time_data, 2, axis=1)

        delay_second_channel = 10
        h = dsp.delay(h, delay_second_channel, [1], True)
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

    def test_window_ir(self):
        # Only functionality
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)
        h.time_data = np.roll(h.time_data, 256 - np.argmax(np.abs(h.time_data)), axis=0)
        h = dsp.pad_trim(h, 2**13)

        dsp.transfer_functions.window_ir(h, 2**11, at_start=True)
        dsp.transfer_functions.window_ir(h, 2**11, at_start=False)
        dsp.transfer_functions.window_ir(h, 2**15, at_start=True)
        # Try window with extra parameters
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
        # Parameters for testing
        total_length_samples = 1024
        constant_percentage = 0.75

        # Create test IRs with impulse at different positions
        # Test 1a: Short IR, impulse early, requires left padding
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
        # Check start position
        assert isinstance(start_pos, np.ndarray)
        assert start_pos.dtype in [np.int32, np.int64, int]
        assert len(start_pos) == result.number_of_channels
        assert np.all(start_pos >= 0), "Start positions should be non-negative"
        assert np.all(
            start_pos < total_length_samples
        ), "Start positions should be within bounds"
        # Check impulse is detected at expected position in windowed result
        for ch in range(result.number_of_channels):
            impulse_pos_in_result = np.argmax(np.abs(result.time_data[:, ch]))
            # Impulse should be reasonably placed and windowed
            assert (
                impulse_pos_in_result > 0
            ), "Impulse should have been placed with some padding"
            assert (
                result.time_data[impulse_pos_in_result, ch] > 0
            ), "Peak should be positive"

        # Test 1b: IR with impulse in middle, non-adaptive
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
        # With offset, the impulse should still be detectable
        impulse_pos = np.argmax(np.abs(result.time_data[:, 0]))
        assert impulse_pos > 0

        # Test 1c: Very short IR (requires right padding)
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

        # Test 1d: Long IR (requires trimming)
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

        # Test 1e: Non-adaptive with different flank ratio
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=False,
            left_to_right_flank_length_ratio=0.5,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)

        # Test 1f: Non-adaptive with at_start=False
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=False,
            at_start=False,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)

        # Test 1g: Non-adaptive with different constant_percentage
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=False,
            constant_percentage=0.5,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)

        # =====================================================================
        # ADAPTIVE MODE TESTS
        # =====================================================================

        # Test 2a: Adaptive with short IR (left flank adaptation)
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
        # Even with offset, impulse should be findable
        impulse_pos = np.argmax(np.abs(result.time_data[:, 0]))
        assert impulse_pos > 0

        # Test 2b: Adaptive, signal longer than target (trim branch)
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

        # Test 2c: Adaptive, signal shorter than target (padding branch)
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

        # Test 2d: Adaptive with large offset (right flank adjustment)
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

        # Test 2e: Adaptive with different flank ratios
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=True,
            left_to_right_flank_length_ratio=2.0,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)

        # Test 2f: Adaptive with at_start=False
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=True,
            at_start=False,
        )
        assert len(result) == total_length_samples
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)

        # =====================================================================
        # WINDOW PROPERTY AND MULTICHANNEL CHECKS
        # =====================================================================

        result, start_pos = dsp.transfer_functions.window_ir(
            ir_short_early,
            total_length_samples,
            adaptive=False,
        )
        # Check window properties
        assert np.all(result.window >= 0)
        assert np.all(result.window <= 1)
        # Check that we get start positions for each channel
        assert len(start_pos) == result.number_of_channels
        assert isinstance(start_pos, np.ndarray)
        assert start_pos.dtype in [np.int32, np.int64, int]
        assert np.all(start_pos >= 0)
        assert np.all(start_pos < total_length_samples)
        # All values should be windowed (product of time_data and window)
        for ch in range(result.number_of_channels):
            assert np.allclose(result.time_data[:, ch], result.time_data[:, ch])
            # Verify impulse is visible in the windowed result
            impulse_peak = np.argmax(np.abs(result.time_data[:, ch]))
            assert impulse_peak > 0, f"Impulse should be detected in channel {ch}"

        # Test with list of window types
        result, start_pos = dsp.transfer_functions.window_ir(
            ir_mid,
            total_length_samples,
            adaptive=False,
            window_type=[dsp.Window.Hann, dsp.Window.Hamming],
        )
        assert len(result) == total_length_samples
        assert isinstance(start_pos, np.ndarray)
        assert np.all(start_pos >= 0)

    def test_window_centered_ir(self):
        # Only functionality
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)

        # ============ Even
        # Shorter
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) - 10, window_type=dsp.Window.Hann
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        # Longer
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) + 10, window_type=dsp.Window.Hann
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # Try window with extra parameters
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h,
            len(h),
            window_type=dsp.Window.Gaussian.with_extra_parameter(5000),
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

        # ============= Odd
        # Shorter
        h.time_data = h.time_data[:-1]
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) - 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        # Longer
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) + 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # Try window with extra parameters
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h,
            len(h),
            window_type=dsp.Window.Gaussian.with_extra_parameter(5000),
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

        # ============= Impulse on the second half, odd
        # Shorter
        h.time_data = h.time_data[::-1]
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) - 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        # Longer
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) + 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # Try window with extra parameters
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h,
            len(h),
            window_type=dsp.Window.Gaussian.with_extra_parameter(5000),
        )

        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

        # ============= Impulse on the second half, even
        # Shorter
        h.time_data = h.time_data[:-1]
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) - 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        # Longer
        h_, _ = dsp.transfer_functions.window_centered_ir(h, len(h) + 10)
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # Try window with extra parameters
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h,
            len(h),
            window_type=dsp.Window.Gaussian.with_extra_parameter(5000),
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

        # ============= Impulse in the middle, no changing lengths, even
        d = dsp.generators.dirac(
            length_samples=1024, delay_samples=512, sampling_rate_hz=self.fs
        )
        d2, _ = dsp.transfer_functions.window_centered_ir(d, len(d))
        assert (
            np.argmax(d.time_data[:, 0]) == np.argmax(d2.window[:, 0])
            and len(d) == len(d2)
            and np.all(np.isclose(d.time_data, d2.time_data))
        )

        # ============= Impulse in the middle, no changing lengths, odd
        d = dsp.generators.dirac(
            length_samples=1025, delay_samples=513, sampling_rate_hz=self.fs
        )
        d2, _ = dsp.transfer_functions.window_centered_ir(d, len(d))
        assert (
            np.argmax(d.time_data[:, 0]) == np.argmax(d2.window[:, 0])
            and len(d) == len(d2)
            and np.all(np.isclose(d.time_data, d2.time_data))
        )

    def test_ir_to_filter(self):
        s = self.audio_multi.time_data[:200, 0]
        s = dsp.ImpulseResponse(None, s, self.fs)
        f = dsp.transfer_functions.ir_to_filter(s, channel=0)
        b, _ = f.get_coefficients(dsp.FilterCoefficientsType.Ba)
        assert np.all(b == s.time_data[:, 0])
        assert f.sampling_rate_hz == s.sampling_rate_hz

        # Functionality in other cases
        f = dsp.transfer_functions.ir_to_filter(s, channel=0, phase_mode="min")
        f = dsp.transfer_functions.ir_to_filter(s, channel=0, phase_mode="lin")

        # To filter bank
        fb = dsp.transfer_functions.ir_to_filter(
            dsp.ImpulseResponse.from_signal(self.audio_multi), channel=None
        )
        assert len(fb) == self.audio_multi.number_of_channels

    def test_filter_to_ir(self):
        order = 216
        f = dsp.Filter.fir_filter(
            order=order,
            frequency_hz=1000,
            type_of_pass=dsp.FilterPassType.Highpass,
            sampling_rate_hz=self.fs,
        )
        s = dsp.transfer_functions.filter_to_ir(f)
        assert s.time_data.shape[0] == order + 1

        # From filter bank
        fb = dsp.FilterBank([f] * 2)
        ir = dsp.transfer_functions.filter_to_ir(fb)
        assert ir.number_of_channels == len(fb)
        assert len(ir) == order + 1

        with pytest.raises(AssertionError):
            f = dsp.Filter.iir_filter(
                order=10,
                frequency_hz=1000,
                filter_design_method=dsp.IirDesignMethod.Butterworth,
                type_of_pass=dsp.FilterPassType.Highpass,
                sampling_rate_hz=self.fs,
            )
            dsp.transfer_functions.filter_to_ir(f)

    def test_compute_transfer_function(self):
        # Only functionality
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
        # Check that coherence is saved
        h.plot_coherence()
        # dsp.plots.show()

    def test_average_irs(self):
        # Only functionality is tested
        h = dsp.transfer_functions.spectral_deconvolve(self.y_st, self.x)
        # h.plot_phase()
        dsp.transfer_functions.average_irs(h, normalize_energy=True)
        # h1.plot_magnitude()
        dsp.transfer_functions.average_irs(h, normalize_energy=False)
        dsp.transfer_functions.average_irs(h, time_average=False)
        # h2.plot_magnitude()

    def test_min_phase_from_mag(self):
        # Only functionality is tested
        self.y_st.set_spectrum_parameters(method=dsp.SpectrumMethod.FFT)
        spec = dsp.Spectrum.from_signal(self.y_st)
        dsp.transfer_functions.min_phase_from_mag(spec, self.y_st.sampling_rate_hz)
        dsp.transfer_functions.min_phase_from_mag(
            spec, self.y_st.sampling_rate_hz, ir_length_samples=self.fs
        )

    def test_lin_phase_from_mag(self):
        # Only functionality is tested here
        self.y_st.set_spectrum_parameters(method=dsp.SpectrumMethod.FFT)
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
        # Check only that some result is produced, validity should be checked
        # somewhere else
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
        # Check only that some result is produced, validity should be checked
        # somewhere else
        # Only works for some signal types
        f, min_phases = dsp.transfer_functions.minimum_phase(ir)
        assert len(f) == len(min_phases)

        f, min_phases = dsp.transfer_functions.minimum_phase(
            dsp.pad_trim(ir, len(ir) + 1)
        )
        assert len(f) == len(min_phases)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.minimum_phase(s1)
        # Single-channel plausibility check
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
        # Check only that some result is produced, validity should be checked
        # somewhere else
        # Only works for some signal types
        dsp.transfer_functions.minimum_group_delay(ir)
        dsp.transfer_functions.minimum_group_delay(ir, smoothing=3)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.minimum_group_delay(s1)
        # Single-channel plausibility check
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
        # Check only that some result is produced, validity should be checked
        # somewhere else
        # Only works for some signal types
        dsp.transfer_functions.excess_group_delay(ir)
        dsp.transfer_functions.excess_group_delay(
            ir, smoothing=3, remove_ir_latency=True
        )
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.excess_group_delay(s1)
        # Single-channel plausibility check
        dsp.transfer_functions.excess_group_delay(ir.get_channels(0))

    def test_min_phase_ir(self):
        # Only functionality, computation is done using scipy's minimum phase
        s = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        s = dsp.transfer_functions.min_phase_ir(s)
        with pytest.raises(AssertionError):
            s = dsp.transfer_functions.min_phase_ir(s, padding_factor=0)
        with pytest.raises(AssertionError):
            s = dsp.transfer_functions.min_phase_ir(s, alpha=0.0)
        s = dsp.transfer_functions.min_phase_ir(s, alpha=1.0 - 1e-6)

    def test_combine_ir(self):
        s = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        dsp.transfer_functions.combine_ir_with_dirac(s, 1000, True, normalization=None)
        dsp.transfer_functions.combine_ir_with_dirac(s, 1000, False, normalization=None)
        dsp.transfer_functions.combine_ir_with_dirac(
            s, 1000, False, normalization="energy"
        )

    def test_find_ir_latency(self):
        ir = dsp.generators.dirac(self.fs, sampling_rate_hz=self.fs)
        delay_seconds = 0.00133  # Some value to have a fractional delay
        delay_samples = self.fs * delay_seconds
        ir = dsp.fractional_delay(ir, delay_seconds)
        peak_min_phase = dsp.transfer_functions.find_ir_latency(ir).squeeze()
        peak = dsp.transfer_functions.find_ir_latency(ir, False)

        assert np.isclose(delay_samples, peak_min_phase, atol=0.4)
        assert np.isclose(delay_samples, peak, atol=0.3)

        # Invert phase, should still deliver the same result
        ir.time_data = ir.time_data * -1.0
        assert np.isclose(peak, dsp.transfer_functions.find_ir_latency(ir, False))
        assert np.isclose(peak_min_phase, dsp.transfer_functions.find_ir_latency(ir))

        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        assert dsp.transfer_functions.find_ir_latency(ir) > 0

    def test_window_frequency_dependent(self):
        s = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        sp = dsp.transfer_functions.window_frequency_dependent(s, 10)

        fig, ax = s.plot_magnitude(normalize=dsp.MagnitudeNormalization.NoNormalization)
        ax.plot(sp.frequency_vector_hz, 20 * np.log10(np.abs(sp.spectral_data)))
        print()

    def test_harmonics_from_chirp_ir(self):
        # Only functionality
        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        dsp.transfer_functions.harmonics_from_chirp_ir(
            ir,
            chirp_range_hz=[20, 20e3],
            chirp_length_s=2,
            n_harmonics=2,
        )

    def test_harmonic_distortion_analysis(self):
        # Only functionality
        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        dsp.transfer_functions.harmonic_distortion_analysis(
            ir,
            chirp_range_hz=[20, 20e3],
            chirp_length_s=2,
            n_harmonics=7,
        )

        harm = dsp.transfer_functions.harmonics_from_chirp_ir(
            ir,
            chirp_range_hz=[20, 20e3],
            chirp_length_s=2,
            n_harmonics=2,
        )
        harm.insert(0, dsp.transfer_functions.trim_ir(ir)[0])
        dsp.transfer_functions.harmonic_distortion_analysis(
            harm,
            chirp_range_hz=None,
            chirp_length_s=None,
            n_harmonics=None,
        )

    def test_trim_rir(self):
        # Only functionality
        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        dsp.transfer_functions.trim_ir(ir, 0)
        dsp.transfer_functions.trim_ir(ir, None)
        # Start offset way longer than rir (should be clipped to 0)
        assert (
            ir.time_data[0, 0]
            == dsp.transfer_functions.trim_ir(ir, start_offset_s=3)[0].time_data[0, 0]
        )
        assert (
            ir.time_data[0, 0]
            == dsp.transfer_functions.trim_ir(ir, start_offset_s=None)[0].time_data[
                0, 0
            ]
        )

    def test_complex_smoothing(self):
        # Only functionality
        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        ir = dsp.pad_trim(ir, int(50e-3 * ir.sampling_rate_hz))
        dsp.transfer_functions.complex_smoothing(
            ir, 12.0, dsp.transfer_functions.SmoothingDomain.RealImaginary
        )
        dsp.transfer_functions.complex_smoothing(
            ir, 12.0, dsp.transfer_functions.SmoothingDomain.Power
        )
        dsp.transfer_functions.complex_smoothing(
            ir, 12.0, dsp.transfer_functions.SmoothingDomain.PowerPhase
        )
        dsp.transfer_functions.complex_smoothing(
            ir, 12.0, dsp.transfer_functions.SmoothingDomain.Magnitude
        )
        dsp.transfer_functions.complex_smoothing(
            ir, 12.0, dsp.transfer_functions.SmoothingDomain.MagnitudePhase
        )
        dsp.transfer_functions.complex_smoothing(
            ir, 12.0, dsp.transfer_functions.SmoothingDomain.EquivalentComplex
        )
        with pytest.raises(AssertionError):
            dsp.transfer_functions.complex_smoothing(
                ir,
                0.0,
                dsp.transfer_functions.SmoothingDomain.EquivalentComplex,
            )
