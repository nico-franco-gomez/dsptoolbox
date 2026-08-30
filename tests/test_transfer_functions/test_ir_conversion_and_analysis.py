"""
Tests for IR<->filter conversion, IR latency/trimming, and dirac-combination
helpers.
"""

import os
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp


def _seeded(seed: int, func, *args, **kwargs):
    """Call `func` with the global `numpy.random` state pinned to `seed`,
    then restore whatever state it had before. `test_ir_to_filter` builds
    a minimum/linear-phase reconstruction from a short noise-derived IR,
    which can be numerically marginal for some noise realizations; pinning
    the seed here makes the class-level `audio_multi` fixture reproducible
    regardless of how much of the shared global RNG state prior tests in a
    full-suite run have already consumed.

    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return func(*args, **kwargs)
    finally:
        np.random.set_state(state)


class TestTransferFunctionsModule:
    fs = 5_000
    audio_multi = _seeded(0, dsp.generators.noise, 2.0, 5_000, number_of_channels=3)

    def test_ir_to_filter(self):
        s = self.audio_multi.time_data[:200, 0]
        s = dsp.ImpulseResponse(None, s, self.fs)
        f = dsp.transfer_functions.ir_to_filter(s, channel=0)
        b, _ = f.get_coefficients(dsp.FilterCoefficientsType.Ba)
        assert np.all(b == s.time_data[:, 0])
        assert f.sampling_rate_hz == s.sampling_rate_hz

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

    def test_min_phase_ir(self):
        s = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")
        )
        s = dsp.transfer_functions.min_phase_ir(s)
        with pytest.raises(AssertionError):
            s = dsp.transfer_functions.min_phase_ir(s, padding_factor=0)
        with pytest.raises(AssertionError):
            s = dsp.transfer_functions.min_phase_ir(s, alpha=0.0)
        s = dsp.transfer_functions.min_phase_ir(s, alpha=1.0 - 1e-6)

    def test_combine_ir(self):
        s = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")
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
        ir = ir.fractional_delay(delay_seconds)
        peak_min_phase = dsp.transfer_functions.find_ir_latency(ir).squeeze()
        peak = dsp.transfer_functions.find_ir_latency(ir, False)

        assert np.isclose(delay_samples, peak_min_phase, atol=0.4)
        assert np.isclose(delay_samples, peak, atol=0.3)

        # Phase inversion should not change the result
        ir.time_data = ir.time_data * -1.0
        assert np.isclose(peak, dsp.transfer_functions.find_ir_latency(ir, False))
        assert np.isclose(peak_min_phase, dsp.transfer_functions.find_ir_latency(ir))

        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")
        )
        assert dsp.transfer_functions.find_ir_latency(ir) > 0

    def test_window_frequency_dependent(self):
        s = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")
        )
        sp = dsp.transfer_functions.window_frequency_dependent(s, 10)

        fig, ax = s.plot_magnitude(normalize=dsp.MagnitudeNormalization.NoNormalization)
        ax.plot(sp.frequency_vector_hz, 20 * np.log10(np.abs(sp.spectral_data)))

    def test_trim_rir(self):
        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")
        )
        dsp.transfer_functions.trim_ir(ir, 0)
        dsp.transfer_functions.trim_ir(ir, None)
        # Start offset way longer than the RIR should be clipped to 0
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

    def test_trim_ir_recovers_known_offset(self):
        """A dirac at a known sample offset, once trimmed, must still have
        its peak at exactly `original_peak_index - start` in the trimmed
        signal (exact for this integer-sample case, regardless of the
        internal offset-rounding convention `trim_ir` uses for `start`).

        """
        fs = 8_000
        delay_samples = 500
        ir = dsp.generators.dirac(
            length_samples=2_000, delay_samples=delay_samples, sampling_rate_hz=fs
        )
        trimmed, start, _stop = dsp.transfer_functions.trim_ir(
            ir, channel=0, start_offset_s=0.01
        )
        peak = np.argmax(np.abs(trimmed.time_data[:, 0]))
        assert peak == delay_samples - start
        np.testing.assert_allclose(trimmed.time_data[peak, 0], 1.0, atol=1e-12)
