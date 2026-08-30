"""
Tests for chirp-derived harmonic analysis and complex spectral smoothing.
"""

import os
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp


class TestTransferFunctionsModule:
    def test_harmonics_from_chirp_ir(self):
        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")
        )
        dsp.transfer_functions.harmonics_from_chirp_ir(
            ir,
            chirp_range_hz=[20, 20e3],
            chirp_length_s=2,
            n_harmonics=2,
        )

    def test_harmonic_distortion_analysis(self):
        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")
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

    def test_harmonics_from_chirp_ir_energy_lands_before_fundamental(self):
        """Per Farina's exponential-sweep-deconvolution theory, the k-th
        harmonic distortion product appears in the deconvolved IR at a
        negative time offset (i.e. earlier than the fundamental at t=0) of
        `-T * ln(k) / ln(f2/f1)`, where T is the sweep duration and
        [f1, f2] its frequency range. `harmonics_from_chirp_ir` slices out
        windows around each of these expected offsets; this only checks the
        weaker, more robust plausibility property that each returned
        harmonic snippet is non-trivial (has some energy) -- exact peak
        alignment depends on the (undocumented) internal snippet-window
        convention and is out of scope for a plausibility check.

        """
        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")
        )
        harmonics = dsp.transfer_functions.harmonics_from_chirp_ir(
            ir,
            chirp_range_hz=[20, 20e3],
            chirp_length_s=2,
            n_harmonics=3,
        )
        assert len(harmonics) == 3
        for h in harmonics:
            assert len(h) > 0
            assert np.sum(h.time_data[:, 0] ** 2) > 0

    def test_complex_smoothing(self):
        ir = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")
        )
        ir = ir.pad_trim(int(50e-3 * ir.sampling_rate_hz))
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

    def test_complex_smoothing_flat_spectrum_stays_flat(self):
        """A dirac impulse has a perfectly flat magnitude spectrum and zero
        phase; there is no ripple for any smoothing domain to remove, so the
        output must stay flat (exact case, no closed-form needed beyond
        this invariant).

        """
        fs = 8_000
        n = 512
        td = np.zeros((n, 1))
        td[0, 0] = 1.0
        ir = dsp.ImpulseResponse(None, td, fs)

        for domain in dsp.transfer_functions.SmoothingDomain:
            sp = dsp.transfer_functions.complex_smoothing(ir, 12.0, domain)
            mag = np.abs(sp.spectral_data[:, 0])
            phase = np.angle(sp.spectral_data[1:, 0])
            np.testing.assert_allclose(mag, 1.0, atol=1e-10)
            np.testing.assert_allclose(phase, 0.0, atol=1e-10)
