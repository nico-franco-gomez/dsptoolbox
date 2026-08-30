"""
Tests for `Spectrum` octave smoothing and frequency warping.
"""

import os

import numpy as np
import pytest

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

    rir = dsp.ImpulseResponse.from_file(RIR_PATH)
    rir_spec_real = dsp.Spectrum.from_signal(
        dsp.ImpulseResponse.from_file(RIR_PATH), False
    )

    def get_spectrum_from_rir(self, complex=False):
        return self.rir_spec_real.copy()

    def test_apply_octave_smoothing(self):
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

    def test_warp(self):
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
