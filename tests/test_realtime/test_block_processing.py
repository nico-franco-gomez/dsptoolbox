"""
Tests for the block-processing interface shared by every RealtimeFilter.
"""

import numpy as np
import scipy.signal as sig

import dsptoolbox as dsp


class TestBlockProcessing:
    fs_hz = 44100
    blocksize = 128
    x = np.random.default_rng(0).normal(0, 0.1, 1000)

    def _filter_in_blocks(self, filt: dsp.realtime.RealtimeFilter):
        return np.concatenate(
            [
                filt.process_block(self.x[i : i + self.blocksize], 0)
                for i in range(0, len(self.x), self.blocksize)
            ]
        )

    def _filter_per_sample(self, filt: dsp.realtime.RealtimeFilter):
        return np.array([filt.process_sample(v, 0) for v in self.x])

    def test_iir_block_matches_sample_and_lfilter(self):
        b, a = sig.butter(4, 0.3)
        filt = dsp.realtime.IIRFilter(b, a)

        per_sample = self._filter_per_sample(filt)
        filt.reset_state()
        per_block = self._filter_in_blocks(filt)

        np.testing.assert_allclose(per_block, per_sample, atol=1e-12)
        np.testing.assert_allclose(per_block, sig.lfilter(b, a, self.x), atol=1e-12)

    def test_fir_block_matches_sample_and_lfilter(self):
        b = sig.firwin(31, 0.3)
        filt = dsp.realtime.FIRFilter(b)

        per_sample = self._filter_per_sample(filt)
        filt.reset_state()
        per_block = self._filter_in_blocks(filt)

        np.testing.assert_allclose(per_block, per_sample, atol=1e-12)
        np.testing.assert_allclose(per_block, sig.lfilter(b, [1.0], self.x), atol=1e-12)

    def test_filter_chain_blocks_through_every_stage(self):
        b, a = sig.butter(4, 0.3)
        b_fir = sig.firwin(15, 0.4)
        chain = dsp.realtime.FilterChain(
            [dsp.realtime.IIRFilter(b, a), dsp.realtime.FIRFilter(b_fir)]
        )

        per_sample = self._filter_per_sample(chain)
        chain.reset_state()
        per_block = self._filter_in_blocks(chain)

        np.testing.assert_allclose(per_block, per_sample, atol=1e-12)

    def test_generic_block_implementation_matches_per_sample(self):
        """`WarpedFIR` does not override `process_block`, so it uses the base
        implementation."""
        filt = dsp.realtime.WarpedFIR(
            np.hanning(15), dsp.WarpingFactor.Custom.with_factor(-0.6), self.fs_hz
        )
        per_sample = self._filter_per_sample(filt)

        filt.reset_state()
        per_block = self._filter_in_blocks(filt)
        np.testing.assert_allclose(per_block, per_sample)

    def test_state_variable_filter_block_returns_all_four_modes(self):
        filt = dsp.realtime.StateVariableFilter(1000.0, 1.0, self.fs_hz)
        per_sample = np.array([filt.process_sample(v, 0) for v in self.x])

        filt.reset_state()
        per_block = np.concatenate(
            [
                filt.process_block(self.x[i : i + self.blocksize], 0)
                for i in range(0, len(self.x), self.blocksize)
            ]
        )
        assert per_block.shape == (len(self.x), 4)
        np.testing.assert_allclose(per_block, per_sample)

    def test_block_length_shorter_than_filter_order(self):
        b = sig.firwin(63, 0.3)
        filt = dsp.realtime.FIRFilter(b)
        per_block = np.concatenate(
            [filt.process_block(self.x[i : i + 8], 0) for i in range(0, 200, 8)]
        )
        np.testing.assert_allclose(
            per_block, sig.lfilter(b, [1.0], self.x[:200]), atol=1e-12
        )
