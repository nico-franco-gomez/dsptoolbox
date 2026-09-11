"""
Tests for the various Filter/FilterBank topology classes (lattice-ladder,
state-space, parallel, Kautz, real-time FIR/IIR, warped IIR, etc.).
"""

import os

import numpy as np
import pytest
import scipy.signal as sig
from matplotlib.pyplot import close

import dsptoolbox as dsp

_rng = np.random.default_rng(8)

RIR_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "example_data",
    "rir.wav",
)


class TestFilterTopologies:
    fs_hz = 24_000

    def get_noise(self):
        """Seeded on purpose: these tests compare a per-sample recursive
        implementation against a vectorized scipy reference with a tight
        tolerance, which the specific noise realization can otherwise push
        past.

        """
        return dsp.generators.noise(
            length_seconds=1, sampling_rate_hz=self.fs_hz, rng=0
        )

    def test_svfilter(self):
        PLOT = False
        sv_filt = dsp.realtime.StateVariableFilter(1000.0, 1.0, self.fs_hz)
        n = self.get_noise()

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = sv_filt.process_sample(td[ind], 0)[0]

        sv_filt.reset_state()
        mb = sv_filt.filter_signal(n)
        n2 = mb.get_all_bands(0)

        np.testing.assert_array_equal(td, n2.time_data[:, 0])

        if PLOT:
            n2.spectrum_method = dsp.SpectrumMethod.FFT
            _, ax = n2.plot_magnitude(normalize=None)
            ax.plot(
                np.fft.rfftfreq(len(td), 1 / self.fs_hz),
                dsp.tools.to_db(np.fft.rfft(td), True),
            )
            dsp.plots.show()

    def test_lattice_ladder_filter(self):
        PLOT = False
        n = self.get_noise()

        # From a/b coefficients in SOS form
        iir = dsp.Filter.iir_filter(
            4,
            1000.0,
            type_of_pass=dsp.FilterPassType.Lowpass,
            filter_design_method=dsp.IirDesignMethod.Butterworth,
            sampling_rate_hz=self.fs_hz,
        )
        llf = dsp.realtime.LatticeLadderFilter.from_filter(iir)

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = llf.process_sample(td[ind], 0)

        llf.reset_state()
        n2 = llf.filter_signal(n)
        np.testing.assert_array_equal(td, n2.time_data[:, 0])
        np.testing.assert_allclose(
            td,
            sig.sosfilt(
                iir.get_coefficients(dsp.FilterCoefficientsType.Sos),
                n.time_data.squeeze(),
            ),
            atol=1e-12,
        )

        # From plain a/b coefficients
        iir = dsp.Filter.from_ba(
            *iir.get_coefficients(dsp.FilterCoefficientsType.Ba),
            sampling_rate_hz=self.fs_hz,
        )
        llf = dsp.realtime.LatticeLadderFilter.from_filter(iir)

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = llf.process_sample(td[ind], 0)

        llf.reset_state()
        n2 = llf.filter_signal(n)
        np.testing.assert_array_equal(td, n2.time_data[:, 0])
        np.testing.assert_allclose(
            td,
            sig.lfilter(
                *iir.get_coefficients(dsp.FilterCoefficientsType.Ba),
                n.time_data.squeeze(),
            ),
            atol=1e-12,
        )

        # An IR-derived FIR case is intentionally not covered here: it does
        # not round-trip through the reflection-coefficient conversion for
        # this filter.

        if PLOT:
            n2.spectrum_method = dsp.SpectrumMethod.FFT
            _, ax = n2.plot_magnitude(normalize=None)
            ax.plot(
                np.fft.rfftfreq(len(td), 1 / self.fs_hz),
                dsp.tools.to_db(np.fft.rfft(td), True),
            )
            dsp.plots.show()

    def test_iir_filter(self):
        iir_original = dsp.Filter.iir_filter(
            4,
            1000.0,
            type_of_pass=dsp.FilterPassType.Highpass,
            filter_design_method=dsp.IirDesignMethod.Butterworth,
            sampling_rate_hz=self.fs_hz,
        )
        b, a = iir_original.get_coefficients(dsp.FilterCoefficientsType.Ba)
        iir = dsp.realtime.IIRFilter(b, a)
        n = self.get_noise()

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = iir.process_sample(td[ind], 0)

        np.testing.assert_allclose(td, sig.lfilter(b, a, n.time_data[:, 0]), atol=1e-12)

        dsp.realtime.IIRFilter.from_filter(iir_original)

    def test_fir_filter(self):
        fir_original = dsp.Filter.fir_filter(
            25,
            1000.0,
            type_of_pass=dsp.FilterPassType.Lowpass,
            window=dsp.Window.Blackman,
            sampling_rate_hz=self.fs_hz,
        )
        b, _ = fir_original.get_coefficients(dsp.FilterCoefficientsType.Ba)
        b = b[: len(b) // 2 + 3]  # some asymmetrical window
        fir = dsp.realtime.FIRFilter(b)
        n = self.get_noise()

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = fir.process_sample(td[ind], 0)

        np.testing.assert_allclose(
            td, sig.lfilter(b, [1], n.time_data[:, 0]), atol=1e-12
        )

        dsp.realtime.FIRFilter.from_filter(fir_original)

    def test_kautz_filters(self):
        fs_hz = 48000

        # Define some poles for smoothing according to Bank, B. (2022). Warped,
        # Kautz, and Fixed-Pole Parallel Filters: A Review. Journal of the
        # Audio Engineering Society.
        fractional_octave_smoothing = 24  # beta
        K = int(10 * (fractional_octave_smoothing / 2) + 1)
        pole_freqs_hz = np.logspace(np.log10(20), np.log10(20480), K, endpoint=True)
        pole_freqs_rad = 2 * np.pi * pole_freqs_hz / fs_hz
        bandwidth = np.zeros_like(pole_freqs_rad)
        bandwidth[0] = pole_freqs_rad[1] - pole_freqs_rad[0]
        bandwidth[-1] = pole_freqs_rad[-1] - pole_freqs_rad[-2]
        bandwidth[1:-1] = (pole_freqs_rad[2:] - pole_freqs_rad[:-2]) / 2  # Eq. 24
        poles = np.exp(-bandwidth / 2 + 1j * pole_freqs_rad)  # Eq. 25

        # Add two real poles just for testing
        poles = np.hstack([0.1, poles, -0.4])

        filter = dsp.realtime.KautzFilter(poles, fs_hz)

        # Per-sample processing must match the block IR
        d = dsp.generators.dirac(2**11, sampling_rate_hz=fs_hz)
        d.constrain_amplitude = False
        td = d.time_data.squeeze()
        for ind in np.arange(len(td)):
            td[ind] = filter.process_sample(td[ind], 0)
        dd = filter.get_ir(2**11)

        td /= np.max(np.abs(td))
        dd = dd.normalize(norm_dbfs=0.0)
        np.testing.assert_allclose(td, dd.time_data.squeeze(), rtol=1e-6)

        filter.fit_coefficients_to_ir(d)
        assert np.any(filter.coefficients_complex_poles != 1.0)
        assert np.any(filter.coefficients_real_poles != 1.0)

    def test_exponential_averager(self):
        n = _rng.normal(0, 0.1, 200)
        f = dsp.realtime.ExponentialAverageFilter(1e-3, 1e-3, self.fs_hz)
        for i in n:
            f.process_sample(i, 0)

    def test_parallel_filterbank(self):
        rir = dsp.ImpulseResponse(RIR_PATH)
        poles = np.logspace(np.log10(1e-2), np.log10(np.pi * 0.95), 3, endpoint=True)
        poles = 0.5 * np.exp(1j * poles)

        fb = dsp.realtime.ParallelFilter(poles, 0, rir.sampling_rate_hz)
        fb.fit_to_ir(rir)
        fb.get_ir(256)
        fb.set_n_channels(3)

        for i in rir.time_data[:, 0]:
            fb.process_sample(i, 1)
        fb.reset_state()
        iir_coeffs = _rng.normal(0, 0.1, (len(poles), 2))
        fb.set_coefficients(iir_coeffs, _rng.normal(0, 0.01, 10))

        fb = dsp.realtime.ParallelFilter(poles, 1, rir.sampling_rate_hz)
        fb.set_parameters(4, 0.0)
        fb.fit_to_ir(rir)
        fb = dsp.realtime.ParallelFilter(poles, 3, rir.sampling_rate_hz)
        fb.set_parameters(10, 1e-3)
        fb.fit_to_ir(rir)

    def test_parallel_filterbank_process_sample_matches_scipy_sosfilt(self):
        """`fit_to_ir` is a frequency-domain least-squares fit -- too
        complex for an exact reference (per the plan's own carve-out for
        IR-fitting optimizers) -- but with *known* coefficients set
        directly via `set_coefficients`, the filter bank becomes a
        deterministic sum of pole-fixed SOS sections, which can be
        reproduced independently with `scipy.signal.sosfilt` composed by
        hand from the same poles/numerators (via `scipy.signal.zpk2sos`,
        not by reading back the library's own internal SOS array).

        Tolerance is not machine-precision: `process_sample` runs each
        section through this codebase's own Transposed-Direct-Form-II
        `IIRFilter`, a different (if mathematically equivalent) per-sample
        recursion from `scipy.signal.sosfilt`'s internal state machine, so
        a small but bounded (non-growing with signal length, empirically
        checked up to 500 samples) floating-point discrepancy is expected.

        """
        rir = dsp.ImpulseResponse.from_file(RIR_PATH).pad_trim(2000)
        poles = np.array([0.6 * np.exp(1j * 0.5), 0.3 * np.exp(1j * 1.5)])
        fb = dsp.realtime.ParallelFilter(poles, 0, rir.sampling_rate_hz)
        # `fit_to_ir` is only used here to allocate the SOS array from the
        # poles; the fitted numerator coefficients are overwritten next.
        fb.fit_to_ir(rir)

        rng = np.random.default_rng(3)
        b_numerators = rng.normal(0, 0.1, (len(poles), 2))
        fb.set_coefficients(b_numerators, None)

        poles_with_conjugates = np.hstack([poles, poles.conjugate()])
        sos_reference = sig.zpk2sos([], poles_with_conjugates, 1.0)
        sos_reference[:, :2] = b_numerators

        x = rng.normal(0, 0.1, 500)
        fb.reset_state()
        out = np.array([fb.process_sample(v, 0) for v in x])
        expected = np.zeros_like(x)
        for row in sos_reference:
            expected += sig.sosfilt(row[None, :], x)

        np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_filter_chain(self):
        fc = dsp.realtime.FilterChain(
            [
                dsp.realtime.IIRFilter(np.array([0.5]), np.array([0.5, 0.1])),
                dsp.realtime.FIRFilter(np.array([0.5, 0.5])),
            ]
        )
        assert fc.n_filters == 2

        n = _rng.normal(0, 0.1, 50)

        for nn in n:
            fc.process_sample(nn, 0)

        fc.reset_state()
        fc.set_n_channels(1)

    def test_filter_chain_matches_cascaded_lfilter(self):
        """`FilterChain.process_sample` applies each filter's
        `process_sample` sequentially, so the whole chain is equivalent to
        cascading `scipy.signal.lfilter` calls with each filter's own
        coefficients in order (exact reference).

        """
        b_iir, a_iir = np.array([0.5]), np.array([0.5, 0.1])
        b_fir = np.array([0.5, 0.5])
        fc = dsp.realtime.FilterChain(
            [
                dsp.realtime.IIRFilter(b_iir.copy(), a_iir.copy()),
                dsp.realtime.FIRFilter(b_fir.copy()),
            ]
        )

        rng = np.random.default_rng(2)
        x = rng.normal(0, 0.1, 200)
        out = np.array([fc.process_sample(v, 0) for v in x])

        expected = sig.lfilter(b_iir, a_iir, x)
        expected = sig.lfilter(b_fir, [1], expected)
        np.testing.assert_allclose(out, expected)

    def test_state_space_filtering(self):
        ff = dsp.Filter.biquad(dsp.BiquadEqType.Peaking, 100, 6, 0.7, self.fs_hz)
        b, a = ff.get_coefficients(dsp.FilterCoefficientsType.Ba)
        A, B, C, D = sig.tf2ss(b, a)
        noise = dsp.generators.noise(
            length_seconds=1.0,
            type_of_noise=-2.0,
            sampling_rate_hz=self.fs_hz,
            number_of_channels=2,
            rng=0,
        )
        ff2 = dsp.realtime.StateSpaceFilter(A, B, C, D)
        ff2.set_n_channels(noise.number_of_channels)
        reference = ff.filter_signal(noise)

        channel = 0
        for ch_n in noise:
            output = np.zeros(len(ch_n))
            for ind in range(len(ch_n)):
                output[ind] = ff2.process_sample(ch_n[ind], channel)
            np.testing.assert_allclose(reference.time_data[:, channel], output)
            channel += 1

        # A cutoff of 2000 Hz (relative to the 24 kHz sampling rate) keeps
        # the order-12 filter well-conditioned; a much narrower relative
        # bandwidth (e.g. 500 Hz here) pushes the direct Ba coefficients
        # into scipy's "badly conditioned filter coefficients" territory,
        # which is a numerical-stability property of tf2ss itself and not
        # what this test is meant to check.
        iir = dsp.Filter.iir_filter(12, 2000.0, dsp.FilterPassType.Lowpass, self.fs_hz)
        reference = iir.filter_signal(noise).time_data[:, 0]
        x = noise.time_data[:, 0]

        # Single state-space filter built directly from the Ba coefficients
        ff_from_filter = dsp.realtime.StateSpaceFilter.from_filter(iir)
        out_from_filter = np.array([ff_from_filter.process_sample(v, 0) for v in x])
        np.testing.assert_allclose(reference, out_from_filter, atol=1e-8)

        # Cascade of second-order state-space filters, one per SOS section
        sos_filters = dsp.realtime.StateSpaceFilter.from_filter_as_sos_list(iir)
        assert len(sos_filters) == 6
        out_from_sos = x.copy()
        for section in sos_filters:
            out_from_sos = np.array(
                [section.process_sample(v, 0) for v in out_from_sos]
            )
        np.testing.assert_allclose(reference, out_from_sos, atol=1e-8)

    @pytest.mark.parametrize(
        "implementation",
        [
            dsp.realtime.FIRFilterOverlapSave,
            dsp.realtime.FIRUniformPartitioned,
        ],
    )
    def test_fir_filter_other_implementations(
        self, implementation: dsp.realtime.FIRFilterOverlapSave
    ):
        rir = dsp.ImpulseResponse.from_file(RIR_PATH)
        noise = (self.get_noise()).resample(rir.sampling_rate_hz)
        fir = implementation.from_filter(dsp.transfer_functions.ir_to_filter(rir))

        blocksize = 512
        fir.prepare(blocksize, 1)
        n_blocks = len(noise) // blocksize + 1
        noise = noise.pad_trim(n_blocks * blocksize)
        accumulator = np.zeros_like(noise.time_data)
        for n in range(n_blocks):
            stop = min((n + 1) * blocksize, len(accumulator))
            sl = slice(n * blocksize, stop)
            accumulator[sl, 0] = fir.process_block(noise.time_data[sl, 0], 0)

        reference = sig.oaconvolve(
            noise.time_data[:, 0], rir.time_data[:, 0], mode="full"
        )
        diff = accumulator.squeeze() - reference.squeeze()[: len(accumulator)]
        np.testing.assert_array_almost_equal(diff, 0.0)

    def test_fir_filter_multichannel_uniform_partitioned(self):
        rir = dsp.ImpulseResponse.from_file(RIR_PATH)
        noise = self.get_noise().resample(rir.sampling_rate_hz)
        rir = rir.append_signals([rir.copy()])
        noise = noise.append_signals([noise.copy()])
        fir = dsp.realtime.FIRUniformPartitionedMultichannel(rir.time_data)

        blocksize = 512
        fir.prepare(blocksize)
        n_blocks = len(noise) // blocksize + 1
        noise = noise.pad_trim(n_blocks * blocksize)
        accumulator = np.zeros_like(noise.time_data)
        for n in range(n_blocks):
            stop = min((n + 1) * blocksize, len(accumulator))
            sl = slice(n * blocksize, stop)
            accumulator[sl, :] = fir.process_block(noise.time_data[sl, :])

        reference = sig.oaconvolve(
            noise.time_data[:, 0], rir.time_data[:, 0], mode="full"
        )
        diff = accumulator[:, 0] - reference.squeeze()[: len(accumulator)]
        np.testing.assert_array_almost_equal(diff, 0.0)
        diff = accumulator[:, 1] - reference.squeeze()[: len(accumulator)]
        np.testing.assert_array_almost_equal(diff, 0.0)

    def test_warped_fir_filter(self):
        rir = (dsp.ImpulseResponse.from_file(RIR_PATH)).pad_trim(300)
        fir = dsp.realtime.WarpedFIR(
            np.hanning(15),
            dsp.WarpingFactor.Custom.with_factor(-0.6),
            rir.sampling_rate_hz,
        )
        [fir.process_sample(x, 0) for x in rir.time_data[:, 0]]

        # Multichannel
        rir.time_data = np.repeat(rir.time_data, 2, axis=1)
        fir.filter_signal(rir)

        dsp.realtime.WarpedFIR.from_filter(
            dsp.Filter.from_ba(np.hanning(20), [1], rir.sampling_rate_hz),
            dsp.WarpingFactor.Custom.with_factor(0.1),
        )

    def test_warped_iir_filter(self):
        rir = (dsp.ImpulseResponse.from_file(RIR_PATH)).pad_trim(300)
        iir_coefficients = dsp.Filter.biquad(
            dsp.BiquadEqType.Peaking, 200.0, 4, 0.7, rir.sampling_rate_hz
        )

        iir_w = dsp.realtime.WarpedIIR(
            iir_coefficients.ba[0].copy(),
            iir_coefficients.ba[1].copy(),
            dsp.WarpingFactor.Custom.with_factor(-0.6),
            rir.sampling_rate_hz,
        )
        [iir_w.process_sample(x, 0) for x in rir.time_data[:, 0]]

        # Multichannel
        rir.time_data = np.repeat(rir.time_data, 2, axis=1)
        iir_w.filter_signal(rir)

        # a and b coefficients of different lengths
        iir_w = dsp.realtime.WarpedIIR(
            np.pad(iir_coefficients.ba[0], ((0, 4))),
            np.pad(iir_coefficients.ba[1], ((0, 10))),
            dsp.WarpingFactor.Custom.with_factor(-0.6),
            rir.sampling_rate_hz,
        )
        [iir_w.process_sample(x, 0) for x in rir.time_data[:, 0]]

        dsp.realtime.WarpedIIR.from_filter(
            iir_coefficients, dsp.WarpingFactor.Custom.with_factor(0.1)
        )

    def test_warped_fir_filter_zero_warp_matches_scipy_lfilter(self):
        """At `warping_factor=0`, the allpass warping stage
        (`(buffer[nn+1]-residue)*warp + buffer[nn]`) reduces exactly to
        `buffer[nn]`: a plain shift register, i.e. `WarpedFIR` collapses to
        an ordinary FIR filter -- an exact reference against
        `scipy.signal.lfilter`.

        """
        rng = np.random.default_rng(0)
        b = rng.normal(0, 1, 7)
        x = rng.normal(0, 1, 200)

        fir = dsp.realtime.WarpedFIR(
            b, dsp.WarpingFactor.Custom.with_factor(0.0), self.fs_hz
        )
        out = np.array([fir.process_sample(v, 0) for v in x])
        expected = sig.lfilter(b, [1], x)
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_warped_iir_filter_zero_warp_matches_scipy_lfilter(self):
        """Same zero-warp collapse as `WarpedFIR`, but for the IIR variant:
        at `warping_factor=0`, `WarpedIIR` reduces to an ordinary direct-
        form IIR filter -- an exact reference against
        `scipy.signal.lfilter`.

        """
        rng = np.random.default_rng(1)
        b = rng.normal(0, 1, 4)
        a = np.array([1.0, -0.5, 0.2])
        x = rng.normal(0, 1, 200)

        iir = dsp.realtime.WarpedIIR(
            b, a, dsp.WarpingFactor.Custom.with_factor(0.0), self.fs_hz
        )
        out = np.array([iir.process_sample(v, 0) for v in x])
        expected = sig.lfilter(b, a, x)
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_warped_rust_backend_parity(self):
        from dsptoolbox.realtime.warped_filters import (
            _warped_fir_filtering_block_rust,
            _warped_fir_filtering_python,
            _warped_fir_filtering_rust,
            _warped_fir_filtering_sample_rust,
            _warped_iir_filtering_block_rust,
            _warped_iir_filtering_python,
            _warped_iir_filtering_rust,
            _warped_iir_filtering_sample_rust,
        )

        if _warped_fir_filtering_rust is None:
            pytest.skip("Rust extension is not available")

        rng = np.random.default_rng(11)
        b = rng.normal(0.0, 0.2, 9)
        td = rng.normal(size=(257, 3))
        state = rng.normal(size=(len(b), 3))

        expected = td.copy()
        expected_state = state.copy()
        _warped_fir_filtering_python(b, -0.6, expected, expected_state)
        actual = td.copy()
        actual_state = state.copy()
        _warped_fir_filtering_rust(b, -0.6, actual, actual_state)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_state, expected_state, rtol=1e-12, atol=1e-12)

        filt = dsp.realtime.WarpedIIR(
            rng.normal(0.0, 0.2, 7),
            np.array([1.0, -0.4, 0.15, -0.05]),
            dsp.WarpingFactor.Custom.with_factor(-0.6),
            self.fs_hz,
        )
        expected = td.copy()
        expected_state = state[: filt.N].copy()
        _warped_iir_filtering_python(
            filt.b, filt.sigmas, filt.warp, expected, expected_state
        )
        actual = td.copy()
        actual_state = state[: filt.N].copy()
        _warped_iir_filtering_rust(filt.b, filt.sigmas, filt.warp, actual, actual_state)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_state, expected_state, rtol=1e-12, atol=1e-12)

        assert _warped_fir_filtering_block_rust is not None
        assert _warped_fir_filtering_sample_rust is not None
        assert _warped_iir_filtering_block_rust is not None
        assert _warped_iir_filtering_sample_rust is not None

        sample = td[0, 0]
        expected_sample_data = np.array([[sample]])
        expected_state = state[:, :1].copy()
        _warped_fir_filtering_python(b, -0.6, expected_sample_data, expected_state)
        actual_state = state[:, :1].copy()
        actual_sample = _warped_fir_filtering_sample_rust(
            b, -0.6, sample, actual_state, 0
        )
        np.testing.assert_allclose(actual_sample, expected_sample_data[0, 0])
        np.testing.assert_allclose(actual_state, expected_state)

        fir = dsp.realtime.WarpedFIR(
            b, dsp.WarpingFactor.Custom.with_factor(-0.6), self.fs_hz
        )
        fir.buffer = state[:, :1].copy()
        np.testing.assert_allclose(
            fir.process_sample(sample, 0), expected_sample_data[0, 0]
        )
        np.testing.assert_allclose(fir.buffer, expected_state)

        sample = td[0, 0]
        expected_sample_data = np.array([[sample]])
        expected_state = state[: filt.N, :1].copy()
        _warped_iir_filtering_python(
            filt.b,
            filt.sigmas,
            filt.warp,
            expected_sample_data,
            expected_state,
        )
        actual_state = state[: filt.N, :1].copy()
        actual_sample = _warped_iir_filtering_sample_rust(
            filt.b, filt.sigmas, filt.warp, sample, actual_state, 0
        )
        np.testing.assert_allclose(actual_sample, expected_sample_data[0, 0])
        np.testing.assert_allclose(actual_state, expected_state)

        filt.reset_state()
        filt.buffer = state[: filt.N, :1].copy()
        np.testing.assert_allclose(
            filt.process_sample(sample, 0), expected_sample_data[0, 0]
        )
        np.testing.assert_allclose(filt.buffer, expected_state)

        block = td[:, 0]
        expected_block = np.empty(len(block))
        expected_state = state[:, :1].copy()
        expected_data = block[:, None].copy()
        _warped_fir_filtering_python(b, -0.6, expected_data, expected_state)
        expected_block[:] = expected_data[:, 0]
        actual_block = np.empty(len(block))
        actual_state = state[:, :1].copy()
        _warped_fir_filtering_block_rust(b, -0.6, block, actual_block, actual_state, 0)
        np.testing.assert_allclose(actual_block, expected_block, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_state, expected_state, rtol=1e-12, atol=1e-12)

        block = td[:, 0]
        expected_block = np.empty(len(block))
        expected_state = state[: filt.N, :1].copy()
        expected_data = block[:, None].copy()
        _warped_iir_filtering_python(
            filt.b, filt.sigmas, filt.warp, expected_data, expected_state
        )
        expected_block[:] = expected_data[:, 0]
        actual_block = np.empty(len(block))
        actual_state = state[: filt.N, :1].copy()
        _warped_iir_filtering_block_rust(
            filt.b,
            filt.sigmas,
            filt.warp,
            block,
            actual_block,
            actual_state,
            0,
        )
        np.testing.assert_allclose(actual_block, expected_block, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_state, expected_state, rtol=1e-12, atol=1e-12)

    def test_iir_filter_does_not_modify_coefficients(self):
        """The constructor normalizes by a[0] and must copy to do so."""
        b = np.array([1.0, 0.5])
        a = np.array([2.0, 0.3])
        b_before, a_before = b.copy(), a.copy()
        dsp.realtime.IIRFilter(b, a)
        np.testing.assert_array_equal(b, b_before)
        np.testing.assert_array_equal(a, a_before)

    def test_iir_filter_accepts_integer_coefficients(self):
        iir = dsp.realtime.IIRFilter(np.array([2, 1]), np.array([2, 0]))
        assert np.isclose(iir.process_sample(1.0, 0), 1.0)

    def test_state_variable_filter_plots(self):
        """All three plot methods must run: they used to pass arguments that
        the underlying Signal/MultiBandSignal plots do not accept.

        """
        svf = dsp.realtime.StateVariableFilter(1000.0, 0.7, self.fs_hz)
        for fig, _ in (
            svf.plot_magnitude(1024),
            svf.plot_phase(1024),
            svf.plot_group_delay(1024),
        ):
            close(fig)
