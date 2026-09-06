"""
Tests for the FilterBank class.
"""

import os
import pickle
import tempfile
from os.path import join

import numpy as np
import pytest
import scipy.signal as sig
from matplotlib.pyplot import close, subplots

import dsptoolbox as dsp

_rng = np.random.default_rng(3)

RIR_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "example_data",
    "rir.wav",
)
CHIRP_STEREO_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "example_data",
    "chirp_stereo.wav",
)


class TestFilterBankClass:
    fs = 44100

    def get_iir_filter(self) -> dsp.Filter:
        return dsp.Filter.iir_filter(
            5,
            frequency_hz=[1510, 2000],
            type_of_pass=dsp.FilterPassType.Bandpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=self.fs,
        )

    def get_fir_filter(self, other_sampling_rate=False) -> dsp.Filter:
        return dsp.Filter.fir_filter(
            order=150,
            frequency_hz=[1500, 2000],
            type_of_pass=dsp.FilterPassType.Bandpass,
            sampling_rate_hz=(self.fs if not other_sampling_rate else self.fs // 2),
        )

    def test_create_filter_bank(self):
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())

        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == self.fs

        fb = fb.add_filter(self.get_fir_filter())

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Create filter bank passing a list
        filters = []
        filters.append(self.get_iir_filter())
        filters.append(self.get_fir_filter())
        fb = dsp.FilterBank(
            filters=filters,
            same_sampling_rate=True,
            info={"Type of filter bank": "Test"},
        )
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Reading FIRs from files
        firs = dsp.FilterBank.firs_from_file(RIR_PATH)
        assert len(firs) == 1
        firs = dsp.FilterBank.firs_from_file(CHIRP_STEREO_PATH)
        assert len(firs) == 2

    def test_save_filterbank_round_trip_and_format_checking(self):
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        with tempfile.TemporaryDirectory() as d:
            fb.save_filterbank(join(d, "with_ext.pkl"))
            with open(join(d, "with_ext.pkl"), "rb") as fh:
                reloaded = pickle.load(fh)
            assert reloaded.number_of_filters == fb.number_of_filters
            assert reloaded.sampling_rate_hz == fb.sampling_rate_hz

            # The extension is required and has to match
            with pytest.raises(ValueError):
                fb.save_filterbank(join(d, "no_ext"))
            with pytest.raises(ValueError):
                fb.save_filterbank(join(d, "wrong_ext.txt"))

    def test_plots(self):
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        fb.plot_magnitude(length_samples=512, mode=dsp.FilterBankMode.Parallel)
        fb.plot_magnitude(length_samples=512, mode=dsp.FilterBankMode.Sequential)
        fb.plot_magnitude(length_samples=512, mode=dsp.FilterBankMode.Summed)
        fb.plot_magnitude(length_samples=512, mode=dsp.FilterBankMode.Parallel)

        fb.plot_phase(length_samples=512, mode=dsp.FilterBankMode.Parallel)
        fb.plot_phase(length_samples=512, mode=dsp.FilterBankMode.Sequential)
        fb.plot_phase(length_samples=512, mode=dsp.FilterBankMode.Summed)
        fb.plot_phase(length_samples=512, mode=dsp.FilterBankMode.Parallel)

        fb.plot_group_delay(length_samples=512, mode=dsp.FilterBankMode.Parallel)
        fb.plot_group_delay(length_samples=512, mode=dsp.FilterBankMode.Sequential)
        fb.plot_group_delay(length_samples=512, mode=dsp.FilterBankMode.Summed)
        fb.plot_group_delay(length_samples=512, mode=dsp.FilterBankMode.Parallel)

    def test_plot_modes_reuse_passed_axis(self):
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        for plot_method in (
            fb.plot_magnitude,
            fb.plot_phase,
            fb.plot_group_delay,
        ):
            for mode in dsp.FilterBankMode:
                fig, ax = subplots()
                _, returned_ax = plot_method(512, mode=mode, ax=ax)
                assert returned_ax is ax
                close(fig)

    def test_filterbank_functionalities(self):
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        fb = fb.remove_filter(0)
        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == self.fs

        fb = fb.add_filter(self.get_fir_filter())
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        fb = fb.swap_filters([1, 0])
        assert fb.number_of_filters == 2
        assert len(fb) == 2
        assert fb.sampling_rate_hz == self.fs

        with pytest.raises(AssertionError):
            fb = fb.swap_filters([1, 1])
        with pytest.raises(AssertionError):
            fb = fb.swap_filters([1, 2])

        fb.get_ir(128, dsp.FilterBankMode.Parallel)
        fb.copy()
        fb.show_info()
        print(fb)

    def test_pop_filter_returns_the_removed_filter(self):
        fb = dsp.FilterBank()
        iir = self.get_iir_filter()
        fir = self.get_fir_filter()
        fb = fb.add_filter(iir).add_filter(fir)

        new_fb, removed = fb.pop_filter()
        assert new_fb.number_of_filters == 1
        assert removed.is_fir
        assert fb.remove_filter().number_of_filters == 1

        with pytest.raises(AssertionError):
            fb.pop_filter(5)

    def test_add_filter_at_index(self):
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter(), index=0)
        assert fb.filters[0].is_fir
        fb = fb.add_filter(self.get_iir_filter())
        assert fb.filters[-1].is_iir

    def test_filtering(self):
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        t_vec = _rng.normal(0, 0.01, (self.fs * 3, 2))
        s = dsp.Signal(None, t_vec, self.fs)

        filt1 = fb.filters[0].get_coefficients(
            coefficients_mode=dsp.FilterCoefficientsType.Sos
        )
        filt2, _ = fb.filters[1].get_coefficients(
            coefficients_mode=dsp.FilterCoefficientsType.Ba
        )
        # Parallel
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Parallel, activate_zi=False)
        assert type(s_) is dsp.MultiBandSignal
        assert s_.number_of_bands == fb.number_of_filters
        assert np.all(
            np.isclose(s_.bands[0].time_data[:, 0], sig.sosfilt(filt1, t_vec[:, 0]))
        )
        assert np.all(
            np.isclose(
                s_.bands[1].time_data[:, 0],
                sig.lfilter(filt2, [1], t_vec[:, 0]),
            )
        )

        # Sequential mode
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Sequential, activate_zi=False)
        assert type(s_) is dsp.Signal
        # The cascade order can be swapped since both are linear systems
        temp = sig.lfilter(filt2, [1], s.time_data[:, 1])
        temp = sig.sosfilt(filt1, temp)
        assert np.all(np.isclose(s_.time_data[:, 1], temp))

        # Summed mode
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Summed, activate_zi=False)
        assert type(s_) is dsp.Signal
        temp = sig.lfilter(filt2, [1], s.time_data[:, 1])
        temp += sig.sosfilt(filt1, s.time_data[:, 1])
        assert np.all(np.isclose(s_.time_data[:, 1], temp))

        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Parallel, activate_zi=True)
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Sequential, activate_zi=True)
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Summed, activate_zi=True)

        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Parallel, zero_phase=True)
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Sequential, zero_phase=True)
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Summed, zero_phase=True)

        # No zi and zero phase filtering at the same time!
        with pytest.raises(AssertionError):
            s_ = fb.filter_signal(
                s,
                mode=dsp.FilterBankMode.Summed,
                activate_zi=True,
                zero_phase=True,
            )

    def test_multirate(self):
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb = fb.add_filter(self.get_iir_filter())

        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == [self.fs]

        fb = fb.add_filter(self.get_fir_filter(True))

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs, self.fs // 2]

        fb = fb.remove_filter(0)
        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == [self.fs // 2]

        fb = fb.add_filter(self.get_fir_filter())
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs // 2, self.fs]

        fb = fb.swap_filters([1, 0])
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs, self.fs // 2]

        # Should not be possible to create
        with pytest.raises(AssertionError):
            fb = dsp.FilterBank(same_sampling_rate=True)
            fb = fb.add_filter(self.get_iir_filter())
            fb = fb.add_filter(self.get_fir_filter(True))

        # Create filter bank passing a list
        filters = []
        filters.append(self.get_iir_filter())
        filters.append(self.get_fir_filter(True))
        fb = dsp.FilterBank(
            filters=filters,
            same_sampling_rate=False,
            info={"Type of filter bank": "Test"},
        )

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs, self.fs // 2]

        with pytest.raises(AssertionError):
            fb = dsp.FilterBank(
                filters=filters,
                same_sampling_rate=True,
                info={"Type of filter bank": "Test"},
            )

    def test_plotting_multirate(self):
        # Should not fail but no plots are created
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter(True))

        fb.plot_magnitude(length_samples=512, mode=dsp.FilterBankMode.Parallel)
        fb.plot_phase(length_samples=512, mode=dsp.FilterBankMode.Parallel)
        fb.plot_group_delay(length_samples=512, mode=dsp.FilterBankMode.Parallel)
        fb.get_ir(128, dsp.FilterBankMode.Parallel)
        with pytest.raises(AssertionError):
            fb.get_ir(128, mode=dsp.FilterBankMode.Summed)

    def test_filtering_multirate_multiband(self):
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter(True))

        s1 = dsp.generators.noise(length_seconds=1, sampling_rate_hz=self.fs, rng=106)
        s2 = dsp.generators.noise(
            length_seconds=2, sampling_rate_hz=self.fs // 2, rng=107
        )

        mb = dsp.MultiBandSignal(bands=[s1, s2], same_sampling_rate=False)
        assert np.all(mb.sampling_rate_hz == [self.fs, self.fs // 2])

        mb_ = fb.filter_multiband_signal(mb, activate_zi=False, zero_phase=False)
        assert np.all(mb_.sampling_rate_hz == [self.fs, self.fs // 2])
        fb.filter_multiband_signal(mb, activate_zi=True, zero_phase=False)
        fb.filter_multiband_signal(mb, activate_zi=False, zero_phase=True)

    def test_iterator(self):
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter(True))
        for n in fb:
            assert type(n) is dsp.Filter

    def test_transfer_function(self):
        # Create
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        freqs = np.linspace(1, 2e3, 400)
        fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Parallel)
        fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Sequential)
        fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Summed)

        with pytest.raises(AssertionError):
            freqs = np.linspace(1, self.fs, 40)
            fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Parallel)

    def test_transfer_function_matches_manual_freqz_composition(self):
        """Cross-check `FilterBank.get_transfer_function` against transfer
        functions computed directly with `scipy.signal.sosfreqz`/`freqz`
        from each filter's own coefficients -- an independent reference,
        not a call into `Filter.get_transfer_function` (which internally
        also uses `freqz`, so this isn't circular re-verification of the
        same code path, just the same underlying scipy primitive).
        `Parallel` keeps per-filter transfer functions, `Sequential` is
        their product (cascade), `Summed` is their sum -- matching the same
        composition already verified in the time domain by `test_filtering`.

        """
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        freqs = np.linspace(1, 2e3, 200)
        sos1 = fb.filters[0].get_coefficients(
            coefficients_mode=dsp.FilterCoefficientsType.Sos
        )
        b2, a2 = fb.filters[1].get_coefficients(
            coefficients_mode=dsp.FilterCoefficientsType.Ba
        )
        w = 2 * np.pi * freqs / self.fs
        _, h1 = sig.sosfreqz(sos1, worN=w)
        _, h2 = sig.freqz(b2, a2, worN=w)

        h_parallel = fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Parallel)
        np.testing.assert_allclose(h_parallel[:, 0], h1, atol=1e-10)
        np.testing.assert_allclose(h_parallel[:, 1], h2, atol=1e-8)

        h_seq = fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Sequential)
        np.testing.assert_allclose(h_seq, h1 * h2, atol=1e-8)

        h_sum = fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Summed)
        np.testing.assert_allclose(h_sum, h1 + h2, atol=1e-8)

    def _two_band_filterbank(self) -> dsp.FilterBank:
        return dsp.FilterBank(
            [
                dsp.Filter.iir_filter(
                    4,
                    500.0,
                    type_of_pass=dsp.FilterPassType.Lowpass,
                    filter_design_method=dsp.IirDesignMethod.Butterworth,
                    sampling_rate_hz=self.fs,
                ),
                dsp.Filter.iir_filter(
                    4,
                    2000.0,
                    type_of_pass=dsp.FilterPassType.Highpass,
                    filter_design_method=dsp.IirDesignMethod.Butterworth,
                    sampling_rate_hz=self.fs,
                ),
            ]
        )

    def test_all_plots_accept_zero_phase(self):
        """`plot_phase` and `plot_group_delay` share `get_ir`'s prologue with
        `plot_magnitude`, so they must accept `zero_phase` as well.

        """
        fb = self._two_band_filterbank()
        for mode in dsp.FilterBankMode:
            for zero_phase in (False, True):
                fb.plot_magnitude(2048, mode, zero_phase=zero_phase)
                fb.plot_phase(2048, mode, zero_phase=zero_phase)
                fb.plot_group_delay(2048, mode, zero_phase=zero_phase)
                close("all")

    def test_get_ir_and_plots_adapt_a_too_short_length(self):
        fir = dsp.Filter.fir_filter(
            600,
            1000.0,
            type_of_pass=dsp.FilterPassType.Lowpass,
            sampling_rate_hz=self.fs,
        )
        fb = dsp.FilterBank([fir])

        with pytest.warns(UserWarning):
            ir = fb.get_ir(64, dsp.FilterBankMode.Parallel)
        assert ir.bands[0].length_samples == 600 + 100

        for plot in (fb.plot_magnitude, fb.plot_phase, fb.plot_group_delay):
            with pytest.warns(UserWarning):
                plot(64, dsp.FilterBankMode.Parallel)
            close("all")

    def test_plots_are_skipped_for_a_multirate_filterbank(self):
        fb = dsp.FilterBank(
            [
                dsp.Filter.iir_filter(
                    2,
                    100.0,
                    type_of_pass=dsp.FilterPassType.Lowpass,
                    filter_design_method=dsp.IirDesignMethod.Butterworth,
                    sampling_rate_hz=self.fs,
                ),
                dsp.Filter.iir_filter(
                    2,
                    100.0,
                    type_of_pass=dsp.FilterPassType.Lowpass,
                    filter_design_method=dsp.IirDesignMethod.Butterworth,
                    sampling_rate_hz=self.fs // 2,
                ),
            ],
            same_sampling_rate=False,
        )
        for plot in (fb.plot_magnitude, fb.plot_phase, fb.plot_group_delay):
            with pytest.warns(UserWarning):
                assert plot(2048, dsp.FilterBankMode.Parallel) is None
