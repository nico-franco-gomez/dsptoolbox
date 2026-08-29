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

import dsptoolbox as dsp

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
        # Create filter bank sequentially
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
            # No extension -> ".pkl" gets appended
            fb.save_filterbank(join(d, "no_ext"))
            with open(join(d, "no_ext.pkl"), "rb") as fh:
                reloaded = pickle.load(fh)
            assert reloaded.number_of_filters == fb.number_of_filters
            assert reloaded.sampling_rate_hz == fb.sampling_rate_hz

            # Matching ".pkl" extension is accepted as is
            fb.save_filterbank(join(d, "with_ext.pkl"))
            assert os.path.exists(join(d, "with_ext.pkl"))

            # A mismatched extension is rejected
            with pytest.raises(AssertionError):
                fb.save_filterbank(join(d, "wrong_ext.txt"))

    def test_plots(self):
        # Create
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        # Get plots
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

    def test_filterbank_functionalities(self):
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Remove
        fb = fb.remove_filter(0)
        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == self.fs

        # Readd
        fb = fb.add_filter(self.get_fir_filter())
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Swap (and Assertions)
        fb = fb.swap_filters([1, 0])
        assert fb.number_of_filters == 2
        assert len(fb) == 2
        assert fb.sampling_rate_hz == self.fs

        with pytest.raises(AssertionError):
            fb = fb.swap_filters([1, 1])
        with pytest.raises(AssertionError):
            fb = fb.swap_filters([1, 2])

        # Others
        fb.get_ir(128, dsp.FilterBankMode.Parallel)
        fb.copy()
        fb.show_info()
        print(fb)

    def test_filtering(self):
        # Create
        fb = dsp.FilterBank()
        fb = fb.add_filter(self.get_iir_filter())
        fb = fb.add_filter(self.get_fir_filter())

        t_vec = np.random.normal(0, 0.01, (self.fs * 3, 2))
        s = dsp.Signal(None, t_vec, self.fs)

        # Type of output and filter results
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
        # Change order (just because they're linear systems)
        temp = sig.lfilter(filt2, [1], s.time_data[:, 1])
        temp = sig.sosfilt(filt1, temp)
        # Try second channel
        assert np.all(np.isclose(s_.time_data[:, 1], temp))

        # Summed mode
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Summed, activate_zi=False)
        assert type(s_) is dsp.Signal
        # Add together
        temp = sig.lfilter(filt2, [1], s.time_data[:, 1])
        temp += sig.sosfilt(filt1, s.time_data[:, 1])
        assert np.all(np.isclose(s_.time_data[:, 1], temp))

        # Filter's zi
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Parallel, activate_zi=True)
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Sequential, activate_zi=True)
        s_ = fb.filter_signal(s, mode=dsp.FilterBankMode.Summed, activate_zi=True)

        # Zero-phase filtering
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

        # Remove
        fb = fb.remove_filter(0)
        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == [self.fs // 2]

        # Readd
        fb = fb.add_filter(self.get_fir_filter())
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs // 2, self.fs]

        # Swap (and Assertions)
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

        s1 = dsp.generators.noise(length_seconds=1, sampling_rate_hz=self.fs)
        s2 = dsp.generators.noise(length_seconds=2, sampling_rate_hz=self.fs // 2)

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
