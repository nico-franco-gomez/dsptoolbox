"""
Tests for the MultiBandSignal class.
"""

import os
import pickle
import tempfile
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp


class TestMultiBandSignal:
    fs = 44100
    s = dsp.Signal(None, np.random.normal(0, 0.01, (fs * 3, 3)), fs)
    fb = dsp.filterbanks.auditory_filters_gammatone(
        frequency_range_hz=[500, 1200], sampling_rate_hz=fs
    )

    def get_mb(self) -> dsp.MultiBandSignal:
        return self.fb.filter_signal(self.s, dsp.FilterBankMode.Parallel)

    def test_create_and_general_functionalities(self):
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        assert mbs.number_of_bands == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        assert mbs.sampling_rate_hz == self.s.sampling_rate_hz

        mbs = mbs.add_band(self.s)
        assert mbs.number_of_bands == 3
        assert mbs.number_of_channels == self.s.number_of_channels
        assert mbs.sampling_rate_hz == self.s.sampling_rate_hz
        mbs = mbs.remove_band(0)
        assert mbs.number_of_bands == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        mbs = mbs.swap_bands([1, 0])
        mbs.show_info()
        print(mbs)
        mbs.copy()

        with pytest.raises(IndexError):
            mbs.remove_band(4)
        with pytest.raises(AssertionError):
            mbs.swap_bands([1, 1])
        with pytest.raises(AssertionError):
            mbs.swap_bands([5, 0])
        with pytest.raises(AssertionError):
            # Inconsistent data in regards to complex values
            s2 = self.s.copy()
            s2.time_data = s2.time_data + 1j
            mbs = dsp.MultiBandSignal(
                bands=[self.s, s2],
                same_sampling_rate=True,
                info=dict(information="test filter bank"),
            )

        # Create from filter bank
        mbs = self.fb.filter_signal(self.s, dsp.FilterBankMode.Parallel)
        assert type(mbs) is dsp.MultiBandSignal

    def test_save_signal_round_trip_and_format_checking(self):
        mbs = self.get_mb()
        with tempfile.TemporaryDirectory() as d:
            # No extension -> ".pkl" gets appended
            mbs.save_signal(join(d, "no_ext"))
            with open(join(d, "no_ext.pkl"), "rb") as fh:
                reloaded = pickle.load(fh)
            assert reloaded.number_of_bands == mbs.number_of_bands
            assert reloaded.number_of_channels == mbs.number_of_channels

            # Matching ".pkl" extension is accepted as is
            mbs.save_signal(join(d, "with_ext.pkl"))
            assert os.path.exists(join(d, "with_ext.pkl"))

            # A mismatched extension is rejected
            with pytest.raises(AssertionError):
                mbs.save_signal(join(d, "wrong_ext.txt"))

    def test_collapse(self):
        td = self.s.time_data.copy()
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        mbs_ = mbs.collapse()

        assert np.all(mbs_.time_data == td + td)

    def test_get_all_bands(self):
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        mbs_ = mbs.get_all_bands(0)
        assert type(mbs_) is dsp.Signal
        # Number of channels has to match number of bands
        assert mbs_.number_of_channels == mbs.number_of_bands

    def test_get_all_time_data(self):
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        td, fs = mbs.get_all_time_data()

        td_s = self.s.time_data
        td_s = np.concatenate([td_s[:, None, :], td_s[:, None, :]], axis=1)

        assert np.all(td == td_s)
        assert fs == self.s.sampling_rate_hz

        # Complex time data
        s2 = self.s.copy()
        s2.time_data = s2.time_data + 1j
        mbs = dsp.MultiBandSignal(
            bands=[s2, s2],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        td, fs = mbs.get_all_time_data()

        td_s = s2.time_data + 1j * s2.time_data_imaginary
        td_s = np.concatenate([td_s[:, None, :], td_s[:, None, :]], axis=1)

        assert np.all(td == td_s)
        assert fs == self.s.sampling_rate_hz

        # Multirate
        s2 = self.s.resample(self.s.sampling_rate_hz // 2)
        mbs = dsp.MultiBandSignal(
            bands=[self.s, s2],
            same_sampling_rate=False,
            info=dict(information="test filter bank"),
        )
        tds = mbs.get_all_time_data()

        assert np.all(tds[0][0] == self.s.time_data)
        assert np.all(tds[1][0] == s2.time_data)

        assert np.all(tds[0][1] == self.s.sampling_rate_hz)
        assert np.all(tds[1][1] == s2.sampling_rate_hz)

    def test_multirate(self):
        s2 = self.s.resample(self.s.sampling_rate_hz // 2)

        # Parameter same sampling rate has to be False
        with pytest.raises(AssertionError):
            mbs = dsp.MultiBandSignal(
                bands=[self.s, s2],
                same_sampling_rate=True,
                info=dict(information="test filter bank"),
            )

        mbs = dsp.MultiBandSignal(
            bands=[self.s, s2],
            same_sampling_rate=False,
            info=dict(information="test filter bank"),
        )
        assert mbs.number_of_bands == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        assert np.all(
            mbs.sampling_rate_hz == [self.s.sampling_rate_hz, s2.sampling_rate_hz]
        )

        mbs = mbs.add_band(self.s)
        assert mbs.number_of_bands == 3
        assert mbs.number_of_channels == self.s.number_of_channels
        assert np.all(
            mbs.sampling_rate_hz
            == [
                self.s.sampling_rate_hz,
                s2.sampling_rate_hz,
                self.s.sampling_rate_hz,
            ]
        )

        mbs = mbs.remove_band(0)
        assert mbs.number_of_bands == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        assert np.all(
            mbs.sampling_rate_hz == [s2.sampling_rate_hz, self.s.sampling_rate_hz]
        )

        mbs = mbs.swap_bands([1, 0])
        assert mbs.number_of_bands == 2
        assert len(mbs) == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        assert np.all(
            mbs.sampling_rate_hz == [self.s.sampling_rate_hz, s2.sampling_rate_hz]
        )
        mbs.show_info()

    def test_iterator(self):
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        for n in mbs:
            assert type(n) is dsp.Signal

    def test_multibandsignal_properties(self):
        mb = self.get_mb()

        # Get
        _ = mb.length_seconds
        _ = mb.number_of_bands
        _ = mb.number_of_channels
        _ = mb.length_samples

        _ = mb.bands

        # Read-only properties
        with pytest.raises(AttributeError):
            mb.length_seconds = 1.0
        with pytest.raises(AttributeError):
            mb.number_of_bands = 1
        with pytest.raises(AttributeError):
            mb.number_of_channels = 1
        with pytest.raises(AttributeError):
            mb.length_samples = 1

    def test_collapse_does_not_modify_bands(self):
        """`collapse` must not accumulate into the first band's array.

        A real-valued filter bank is used on purpose: the complex branch
        allocates its own accumulator and never had the defect.

        """
        crossover = dsp.filterbanks.linkwitz_riley_crossovers([1000], [4], self.fs)
        mb = crossover.filter_signal(self.s, dsp.FilterBankMode.Parallel)
        assert not mb.bands[0].is_complex_signal

        first_band = mb.bands[0].time_data.copy()
        collapsed = mb.collapse()
        np.testing.assert_array_equal(first_band, mb.bands[0].time_data)
        np.testing.assert_allclose(
            collapsed.time_data,
            sum(band.time_data for band in mb.bands),
            atol=1e-12,
        )

        # A second call must give the same result
        np.testing.assert_allclose(
            collapsed.time_data, mb.collapse().time_data, atol=1e-12
        )

    def test_sampling_rate_count_must_match_bands(self):
        mb = self.get_mb()
        mb.same_sampling_rate = False
        with pytest.raises(AssertionError):
            mb.sampling_rate_hz = [self.fs] * (mb.number_of_bands + 1)

    def test_metadata_str_underline_matches_header(self):
        mb = self.get_mb()
        lines = mb.metadata_str.splitlines()
        assert lines[1] == "\u2013" * len(lines[0])
