"""
Tests for the ImpulseResponse class.
"""

import os
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp


class TestImpulseResponse:
    fs_hz = 10_000
    seconds = 2
    d = dsp.generators.dirac(seconds * fs_hz, sampling_rate_hz=fs_hz)

    path_rir = join(os.path.dirname(__file__), "..", "..", "example_data", "rir.wav")

    def get_ir(self):
        return dsp.ImpulseResponse.from_file(self.path_rir)

    def test_constructors(self):
        rir = self.get_ir()
        dsp.ImpulseResponse.from_time_data(rir.time_data, rir.sampling_rate_hz)
        dsp.ImpulseResponse.from_signal(dsp.Signal.from_file(self.path_rir))

    def test_channel_handling_with_window(self):
        rir = self.get_ir()
        rir = dsp.transfer_functions.window_centered_ir(rir, len(rir))[0]

        # Add channel
        rir = rir.add_channel(self.path_rir)
        assert not hasattr(rir, "window")

        # Window again
        rir = dsp.transfer_functions.window_centered_ir(rir, len(rir))[0]
        assert rir.window.shape == rir.time_data.shape
        np.testing.assert_array_equal(rir.window[:, 1], rir.window[:, 0])

        # Remove channel
        rir = rir.remove_channel(1)

        # Swap channels
        rir = rir.add_channel(self.path_rir)
        rir = rir.add_channel(self.path_rir)
        rir = rir.swap_channels([2, 1, 0])

    def test_plotting_with_window(self):
        rir = self.get_ir()
        rir = dsp.transfer_functions.window_centered_ir(rir, len(rir))[0]
        rir.plot_time()
        rir.plot_spl()
        rir.add_channel(self.path_rir)
        rir.plot_time()
        rir.plot_spl()
        # dsp.plots.show()

    def test_other_plotting(self):
        rir = self.get_ir()
        rir.plot_bode()
        rir.plot_bode(show_group_delay=True)
        # dsp.plots.show()

    def test_set_window(self):
        rir = self.get_ir()
        window = np.hanning(rir.time_data.shape[0])[:, None] * np.ones(
            (1, rir.number_of_channels)
        )
        windowed = rir.set_window(window)

        assert hasattr(windowed, "window")
        np.testing.assert_array_equal(windowed.window, window)
        # `set_window` only attaches the window array; it does not apply it
        # to the stored time data (that happens in the windowing functions
        # under `dsp.transfer_functions`, e.g. `window_ir`/`window_centered_ir`).
        np.testing.assert_array_equal(windowed.time_data, rir.time_data)
        # Returns a copy, the original is unaffected
        assert not hasattr(rir, "window")

        with pytest.raises(AssertionError):
            rir.set_window(np.hanning(rir.time_data.shape[0] - 1)[:, None])
