"""
Tests for drawing several results onto one set of axes via `ax=`.
"""

import numpy as np
from matplotlib.pyplot import close, subplots

import dsptoolbox as dsp


class TestSharedAxes:
    fs = 44_100

    def get_signal(self, gain_db=0.0):
        rng = np.random.default_rng(0)
        return dsp.Signal(None, rng.normal(0, 0.1, (2048, 1)), self.fs).apply_gain(
            gain_db
        )

    def test_signal_plot_magnitude_shares_one_axis(self):
        fig, ax = subplots(1, 1)
        first = self.get_signal()
        second = self.get_signal(-6.0)

        _, returned = first.plot_magnitude(ax=ax)
        assert returned is ax
        second.plot_magnitude(ax=ax)

        assert len(ax.get_lines()) == 2
        close(fig)

    def test_filter_and_impulse_response_share_one_axis(self):
        fig, ax = subplots(1, 1)
        filt = dsp.Filter.biquad(dsp.BiquadEqType.Peaking, 500.0, 6.0, 1.0, self.fs)

        filt.plot_magnitude(length_samples=1024, ax=ax)
        filt.get_ir(1024).plot_magnitude(ax=ax)

        assert len(ax.get_lines()) == 2
        close(fig)

    def test_filterbank_plot_magnitude_shares_one_axis(self):
        fig, ax = subplots(1, 1)
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [500, 1000], order=4, sampling_rate_hz=self.fs
        )
        fb.plot_magnitude(length_samples=1024, ax=ax)
        assert len(ax.get_lines()) > 0
        close(fig)

    def test_spectrum_plot_magnitude_shares_one_axis(self):
        fig, ax = subplots(1, 1)
        spectrum = dsp.Spectrum.from_signal(self.get_signal())

        spectrum.plot_magnitude(ax=ax)
        spectrum.apply_gain(-6.0).plot_magnitude(ax=ax)

        assert len(ax.get_lines()) == 2
        close(fig)

    def test_subplot_based_plots_accept_a_list_of_axes(self):
        signal = self.get_signal().append_signals([self.get_signal(-6.0)])
        fig, ax = subplots(signal.number_of_channels, 1)

        _, returned = signal.plot_time(ax=list(ax))
        assert list(returned) == list(ax)
        close(fig)


class TestNoImportSideEffects:
    def test_importing_the_library_does_not_change_matplotlib_defaults(self):
        """`use_default_style()` is opt-in: importing the library must leave
        matplotlib's global settings alone."""
        import matplotlib as mpl

        assert mpl.rcParams["axes.grid"] == mpl.rcParamsDefault["axes.grid"]
        assert mpl.rcParams["axes.facecolor"] == mpl.rcParamsDefault["axes.facecolor"]

    def test_use_default_style_applies_the_seaborn_grid(self):
        import matplotlib as mpl

        before = dict(mpl.rcParams)
        try:
            assert dsp.plots.use_default_style()
            assert mpl.rcParams["axes.grid"]
        finally:
            mpl.rcParams.update(before)

    def test_importing_the_library_does_not_set_the_asio_variable(self):
        import os

        assert "SD_ENABLE_ASIO" not in os.environ

    def test_importing_the_library_does_not_import_sounddevice(self):
        """`audio_io` imports sounddevice on first use, so that a plain
        `import dsptoolbox` neither needs the audio backend nor locks in its
        configuration."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, dsptoolbox; print('sounddevice' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"
