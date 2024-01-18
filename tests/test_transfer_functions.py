import dsptoolbox as dsp
from os.path import join
import numpy as np
import pytest


class TestTransferFunctionsModule:
    y_m = dsp.Signal(join("examples", "data", "chirp_mono.wav"))
    y_st = dsp.Signal(join("examples", "data", "chirp_stereo.wav"))
    x = dsp.Signal(join("examples", "data", "chirp.wav"))
    fs = 5_000
    audio_multi = dsp.generators.noise("white", 2, 5_000, number_of_channels=3)

    def test_deconvolve(self):
        # Only functionality is tested here
        # Regularized
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            mode="regularized",
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            mode="regularized",
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            mode="regularized",
            start_stop_hz=[30, 15e3],
            threshold_db=None,
            padding=False,
            keep_original_length=False,
        )
        # Window
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            mode="window",
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            mode="window",
            start_stop_hz=[30, 15e3],
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        # Standard
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            mode="standard",
            start_stop_hz=None,
            threshold_db=None,
            padding=False,
            keep_original_length=False,
        )
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            mode="standard",
            start_stop_hz=None,
            threshold_db=None,
            padding=True,
            keep_original_length=False,
        )
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m,
            self.x,
            mode="standard",
            start_stop_hz=None,
            threshold_db=None,
            padding=True,
            keep_original_length=True,
        )

    def test_window_ir(self):
        # Only functionality
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)

        dsp.transfer_functions.window_ir(h, window_type="hann", at_start=True)
        dsp.transfer_functions.window_ir(h, window_type="hann", at_start=False)
        dsp.transfer_functions.window_ir(
            h, exp2_trim=None, window_type="hann", at_start=True
        )

        # Try window with extra parameters
        dsp.transfer_functions.window_ir(
            h, exp2_trim=None, window_type=("chebwin", 50), at_start=True
        )

    def test_window_centered_ir(self):
        # Only functionality
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)

        # ============ Even
        # Shorter
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) - 10, window_type="hann"
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        # Longer
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) + 10, window_type="hann"
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # Try window with extra parameters
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h), window_type=("gauss", 8)
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # ============= Odd
        # Shorter
        h.time_data = h.time_data[:-1]
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) - 10, window_type="hann"
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        # Longer
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) + 10, window_type="hann"
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # Try window with extra parameters
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h), window_type=("gauss", 8)
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # ============= Impulse on the second half, odd
        # Shorter
        h.time_data = h.time_data[::-1]
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) - 10, window_type="hann"
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        # Longer
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) + 10, window_type="hann"
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # Try window with extra parameters
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h), window_type=("gauss", 8)
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # ============= Impulse on the second half, even
        # Shorter
        h.time_data = h.time_data[:-1]
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) - 10, window_type="hann"
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])
        # Longer
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h) + 10, window_type="hann"
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

        # Try window with extra parameters
        h_, _ = dsp.transfer_functions.window_centered_ir(
            h, len(h), window_type=("gauss", 8)
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(h_.window[:, 0])

    def test_ir_to_filter(self):
        s = self.audio_multi.time_data[:200, 0]
        s = dsp.Signal(None, s, self.fs, signal_type="rir")
        f = dsp.transfer_functions.ir_to_filter(s, channel=0)
        b, _ = f.get_coefficients(mode="ba")
        assert np.all(b == s.time_data[:, 0])
        assert f.sampling_rate_hz == s.sampling_rate_hz

    def test_filter_to_ir(self):
        f = dsp.Filter(
            "fir",
            dict(order=216, freqs=1000, type_of_pass="highpass"),
            self.fs,
        )
        s = dsp.transfer_functions.filter_to_ir(f)
        assert s.time_data.shape[0] == 216 + 1

        with pytest.raises(AssertionError):
            f = dsp.Filter(
                "iir",
                dict(order=10, freqs=1000, type_of_pass="highpass"),
                self.fs,
            )
            dsp.transfer_functions.filter_to_ir(f)

    def test_compute_transfer_function(self):
        # Only functionality
        # Multi-channel
        dsp.transfer_functions.compute_transfer_function(
            self.y_st,
            self.x,
            mode="H1",
            window_length_samples=1024,
            spectrum_parameters=None,
        )
        # Single-channel with other windows
        h, h_numpy = dsp.transfer_functions.compute_transfer_function(
            self.y_m,
            self.x,
            mode="H2",
            window_length_samples=1024,
            spectrum_parameters=dict(window_type=("chebwin", 40)),
        )
        # Check that coherence is saved
        h.get_coherence()

    def test_spectral_average(self):
        # Only functionality is tested
        h = dsp.transfer_functions.spectral_deconvolve(self.y_st, self.x)
        # h.plot_phase()
        dsp.transfer_functions.spectral_average(h, True)
        # h1.plot_magnitude()
        dsp.transfer_functions.spectral_average(h, False)
        # h2.plot_magnitude()

    def test_min_phase_from_mag(self):
        # Only functionality is tested
        self.y_st.set_spectrum_parameters(method="standard")
        f, sp = self.y_st.get_spectrum()
        dsp.transfer_functions.min_phase_from_mag(
            sp, self.y_st.sampling_rate_hz
        )

    def test_lin_phase_from_mag(self):
        # Only functionality is tested here
        self.y_st.set_spectrum_parameters(method="standard")
        f, sp = self.y_st.get_spectrum()
        dsp.transfer_functions.lin_phase_from_mag(
            sp, self.y_st.sampling_rate_hz, group_delay_ms="minimum"
        )

    def test_group_delay(self):
        ir = dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            mode="regularized",
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        ir, _ = dsp.transfer_functions.window_ir(
            ir, window_type="hann", exp2_trim=12, at_start=True
        )
        # Check only that some result is produced, validity should be checked
        # somewhere else
        dsp.transfer_functions.group_delay(ir, method="matlab")
        dsp.transfer_functions.group_delay(ir, method="direct")

        # Single-channel plausibility check
        dsp.transfer_functions.group_delay(ir.get_channels(0))

    def test_minimum_phase(self):
        ir = dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            mode="regularized",
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        ir, _ = dsp.transfer_functions.window_ir(
            ir, window_type="hann", exp2_trim=12, at_start=True
        )
        # Check only that some result is produced, validity should be checked
        # somewhere else
        # Only works for some signal types
        f, min_phases = dsp.transfer_functions.minimum_phase(ir)
        assert len(f) == len(min_phases)

        f, min_phases = dsp.transfer_functions.minimum_phase(
            dsp.pad_trim(ir, len(ir) + 1)
        )
        assert len(f) == len(min_phases)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.minimum_phase(s1)
        # Single-channel plausibility check
        dsp.transfer_functions.minimum_phase(ir.get_channels(0))

    def test_minimum_group_delay(self):
        ir = dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            mode="regularized",
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        ir, _ = dsp.transfer_functions.window_ir(
            ir, window_type="hann", exp2_trim=12, at_start=True
        )
        # Check only that some result is produced, validity should be checked
        # somewhere else
        # Only works for some signal types
        dsp.transfer_functions.minimum_group_delay(ir)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.minimum_group_delay(s1)
        # Single-channel plausibility check
        dsp.transfer_functions.minimum_group_delay(ir.get_channels(0))

    def test_excess_group_delay(self):
        ir = dsp.transfer_functions.spectral_deconvolve(
            self.y_st,
            self.x,
            mode="regularized",
            start_stop_hz=None,
            threshold_db=-30,
            padding=False,
            keep_original_length=False,
        )
        ir, _ = dsp.transfer_functions.window_ir(
            ir, window_type="hann", exp2_trim=12, at_start=True
        )
        # Check only that some result is produced, validity should be checked
        # somewhere else
        # Only works for some signal types
        dsp.transfer_functions.excess_group_delay(ir)
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.excess_group_delay(s1)
        # Single-channel plausibility check
        dsp.transfer_functions.excess_group_delay(ir.get_channels(0))

    def test_min_phase_ir(self):
        # Only functionality, computation is done using scipy's minimum phase
        s = dsp.Signal(join("examples", "data", "rir.wav"), signal_type="rir")
        s = dsp.transfer_functions.min_phase_ir(s)

    def test_combine_ir(self):
        s = dsp.Signal(join("examples", "data", "rir.wav"), signal_type="rir")
        dsp.transfer_functions.combine_ir_with_dirac(
            s, 1000, True, normalization=None
        )
        dsp.transfer_functions.combine_ir_with_dirac(
            s, 1000, False, normalization=None
        )
        dsp.transfer_functions.combine_ir_with_dirac(
            s, 1000, False, normalization="energy"
        )

    def test_find_ir_latency(self):
        ir = dsp.generators.dirac(self.fs, sampling_rate_hz=self.fs)
        ir.signal_type = "ir"
        delay_seconds = 0.00133  # Some value to have a fractional delay
        delay_samples = self.fs * delay_seconds
        ir = dsp.fractional_delay(ir, delay_seconds)
        output = dsp.transfer_functions.find_ir_latency(ir).squeeze()

        assert np.isclose(delay_samples, output, atol=0.4)

    def test_window_frequency_dependent(self):
        s = dsp.Signal(join("examples", "data", "rir.wav"), signal_type="rir")
        f, sp = dsp.transfer_functions.window_frequency_dependent(
            s, 10, 0, [100, 1000]
        )

        fig, ax = s.plot_magnitude(normalize=None)
        ax.plot(f, 20 * np.log10(np.abs(sp)))
        print()

    def test_warp_ir(self):
        # Only functionality
        s = dsp.Signal(join("examples", "data", "rir.wav"), signal_type="rir")
        dsp.transfer_functions.warp_ir(s, -0.6, True, 2**8)
        dsp.transfer_functions.warp_ir(s, 0.6, False, 2**8)

    def test_harmonics_from_chirp_ir(self):
        # Only functionality
        ir = dsp.Signal(
            "/Users/nico/Downloads/tests/some_new_ir.wav",
            signal_type="rir",
        )
        harms = dsp.transfer_functions.harmonics_from_chirp_ir(
            ir,
            chirp_range_hz=[20, 20e3],
            chirp_length_seconds=2,
            n_harmonics=2,
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        for ind, h in enumerate(harms):
            freqs = np.fft.rfftfreq(len(h), 1 / ir.sampling_rate_hz)
            s = np.fft.rfft(h)
            ax.semilogx(freqs, 20 * np.log10(np.abs(s)))
        plt.show()
