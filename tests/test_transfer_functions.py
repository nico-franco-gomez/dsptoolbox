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
        h.time_data = np.roll(
            h.time_data, 256 - np.argmax(np.abs(h.time_data)), axis=0
        )
        h = dsp.pad_trim(h, 2**13)

        dsp.transfer_functions.window_ir(
            h, 2**11, window_type="hann", at_start=True
        )
        dsp.transfer_functions.window_ir(
            h, 2**11, window_type="hann", at_start=False
        )
        dsp.transfer_functions.window_ir(
            h, 2**15, window_type="hann", at_start=True
        )
        # Try window with extra parameters
        dsp.transfer_functions.window_ir(
            h, 2**12, window_type=("kaiser", 10), at_start=True
        )
        dsp.transfer_functions.window_ir(
            h,
            2**12,
            adaptive=False,
            window_type=("kaiser", 10),
            at_start=True,
        )
        dsp.transfer_functions.window_ir(
            h,
            2**12,
            adaptive=False,
            window_type=("kaiser", 10),
            at_start=True,
            offset_samples=200,
        )
        dsp.transfer_functions.window_ir(
            h,
            2**12,
            adaptive=False,
            window_type=["hann", "hamming"],
            at_start=False,
            offset_samples=200,
        )
        dsp.transfer_functions.window_ir(
            h,
            2**12,
            adaptive=True,
            window_type=["hann", ("kaiser", 10)],
            at_start=False,
            offset_samples=200,
            left_to_right_flank_length_ratio=0.5,
        )
        dsp.transfer_functions.window_ir(
            h,
            2**15,
            adaptive=True,
            window_type=["hann", ("kaiser", 10)],
            at_start=False,
            offset_samples=200,
            left_to_right_flank_length_ratio=0.5,
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
            h, len(h), window_type=("gauss", 5000)
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

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
            h, len(h), window_type=("gauss", 5000)
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

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
            h, len(h), window_type=("gauss", 5000)
        )

        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

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
            h, len(h), window_type=("gauss", 5000)
        )
        assert np.argmax(h_.time_data[:, 0]) == np.argmax(
            h_.window[:, 0]
        ) and np.argmax(h.time_data[:, 0]) == np.argmax(h_.time_data[:, 0])

        # ============= Impulse in the middle, no changing lengths, even
        d = dsp.generators.dirac(1024, 512, sampling_rate_hz=self.fs)
        d2, _ = dsp.transfer_functions.window_centered_ir(d, len(d))
        assert (
            np.argmax(d.time_data[:, 0]) == np.argmax(d2.window[:, 0])
            and len(d) == len(d2)
            and np.all(np.isclose(d.time_data, d2.time_data))
        )

        # ============= Impulse in the middle, no changing lengths, odd
        d = dsp.generators.dirac(1025, 513, sampling_rate_hz=self.fs)
        d2, _ = dsp.transfer_functions.window_centered_ir(d, len(d))
        assert (
            np.argmax(d.time_data[:, 0]) == np.argmax(d2.window[:, 0])
            and len(d) == len(d2)
            and np.all(np.isclose(d.time_data, d2.time_data))
        )

    def test_ir_to_filter(self):
        s = self.audio_multi.time_data[:200, 0]
        s = dsp.ImpulseResponse(None, s, self.fs)
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
        h, h_numpy, coh = dsp.transfer_functions.compute_transfer_function(
            self.y_m,
            self.x,
            mode="H2",
            window_length_samples=1024,
            spectrum_parameters=dict(window_type=("chebwin", 40)),
        )
        # Check that coherence is saved
        h.get_coherence()
        h.plot_coherence()
        # dsp.plots.show()

    def test_average_irs(self):
        # Only functionality is tested
        h = dsp.transfer_functions.spectral_deconvolve(self.y_st, self.x)
        # h.plot_phase()
        dsp.transfer_functions.average_irs(h, normalize_energy=True)
        # h1.plot_magnitude()
        dsp.transfer_functions.average_irs(h, normalize_energy=False)
        dsp.transfer_functions.average_irs(h, mode="spectral")
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
            ir, window_type="hann", total_length_samples=2**12, at_start=True
        )
        # Check only that some result is produced, validity should be checked
        # somewhere else
        dsp.transfer_functions.group_delay(ir, method="matlab")
        dsp.transfer_functions.group_delay(ir, method="direct")

        dsp.transfer_functions.group_delay(ir, method="matlab", smoothing=4)
        dsp.transfer_functions.group_delay(
            ir, method="direct", smoothing=4, remove_ir_latency=True
        )

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
            ir, 2**12, window_type="hann", at_start=True
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
            ir, 2**12, window_type="hann", at_start=True
        )
        # Check only that some result is produced, validity should be checked
        # somewhere else
        # Only works for some signal types
        dsp.transfer_functions.minimum_group_delay(ir)
        dsp.transfer_functions.minimum_group_delay(ir, smoothing=3)
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
            ir, 2**12, window_type="hann", at_start=True
        )
        # Check only that some result is produced, validity should be checked
        # somewhere else
        # Only works for some signal types
        dsp.transfer_functions.excess_group_delay(ir)
        dsp.transfer_functions.excess_group_delay(
            ir, smoothing=3, remove_ir_latency=True
        )
        with pytest.raises(AssertionError):
            s1 = dsp.Signal(None, ir.time_data, ir.sampling_rate_hz)
            dsp.transfer_functions.excess_group_delay(s1)
        # Single-channel plausibility check
        dsp.transfer_functions.excess_group_delay(ir.get_channels(0))

    def test_min_phase_ir(self):
        # Only functionality, computation is done using scipy's minimum phase
        s = dsp.ImpulseResponse(join("examples", "data", "rir.wav"))
        s = dsp.transfer_functions.min_phase_ir(s)

    def test_combine_ir(self):
        s = dsp.ImpulseResponse(join("examples", "data", "rir.wav"))
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
        delay_seconds = 0.00133  # Some value to have a fractional delay
        delay_samples = self.fs * delay_seconds
        ir = dsp.fractional_delay(ir, delay_seconds)
        output = dsp.transfer_functions.find_ir_latency(ir).squeeze()

        assert np.isclose(delay_samples, output, atol=0.4)

        ir = dsp.ImpulseResponse(join("examples", "data", "rir.wav"))
        assert dsp.transfer_functions.find_ir_latency(ir) > 0

    def test_window_frequency_dependent(self):
        s = dsp.ImpulseResponse(join("examples", "data", "rir.wav"))
        f, sp = dsp.transfer_functions.window_frequency_dependent(
            s, 10, 0, [100, 1000]
        )

        fig, ax = s.plot_magnitude(normalize=None)
        ax.plot(f, 20 * np.log10(np.abs(sp)))
        print()

    def test_harmonics_from_chirp_ir(self):
        # Only functionality
        ir = dsp.ImpulseResponse(join("examples", "data", "rir.wav"))
        dsp.transfer_functions.harmonics_from_chirp_ir(
            ir,
            chirp_range_hz=[20, 20e3],
            chirp_length_s=2,
            n_harmonics=2,
        )

    def test_harmonic_distortion_analysis(self):
        # Only functionality
        ir = dsp.ImpulseResponse(join("examples", "data", "rir.wav"))
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

    def test_trim_rir(self):
        # Only functionality
        ir = dsp.ImpulseResponse(join("examples", "data", "rir.wav"))
        dsp.transfer_functions.trim_ir(ir)
        # Start offset way longer than rir (should be clipped to 0)
        assert (
            ir.time_data[0, 0]
            == dsp.transfer_functions.trim_ir(ir, start_offset_s=3)[
                0
            ].time_data[0, 0]
        )
