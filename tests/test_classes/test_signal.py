"""
Tests for the Signal class.
"""

import os
import pickle
import tempfile
from dataclasses import FrozenInstanceError
from os.path import join

import numpy as np
import pytest
import scipy.signal as sig
import soundfile as sf
from matplotlib.pyplot import close

import dsptoolbox as dsp

_rng = np.random.default_rng(4)


class TestSignal:
    fs = 44100
    length_samp = 2 * fs
    channels = 4
    time_vec = _rng.normal(0, 0.1, (length_samp, channels))
    imag = _rng.normal(0, 0.1, (length_samp, channels))
    complex_time_vec = time_vec + 1j * imag

    def test_importing_from_file(self):
        path = join(os.path.dirname(__file__), "..", "..", "example_data", "chirp.wav")
        s = dsp.Signal(path)
        _ = s.number_of_channels

    def test_from_file_nonexistent_path_raises(self):
        with pytest.raises(sf.LibsndfileError):
            dsp.Signal.from_file(join(os.path.dirname(__file__), "does-not-exist.wav"))

    def test_save_signal_round_trip_wav_flac_pkl(self):
        """`save_signal` no longer takes a `mode` parameter -- the saving
        format is inferred purely from `path`'s extension. Round-tripping
        through each supported format should recover the original signal:
        exactly for `.pkl` (a plain pickle of the object) and up to
        quantization error for `.wav`/`.flac` (PCM/float encoding via
        soundfile).

        """
        rng = np.random.default_rng(0)
        s = dsp.Signal(None, rng.normal(0, 0.1, (2000, 2)), self.fs)

        with tempfile.TemporaryDirectory() as d:
            # Float32 PCM: near-exact round trip
            s.save_signal(join(d, "x.wav"), bit_depth=32)
            reloaded_wav = dsp.Signal.from_file(join(d, "x.wav"))
            assert reloaded_wav.sampling_rate_hz == s.sampling_rate_hz
            np.testing.assert_allclose(reloaded_wav.time_data, s.time_data, atol=1e-6)

            # 16-bit PCM flac: quantization noise expected, format still
            # inferred correctly from ".flac"
            s.save_signal(join(d, "x.flac"), bit_depth=16)
            reloaded_flac = dsp.Signal.from_file(join(d, "x.flac"))
            np.testing.assert_allclose(reloaded_flac.time_data, s.time_data, atol=1e-4)

            # Pickle: exact round trip of the whole object
            s.save_signal(join(d, "x.pkl"))
            with open(join(d, "x.pkl"), "rb") as f:
                reloaded_pkl = pickle.load(f)
            np.testing.assert_array_equal(reloaded_pkl.time_data, s.time_data)
            assert reloaded_pkl.sampling_rate_hz == s.sampling_rate_hz

    def test_save_signal_invalid_path_or_bit_depth_raises(self):
        s = dsp.Signal(None, np.zeros((100, 1)), self.fs)
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError):
                s.save_signal(join(d, "x.mp3"))  # unsupported format
            with pytest.raises(ValueError):
                s.save_signal(join(d, "x"))  # no extension to infer from
            with pytest.raises(ValueError):
                s.save_signal(join(d, "x.wav"), bit_depth=8)  # invalid bit depth

    def test_creating_signal_from_vector(self):
        # Multichannel
        s = dsp.Signal(None, self.complex_time_vec, self.fs)
        real_cond = np.all(s.time_data == self.time_vec)
        imag_cond = np.all(s.time_data_imaginary == self.imag)
        assert real_cond and imag_cond

        # Single channel
        one_ch = self.time_vec[:, 0]
        one_ch_c = self.imag[:, 0]
        s = dsp.Signal(None, one_ch + 1j * one_ch_c, self.fs)
        real_cond = np.all(s.time_data == one_ch[..., None])
        imag_cond = np.all(s.time_data_imaginary == one_ch_c[..., None])
        assert real_cond and imag_cond

        # Broadcasting with too many dimensions
        r = _rng.normal(0, 0.1, (self.length_samp, self.channels, 1))
        s = dsp.Signal(None, r, self.fs)

        # Not broadcastable to time data vector
        with pytest.raises(AssertionError):
            r = _rng.normal(0, 0.1, (self.length_samp, self.channels, 4))
            s = dsp.Signal(None, r, self.fs)

        li = [self.time_vec[:, i] for i in range(self.time_vec.shape[1])]
        s = dsp.Signal(None, li, self.fs)

        tu = tuple(self.time_vec.T)
        s = dsp.Signal(None, tu, self.fs)

        # Not broadcastable to time data vector (with lists)
        with pytest.raises(AssertionError):
            r = _rng.normal(0, 0.1, (self.length_samp, self.channels, 4))
            r = list(r)
            s = dsp.Signal(None, r, self.fs)

    def test_get_spectrum(self):
        sp = np.fft.rfft(self.time_vec, axis=0)

        s = dsp.Signal(None, self.time_vec, self.fs)
        s = s.set_spectrum_parameters(
            method=dsp.SpectrumMethod.FFT,
            scaling=dsp.SpectrumScaling.FFTBackward,
            pad_to_fast_length=False,
        )
        _, sp_sig = s.get_spectrum()
        np.testing.assert_allclose(sp, sp_sig)

        s = s.set_spectrum_parameters(
            method=dsp.SpectrumMethod.FFT,
            scaling=dsp.SpectrumScaling.PowerSpectrum,
            pad_to_fast_length=False,
        )
        _, sp_sig = s.get_spectrum()
        _, sp_reference = sig.periodogram(
            self.time_vec.squeeze(),
            fs=self.fs,
            detrend=False,
            scaling="spectrum",
            axis=0,
        )
        assert np.all(np.isclose(sp_reference, sp_sig.squeeze()))

        s = s.set_spectrum_parameters(
            method=dsp.SpectrumMethod.FFT,
            scaling=dsp.SpectrumScaling.PowerSpectralDensity,
            pad_to_fast_length=False,
        )
        _, sp_sig = s.get_spectrum()
        _, sp_reference = sig.periodogram(
            self.time_vec.squeeze(),
            axis=0,
            detrend=False,
            scaling="density",
            fs=self.fs,
        )
        assert np.all(np.isclose(sp_reference, sp_sig.squeeze()))

        s = s.set_spectrum_parameters(
            method=dsp.SpectrumMethod.FFT,
            scaling=dsp.SpectrumScaling.AmplitudeSpectrum,
            pad_to_fast_length=False,
            smoothing=3,
        )
        s.get_spectrum()

    def test_managing_channels(self):
        new_ch = _rng.normal(0, 0.1, (self.length_samp, 1))
        t_vec = np.append(self.time_vec, new_ch, axis=1)
        s = dsp.Signal(None, self.time_vec.copy(), self.fs)
        s = s.add_channel(None, new_ch, s.sampling_rate_hz)
        assert np.all(t_vec == s.time_data)

        s = s.remove_channel()
        assert np.all(self.time_vec == s.time_data)

        with pytest.raises(AssertionError):
            s.remove_channel(self.channels + 10)

        ch = s.get_channels(0)
        assert np.all(self.time_vec[:, 0][..., None] == ch.time_data)

        with pytest.raises(IndexError):
            s.get_channels(self.channels + 10)

        new_order = np.arange(0, self.channels)[::-1]
        assert np.all(self.time_vec[:, ::-1] == s.swap_channels(new_order).time_data)

        with pytest.raises(AssertionError):
            # Order vector with too few elements
            s.swap_channels(new_order[:-2])
        with pytest.raises(AssertionError):
            # Order vector with too many elements
            s.swap_channels(np.append(new_order, new_order))
        with pytest.raises(AssertionError):
            # Order vector with repeated elements
            s.swap_channels(np.append(new_order[:-1], new_order[0]))

    def test_setting_properties(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)

        fs = 22000
        s.sampling_rate_hz = fs
        assert fs == s.sampling_rate_hz

        with pytest.raises(AssertionError):
            s.sampling_rate_hz = 44100.5

        assert s.number_of_channels == self.channels

        s.spectrum_method = dsp.SpectrumMethod.FFT
        assert s.spectrum_method == dsp.SpectrumMethod.FFT
        s.spectrum_scaling = dsp.SpectrumScaling.FFTOrthogonal
        assert s.spectrum_scaling == dsp.SpectrumScaling.FFTOrthogonal

        _ = s.number_of_channels
        _ = s.length_samples
        _ = s.length_seconds
        _ = s.time_vector_s

        with pytest.raises(AttributeError):
            s.number_of_channels = 10
        with pytest.raises(AttributeError):
            s.length_samples = 10
        with pytest.raises(AttributeError):
            s.length_seconds = 10.0
        with pytest.raises(AttributeError):
            s.time_vector_s = np.array([0.0, 1.0])

    def test_ir_latency_removal_members(self):
        """`Custom` carries the delay, so the parameter has one type and
        `NoRemoval` replaces the `None` that used to mean "leave it".

        """
        delay_samples = 64
        ir = dsp.ImpulseResponse.from_time_data(
            np.eye(512, 2, -delay_samples), self.fs
        ).set_spectrum_parameters(method=dsp.SpectrumMethod.FFT)

        removal = dsp.IrLatencyRemoval
        fig, ax_untouched = ir.plot_phase(remove_ir_latency=removal.NoRemoval)
        untouched = ax_untouched.lines[0].get_ydata()
        close(fig)

        # The peak sits exactly on a sample, so removing the estimated delay
        # and removing it explicitly must agree
        _, ax_estimated = ir.plot_phase(remove_ir_latency=removal.Peak)
        _, ax_custom = ir.plot_phase(
            remove_ir_latency=removal.Custom.with_delay_samples(delay_samples)
        )
        estimated = ax_estimated.lines[0].get_ydata()
        np.testing.assert_allclose(estimated, ax_custom.lines[0].get_ydata(), atol=1e-9)
        assert not np.allclose(estimated, untouched)
        close("all")

        # A mismatched number of delays is rejected
        with pytest.raises(ValueError):
            ir.plot_phase(
                remove_ir_latency=removal.Custom.with_delay_samples([64, 60, 55])
            )
        with pytest.raises(ValueError):
            removal.Peak.with_delay_samples(10)
        with pytest.raises(ValueError):
            removal.Custom.with_delay_samples(np.zeros((2, 2)))

    def test_plot_csm_with_phase(self):
        """`with_phase` used to be accepted and ignored, so the phase was
        drawn on a twin axis no matter what.

        """
        s = dsp.Signal(time_data=self.time_vec[:, :2], sampling_rate_hz=self.fs)

        fig, _ = s.plot_csm(with_phase=True)
        n_axes_with_phase = len(fig.axes)
        close(fig)

        fig, _ = s.plot_csm(with_phase=False)
        assert len(fig.axes) < n_axes_with_phase
        close(fig)

    def test_plot_generation(self):
        s = dsp.ImpulseResponse(time_data=self.time_vec, sampling_rate_hz=self.fs)
        s.plot_magnitude()
        s.plot_magnitude(show_info_box=True)
        s.plot_time()
        s.plot_spectrogram(channel_number=0, log_freqs=True)
        s.plot_csm()
        s.plot_csm(with_phase=False)
        s.plot_spl(False)
        s.plot_spl(True)

        s = s.set_spectrum_parameters(method=dsp.SpectrumMethod.FFT)
        s.plot_phase()
        s.plot_phase(
            unwrap=True,
            smoothing=4,
            remove_ir_latency=dsp.IrLatencyRemoval.NoRemoval,
        )
        s.plot_phase(remove_ir_latency=dsp.IrLatencyRemoval.MinimumPhase)
        s.plot_phase(remove_ir_latency=dsp.IrLatencyRemoval.Peak)
        s.plot_phase(
            remove_ir_latency=dsp.IrLatencyRemoval.Custom.with_delay_samples(
                [10] * s.number_of_channels
            )
        )
        # A single delay applies to every channel
        s.plot_phase(
            remove_ir_latency=dsp.IrLatencyRemoval.Custom.with_delay_samples(10.5)
        )
        # Custom without a bound delay is not a valid value
        with pytest.raises(ValueError):
            s.plot_phase(remove_ir_latency=dsp.IrLatencyRemoval.Custom)
        s.plot_group_delay()

        # Welch's method is incompatible with phase plotting
        with pytest.raises(AssertionError):
            s = s.set_spectrum_parameters(
                method=dsp.SpectrumMethod.WelchPeriodogram,
                window_length_samples=32,
            )
            s.plot_phase()

        d = dsp.generators.dirac(
            length_samples=1024, delay_samples=512, sampling_rate_hz=self.fs
        )
        d, _ = dsp.transfer_functions.window_centered_ir(d, len(d))
        d = dsp.transforms.hilbert(d)
        d.plot_time()
        d.plot_spl()
        close("all")

    def test_get_power_spectrum_welch(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        s.spectrum_scaling = dsp.SpectrumScaling.FFTBackward
        s.spectrum_method = dsp.SpectrumMethod.WelchPeriodogram
        s.get_spectrum()
        s.spectrum_method = dsp.SpectrumMethod.FFT
        s.get_spectrum()

        s.spectrum_scaling = dsp.SpectrumScaling.PowerSpectralDensity
        s.spectrum_method = dsp.SpectrumMethod.WelchPeriodogram
        s.get_spectrum()
        s.spectrum_method = dsp.SpectrumMethod.FFT
        s.get_spectrum()

    def test_get_csm(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        s.spectrum_scaling = dsp.SpectrumScaling.FFTBackward
        s.spectrum_method = dsp.SpectrumMethod.WelchPeriodogram
        s.get_csm()
        s.spectrum_method = dsp.SpectrumMethod.FFT
        s.get_csm()

        s.spectrum_scaling = dsp.SpectrumScaling.PowerSpectralDensity
        s.spectrum_method = dsp.SpectrumMethod.WelchPeriodogram
        s.get_csm()
        s.spectrum_method = dsp.SpectrumMethod.FFT
        s.get_csm()

    def test_get_stft(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        # Use parameters just like librosa for validation
        s = s.set_spectrogram_parameters(
            window_length_samples=1024,
            window_type=dsp.Window.Hann,
            overlap_percent=50,
            fft_length_samples=4096,
            detrend=False,
            padding=False,
            scaling=dsp.SpectrumScaling.FFTBackward,
        )
        t, f, stft = s.get_spectrogram()
        s = s.set_spectrogram_parameters(
            window_length_samples=1024,
            window_type=dsp.Window.Hann,
            overlap_percent=50,
            fft_length_samples=None,
            detrend=False,
            padding=False,
            scaling=dsp.SpectrumScaling.PowerSpectrum,
        )
        t, f, stft = s.get_spectrogram()

        # Validate result with librosa library if installed
        try:
            import librosa

            y = librosa.stft(
                self.time_vec[:, 0],
                n_fft=1024,
                hop_length=1024 // 2,
                window="hann",
                center=False,
            )
            # There are some extra frames in the dsptoolbox version...
            assert np.all(np.isclose(stft[:, : y.shape[1], 0], y))
        except ModuleNotFoundError as e:
            print(e)
            pass
        except Exception as e:
            print(e)
            raise AssertionError() from e

    def test_copying_signal(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        s.copy()

    def test_show_info(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        s.show_info()
        print(s)

    def test_time_vec(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        t = s.time_vector_s
        le = s.time_data.shape[0]
        t_ = np.arange(le) / self.fs
        np.testing.assert_almost_equal(t, t_)

    def test_length_signal(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        assert len(s) == s.time_data.shape[0]
        assert s.length_samples == len(s)
        assert s.length_seconds == len(s) / s.sampling_rate_hz
        # The last sample sits one sampling period before the signal's end
        assert s.time_vector_s[-1] == (len(s) - 1) / s.sampling_rate_hz

    def test_constrain_amplitude(self):
        t = _rng.normal(0, 1, 200)
        s = dsp.Signal(None, t, sampling_rate_hz=100, constrain_amplitude=True)
        assert np.all(s.time_data <= 1)

        s = dsp.Signal(None, t, sampling_rate_hz=100, constrain_amplitude=False)
        assert np.all(t == s.time_data.squeeze())

    def test_sum_channels(self):
        n = _rng.normal(0, 0.01, (300, 2))
        nn = dsp.Signal.from_time_data(n, 10_000)
        np.testing.assert_array_equal(
            nn.sum_channels().time_data, np.sum(n, axis=1, keepdims=True)
        )

    def test_copy_with_new_time_data(self):
        n = dsp.Signal.from_time_data(self.time_vec, self.fs, False)

        n.spectrum_method = dsp.SpectrumMethod.FFT
        n.spectrum_scaling = dsp.SpectrumScaling.PowerSpectrum
        n = n.set_spectrogram_parameters(256, window_type=dsp.Window.Blackman)
        n2 = n.copy_with_new_time_data(np.zeros((100, 1)))

        assert n2.spectrum_scaling == dsp.SpectrumScaling.PowerSpectrum
        assert n2.spectrum_method == dsp.SpectrumMethod.FFT
        assert n2.spectrogram_parameters.window_length_samples == 256
        assert n2.spectrogram_parameters.window_type == dsp.Window.Blackman
        assert n2.constrain_amplitude == n.constrain_amplitude
        assert n2.time_data_imaginary == n.time_data_imaginary

        # Complex signal
        n_comp = dsp.Signal.from_time_data(self.complex_time_vec, self.fs, True)
        n2_comp = n_comp.copy_with_new_time_data(np.zeros((100, 1)))
        assert not n2_comp.is_complex_signal

        # New time data should not share memory with the original
        n = dsp.Signal.from_time_data(np.zeros((100, 2)), self.fs)
        n2 = n.copy_with_new_time_data(n.time_data[:, 0])
        n.time_data[0, ...] = 1.0
        assert np.all(n2.time_data[0] == 0.0)

    def test_spectrum_smoothing_invalidates_cache(self):
        """Setting the smoothing must take effect even when caching is on."""
        s = dsp.Signal(
            None, np.random.default_rng(0).normal(0, 0.1, (4096, 1)), self.fs
        )
        s.activate_cache = True
        s.spectrum_method = dsp.SpectrumMethod.FFT
        _, without_smoothing = s.get_spectrum()
        s.spectrum_smoothing = 3
        _, with_smoothing = s.get_spectrum()
        assert not np.array_equal(without_smoothing, with_smoothing)

    def test_metadata_str_has_underline(self):
        s = dsp.Signal(None, np.zeros((128, 1)), self.fs)
        lines = s.metadata_str.splitlines()
        assert lines[0] == "Signal:"
        assert lines[1] == "-" * len(lines[0])

    def test_plot_spectrogram_skips_dc_bin(self):
        s = dsp.Signal(
            None, np.random.default_rng(0).normal(0, 0.1, (8192, 1)), self.fs
        )
        fig, ax = s.plot_spectrogram()
        assert ax is not None
        close(fig)

    def test_spectrogram_time_vector_follows_hop_size(self):
        s = dsp.Signal(
            None, np.random.default_rng(0).normal(0, 0.1, (8192, 1)), self.fs
        )
        window_length = s.spectrogram_parameters.window_length_samples
        overlap = int(
            s.spectrogram_parameters.overlap_percent / 100 * window_length + 0.5
        )
        step = window_length - overlap

        t, _, _ = s.get_spectrogram()

        np.testing.assert_allclose(np.diff(t), step / self.fs)
        np.testing.assert_allclose(t[0], ((window_length - 1) / 2 - overlap) / self.fs)

    def test_spectrogram_time_vector_locates_a_transient(self):
        td = np.zeros((16384, 1))
        peak_sample = 8000
        td[peak_sample, 0] = 1.0
        s = dsp.Signal(None, td, self.fs)

        t, _, stft = s.get_spectrogram()
        loudest_frame = np.argmax(np.sum(np.abs(stft[..., 0]) ** 2.0, axis=0))

        window_length = s.spectrogram_parameters.window_length_samples
        assert abs(t[loudest_frame] - peak_sample / self.fs) < window_length / self.fs

    def test_spectrum_shape_is_independent_of_the_method(self):
        """`get_spectrum` must keep the channel axis for a single-channel
        signal, whichever method computes it."""
        s = dsp.Signal(None, _rng.normal(0, 0.1, (4096, 1)), self.fs)

        _, welch = s.set_spectrum_parameters(
            method=dsp.SpectrumMethod.WelchPeriodogram
        ).get_spectrum()
        _, fft = s.set_spectrum_parameters(method=dsp.SpectrumMethod.FFT).get_spectrum()

        assert welch.ndim == fft.ndim == 2
        assert welch.shape[1] == fft.shape[1] == 1

    def test_spectrum_parameters_are_a_frozen_dataclass(self):
        s = dsp.Signal(None, np.zeros((256, 1)), self.fs)
        parameters = s.spectrum_parameters
        assert parameters.method == dsp.SpectrumMethod.WelchPeriodogram

        with pytest.raises(FrozenInstanceError):
            parameters.method = dsp.SpectrumMethod.FFT

        new = s.with_spectrum_parameters(
            parameters.replace(method=dsp.SpectrumMethod.FFT)
        )
        assert new.spectrum_method == dsp.SpectrumMethod.FFT
        assert s.spectrum_method == dsp.SpectrumMethod.WelchPeriodogram
