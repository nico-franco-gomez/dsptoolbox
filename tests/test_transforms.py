import dsptoolbox as dsp
import pytest
from os.path import join
import numpy as np
from matplotlib.pyplot import close
from scipy.signal import hilbert
import os


class TestTransformsModule:
    speech = dsp.Signal(
        join(os.path.dirname(__file__), "..", "example_data", "speech.flac")
    )

    def test_cepstrum(self):
        # Only functionality
        dsp.transforms.cepstrum(
            self.speech, mode=dsp.transforms.CepstrumType.Power
        )
        dsp.transforms.cepstrum(
            self.speech, mode=dsp.transforms.CepstrumType.Real
        )
        dsp.transforms.cepstrum(
            self.speech, mode=dsp.transforms.CepstrumType.Complex
        )

    def test_log_mel_spectrogram(self):
        # Only functionality
        dsp.transforms.log_mel_spectrogram(
            self.speech,
            range_hz=None,
            n_bands=40,
            generate_plot=False,
            stft_parameters=None,
        )
        dsp.transforms.log_mel_spectrogram(
            self.speech,
            range_hz=[20, 20e3],
            n_bands=10,
            generate_plot=False,
            stft_parameters=None,
        )
        dsp.transforms.log_mel_spectrogram(
            self.speech,
            range_hz=None,
            n_bands=40,
            generate_plot=True,
            stft_parameters=None,
        )
        dsp.transforms.log_mel_spectrogram(
            self.speech,
            range_hz=None,
            n_bands=40,
            generate_plot=False,
            stft_parameters=dict(
                window_type=dsp.Window.Chebwin.with_extra_parameter(40)
            ),
        )

        # Raise Assertion error if set range is larger than the nyquist
        # frequency
        with pytest.raises(AssertionError):
            dsp.transforms.log_mel_spectrogram(
                self.speech,
                range_hz=[20, 30e3],
                n_bands=10,
                generate_plot=False,
                stft_parameters=None,
            )
        close("all")

    def test_mel_filters(self):
        # Only functionality
        f = np.linspace(0, 24000, 2048)
        dsp.transforms.mel_filterbank(
            f_hz=f, range_hz=None, n_bands=30, normalize=False
        )
        dsp.transforms.mel_filterbank(
            f_hz=f, range_hz=[1e3, 5e3], n_bands=10, normalize=False
        )
        dsp.transforms.mel_filterbank(
            f_hz=f, range_hz=None, n_bands=30, normalize=True
        )

    def test_plot_waterfall(self):
        # Only functionality
        dsp.transforms.plot_waterfall(self.speech)
        with pytest.raises(AssertionError):
            dsp.transforms.plot_waterfall(self.speech, dynamic_range_db=-10)
        dsp.transforms.plot_waterfall(
            self.speech,
            stft_parameters=dict(
                window_type=dsp.Window.Chebwin.with_extra_parameter(40)
            ),
        )

    def test_mfcc(self):
        # Only functionality
        t, f, s = self.speech.get_spectrogram()

        mels, _ = dsp.transforms.mel_filterbank(f, [20, 10e3], n_bands=4)
        t, mel, mf, fig, ax = dsp.transforms.mfcc(
            self.speech, mel_filters=mels
        )
        t, mel, mf = dsp.transforms.mfcc(self.speech, generate_plot=False)

    def test_istft(self):
        # Test reconstruction fidelity
        # This would most likely fail if padding=False or detrend=True
        t, f, sp = self.speech.get_spectrogram()
        speech_rec = dsp.transforms.istft(sp, original_signal=self.speech)
        assert np.all(np.isclose(self.speech.time_data, speech_rec.time_data))

        speech_rec = dsp.transforms.istft(
            sp,
            parameters=self.speech._spectrogram_parameters,
            sampling_rate_hz=self.speech.sampling_rate_hz,
        )
        assert np.all(
            np.isclose(
                self.speech.time_data, speech_rec.time_data[: len(self.speech)]
            )
        )

        # With longer fft length than window
        wl = 512
        self.speech.set_spectrogram_parameters(
            window_length_samples=wl, fft_length_samples=wl * 2
        )
        t, f, sp = self.speech.get_spectrogram()
        speech_rec = dsp.transforms.istft(sp, original_signal=self.speech)
        assert np.all(np.isclose(self.speech.time_data, speech_rec.time_data))

        speech_rec = dsp.transforms.istft(
            sp,
            parameters=self.speech._spectrogram_parameters,
            sampling_rate_hz=self.speech.sampling_rate_hz,
        )
        assert np.all(
            np.isclose(
                self.speech.time_data, speech_rec.time_data[: len(self.speech)]
            )
        )

    def test_chroma(self):
        # Only functionality
        dsp.transforms.chroma_stft(self.speech.copy())
        dsp.transforms.chroma_stft(self.speech.copy(), plot_channel=0)

    def test_cwt(self):
        # Only functionality
        query_f = np.linspace(100, 200, 50)
        morlet = dsp.transforms.MorletWavelet(b=None, h=3, step=1e-3)
        dsp.transforms.cwt(self.speech, query_f, morlet, False)
        dsp.transforms.cwt(self.speech, query_f, morlet, True)

    def test_hilbert(self):
        # Results compared with scipy hilbert
        speech = self.speech.copy()
        speech.constrain_amplitude = False
        s = dsp.transforms.hilbert(speech)
        s = s.time_data + s.time_data_imaginary * 1j
        s2 = speech.time_data

        s2 = hilbert(s2, axis=0)
        np.testing.assert_allclose(s, s2)

        # Now other length (even vs. odd)
        s = dsp.transforms.hilbert(dsp.pad_trim(speech, len(speech) - 1))
        s = s.time_data + s.time_data_imaginary * 1j
        s2 = speech.time_data[:-1, ...]

        s2 = hilbert(s2, axis=0)
        np.testing.assert_allclose(s, s2)

        # Functionality for multiband signals
        s_mb = dsp.filterbanks.linkwitz_riley_crossovers(
            [400], 2, speech.sampling_rate_hz
        ).filter_signal(speech)
        dsp.transforms.hilbert(s_mb)

    def test_stereo_mid_side(self):
        sp = dsp.append_signals([self.speech, self.speech])
        sp_aft = dsp.transforms.stereo_mid_side(sp, True)
        sp_aft = dsp.transforms.stereo_mid_side(sp_aft, False)
        assert np.all(np.isclose(sp.time_data, sp_aft.time_data))

    def test_laguerre(self):
        # Only functionality
        sp = dsp.pad_trim(self.speech, 128)
        dsp.transforms.laguerre(sp, -0.7)

    def test_warp(self):
        # Only functionality
        s = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        dsp.transforms.warp(s, -0.6, True, 2**8)
        dsp.transforms.warp(s, 0.6, False, 2**8)

        # warping scales
        dsp.transforms.warp(s, "bark", False, 2**7)
        dsp.transforms.warp(s, "bark-", False, 2**7)
        dsp.transforms.warp(s, "erb", False, 2**7)
        dsp.transforms.warp(s, "erb-", False, 2**7)

    def test_warp_filter(self):
        # Only functionality
        i = dsp.Filter.iir_filter(
            3,
            100.0,
            type_of_pass=dsp.FilterPassType.Highpass,
            filter_design_method=dsp.IirDesignMethod.Butterworth,
            sampling_rate_hz=24000,
        )
        dsp.transforms.warp_filter(i, -0.6)
        # dsp.FilterBank([i, ii]).plot_magnitude(length_samples=2**14)
        dsp.transforms.warp_filter(i, 0.6)
        # dsp.plots.show()

    def test_lpc(self):
        # Only functionality
        speech = dsp.resample(self.speech, 8000)
        dsp.transforms.lpc(speech, 10, 1024, False, "burg", 512)
        dsp.transforms.lpc(speech, 10, 1024, True, "burg", 512)

        dsp.transforms.lpc(speech, 10, 1024, False, "yw", 512)
        dsp.transforms.lpc(speech, 10, 1024, True, "yw", 512)

    def test_dft(self):
        s = dsp.pad_trim(self.speech, 20_000)
        s.spectrum_method = dsp.SpectrumMethod.FFT
        f, spectrum = s.get_spectrum()

        select = slice(20, 40)
        dft = dsp.transforms.dft(s, f[select])
        np.testing.assert_allclose(dft, spectrum[select, ...])
