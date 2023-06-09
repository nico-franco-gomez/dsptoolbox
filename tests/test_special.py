import dsptoolbox as dsp
import pytest
from os.path import join
from numpy import linspace


class TestSpecialModule():
    speech = dsp.Signal(join('examples', 'data', 'speech.flac'))

    def test_cepstrum(self):
        # Only functionality
        dsp.special.cepstrum(self.speech, mode='power')
        dsp.special.cepstrum(self.speech, mode='real')
        dsp.special.cepstrum(self.speech, mode='complex')

    def test_log_mel_spectrogram(self):
        # Only functionality
        dsp.special.log_mel_spectrogram(
            self.speech, range_hz=None, n_bands=40, generate_plot=False,
            stft_parameters=None)
        dsp.special.log_mel_spectrogram(
            self.speech, range_hz=[20, 20e3], n_bands=10, generate_plot=False,
            stft_parameters=None)
        dsp.special.log_mel_spectrogram(
            self.speech, range_hz=None, n_bands=40, generate_plot=True,
            stft_parameters=None)
        dsp.special.log_mel_spectrogram(
            self.speech, range_hz=None, n_bands=40, generate_plot=False,
            stft_parameters=dict(window_type=('chebwin', 40)))

        # Raise Assertion error if set range is larger than the nyquist
        # frequency
        with pytest.raises(AssertionError):
            dsp.special.log_mel_spectrogram(
                self.speech, range_hz=[20, 30e3], n_bands=10,
                generate_plot=False, stft_parameters=None)

    def test_mel_filters(self):
        # Only functionality
        f = linspace(0, 24000, 2048)
        dsp.special.mel_filterbank(
            f_hz=f, range_hz=None, n_bands=30, normalize=False)
        dsp.special.mel_filterbank(
            f_hz=f, range_hz=[1e3, 5e3], n_bands=10, normalize=False)
        dsp.special.mel_filterbank(
            f_hz=f, range_hz=None, n_bands=30, normalize=True)

    def test_plot_waterfall(self):
        # Only functionality
        dsp.special.plot_waterfall(self.speech)
        with pytest.raises(AssertionError):
            dsp.special.plot_waterfall(self.speech, dynamic_range_db=-10)
        dsp.special.plot_waterfall(
            self.speech, stft_parameters=dict(window_type=('chebwin', 40)))

    def test_mfcc(self):
        # Only functionality
        t, f, s = self.speech.get_spectrogram()

        mels, _ = dsp.special.mel_filterbank(f, [20, 10e3], n_bands=4)
        t, mel, mf, fig, ax = dsp.special.mfcc(self.speech, mel_filters=mels)
        t, mel, mf = dsp.special.mfcc(self.speech, generate_plot=False)
