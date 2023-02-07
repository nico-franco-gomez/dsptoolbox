import dsptoolbox as dsp
import pytest
from os.path import join

stereo_signal = dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))


class TestDistancesModule():
    sig1 = stereo_signal.get_channels(0)
    sig2 = stereo_signal.get_channels(1)

    def test_log_spectral(self):
        # Only functionality
        dsp.distances.log_spectral(
            self.sig1, self.sig2, method='standard', f_range_hz=[20, 20e3],
            energy_normalization=True, spectrum_parameters=None)
        dsp.distances.log_spectral(
            self.sig1, self.sig2, method='welch', f_range_hz=[200, 5000],
            energy_normalization=True, spectrum_parameters=None)
        with pytest.raises(AssertionError):
            dsp.distances.log_spectral(
                self.sig1, self.sig2, method='welch', f_range_hz=[20, 30e3],
                energy_normalization=True, spectrum_parameters=None)

        # Pass some spectrum parameters
        dsp.distances.log_spectral(
            self.sig1, self.sig2, method='welch', f_range_hz=[20, 20e3],
            energy_normalization=False,
            spectrum_parameters=dict(window_type=('chebwin', 40)))

    def test_itakura_saito(self):
        # Only functionality
        dsp.distances.itakura_saito(
            self.sig1, self.sig2, method='standard', f_range_hz=[20, 20e3],
            energy_normalization=True, spectrum_parameters=None)
        dsp.distances.itakura_saito(
            self.sig1, self.sig2, method='welch', f_range_hz=[200, 5000],
            energy_normalization=True, spectrum_parameters=None)
        with pytest.raises(AssertionError):
            dsp.distances.itakura_saito(
                self.sig1, self.sig2, method='welch', f_range_hz=[20, 30e3],
                energy_normalization=True, spectrum_parameters=None)

        # Pass some spectrum parameters
        dsp.distances.itakura_saito(
            self.sig1, self.sig2, method='welch', f_range_hz=[20, 20e3],
            energy_normalization=False,
            spectrum_parameters=dict(window_type=('chebwin', 40)))

    def test_snr(self):
        # Only functionality
        speech = dsp.Signal(join('examples', 'data', 'speech.flac'))
        noise = dsp.generators.noise(
            peak_level_dbfs=-30, sampling_rate_hz=speech.sampling_rate_hz)
        dsp.distances.snr(speech, noise)

    def test_si_sdr(self):
        dsp.distances.si_sdr(self.sig1, self.sig2)

    def test_fw_snr_seg(self):
        dsp.distances.fw_snr_seg(
            self.sig1, self.sig2, f_range_hz=[500, 4000],
            snr_range_db=[-10, 35], gamma=0.5)
