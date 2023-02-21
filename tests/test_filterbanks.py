import dsptoolbox as dsp
import pytest


class TestFilterbanksModule():
    def test_linkwitz(self):
        # Only functionality
        fb = dsp.filterbanks.linkwitz_riley_crossovers(
            [500, 1000], order=4, sampling_rate_hz=5_000)
        with pytest.raises(AssertionError):
            dsp.filterbanks.linkwitz_riley_crossovers(
                [500, 1000], order=[2, 4, 4], sampling_rate_hz=5_000)
        with pytest.raises(AssertionError):
            dsp.filterbanks.linkwitz_riley_crossovers(
                [500, 5000], order=4, sampling_rate_hz=5_000)

        # Test filtering
        s = dsp.generators.noise('white', sampling_rate_hz=5_000)
        fb.filter_signal(s, mode='parallel')

    def test_reconstructing_fractional_octave_bands(self):
        # Only functionality
        dsp.filterbanks.reconstructing_fractional_octave_bands(
            octave_fraction=1, frequency_range_hz=[63, 1024], overlap=0.5,
            slope=1, n_samples=2**10, sampling_rate_hz=5_000)

    def test_auditory_filters_gammatone(self):
        # Only functionality
        fb = dsp.filterbanks.auditory_filters_gammatone(
            frequency_range_hz=[500, 1000], sampling_rate_hz=4_000)
        with pytest.raises(AssertionError):
            dsp.filterbanks.auditory_filters_gammatone(
                frequency_range_hz=[500, 3000], sampling_rate_hz=4_000)

        # Reconstruct signal
        s = dsp.generators.noise(type_of_noise='pink', sampling_rate_hz=4_000)
        mb = fb.filter_signal(s)
        fb.reconstruct(mb)

    def test_qmf_crossover(self):
        # Only functionality
        fs_hz = 4_000
        ny_hz = fs_hz//2
        lp = dsp.Filter('fir', {'order': 10, 'freqs': ny_hz//2,
                                'type_of_pass': 'lowpass'},
                        sampling_rate_hz=fs_hz)
        fb = dsp.filterbanks.qmf_crossover(lp)
        s = dsp.generators.noise('white', sampling_rate_hz=fs_hz)
        fb.filter_signal(
            s, mode='parallel', activate_zi=False, downsample=False)
        fb.filter_signal(
            s, mode='parallel', activate_zi=True, downsample=False)
        mb_ = fb.filter_signal(
            s, mode='parallel', activate_zi=False, downsample=True)

        # Reconstruction
        fb.reconstruct_signal(mb_, upsample=True)
