import numpy as np
import dsptoolbox as dsp
import pytest
from os.path import join


class TestRoomAcousticsModule():
    rir = dsp.Signal(join('examples', 'data', 'rir.wav'), signal_type='rir')

    def test_reverb_time(self):
        # Only functionality
        dsp.room_acoustics.reverb_time(self.rir, mode='t20', ir_start=None)
        dsp.room_acoustics.reverb_time(self.rir, mode='t30', ir_start=None)
        dsp.room_acoustics.reverb_time(self.rir, mode='t60', ir_start=None)
        dsp.room_acoustics.reverb_time(self.rir, mode='edt', ir_start=None)

        # Check Index
        ind = np.argmax(np.abs(self.rir.time_data))
        dsp.room_acoustics.reverb_time(self.rir, mode='edt', ir_start=ind)

        # Check MultiBandSignal
        fb = dsp.filterbanks.auditory_filters_gammatone(
            [500, 800], sampling_rate_hz=self.rir.sampling_rate_hz)
        mb = fb.filter_signal(self.rir)
        dsp.room_acoustics.reverb_time(mb, mode='t20', ir_start=None)
        dsp.room_acoustics.reverb_time(mb, mode='t20', ir_start=ind)

    def test_room_modes(self):
        # Only functionality
        # Take a multi-channel signal in order to find modes
        y = dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
        x = dsp.Signal(join('examples', 'data', 'chirp.wav'))
        h = dsp.transfer_functions.spectral_deconvolve(
            y, x, padding=True, keep_original_length=True)
        h = dsp.transfer_functions.window_ir(h, exp2_trim=10)

        dsp.room_acoustics.find_modes(
            h, f_range_hz=[50, 150], proximity_effect=False, dist_hz=5,
            prune_antimodes=False)
        dsp.room_acoustics.find_modes(
            h, f_range_hz=[50, 150], proximity_effect=True, dist_hz=5,
            prune_antimodes=False)
        dsp.room_acoustics.find_modes(
            h, f_range_hz=[50, 150], proximity_effect=False, dist_hz=5,
            prune_antimodes=True)
        dsp.room_acoustics.find_modes(
            h, f_range_hz=[50, 150], proximity_effect=False, dist_hz=0,
            prune_antimodes=False)

    def test_convolve_rir_on_signal(self):
        # Only functionality
        speech = dsp.Signal(join('examples', 'data', 'speech.flac'))
        dsp.room_acoustics.convolve_rir_on_signal(
            speech, self.rir, keep_peak_level=True, keep_length=True)
        dsp.room_acoustics.convolve_rir_on_signal(
            speech, self.rir, keep_peak_level=False, keep_length=False)

    def test_find_ir_start(self):
        # Only functionality
        dsp.room_acoustics.find_ir_start(self.rir)
        # Positive dBFS value for threshold throws assertion error
        with pytest.raises(AssertionError):
            dsp.room_acoustics.find_ir_start(self.rir, 20)

    def test_generate_synthetic_rir(self):
        # Only functionality
        dsp.room_acoustics.generate_synthetic_rir(
            room_dimensions_meters=[5, 4, 3], source_position=[2, 2, 2],
            receiver_position=[1, 1, 1], total_length_seconds=0.4,
            sampling_rate_hz=44100, desired_reverb_time_seconds=0.15,
            apply_bandpass=False)
        dsp.room_acoustics.generate_synthetic_rir(
            room_dimensions_meters=[5, 4, 3], source_position=[2, 2, 2],
            receiver_position=[1, 1, 1], total_length_seconds=0.4,
            sampling_rate_hz=44100, desired_reverb_time_seconds=0.15,
            apply_bandpass=True)
