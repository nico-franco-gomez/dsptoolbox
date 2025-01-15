import numpy as np
import dsptoolbox as dsp
import pytest
from os.path import join
import scipy.signal as sig


class TestRoomAcousticsModule:
    rir = dsp.ImpulseResponse(join("examples", "data", "rir.wav"))

    def test_reverb_time(self):
        # Only functionality
        dsp.room_acoustics.reverb_time(
            self.rir,
            mode=dsp.room_acoustics.ReverbTime.Adaptive,
            ir_start=None,
        )
        dsp.room_acoustics.reverb_time(
            self.rir, dsp.room_acoustics.ReverbTime.T20, ir_start=None
        )
        dsp.room_acoustics.reverb_time(
            self.rir, dsp.room_acoustics.ReverbTime.T30, ir_start=None
        )
        dsp.room_acoustics.reverb_time(
            self.rir, dsp.room_acoustics.ReverbTime.T60, ir_start=None
        )

        dsp.room_acoustics.reverb_time(
            self.rir,
            dsp.room_acoustics.ReverbTime.EDT,
            ir_start=None,
            automatic_trimming=False,
        )
        dsp.room_acoustics.reverb_time(
            self.rir,
            dsp.room_acoustics.ReverbTime.T60,
            ir_start=None,
            automatic_trimming=False,
        )
        dsp.room_acoustics.reverb_time(
            self.rir,
            dsp.room_acoustics.ReverbTime.EDT,
            ir_start=None,
            automatic_trimming=False,
        )

        # Check Index
        ind = np.argmax(np.abs(self.rir.time_data))
        dsp.room_acoustics.reverb_time(
            self.rir, dsp.room_acoustics.ReverbTime.EDT, ir_start=ind
        )
        combined = dsp.append_signals([self.rir, self.rir])
        dsp.room_acoustics.reverb_time(
            combined,
            dsp.room_acoustics.ReverbTime.EDT,
            ir_start=[ind, ind - 1],
        )

        # Check MultiBandSignal
        fb = dsp.filterbanks.auditory_filters_gammatone(
            [500, 800], sampling_rate_hz=self.rir.sampling_rate_hz
        )
        mb = fb.filter_signal(
            self.rir, dsp.FilterBankMode.Parallel, zero_phase=True
        )
        dsp.room_acoustics.reverb_time(
            mb, dsp.room_acoustics.ReverbTime.T20, ir_start=None
        )
        dsp.room_acoustics.reverb_time(
            mb, dsp.room_acoustics.ReverbTime.T20, ir_start=ind
        )

        mb = fb.filter_signal(
            combined, dsp.FilterBankMode.Parallel, zero_phase=True
        )
        dsp.room_acoustics.reverb_time(
            mb, dsp.room_acoustics.ReverbTime.T20, ir_start=[ind, ind - 1]
        )

        starts = np.ones((mb.number_of_bands, mb.number_of_channels)) * ind
        dsp.room_acoustics.reverb_time(
            mb, dsp.room_acoustics.ReverbTime.T20, ir_start=starts
        )

    def test_room_modes(self):
        # Only functionality
        # Take a multi-channel signal in order to find modes
        y = dsp.Signal(join("examples", "data", "chirp_stereo.wav"))
        x = dsp.Signal(join("examples", "data", "chirp.wav"))
        h = dsp.transfer_functions.spectral_deconvolve(
            y, x, padding=True, keep_original_length=True
        )
        h, _ = dsp.transfer_functions.window_ir(h, 2**10)

        dsp.room_acoustics.find_modes(h, f_range_hz=[50, 150], dist_hz=5)

        h = h.get_channels(0)
        dsp.room_acoustics.find_modes(h, f_range_hz=[50, 150], dist_hz=5)

    def test_convolve_rir_on_signal(self):
        speech = dsp.Signal(join("examples", "data", "speech.flac"))
        speech_2 = dsp.append_signals([speech, speech])
        result = dsp.room_acoustics.convolve_rir_on_signal(
            speech, self.rir, keep_peak_level=False, keep_length=True
        )
        assert len(result) == len(speech)

        result = dsp.room_acoustics.convolve_rir_on_signal(
            speech_2, self.rir, keep_peak_level=True, keep_length=False
        )
        np.testing.assert_allclose(
            np.max(np.abs(result.time_data), axis=0),
            np.max(np.abs(speech_2.time_data), axis=0),
        )

        # Double-channel
        conv = dsp.room_acoustics.convolve_rir_on_signal(
            speech_2,
            self.rir,
            keep_peak_level=False,
            keep_length=False,
        ).time_data
        td = speech.time_data.squeeze()
        ir = self.rir.time_data.squeeze()
        expected = sig.convolve(td, ir)
        np.testing.assert_allclose(conv[:, 0], expected)
        np.testing.assert_allclose(conv[:, 1], expected)

        # Length to trigger oaconvolve
        length_ir = len(td) // 11
        oaconv = dsp.room_acoustics.convolve_rir_on_signal(
            speech_2,
            dsp.pad_trim(self.rir, length_ir),
            keep_peak_level=False,
            keep_length=False,
        ).time_data
        expected = sig.convolve(td, ir[:length_ir])
        np.testing.assert_allclose(oaconv[:, 0], expected)
        np.testing.assert_allclose(oaconv[:, 1], expected)

    def test_find_ir_start(self):
        # Only functionality
        dsp.room_acoustics.find_ir_start(self.rir)
        # Positive dBFS value for threshold throws assertion error
        with pytest.raises(AssertionError):
            dsp.room_acoustics.find_ir_start(self.rir, 20)

    def test_generate_synthetic_rir(self):
        r = dsp.room_acoustics.ShoeboxRoom([3, 4, 5], None, 0.97)
        # Standard case
        dsp.room_acoustics.generate_synthetic_rir(
            room=r,
            source_position=[2, 2, 2],
            receiver_position=[1, 1, 1],
            total_length_seconds=0.3,
            sampling_rate_hz=44100,
            apply_bandpass=False,
            add_noise_reverberant_tail=False,
            use_detailed_absorption=False,
            max_order=None,
        )
        # rir.plot_magnitude(smoothe=0)
        # dsp.plots.show()
        # exit()
        # Detailed absorption
        d = {}
        for i in ["north", "south", "east", "west", "floor", "ceiling"]:
            d[i] = np.random.uniform(0.94, 0.96, size=4)
        r.add_detailed_absorption(d)
        # Use max order, detailed absorption, reverberant tail and bandpass
        # (they are all independent from each other)
        dsp.room_acoustics.generate_synthetic_rir(
            room=r,
            source_position=[2, 2, 2],
            receiver_position=[1, 1, 1],
            total_length_seconds=0.3,
            sampling_rate_hz=44100,
            apply_bandpass=True,
            add_noise_reverberant_tail=True,
            use_detailed_absorption=True,
            max_order=4,
        )

    def test_shoebox_room(self):
        r = dsp.room_acoustics.ShoeboxRoom([3, 4, 5], t60_s=0.6)
        r.get_mixing_time(mode="perceptual")
        r.get_mixing_time(mode="physical", n_reflections=1000)
        r.get_room_modes(3)
        assert r.check_if_in_room([1, 1, 1])
        assert not r.check_if_in_room([7, 7, 7])
        f = np.linspace(50, 200, 100)
        r.get_analytical_transfer_function(
            [1, 1, 1], [2, 2, 2], freqs=f, max_mode_order=5
        )
        with pytest.raises(AssertionError):
            dsp.room_acoustics.ShoeboxRoom([10, 10, 10], t60_s=0.01)

        # Check detailed absorption, it should deliver the same value as the
        # mean absorption when all given coefficients are the same
        r = dsp.room_acoustics.ShoeboxRoom([3, 4, 5], None, 0.97)
        d = {}
        for i in ["north", "south", "east", "west", "floor", "ceiling"]:
            d[i] = 0.97
        old_value = r.t60_s
        r.add_detailed_absorption(d)
        assert np.isclose(old_value, r.t60_s)

    def test_descriptors(self):
        # Only functionality
        # Single channel
        dsp.room_acoustics.descriptors(
            self.rir, dsp.room_acoustics.RoomAcousticsDescriptor.D50
        )
        dsp.room_acoustics.descriptors(
            self.rir, dsp.room_acoustics.RoomAcousticsDescriptor.C80
        )
        dsp.room_acoustics.descriptors(
            self.rir, dsp.room_acoustics.RoomAcousticsDescriptor.CenterTime
        )
        dsp.room_acoustics.descriptors(
            self.rir, dsp.room_acoustics.RoomAcousticsDescriptor.BassRatio
        )

        # MultiBand
        fb = dsp.filterbanks.fractional_octave_bands(
            [125, 1000], sampling_rate_hz=self.rir.sampling_rate_hz
        )
        rir_filt = fb.filter_signal(
            self.rir, dsp.FilterBankMode.Parallel, zero_phase=True
        )
        dsp.room_acoustics.descriptors(
            rir_filt, dsp.room_acoustics.RoomAcousticsDescriptor.D50
        )
        dsp.room_acoustics.descriptors(
            rir_filt, dsp.room_acoustics.RoomAcousticsDescriptor.C80
        )
        dsp.room_acoustics.descriptors(
            rir_filt, dsp.room_acoustics.RoomAcousticsDescriptor.CenterTime
        )

        with pytest.raises(AssertionError):
            dsp.room_acoustics.descriptors(
                rir_filt, dsp.room_acoustics.RoomAcousticsDescriptor.BassRatio
            )
