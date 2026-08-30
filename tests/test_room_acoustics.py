import os
from os.path import join

import numpy as np
import pytest
import scipy.signal as sig

import dsptoolbox as dsp


class TestRoomAcousticsModule:
    rir = dsp.ImpulseResponse(
        join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
    )

    def test_reverb_time(self):
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

        ind = np.argmax(np.abs(self.rir.time_data))
        dsp.room_acoustics.reverb_time(
            self.rir, dsp.room_acoustics.ReverbTime.EDT, ir_start=ind
        )
        combined = self.rir.append_signals([self.rir])
        dsp.room_acoustics.reverb_time(
            combined,
            dsp.room_acoustics.ReverbTime.EDT,
            ir_start=[ind, ind - 1],
        )

        fb = dsp.filterbanks.auditory_filters_gammatone(
            [500, 800], sampling_rate_hz=self.rir.sampling_rate_hz
        )
        mb = fb.filter_signal(self.rir, dsp.FilterBankMode.Parallel, zero_phase=True)
        dsp.room_acoustics.reverb_time(
            mb, dsp.room_acoustics.ReverbTime.T20, ir_start=None
        )
        dsp.room_acoustics.reverb_time(
            mb, dsp.room_acoustics.ReverbTime.T20, ir_start=ind
        )

        mb = fb.filter_signal(combined, dsp.FilterBankMode.Parallel, zero_phase=True)
        dsp.room_acoustics.reverb_time(
            mb, dsp.room_acoustics.ReverbTime.T20, ir_start=[ind, ind - 1]
        )

        starts = np.ones((mb.number_of_bands, mb.number_of_channels)) * ind
        dsp.room_acoustics.reverb_time(
            mb, dsp.room_acoustics.ReverbTime.T20, ir_start=starts
        )

    def test_reverb_time_matches_known_decay_constant(self):
        """A synthetic noise IR shaped by `exp(-t/tau)` has an energy decay
        curve (Schroeder backward integration of the squared IR) that also
        decays exponentially at twice the rate, `exp(-2t/tau)`; in dB this
        is a straight line with slope `-8.686/tau` dB/s (since
        `10*log10(exp(-2/tau)) = -20/(tau*ln(10))`), giving
        `T60 = 60*tau/8.686 = 6.908*tau`. Verified empirically first: T30
        (the least edge-sensitive fit range) matches to within ~0.2% and
        the correlation coefficient is ~-0.9999 for this clean synthetic
        case, well clear of the pre-existing "> -0.95" warning seen
        elsewhere in this suite with real (noisier) RIRs.

        """
        fs = 8_000
        tau = 0.3
        n_samples = int(2.0 * fs)
        t = np.arange(n_samples) / fs
        rng = np.random.default_rng(0)
        td = rng.normal(0, 1, n_samples) * np.exp(-t / tau)
        # Small noise floor to keep the decay curve well-behaved down to
        # the -65 dB range T60's fit needs, avoiding log(~0) instabilities.
        td += rng.normal(0, 1e-4, n_samples)

        ir = dsp.ImpulseResponse(None, td[:, None], fs, constrain_amplitude=False)
        rt, corr = dsp.room_acoustics.reverb_time(
            ir,
            dsp.room_acoustics.ReverbTime.T30,
            ir_start=None,
            automatic_trimming=False,
        )

        np.testing.assert_allclose(rt[0], 6.908 * tau, rtol=0.02)
        assert corr[0] < -0.999

    def test_room_modes(self):
        # A multi-channel signal, to also exercise the per-channel path
        y = dsp.Signal(
            join(
                os.path.dirname(__file__),
                "..",
                "example_data",
                "chirp_stereo.wav",
            )
        )
        x = dsp.Signal(
            join(os.path.dirname(__file__), "..", "example_data", "chirp.wav")
        )
        h = dsp.transfer_functions.spectral_deconvolve(
            y, x, padding=True, keep_original_length=True
        )
        h, _ = dsp.transfer_functions.window_ir(h, 2**10)

        dsp.room_acoustics.find_modes(h, f_range_hz=[50, 150], dist_hz=5)

        h = h.get_channels(0)
        dsp.room_acoustics.find_modes(h, f_range_hz=[50, 150], dist_hz=5)

    def test_find_modes_recovers_known_frequencies(self):
        """A synthetic IR built as a sum of decaying sinusoids at known
        frequencies should have its modes detected via the complex mode
        indicator function at (near) those exact frequencies -- verified
        empirically to be an exact match at ~1 Hz resolution (the function
        pads to a 1-second buffer internally) for this well-separated,
        low-noise case.

        """
        fs = 2_000
        n_samples = int(0.5 * fs)
        t = np.arange(n_samples) / fs
        freqs_true = [80.0, 120.0, 170.0]
        rng = np.random.default_rng(1)
        td = np.zeros(n_samples)
        for f in freqs_true:
            td += np.sin(2 * np.pi * f * t) * np.exp(-t / 0.1)
        td += rng.normal(0, 1e-4, n_samples)

        ir = dsp.ImpulseResponse(None, td[:, None], fs, constrain_amplitude=False)
        modes = dsp.room_acoustics.find_modes(ir, f_range_hz=[50, 200], dist_hz=5)

        np.testing.assert_allclose(modes, freqs_true, atol=1.0)

    def test_convolve_rir_on_signal(self):
        speech = dsp.Signal(
            join(os.path.dirname(__file__), "..", "example_data", "speech.flac")
        )
        speech_2 = speech.append_signals([speech])
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
            self.rir.pad_trim(length_ir),
            keep_peak_level=False,
            keep_length=False,
        ).time_data
        expected = sig.convolve(td, ir[:length_ir])
        np.testing.assert_allclose(oaconv[:, 0], expected)
        np.testing.assert_allclose(oaconv[:, 1], expected)

    def test_find_ir_start(self):
        dsp.room_acoustics.find_ir_start(self.rir)
        # A positive dBFS threshold is invalid
        with pytest.raises(AssertionError):
            dsp.room_acoustics.find_ir_start(self.rir, 20)

    def test_generate_synthetic_rir(self):
        r = dsp.room_acoustics.ShoeboxRoom([3, 4, 5], None, 0.97)
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
        )[0]
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

    def test_descriptors_boundary_case_energy_before_50ms(self):
        """Per the source, D50 = energy(0-50ms) / energy(0-stop) (a
        fraction, no unit) and C80 = 10*log10(energy(0-80ms) /
        energy(80ms-stop)) (in dB). An IR with (almost) all its energy
        concentrated in the first 10 ms should therefore have D50 close to
        1 (nearly all energy already within the 50ms window) and C80 very
        high (the 80ms-to-end denominator is close to the noise floor).

        """
        fs = 8_000
        n_samples = fs
        early_len = int(0.01 * fs)
        rng = np.random.default_rng(0)
        td = np.zeros(n_samples)
        td[:early_len] = rng.normal(0, 1, early_len) * np.exp(
            -np.arange(early_len) / (early_len / 5)
        )
        # Tiny noise floor everywhere else so C80's late-energy denominator
        # is small but not literally zero.
        td[early_len:] = rng.normal(0, 1e-4, n_samples - early_len)

        ir = dsp.ImpulseResponse(None, td[:, None], fs, constrain_amplitude=False)
        d50 = dsp.room_acoustics.descriptors(
            ir,
            dsp.room_acoustics.RoomAcousticsDescriptor.D50,
            automatic_trimming_rir=False,
        )
        c80 = dsp.room_acoustics.descriptors(
            ir,
            dsp.room_acoustics.RoomAcousticsDescriptor.C80,
            automatic_trimming_rir=False,
        )

        assert d50[0] > 0.999
        assert c80[0] > 30.0
