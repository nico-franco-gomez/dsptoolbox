import os
from os.path import join

import numpy as np
import pytest
from matplotlib.pyplot import close
from scipy.signal import hilbert

import dsptoolbox as dsp


def _hz_to_mel(f_hz):
    """Independent re-derivation of the HTK mel scale used by the library
    (`dsptoolbox.helpers.frequency_conversion._hz2mel`), not a call into it.

    """
    return 2595 * np.log10(1 + f_hz / 700)


def _mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


class TestTransformsModule:
    speech = dsp.Signal(
        join(os.path.dirname(__file__), "..", "example_data", "speech.flac")
    )

    def test_cepstrum(self):
        cc = dsp.transforms.cepstrum(self.speech, True)
        dsp.transforms.cepstrum(self.speech, False)
        ss = dsp.transforms.from_complex_cepstrum(cc, self.speech.sampling_rate_hz)
        np.testing.assert_allclose(
            self.speech.time_data,
            ss.time_data,
            atol=dsp.tools.from_db(-100, True),
        )

    def test_log_mel_spectrogram(self):
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

        # Range must not exceed the Nyquist frequency
        with pytest.raises(AssertionError):
            dsp.transforms.log_mel_spectrogram(
                self.speech,
                range_hz=[20, 30e3],
                n_bands=10,
                generate_plot=False,
                stft_parameters=None,
            )
        close("all")

    def test_log_mel_spectrogram_pure_tone_energy_at_correct_bin(self):
        """A pure tone's energy should be concentrated in the mel band
        whose (Hz-converted) center frequency is closest to the tone
        (plausibility -- filter overlap and STFT leakage spread some
        energy into neighboring bands, so this checks the argmax bin only).

        """
        fs = 8_000
        t = np.arange(8_000) / fs
        freq = 1_000.0
        x = np.sin(2 * np.pi * freq * t)
        sig = dsp.Signal(None, x[:, None], fs)

        _, f_mel, log_mel_sp = dsp.transforms.log_mel_spectrogram(
            sig, n_bands=40, generate_plot=False
        )
        mean_energy = log_mel_sp.mean(axis=1)[:, 0]
        peak_idx = np.argmax(mean_energy)
        peak_center_hz = _mel_to_hz(f_mel[peak_idx])

        # Within the two neighboring bands' worth of frequency distance.
        neighbor_idx = min(peak_idx + 1, len(f_mel) - 1)
        band_width_hz = abs(_mel_to_hz(f_mel[neighbor_idx]) - peak_center_hz)
        assert abs(peak_center_hz - freq) <= band_width_hz

    def test_mel_filters(self):
        f = np.linspace(0, 24000, 2048)
        dsp.transforms.mel_filterbank(
            f_hz=f, range_hz=None, n_bands=30, normalize=False
        )
        dsp.transforms.mel_filterbank(
            f_hz=f, range_hz=[1e3, 5e3], n_bands=10, normalize=False
        )
        dsp.transforms.mel_filterbank(f_hz=f, range_hz=None, n_bands=30, normalize=True)

    def test_mel_filterbank_center_frequencies_match_closed_form(self):
        """The returned center-frequency vector is an even split of the mel
        scale (`2595*log10(1+f/700)`) between the range bounds -- not the
        (frequency-bin-snapped) `bands_hz` used to build the filters
        themselves. This is recomputed independently here, not via the
        internal `_hz2mel`/`_mel2hz` helpers.

        """
        f = np.linspace(0, 24000, 2048)
        range_hz = [100.0, 8000.0]
        n_bands = 12
        _, centers_mel = dsp.transforms.mel_filterbank(
            f_hz=f, range_hz=range_hz, n_bands=n_bands, normalize=False
        )

        range_mel = _hz_to_mel(np.asarray(range_hz))
        expected = np.linspace(range_mel[0], range_mel[1], n_bands + 2)[1:-1]
        np.testing.assert_allclose(centers_mel, expected, rtol=1e-12)

    def test_mel_filterbank_matches_librosa(self):
        """Soft comparison against `librosa.filters.mel` using the matching
        HTK mel-scale convention (`htk=True`) and no filter-area
        normalization on either side, skipped entirely if librosa is not
        installed.

        """
        try:
            import librosa
        except ImportError:
            return

        sr = 16000
        n_fft = 2048
        n_bands = 20
        f = np.linspace(0, sr / 2, n_fft // 2 + 1)
        dsp_filters, dsp_centers_mel = dsp.transforms.mel_filterbank(
            f_hz=f, range_hz=None, n_bands=n_bands, normalize=False
        )
        librosa_filters = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_bands, htk=True, norm=None
        )
        # Both are triangular filters over the same frequency axis;
        # compare where their peaks land rather than exact amplitudes,
        # since edge/snapping conventions differ between the two
        # implementations.
        dsp_peaks = np.argmax(dsp_filters, axis=1)
        librosa_peaks = np.argmax(librosa_filters, axis=1)
        np.testing.assert_allclose(dsp_peaks, librosa_peaks, atol=2)

        librosa_centers_hz = _mel_to_hz(dsp_centers_mel)
        assert np.all(librosa_centers_hz > 0)

    def test_mel_filterbank_invalid_parameters_raise(self):
        f = np.linspace(0, 24000, 2048)
        with pytest.raises(AssertionError):
            dsp.transforms.mel_filterbank(f_hz=np.zeros((2, 2)))
        with pytest.raises(AssertionError):
            dsp.transforms.mel_filterbank(f_hz=f, range_hz=[100, 30000])
        with pytest.raises(AssertionError):
            dsp.transforms.mel_filterbank(f_hz=f, range_hz=[-10, 8000])

    def test_plot_waterfall(self):
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
        t, f, s = self.speech.get_spectrogram()

        mels, _ = dsp.transforms.mel_filterbank(f, [20, 10e3], n_bands=4)
        t, mel, mf, fig, ax = dsp.transforms.mfcc(self.speech, mel_filters=mels)
        t, mel, mf = dsp.transforms.mfcc(self.speech, generate_plot=False)

    def test_mfcc_distinguishes_different_tones(self):
        """MFCC coefficients are a DCT of the log-mel spectrum, so a single
        spectral peak's energy is not concentrated in one cepstral bin the
        way it is in the mel/chroma domain themselves -- there is no single
        "correct bin" to check. This is scoped down to the weaker but still
        meaningful plausibility property that two clearly different pure
        tones produce distinguishable (non-identical, finite) MFCC vectors.

        """
        fs = 8_000
        t = np.arange(4_000) / fs

        def make_signal(freq):
            x = np.sin(2 * np.pi * freq * t)
            return dsp.Signal(None, x[:, None], fs)

        _, _, mf_low = dsp.transforms.mfcc(make_signal(300.0), generate_plot=False)
        _, _, mf_high = dsp.transforms.mfcc(make_signal(2000.0), generate_plot=False)

        assert np.all(np.isfinite(mf_low)) and np.all(np.isfinite(mf_high))
        assert not np.allclose(mf_low, mf_high)

    def test_istft(self):
        # This would most likely fail if padding=False or detrend=True
        t, f, sp = self.speech.get_spectrogram()
        speech_rec = dsp.transforms.istft(sp, original_signal=self.speech)
        assert np.all(np.isclose(self.speech.time_data, speech_rec.time_data))

        speech_rec = dsp.transforms.istft(
            sp,
            parameters=self.speech.spectrogram_parameters,
            sampling_rate_hz=self.speech.sampling_rate_hz,
        )
        assert np.all(
            np.isclose(self.speech.time_data, speech_rec.time_data[: len(self.speech)])
        )

        # With longer fft length than window
        wl = 512
        self.speech = self.speech.set_spectrogram_parameters(
            window_length_samples=wl, fft_length_samples=wl * 2
        )
        t, f, sp = self.speech.get_spectrogram()
        speech_rec = dsp.transforms.istft(sp, original_signal=self.speech)
        assert np.all(np.isclose(self.speech.time_data, speech_rec.time_data))

        speech_rec = dsp.transforms.istft(
            sp,
            parameters=self.speech.spectrogram_parameters,
            sampling_rate_hz=self.speech.sampling_rate_hz,
        )
        assert np.all(
            np.isclose(self.speech.time_data, speech_rec.time_data[: len(self.speech)])
        )

    def test_chroma(self):
        dsp.transforms.chroma_stft(self.speech.copy())
        dsp.transforms.chroma_stft(self.speech.copy(), plot_channel=0)

    def test_chroma_stft_pure_tone_at_correct_note(self):
        """A pure 440 Hz tone is concert pitch A4 (MIDI 69), which is note
        index `69 % 12 == 9` in the returned chroma vector (index 0 = C,
        per the docstring's note ordering). This is a closed-form check
        against the standard 12-tone-equal-temperament pitch mapping.

        """
        fs = 8_000
        t = np.arange(8_000) / fs
        x = np.sin(2 * np.pi * 440.0 * t)
        sig = dsp.Signal(None, x[:, None], fs)

        _, chroma, _ = dsp.transforms.chroma_stft(sig)
        mean_energy_per_note = chroma.mean(axis=1)[:, 0]
        assert np.argmax(mean_energy_per_note) == 9

    def test_chroma_stft_invalid_parameters_raise(self):
        with pytest.raises(AssertionError):
            dsp.transforms.chroma_stft(self.speech.copy(), tuning_a_hz=0)
        with pytest.raises(AssertionError):
            dsp.transforms.chroma_stft(self.speech.copy(), compression=0)

    def test_cwt(self):
        query_f = np.linspace(100, 200, 50)
        morlet = dsp.transforms.MorletWavelet(b=None, h=3, step=1e-3)
        dsp.transforms.cwt(self.speech, query_f, morlet, False)
        dsp.transforms.cwt(self.speech, query_f, morlet, True)

    def test_squeeze_scalogram_rust_backend_parity(self):
        from dsptoolbox.transforms._transforms import _squeeze_scalogram_python

        try:
            from dsptoolbox._rust import squeeze_scalogram
        except ImportError:
            pytest.skip("Rust extension is not available")

        freqs = np.linspace(100.0, 1200.0, 7)

        for n_times in (3, 31):
            rng = np.random.default_rng(n_times)
            scalogram = (
                rng.normal(size=(7, n_times, 2)) + 1j * rng.normal(size=(7, n_times, 2))
            ).astype(np.complex128)
            gradient = np.gradient(scalogram, axis=1, edge_order=2)
            for apply_normalization in (False, True):
                expected = _squeeze_scalogram_python(
                    scalogram,
                    freqs,
                    8000,
                    0.05,
                    apply_normalization,
                    gradient,
                )
                actual = squeeze_scalogram(
                    scalogram,
                    freqs,
                    8000,
                    0.05,
                    apply_normalization,
                    gradient,
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cwt_peak_at_true_frequency(self):
        """The scalogram's magnitude, at a time sample away from the
        signal's edges, should peak at the queried frequency nearest the
        pure tone actually present (plausibility -- wavelet time-frequency
        resolution trade-offs mean this isn't an exact closed form, but a
        sufficiently narrow-band Morlet wavelet localizes the peak tightly).

        """
        fs = 8_000
        t = np.arange(4_000) / fs
        freq = 500.0
        x = np.sin(2 * np.pi * freq * t)
        sig = dsp.Signal(None, x[:, None], fs)

        query_f = np.linspace(100, 1000, 46)
        morlet = dsp.transforms.MorletWavelet(b=None, h=8, step=1e-3)
        scalogram = dsp.transforms.cwt(sig, query_f, morlet, channel=None)

        mid = len(sig) // 2
        magnitude = np.abs(scalogram[:, mid, 0])
        peak_freq = query_f[np.argmax(magnitude)]

        bin_width = query_f[1] - query_f[0]
        assert abs(peak_freq - freq) <= bin_width

    def test_vqt_peak_near_true_frequency(self):
        """Same plausibility property as the CWT test, but for the VQT's
        logarithmically-spaced frequency bins: the peak bin should be
        within one bin (a ratio of `2**(1/bins_per_octave)`) of the tone's
        true frequency.

        """
        fs = 8_000
        t = np.arange(8_000) / fs
        freq = 440.0
        x = np.sin(2 * np.pi * freq * t)
        sig = dsp.Signal(None, x[:, None], fs)

        bins_per_octave = 24
        f, vq = dsp.transforms.vqt(sig, octaves=(1, 5), bins_per_octave=bins_per_octave)
        mid = len(sig) // 2
        magnitude = np.abs(vq[:, mid, 0])
        peak_freq = f[np.argmax(magnitude)]

        bin_ratio = 2 ** (1 / bins_per_octave)
        assert (freq / bin_ratio) <= peak_freq <= (freq * bin_ratio)

    def test_hilbert(self):
        speech = self.speech.copy()
        speech.constrain_amplitude = False
        s = dsp.transforms.hilbert(speech)
        s = s.time_data + s.time_data_imaginary * 1j
        s2 = speech.time_data

        s2 = hilbert(s2, axis=0)
        np.testing.assert_allclose(s, s2)

        # Now other length (even vs. odd)
        s = dsp.transforms.hilbert(speech.pad_trim(len(speech) - 1))
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
        sp = self.speech.append_signals([self.speech])
        sp_aft = dsp.transforms.stereo_mid_side(sp, True)
        sp_aft = dsp.transforms.stereo_mid_side(sp_aft, False)
        assert np.all(np.isclose(sp.time_data, sp_aft.time_data))

    def test_laguerre(self):
        sp = self.speech.pad_trim(128)
        dsp.transforms.laguerre(sp, dsp.WarpingFactor.Custom.with_factor(-0.7))
        dsp.transforms.laguerre(sp, dsp.WarpingFactor.Erb)

    def test_laguerre_rust_backend_parity(self):
        from dsptoolbox.transforms._transforms import _laguerre_python

        try:
            from dsptoolbox._rust import laguerre
        except ImportError:
            pytest.skip("Rust extension is not available")

        for shape in ((3, 1), (31, 2), (128, 4)):
            rng = np.random.default_rng(sum(shape))
            for warping_factor in (-0.7, 0.4):
                contiguous = rng.normal(size=shape)
                for time_data in (contiguous, np.asfortranarray(contiguous)):
                    expected = _laguerre_python(time_data, warping_factor)
                    actual = laguerre(time_data, warping_factor)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_laguerre_round_trip_is_identity(self):
        """Per the docstring, applying `laguerre` with a warping factor and
        then again with its negation must reconstruct the original signal
        (round-trip identity). Verified empirically to be exact to machine
        precision for a decaying, IR-like signal (a signal with substantial
        energy right at the buffer edge does not round-trip cleanly, since
        the underlying recursive filter chain needs samples beyond the
        buffer to fully invert -- this is expected of any IIR-based warp
        applied to a finite window, not a bug).

        """
        fs = 8_000
        n = 256
        t = np.arange(n) / fs
        rng = np.random.default_rng(2)
        decay = np.exp(-t * 3000)
        td = (rng.normal(0, 1, n) * decay)[:, None]
        sig = dsp.Signal(None, td, fs)
        sig.constrain_amplitude = False

        for warping_factor in (0.4, -0.6):
            warped = dsp.transforms.laguerre(
                sig, dsp.WarpingFactor.Custom.with_factor(warping_factor)
            )
            unwarped = dsp.transforms.laguerre(
                warped, dsp.WarpingFactor.Custom.with_factor(-warping_factor)
            )
            np.testing.assert_allclose(unwarped.time_data[:, 0], td[:, 0], atol=1e-9)

    def test_laguerre_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            dsp.WarpingFactor.Custom.with_factor(1.0)
        with pytest.raises(ValueError):
            dsp.WarpingFactor.Custom.with_factor(-1.0)

    def test_warp(self):
        s = dsp.ImpulseResponse(
            join(os.path.dirname(__file__), "..", "example_data", "rir.wav")
        )
        dsp.transforms.warp(s, dsp.WarpingFactor.Custom.with_factor(-0.6), True, 2**8)
        dsp.transforms.warp(s, dsp.WarpingFactor.Custom.with_factor(0.6), False, 2**8)

        # warping scales
        for scale in (
            dsp.WarpingFactor.Bark,
            dsp.WarpingFactor.BarkInverse,
            dsp.WarpingFactor.Erb,
            dsp.WarpingFactor.ErbInverse,
        ):
            dsp.transforms.warp(s, scale, False, 2**7)

    def test_warp_round_trip_is_identity(self):
        """`warp` states: "To pre-warp a signal, pass a negative
        `warping_factor`. To de-warp it, use the same positive
        `warping_factor`." -- i.e. `warp(warp(x, -lambda), lambda)` should
        reconstruct `x` (round-trip identity, `shift_ir=False` to avoid the
        non-invertible start-detection/rolling step). Verified empirically
        to be exact to machine precision for a decaying, IR-like signal
        (see `test_laguerre_round_trip_is_identity` for why a signal with
        energy at the buffer's edge would not round-trip as cleanly).

        """
        fs = 8_000
        n = 512
        t = np.arange(n) / fs
        rng = np.random.default_rng(1)
        decay = np.exp(-t * 3000)
        td = (rng.normal(0, 1, n) * decay)[:, None]
        ir = dsp.ImpulseResponse(None, td, fs)
        ir.constrain_amplitude = False

        for warping_factor in (0.3, -0.5, 0.7):
            warped = dsp.transforms.warp(
                ir, dsp.WarpingFactor.Custom.with_factor(-warping_factor), False, n
            )
            unwarped = dsp.transforms.warp(
                warped, dsp.WarpingFactor.Custom.with_factor(warping_factor), False, n
            )
            np.testing.assert_allclose(unwarped.time_data[:, 0], td[:, 0], atol=1e-9)

    def test_warp_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            dsp.WarpingFactor.Custom.with_factor(1.0)
        # Only Custom carries an explicit factor, and it is required
        with pytest.raises(ValueError):
            dsp.WarpingFactor.Erb.with_factor(0.5)
        with pytest.raises(ValueError):
            dsp.WarpingFactor.Custom.get_factor(48_000)

    def test_warp_filter(self):
        i = dsp.Filter.iir_filter(
            3,
            100.0,
            type_of_pass=dsp.FilterPassType.Highpass,
            filter_design_method=dsp.IirDesignMethod.Butterworth,
            sampling_rate_hz=24000,
        )
        dsp.transforms.warp_filter(i, dsp.WarpingFactor.Custom.with_factor(-0.6))
        dsp.transforms.warp_filter(i, dsp.WarpingFactor.Bark)

    def test_warp_filter_fixed_points_at_dc_and_nyquist(self):
        """Per the docstring, poles/zeros are transformed via the Oppenheim
        allpass frequency-warping pole map
        `p_new = (warping_factor + p) / (1 + warping_factor * p)`. This map
        has `p=1` (DC, f=0) and `p=-1` (Nyquist) as fixed points for any
        warping factor -- verified here to be an exact closed-form
        invariant of the transform, for both signs of warping factor.

        """
        fs = 8_000
        zeros = np.array([1.0, -1.0, 0.3 + 0.2j, 0.3 - 0.2j])
        poles = np.array([1.0, -1.0, 0.5 + 0.1j, 0.5 - 0.1j])
        filt = dsp.Filter.from_zpk(zeros, poles, 1.0, fs)

        for warping_factor in (0.3, -0.6):
            warped = dsp.transforms.warp_filter(
                filt, dsp.WarpingFactor.Custom.with_factor(warping_factor)
            )
            zw, pw, _ = warped.get_coefficients(dsp.FilterCoefficientsType.Zpk)
            assert np.any(np.isclose(zw, 1.0)) and np.any(np.isclose(zw, -1.0))
            assert np.any(np.isclose(pw, 1.0)) and np.any(np.isclose(pw, -1.0))

    def test_warp_filter_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            dsp.WarpingFactor.Custom.with_factor(1.0)
        with pytest.raises(ValueError):
            dsp.WarpingFactor.Custom.with_factor(-1.0)

    def test_lpc(self):
        speech = self.speech.resample(8000)
        dsp.transforms.lpc(speech, 10, 1024, False, True, 512)
        dsp.transforms.lpc(speech, 10, 1024, True, True, 512)

        dsp.transforms.lpc(speech, 10, 1024, False, False, 512)
        dsp.transforms.lpc(speech, 10, 1024, True, False, 512)

    def test_lpc_recovers_known_ar_coefficients(self):
        """A synthetic AR(2) resonator process `x[n] = a1*x[n-1] + a2*x[n-2]
        + noise[n]` with known coefficients (built independently via
        `scipy.signal.lfilter`, not the library's own AR estimator) should
        have its coefficients recovered closely by the Yule-Walker method.
        LPC estimation from finite noisy data is not bit-exact, hence the
        (still tight) relative tolerance rather than an exact match.

        """
        from scipy.signal import lfilter

        fs = 8_000
        order = 2
        r = 0.9
        theta = 2 * np.pi * 500 / fs
        a1 = 2 * r * np.cos(theta)
        a2 = -(r**2)
        true_a = np.array([1.0, -a1, -a2])

        rng = np.random.default_rng(0)
        n = 20_000
        noise = rng.normal(0, 1, n)
        x = lfilter([1.0], true_a, noise)[2_000:]  # discard transient

        sig = dsp.Signal(None, x[:, None], fs)
        a, _ = dsp.transforms.lpc(
            sig, order, window_length_samples=len(x), use_burg_method=False
        )
        np.testing.assert_allclose(a[:, 0, 0], true_a, rtol=0.05)

    def test_dft(self):
        s = self.speech.pad_trim(20_000)
        s.spectrum_method = dsp.SpectrumMethod.FFT
        f, spectrum = s.get_spectrum()

        select = slice(20, 40)
        dft = dsp.transforms.dft(s, f[select])
        np.testing.assert_allclose(dft, spectrum[select, ...])

    def test_spectrum_via_filterbank(self):
        s = self.speech.pad_trim(20_000)
        freqs = np.asarray([500, 550, 1000])
        # Linear
        spec1 = dsp.transforms.spectrum_via_filterbank(s, freqs, None, 20.0, 8, False)
        dsp.transforms.spectrum_via_filterbank(s, freqs, None, 20.0, 8, True)

        s_multi = s.append_signals([s.copy()])
        spec2 = dsp.transforms.spectrum_via_filterbank(
            s_multi, freqs, None, 20.0, 8, False
        )
        np.testing.assert_allclose(spec1.spectral_data[:, 0], spec2.spectral_data[:, 0])
        np.testing.assert_allclose(spec1.spectral_data[:, 0], spec2.spectral_data[:, 1])

        # Log
        spec2 = dsp.transforms.spectrum_via_filterbank(
            s_multi, freqs, 0.5, None, 8, False
        )
        with pytest.raises(AssertionError):
            spec2 = dsp.transforms.spectrum_via_filterbank(
                s_multi, freqs, 0.5, 10, 8, False
            )
        with pytest.raises(AssertionError):
            spec2 = dsp.transforms.spectrum_via_filterbank(
                s_multi, freqs, -0.5, None, 8, False
            )
        with pytest.raises(AssertionError):
            spec2 = dsp.transforms.spectrum_via_filterbank(
                s_multi, freqs, None, -10, 8, False
            )

    def test_warp_accepts_integer_and_numpy_warping_factors(self):
        """The factor used to be validated with `type(x) is float`, which
        rejects an int and every numpy float.

        """
        ir = dsp.ImpulseResponse.from_time_data(np.eye(64, 1), 8_000)
        custom = dsp.WarpingFactor.Custom.with_factor
        reference = dsp.transforms.warp(ir, custom(0.3), False).time_data
        for factor in (np.float64(0.3), np.float32(0.3)):
            np.testing.assert_allclose(
                dsp.transforms.warp(ir, custom(factor), False).time_data,
                reference,
                atol=1e-6,
            )
        # An integer factor of 0 is the identity warp
        np.testing.assert_allclose(
            dsp.transforms.warp(ir, custom(0), False).time_data,
            ir.time_data,
            atol=1e-12,
        )

    def test_warping_factor_scales_are_symmetric(self):
        """The inverse members must resolve to exactly the negated factor of
        their forward counterpart, so that a warp/dewarp pair cancels.

        """
        fs = 44_100
        for forward, inverse in (
            (dsp.WarpingFactor.Bark, dsp.WarpingFactor.BarkInverse),
            (dsp.WarpingFactor.Erb, dsp.WarpingFactor.ErbInverse),
        ):
            assert forward.get_factor(fs) == -inverse.get_factor(fs)
            assert abs(forward.get_factor(fs)) < 1.0
