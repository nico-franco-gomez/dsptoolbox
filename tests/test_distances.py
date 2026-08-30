import os
from os.path import join

import numpy as np
import pytest

import dsptoolbox as dsp

stereo_signal = dsp.Signal(
    join(os.path.dirname(__file__), "..", "example_data", "chirp_stereo.wav")
)


class TestDistancesModule:
    sig1 = stereo_signal.get_channels(0)
    sig2 = stereo_signal.get_channels(1)

    def test_log_spectral(self):
        dsp.distances.log_spectral(
            self.sig1,
            self.sig2,
            method="standard",
            f_range_hz=[20, 20e3],
            energy_normalization=True,
            spectrum_parameters=None,
        )
        dsp.distances.log_spectral(
            self.sig1,
            self.sig2,
            method="welch",
            f_range_hz=[200, 5000],
            energy_normalization=True,
            spectrum_parameters=None,
        )
        with pytest.raises(AssertionError):
            dsp.distances.log_spectral(
                self.sig1,
                self.sig2,
                method="welch",
                f_range_hz=[20, 30e3],
                energy_normalization=True,
                spectrum_parameters=None,
            )

        dsp.distances.log_spectral(
            self.sig1,
            self.sig2,
            method="welch",
            f_range_hz=[20, 20e3],
            energy_normalization=False,
            spectrum_parameters=dict(window_type=("chebwin", 40)),
        )

    def test_log_spectral_matches_closed_form(self):
        """The log spectral distance is
        `sqrt(integral((10*log10(psd1/psd2))**2, f))` over the observed
        frequency range, computed directly here from the same PSDs the
        library obtains via its public `get_spectrum` API (energy-
        normalized, as the library does when `energy_normalization=True`).

        """
        from scipy.integrate import simpson

        f_range_hz = [200, 5000]
        s1 = self.sig1.set_spectrum_parameters(method="welch")
        s2 = self.sig2.set_spectrum_parameters(method="welch")
        f, spec1 = s1.get_spectrum()
        _, spec2 = s2.get_spectrum()

        psd1 = np.abs(spec1)
        psd2 = np.abs(spec2)
        if s1.spectrum_scaling.is_amplitude_scaling():
            psd1 = psd1**2
            psd2 = psd2**2

        lo = np.argmin(np.abs(f - f_range_hz[0]))
        hi = np.argmin(np.abs(f - f_range_hz[1]))
        f_sub = f[lo:hi]

        expected = np.zeros(self.sig1.number_of_channels)
        for n in range(self.sig1.number_of_channels):
            x = psd1[lo:hi, n].copy()
            y = psd2[lo:hi, n].copy()
            x /= np.sum(x)
            y /= np.sum(y)
            expected[n] = np.sqrt(simpson((10 * np.log10(x / y)) ** 2, x=f_sub))

        result = dsp.distances.log_spectral(
            self.sig1,
            self.sig2,
            method="welch",
            f_range_hz=f_range_hz,
            energy_normalization=True,
            spectrum_parameters=None,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_itakura_saito(self):
        dsp.distances.itakura_saito(
            self.sig1,
            self.sig2,
            method="standard",
            f_range_hz=[20, 20e3],
            energy_normalization=True,
            spectrum_parameters=None,
        )
        dsp.distances.itakura_saito(
            self.sig1,
            self.sig2,
            method="welch",
            f_range_hz=[200, 5000],
            energy_normalization=True,
            spectrum_parameters=None,
        )
        with pytest.raises(AssertionError):
            dsp.distances.itakura_saito(
                self.sig1,
                self.sig2,
                method="welch",
                f_range_hz=[20, 30e3],
                energy_normalization=True,
                spectrum_parameters=None,
            )

        dsp.distances.itakura_saito(
            self.sig1,
            self.sig2,
            method="welch",
            f_range_hz=[20, 20e3],
            energy_normalization=False,
            spectrum_parameters=dict(window_type=("chebwin", 40)),
        )

    def test_itakura_saito_matches_closed_form(self):
        """The Itakura-Saito measure is
        `integral(psd1/psd2 - log10(psd1/psd2) - 1, f)`, recomputed here
        directly from the same PSDs the library obtains via its public
        `get_spectrum` API. Note the implementation uses `log10` (not the
        natural log of the textbook/Wikipedia definition); this test
        documents the implemented formula, not the literature one.

        """
        from scipy.integrate import simpson

        f_range_hz = [200, 5000]
        s1 = self.sig1.set_spectrum_parameters(method="welch")
        s2 = self.sig2.set_spectrum_parameters(method="welch")
        f, spec1 = s1.get_spectrum()
        _, spec2 = s2.get_spectrum()

        psd1 = np.abs(spec1)
        psd2 = np.abs(spec2)
        if s1.spectrum_scaling.is_amplitude_scaling():
            psd1 = psd1**2
            psd2 = psd2**2

        lo = np.argmin(np.abs(f - f_range_hz[0]))
        hi = np.argmin(np.abs(f - f_range_hz[1]))
        f_sub = f[lo:hi]

        expected = np.zeros(self.sig1.number_of_channels)
        for n in range(self.sig1.number_of_channels):
            x = psd1[lo:hi, n].copy()
            y = psd2[lo:hi, n].copy()
            x /= np.sum(x)
            y /= np.sum(y)
            expected[n] = simpson(x / y - np.log10(x / y) - 1, x=f_sub)

        result = dsp.distances.itakura_saito(
            self.sig1,
            self.sig2,
            method="welch",
            f_range_hz=f_range_hz,
            energy_normalization=True,
            spectrum_parameters=None,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_snr(self):
        speech = dsp.Signal(
            join(os.path.dirname(__file__), "..", "example_data", "speech.flac")
        )
        noise = dsp.generators.noise(
            length_seconds=1.0,
            peak_level_dbfs=-30,
            sampling_rate_hz=speech.sampling_rate_hz,
        )
        dsp.distances.snr(speech, noise)
        # Multichannel
        speech = speech.append_signals([speech])
        dsp.distances.snr(speech, noise)

    def test_snr_matches_closed_form(self):
        """SNR is `20*log10(rms(signal)/rms(noise))` with `rms` defined as
        `sqrt(mean(x**2))`, i.e. including any DC component.

        """
        fs = 8_000
        n_samples = 4_000
        rng = np.random.default_rng(1)
        clean = rng.normal(0, 1.0, n_samples)
        noise = rng.normal(0, 0.1, n_samples)

        sig = dsp.Signal(None, clean[:, None], fs)
        noise_sig = dsp.Signal(None, noise[:, None], fs)

        result = dsp.distances.snr(sig, noise_sig)
        expected = 20 * np.log10(
            np.mean(clean**2.0) ** 0.5 / np.mean(noise**2.0) ** 0.5
        )
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_snr_accounts_for_dc_offset(self):
        """A DC offset raises the RMS, so it must lower the SNR."""
        fs = 8_000
        rng = np.random.default_rng(1)
        clean = rng.normal(0, 1.0, 4_000)
        noise = rng.normal(0, 0.1, 4_000)

        sig = dsp.Signal(None, clean[:, None], fs)
        noise_sig = dsp.Signal(None, noise[:, None], fs)
        offset_noise_sig = dsp.Signal(None, (noise + 0.5)[:, None], fs)

        assert (
            dsp.distances.snr(sig, offset_noise_sig)[0]
            < dsp.distances.snr(sig, noise_sig)[0]
        )

    def test_snr_invalid_parameters_raise(self):
        fs = 8_000
        sig = dsp.Signal(None, np.zeros((100, 2)), fs)
        noise_sig = dsp.Signal(None, np.zeros((100, 3)), fs)
        with pytest.raises(AssertionError):
            dsp.distances.snr(sig, noise_sig)
        other_fs_sig = dsp.Signal(None, np.zeros((100, 2)), fs * 2)
        with pytest.raises(AssertionError):
            dsp.distances.snr(sig, other_fs_sig)

    def test_si_sdr(self):
        # Single-channel
        dsp.distances.si_sdr(self.sig1, self.sig2)
        # Multi-channel
        sig2 = self.sig2.append_signals([self.sig2])
        dsp.distances.si_sdr(self.sig1, sig2)

    def test_si_sdr_matches_closed_form(self):
        """SI-SDR (Le Roux et al., 2019) rescales the target by the
        least-squares-optimal `alpha = (s . shat) / (s . s)` and computes
        `10*log10(||alpha*s||**2 / ||alpha*s - shat||**2)`.

        """
        fs = 8_000
        n_samples = 4_000
        t = np.arange(n_samples) / fs
        target = 0.7 * np.sin(2 * np.pi * 300 * t)

        rng = np.random.default_rng(7)
        gain = 1.8
        modified = gain * target + rng.normal(0, 0.05, n_samples)

        target_sig = dsp.Signal(None, target[:, None], fs)
        modified_sig = dsp.Signal(None, modified[:, None], fs)

        result = dsp.distances.si_sdr(target_sig, modified_sig)

        alpha = (target @ modified) / (target @ target)
        expected = 10 * np.log10(
            np.sum((alpha * target) ** 2) / np.sum((alpha * target - modified) ** 2)
        )
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_si_sdr_near_perfect_reconstruction_is_very_high(self):
        """A modified signal that is an exact scaled copy of the target has
        (up to floating-point rounding) zero distortion, so SI-SDR should
        be extremely high (round-trip identity; not exactly infinite
        because of floating-point rounding in the dot products).

        """
        fs = 8_000
        t = np.arange(4_000) / fs
        target = 0.5 * np.sin(2 * np.pi * 250 * t)
        modified = 3.0 * target

        target_sig = dsp.Signal(None, target[:, None], fs)
        modified_sig = dsp.Signal(None, modified[:, None], fs)

        result = dsp.distances.si_sdr(target_sig, modified_sig)
        assert result[0] > 100

    def test_fw_snr_seg(self):
        dsp.distances.fw_snr_seg(
            self.sig1,
            self.sig2,
            f_range_hz=[500, 4000],
            snr_range_db=[-10, 35],
            gamma=0.5,
        )

    def test_fw_snr_seg_plausibility(self):
        """No closed form exists for fwSNRseg, but two boundary behaviors
        are guaranteed: identical signals should saturate at the upper end
        of `snr_range_db`, and doubling the noise on top of the same clean
        signal should strictly decrease the measured SNR.

        """
        fs = 8_000
        n_samples = fs * 2
        t = np.arange(n_samples) / fs

        rng = np.random.default_rng(3)
        clean = 0.5 * np.sin(2 * np.pi * 500 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)
        noise = rng.normal(0, 0.05, n_samples)

        clean_sig = dsp.Signal(None, clean[:, None], fs)
        noisy_sig = dsp.Signal(None, (clean + noise)[:, None], fs)
        noisier_sig = dsp.Signal(None, (clean + 2 * noise)[:, None], fs)

        snr_range_db = [-10, 35]
        identical_snr = dsp.distances.fw_snr_seg(
            clean_sig, clean_sig, f_range_hz=[100, 3000], snr_range_db=snr_range_db
        )[0]
        assert identical_snr >= snr_range_db[1] - 1e-6

        snr_noisy = dsp.distances.fw_snr_seg(
            clean_sig, noisy_sig, f_range_hz=[100, 3000], snr_range_db=snr_range_db
        )[0]
        snr_noisier = dsp.distances.fw_snr_seg(
            clean_sig,
            noisier_sig,
            f_range_hz=[100, 3000],
            snr_range_db=snr_range_db,
        )[0]
        assert snr_noisy > snr_noisier

    def test_fw_snr_seg_invalid_parameters_raise(self):
        with pytest.raises(AssertionError):
            dsp.distances.fw_snr_seg(self.sig1, self.sig2, gamma=0.05)
        with pytest.raises(AssertionError):
            dsp.distances.fw_snr_seg(self.sig1, self.sig2, gamma=3)
        with pytest.raises(AssertionError):
            dsp.distances.fw_snr_seg(self.sig1, self.sig2, f_range_hz=[20, 1e6])
