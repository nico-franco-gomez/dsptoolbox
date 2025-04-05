"""
Tests for basic functionalities of the classes in dsptoolbox
"""

import pytest
import dsptoolbox as dsp
import numpy as np
from os.path import join
import os
import scipy.signal as sig
from matplotlib.pyplot import close

RIR_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "example_data",
    "rir.wav",
)
CHIRP_STEREO_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "example_data",
    "chirp_stereo.wav",
)


class TestSignal:
    """Testing signal functionalities."""

    # Some vectors to run the tests
    fs = 44100
    length_samp = 2 * fs
    channels = 4
    time_vec = np.random.normal(0, 0.1, (length_samp, channels))
    imag = np.random.normal(0, 0.1, (length_samp, channels))
    complex_time_vec = time_vec + 1j * imag

    def test_importing_from_file(self):
        path = join(
            os.path.dirname(__file__), "..", "example_data", "chirp.wav"
        )
        s = dsp.Signal(path)
        s.number_of_channels

    def test_creating_signal_from_vector(self):
        # Check real and imag (Multichannel)
        s = dsp.Signal(None, self.complex_time_vec, self.fs)
        real_cond = np.all(s.time_data == self.time_vec)
        imag_cond = np.all(s.time_data_imaginary == self.imag)
        assert real_cond and imag_cond

        # Check real and imag (Single channel)
        one_ch = self.time_vec[:, 0]
        one_ch_c = self.imag[:, 0]
        s = dsp.Signal(None, one_ch + 1j * one_ch_c, self.fs)
        real_cond = np.all(s.time_data == one_ch[..., None])
        imag_cond = np.all(s.time_data_imaginary == one_ch_c[..., None])
        assert real_cond and imag_cond

        # Broadcasting with too many dimensions
        r = np.random.normal(0, 0.1, (self.length_samp, self.channels, 1))
        s = dsp.Signal(None, r, self.fs)

        # Not broadcastable to time data vector
        with pytest.raises(AssertionError):
            r = np.random.normal(0, 0.1, (self.length_samp, self.channels, 4))
            s = dsp.Signal(None, r, self.fs)

        # Passing list
        li = [self.time_vec[:, i] for i in range(self.time_vec.shape[1])]
        s = dsp.Signal(None, li, self.fs)

        # Passing tuple
        tu = tuple(self.time_vec.T)
        s = dsp.Signal(None, tu, self.fs)

        # Not broadcastable to time data vector (with lists)
        with pytest.raises(AssertionError):
            r = np.random.normal(0, 0.1, (self.length_samp, self.channels, 4))
            r = list(r)
            s = dsp.Signal(None, r, self.fs)

    def test_get_spectrum(self):
        sp = np.fft.rfft(self.time_vec, axis=0)

        # Check normal FFT
        s = dsp.Signal(None, self.time_vec, self.fs)
        s.set_spectrum_parameters(
            method=dsp.SpectrumMethod.FFT,
            scaling=dsp.SpectrumScaling.FFTBackward,
            pad_to_fast_length=False,
        )
        _, sp_sig = s.get_spectrum()
        np.testing.assert_allclose(sp, sp_sig)

        # Check amplitude spectrum scaling for normal FFT
        s.set_spectrum_parameters(
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

        s.set_spectrum_parameters(
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

        # Try smoothing
        s.set_spectrum_parameters(
            method=dsp.SpectrumMethod.FFT,
            scaling=dsp.SpectrumScaling.AmplitudeSpectrum,
            pad_to_fast_length=False,
            smoothing=3,
        )
        s.get_spectrum()

    def test_managing_channels(self):
        # Add new channel
        new_ch = np.random.normal(0, 0.1, (self.length_samp, 1))
        t_vec = np.append(self.time_vec, new_ch, axis=1)
        s = dsp.Signal(None, self.time_vec.copy(), self.fs)
        assert np.all(
            t_vec == s.add_channel(None, new_ch, s.sampling_rate_hz).time_data
        )

        # Remove channel
        assert np.all(self.time_vec == s.remove_channel(-1).time_data)

        # Try to remove channel that does not exist
        with pytest.raises(AssertionError):
            s.remove_channel(self.channels + 10)

        # Get specific channel
        ch = s.get_channels(0)
        assert np.all(self.time_vec[:, 0][..., None] == ch.time_data)

        # Try to get a channel that does not exist
        with pytest.raises(IndexError):
            s.get_channels(self.channels + 10)

        # Swap channels
        new_order = np.arange(0, self.channels)[::-1]
        assert np.all(
            self.time_vec[:, ::-1] == s.swap_channels(new_order).time_data
        )

        # Try swapping channels wrongly
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

        # Setting sampling rate
        fs = 22000
        s.sampling_rate_hz = fs
        assert fs == s.sampling_rate_hz

        # Setting a float sampling rate
        with pytest.raises(AssertionError):
            s.sampling_rate_hz = 44100.5

        # Number of channels is generated right
        assert s.number_of_channels == self.channels

        # Spectrum parameters - Write+Read
        s.spectrum_method = dsp.SpectrumMethod.FFT
        assert s.spectrum_method == dsp.SpectrumMethod.FFT
        s.spectrum_scaling = dsp.SpectrumScaling.FFTOrthogonal
        assert s.spectrum_scaling == dsp.SpectrumScaling.FFTOrthogonal

        # Read-only properties - check
        s.number_of_channels
        s.length_samples
        s.length_seconds
        s.time_vector_s

        # Some properties are read-only
        with pytest.raises(AttributeError):
            s.number_of_channels = 10
        with pytest.raises(AttributeError):
            s.length_samples = 10
        with pytest.raises(AttributeError):
            s.length_seconds = 10.0
        with pytest.raises(AttributeError):
            s.time_vector_s = np.array([0.0, 1.0])

    def test_plot_generation(self):
        s = dsp.ImpulseResponse(
            time_data=self.time_vec, sampling_rate_hz=self.fs
        )
        # Test that all plots are generated without problems
        s.plot_magnitude()
        s.plot_magnitude(show_info_box=True)
        s.plot_time()
        s.plot_spectrogram(channel_number=0, log_freqs=True)
        s.plot_csm()
        s.plot_csm(with_phase=False)
        s.plot_spl(False)
        s.plot_spl(True)

        # Plot phase and group delay
        s.set_spectrum_parameters(method=dsp.SpectrumMethod.FFT)
        s.plot_phase()
        s.plot_phase(unwrap=True, smoothing=4, remove_ir_latency=None)
        s.plot_phase(remove_ir_latency="min_phase")
        s.plot_phase(remove_ir_latency="peak")
        s.plot_phase(remove_ir_latency=[10] * s.number_of_channels)
        with pytest.raises(ValueError):
            s.plot_phase(remove_ir_latency="no idea what removal method")
        s.plot_group_delay()

        # Try to plot phase having welch's method for magnitude
        with pytest.raises(AssertionError):
            s.set_spectrum_parameters(
                method=dsp.SpectrumMethod.WelchPeriodogram,
                window_length_samples=32,
            )
            s.plot_phase()

        # Plot signal with window and imaginary time data
        d = dsp.generators.dirac(
            length_samples=1024, delay_samples=512, sampling_rate_hz=self.fs
        )
        d, _ = dsp.transfer_functions.window_centered_ir(d, len(d))
        d = dsp.transforms.hilbert(d)
        d.plot_time()
        d.plot_spl()
        close("all")

    def test_get_power_spectrum_welch(self):
        # Try to get power spectrum
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
        s.set_spectrogram_parameters(
            window_length_samples=1024,
            window_type=dsp.Window.Hann,
            overlap_percent=50,
            fft_length_samples=4096,
            detrend=False,
            padding=False,
            scaling=dsp.SpectrumScaling.FFTBackward,
        )
        t, f, stft = s.get_spectrogram()
        s.set_spectrogram_parameters(
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
            assert False

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
        t_ = np.linspace(0, le / self.fs, le, endpoint=True)
        np.testing.assert_almost_equal(t, t_)

    def test_length_signal(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        assert len(s) == s.time_data.shape[0]
        assert s.length_samples == len(s)
        assert s.length_seconds == len(s) / s.sampling_rate_hz
        assert s.length_seconds == s.time_vector_s[-1]

    def test_constrain_amplitude(self):
        t = np.random.normal(0, 1, 200)
        s = dsp.Signal(None, t, sampling_rate_hz=100, constrain_amplitude=True)
        assert np.all(s.time_data <= 1)

        s = dsp.Signal(
            None, t, sampling_rate_hz=100, constrain_amplitude=False
        )
        assert np.all(t == s.time_data.squeeze())

    def test_sum_channels(self):
        n = np.random.normal(0, 0.01, (300, 2))
        nn = dsp.Signal.from_time_data(n, 10_000)
        np.testing.assert_array_equal(
            nn.sum_channels().time_data, np.sum(n, axis=1, keepdims=True)
        )

    def test_copy_with_new_time_data(self):
        n = dsp.Signal.from_time_data(self.time_vec, self.fs, False)

        #
        n.spectrum_method = dsp.SpectrumMethod.FFT
        n.spectrum_scaling = dsp.SpectrumScaling.PowerSpectrum
        n.set_spectrogram_parameters(256, window_type=dsp.Window.Blackman)
        n2 = n.copy_with_new_time_data(np.zeros((100, 1)))

        #
        assert n2.spectrum_scaling == dsp.SpectrumScaling.PowerSpectrum
        assert n2.spectrum_method == dsp.SpectrumMethod.FFT
        assert n2._spectrogram_parameters["window_length_samples"] == 256
        assert n2._spectrogram_parameters["window_type"] == dsp.Window.Blackman
        assert n2.constrain_amplitude == n.constrain_amplitude
        assert n2.time_data_imaginary == n.time_data_imaginary

        # === Complex
        n_comp = dsp.Signal.from_time_data(
            self.complex_time_vec, self.fs, True
        )
        n2_comp = n_comp.copy_with_new_time_data(np.zeros((100, 1)))
        assert not n2_comp.is_complex_signal

        # === Check slicing and ownership
        n = dsp.Signal.from_time_data(np.zeros((100, 2)), self.fs)
        n2 = n.copy_with_new_time_data(n.time_data[:, 0])
        n.time_data[0, ...] = 1.0
        assert np.all(n2.time_data[0] == 0.0)


class TestFilterClass:
    """Tests for the Filter class.

    Plotting
    Saving

    """

    # Create some filters to validate functions
    fs = 44100
    fir = sig.firwin(150, 1000, pass_zero="lowpass", fs=fs)
    iir = sig.iirfilter(
        8,
        1000,
        btype="lowpass",
        analog=False,
        ftype="butter",
        output="sos",
        fs=fs,
    )
    iir_ba = sig.iirfilter(
        8,
        1000,
        btype="lowpass",
        analog=False,
        ftype="butter",
        output="ba",
        fs=fs,
    )

    def get_iir(self, sos: bool = True) -> dsp.Filter:
        if sos:
            return dsp.Filter.from_sos(self.iir, self.fs)
        return dsp.Filter.from_ba(*self.iir_ba, self.fs)

    def get_fir(self):
        return dsp.Filter.from_ba(self.fir, np.array([1.0]), self.fs)

    def test_create_from_coefficients(self):
        # Try creating a filter from the coefficients, recognizing filter
        # type and returning the coefficients in the right way

        # FIR
        f = dsp.Filter(
            filter_coefficients={dsp.FilterCoefficientsType.Ba: [self.fir, 1]},
            sampling_rate_hz=self.fs,
        )
        assert f.is_fir
        b, _ = f.ba
        assert np.all(b == self.fir)

        # IIR
        f = dsp.Filter(
            filter_coefficients={dsp.FilterCoefficientsType.Sos: self.iir},
            sampling_rate_hz=self.fs,
        )
        assert f.is_iir
        sos = f.sos
        assert np.all(sos == self.iir)

    def test_filter_properties(self):
        iir = dsp.Filter.from_ba(*self.iir_ba, sampling_rate_hz=self.fs)
        assert type(iir.ba) is list
        iir.ba[1] = np.array([1.0])
        np.testing.assert_equal(iir.ba[1], np.array([1.0]))
        assert iir.order == len(self.iir_ba[0]) - 1
        assert not iir.has_sos

        with pytest.raises(ValueError):
            iir.ba = [0, "b"]
        with pytest.raises(AssertionError):
            iir.ba = [0, 1, 1]

        iir = dsp.Filter.from_sos(self.iir, sampling_rate_hz=self.fs)
        with pytest.raises(AssertionError):
            iir.sos = ["b"]
        with pytest.raises(AssertionError):
            iir.sos = np.zeros((3, 7))
        assert iir.order == self.iir.shape[0] * 2

        # Check order with sos
        sos = dsp.Filter.iir_filter(
            6,
            100.0,
            type_of_pass=dsp.FilterPassType.Lowpass,
            filter_design_method=dsp.IirDesignMethod.Butterworth,
            sampling_rate_hz=self.fs,
        )
        assert sos.order == 6
        sos = dsp.Filter.iir_filter(
            5,
            100.0,
            type_of_pass=dsp.FilterPassType.Lowpass,
            filter_design_method=dsp.IirDesignMethod.Butterworth,
            sampling_rate_hz=self.fs,
        )
        assert sos.order == 5
        assert sos.has_sos

        # Pass integer a coefficients
        fir = dsp.Filter.from_ba(self.fir, [1], self.fs)
        assert fir.ba[1].dtype == np.float64
        assert not fir.has_sos

    def test_filtering_fir(self):
        # Try filtering compared to scipy's functions
        t_vec = np.random.normal(0, 0.01, self.fs * 2)

        # FIR
        result_scipy = sig.lfilter(self.fir, [1], t_vec)
        s = dsp.Signal.from_time_data(t_vec, self.fs)
        f = self.get_fir()
        result_own = f.filter_signal(s).time_data.squeeze()
        np.testing.assert_allclose(result_scipy, result_own)

        # filtfilt
        result_scipy = sig.filtfilt(self.fir, [1], t_vec)
        result_own = f.filter_signal(s, zero_phase=True).time_data.squeeze()
        np.testing.assert_allclose(result_scipy, result_own)

        # Assert original data remains equal
        np.testing.assert_array_equal(s.time_data.squeeze(), t_vec)

    def test_filtering_iir(self):
        # Try filtering compared to scipy's functions
        t_vec = np.random.normal(0, 0.01, self.fs * 2)
        s = dsp.Signal(None, t_vec, self.fs)
        # IIR
        result_scipy = sig.sosfilt(self.iir, t_vec)
        f = self.get_iir()
        result_own = f.filter_signal(s).time_data.squeeze()
        np.testing.assert_allclose(result_scipy, result_own)

        # filtfilt
        result_scipy = sig.sosfiltfilt(self.iir, t_vec)
        result_own = f.filter_signal(s, zero_phase=True).time_data.squeeze()
        np.testing.assert_allclose(result_scipy, result_own)

        # Assert original data remains equal
        np.testing.assert_array_equal(s.time_data.squeeze(), t_vec)

    def test_plots(self):
        f = self.get_iir()
        # Standard config
        f.plot_magnitude()
        f.plot_phase()
        f.plot_group_delay()
        f.plot_zp()

        # More config
        f.plot_magnitude(show_info_box=True)
        f.plot_phase(show_info_box=True)
        f.plot_group_delay(show_info_box=True)
        f.plot_zp(show_info_box=True)

        f.plot_magnitude(normalize=dsp.MagnitudeNormalization.OneKhz)
        f.plot_magnitude(normalize=dsp.MagnitudeNormalization.Max)
        f.plot_magnitude(normalize=dsp.MagnitudeNormalization.Energy)
        f.plot_magnitude(
            normalize=dsp.MagnitudeNormalization.OneKhzFirstChannel
        )
        f.plot_magnitude(normalize=dsp.MagnitudeNormalization.MaxFirstChannel)
        f.plot_magnitude(
            normalize=dsp.MagnitudeNormalization.EnergyFirstChannel
        )

        with pytest.raises(AssertionError):
            f.plot_taps()

        f2 = self.get_fir()
        f2.plot_taps()
        close("all")

    def test_get_coefficients(self):
        f = self.get_iir()
        f.get_coefficients(coefficients_mode=dsp.FilterCoefficientsType.Ba)
        f.get_coefficients(coefficients_mode=dsp.FilterCoefficientsType.Sos)
        f.get_coefficients(coefficients_mode=dsp.FilterCoefficientsType.Zpk)

    def test_get_ir(self):
        f = self.get_iir()
        f.get_ir()

    def test_other_functionalities(self):
        #
        dsp.Filter.fir_from_file(RIR_PATH)

        #
        f = self.get_iir()
        f.show_info()
        print(f)
        f.copy()
        f.initialize_zi(1)
        with pytest.raises(AssertionError):
            f.initialize_zi(0)

    def test_get_transfer_function(self):
        # Functionality
        f = self.get_iir()
        freqs = np.linspace(1, 4e3, 200)
        f.get_transfer_function(freqs)

        f = self.get_fir()
        f.get_transfer_function(freqs)

        f = dsp.Filter.biquad(
            eq_type=dsp.BiquadEqType.Peaking,
            frequency_hz=200,
            gain_db=3,
            q=0.7,
            sampling_rate_hz=self.fs,
        )
        f.get_transfer_function(freqs)

    def test_all_biquads(self):
        # Only functionality
        for t in [
            dsp.BiquadEqType.Allpass,
            dsp.BiquadEqType.AllpassFirstOrder,
            dsp.BiquadEqType.BandpassPeak,
            dsp.BiquadEqType.BandpassSkirt,
            dsp.BiquadEqType.Highpass,
            dsp.BiquadEqType.HighpassFirstOrder,
            dsp.BiquadEqType.Highshelf,
            dsp.BiquadEqType.Inverter,
            dsp.BiquadEqType.Lowpass,
            dsp.BiquadEqType.LowpassFirstOrder,
            dsp.BiquadEqType.Lowshelf,
            dsp.BiquadEqType.Notch,
            dsp.BiquadEqType.Peaking,
        ]:
            dsp.Filter.biquad(t, 100.0, 2.0, 0.7, 2000)

    def test_filter_and_resampling_IIR(self):
        f = self.get_iir()

        # Time vector
        t_vec = np.random.normal(0, 0.01, self.fs * 2)

        # dsptoolbox
        t_signal = dsp.Signal(None, t_vec, self.fs)
        t_res = f.filter_and_resample_signal(t_signal, self.fs // 2)
        t_res = t_res.time_data.squeeze()

        # Scipy
        t_res_sc = sig.sosfilt(self.iir, t_vec)
        t_res_sc = t_res_sc[::2]
        assert np.all(np.isclose(t_res_sc, t_res))

    def test_filter_and_resampling_FIR(self):
        # Lowpass filter for antialiasing
        b = sig.firwin(
            1500,
            (self.fs // 2 // 2),
            pass_zero="lowpass",
            fs=self.fs,
            window="flattop",
        )
        f = dsp.Filter(
            filter_coefficients={dsp.FilterCoefficientsType.Ba: [b, 1]},
            sampling_rate_hz=self.fs,
        )
        # Time vector
        t_vec = np.random.normal(0, 0.01, self.fs * 2)

        # dsptoolbox
        t_signal = dsp.Signal(None, t_vec, self.fs)
        t_res = f.filter_and_resample_signal(t_signal, self.fs // 2)
        t_res = t_res.time_data.squeeze()

        # Scipy
        t_res_sc = sig.resample_poly(t_vec, up=1, down=2, window=b)

        assert np.all(np.isclose(t_res_sc, t_res))

    def test_filter_length(self):
        b = sig.firwin(
            1500,
            (self.fs // 2 // 2),
            pass_zero="lowpass",
            fs=self.fs,
            window="flattop",
        )
        f = dsp.Filter(
            filter_coefficients={dsp.FilterCoefficientsType.Ba: [b, 1]},
            sampling_rate_hz=self.fs,
        )
        assert len(f) == len(b)

    def test_order(self):
        b = sig.firwin(
            1500,
            (self.fs // 2 // 2),
            pass_zero="lowpass",
            fs=self.fs,
            window="flattop",
        )
        f = dsp.Filter(
            filter_coefficients={dsp.FilterCoefficientsType.Ba: [b, 1]},
            sampling_rate_hz=self.fs,
        )
        assert f.order == len(b) - 1

    def test_group_delay(self):
        f_log = dsp.tools.log_frequency_vector([20, 20e3], 128)
        bb = dsp.Filter.biquad(
            eq_type=dsp.BiquadEqType.Peaking,
            frequency_hz=300,
            gain_db=10,
            q=1.5,
            sampling_rate_hz=48000,
        )
        gd = bb.get_group_delay(f_log)
        ff, gg = dsp.transfer_functions.group_delay(
            bb.get_ir(length_samples=2**14)
        )

        interpolated_gd = dsp.tools.interpolate_fr(
            ff, gg.squeeze(), f_log, interpolation_scheme="cubic"
        )
        np.testing.assert_allclose(interpolated_gd, gd, atol=1e-6)

        # Check it runs
        gd = bb.get_group_delay(f_log, False)


class TestFilterBankClass:
    fs = 44100

    def get_iir_filter(self) -> dsp.Filter:
        return dsp.Filter.iir_filter(
            5,
            frequency_hz=[1510, 2000],
            type_of_pass=dsp.FilterPassType.Bandpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=self.fs,
        )

    def get_fir_filter(self, other_sampling_rate=False) -> dsp.Filter:
        return dsp.Filter.fir_filter(
            order=150,
            frequency_hz=[1500, 2000],
            type_of_pass=dsp.FilterPassType.Bandpass,
            sampling_rate_hz=(
                self.fs if not other_sampling_rate else self.fs // 2
            ),
        )

    def test_create_filter_bank(self):
        # Create filter bank sequentially
        fb = dsp.FilterBank()
        fb.add_filter(self.get_iir_filter())

        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == self.fs

        fb.add_filter(self.get_fir_filter())

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Create filter bank passing a list
        filters = []
        filters.append(self.get_iir_filter())
        filters.append(self.get_fir_filter())
        fb = dsp.FilterBank(
            filters=filters,
            same_sampling_rate=True,
            info={"Type of filter bank": "Test"},
        )
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Reading FIRs from files
        firs = dsp.FilterBank.firs_from_file(RIR_PATH)
        assert len(firs) == 1
        firs = dsp.FilterBank.firs_from_file(CHIRP_STEREO_PATH)
        assert len(firs) == 2

    def test_plots(self):
        # Create
        fb = dsp.FilterBank()
        fb.add_filter(self.get_iir_filter())
        fb.add_filter(self.get_fir_filter())

        # Get plots
        fb.plot_magnitude(mode=dsp.FilterBankMode.Parallel)
        fb.plot_magnitude(mode=dsp.FilterBankMode.Sequential)
        fb.plot_magnitude(mode=dsp.FilterBankMode.Summed)
        fb.plot_magnitude(mode=dsp.FilterBankMode.Parallel, test_zi=True)

        fb.plot_phase(mode=dsp.FilterBankMode.Parallel)
        fb.plot_phase(mode=dsp.FilterBankMode.Sequential)
        fb.plot_phase(mode=dsp.FilterBankMode.Summed)
        fb.plot_phase(mode=dsp.FilterBankMode.Parallel, test_zi=True)

        fb.plot_group_delay(mode=dsp.FilterBankMode.Parallel)
        fb.plot_group_delay(mode=dsp.FilterBankMode.Sequential)
        fb.plot_group_delay(mode=dsp.FilterBankMode.Summed)
        fb.plot_group_delay(mode=dsp.FilterBankMode.Parallel, test_zi=True)

    def test_filterbank_functionalities(self):
        fb = dsp.FilterBank()
        fb.add_filter(self.get_iir_filter())
        fb.add_filter(self.get_fir_filter())

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Remove
        fb.remove_filter(0)
        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == self.fs

        # Readd
        fb.add_filter(self.get_fir_filter())
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Swap (and Assertions)
        fb.swap_filters([1, 0])
        assert fb.number_of_filters == 2
        assert len(fb) == 2
        assert fb.sampling_rate_hz == self.fs

        with pytest.raises(AssertionError):
            fb.swap_filters([1, 1])
        with pytest.raises(AssertionError):
            fb.swap_filters([1, 2])

        # Others
        fb.get_ir(dsp.FilterBankMode.Parallel)
        fb.copy()
        fb.show_info()
        print(fb)

    def test_filtering(self):
        # Create
        fb = dsp.FilterBank()
        fb.add_filter(self.get_iir_filter())
        fb.add_filter(self.get_fir_filter())

        t_vec = np.random.normal(0, 0.01, (self.fs * 3, 2))
        s = dsp.Signal(None, t_vec, self.fs)

        # Type of output and filter results
        filt1 = fb.filters[0].get_coefficients(
            coefficients_mode=dsp.FilterCoefficientsType.Sos
        )
        filt2, _ = fb.filters[1].get_coefficients(
            coefficients_mode=dsp.FilterCoefficientsType.Ba
        )
        # Parallel
        s_ = fb.filter_signal(
            s, mode=dsp.FilterBankMode.Parallel, activate_zi=False
        )
        assert type(s_) is dsp.MultiBandSignal
        assert s_.number_of_bands == fb.number_of_filters
        assert np.all(
            np.isclose(
                s_.bands[0].time_data[:, 0], sig.sosfilt(filt1, t_vec[:, 0])
            )
        )
        assert np.all(
            np.isclose(
                s_.bands[1].time_data[:, 0],
                sig.lfilter(filt2, [1], t_vec[:, 0]),
            )
        )

        # Sequential mode
        s_ = fb.filter_signal(
            s, mode=dsp.FilterBankMode.Sequential, activate_zi=False
        )
        assert type(s_) is dsp.Signal
        # Change order (just because they're linear systems)
        temp = sig.lfilter(filt2, [1], s.time_data[:, 1])
        temp = sig.sosfilt(filt1, temp)
        # Try second channel
        assert np.all(np.isclose(s_.time_data[:, 1], temp))

        # Summed mode
        s_ = fb.filter_signal(
            s, mode=dsp.FilterBankMode.Summed, activate_zi=False
        )
        assert type(s_) is dsp.Signal
        # Add together
        temp = sig.lfilter(filt2, [1], s.time_data[:, 1])
        temp += sig.sosfilt(filt1, s.time_data[:, 1])
        assert np.all(np.isclose(s_.time_data[:, 1], temp))

        # Filter's zi
        s_ = fb.filter_signal(
            s, mode=dsp.FilterBankMode.Parallel, activate_zi=True
        )
        s_ = fb.filter_signal(
            s, mode=dsp.FilterBankMode.Sequential, activate_zi=True
        )
        s_ = fb.filter_signal(
            s, mode=dsp.FilterBankMode.Summed, activate_zi=True
        )

        # Zero-phase filtering
        s_ = fb.filter_signal(
            s, mode=dsp.FilterBankMode.Parallel, zero_phase=True
        )
        s_ = fb.filter_signal(
            s, mode=dsp.FilterBankMode.Sequential, zero_phase=True
        )
        s_ = fb.filter_signal(
            s, mode=dsp.FilterBankMode.Summed, zero_phase=True
        )

        # No zi and zero phase filtering at the same time!
        with pytest.raises(AssertionError):
            s_ = fb.filter_signal(
                s,
                mode=dsp.FilterBankMode.Summed,
                activate_zi=True,
                zero_phase=True,
            )

    def test_multirate(self):
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb.add_filter(self.get_iir_filter())

        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == [self.fs]

        fb.add_filter(self.get_fir_filter(True))

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs, self.fs // 2]

        # Remove
        fb.remove_filter(0)
        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == [self.fs // 2]

        # Readd
        fb.add_filter(self.get_fir_filter())
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs // 2, self.fs]

        # Swap (and Assertions)
        fb.swap_filters([1, 0])
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs, self.fs // 2]

        # Should not be possible to create
        with pytest.raises(AssertionError):
            fb = dsp.FilterBank(same_sampling_rate=True)
            fb.add_filter(self.get_iir_filter())
            fb.add_filter(self.get_fir_filter(True))

        # Create filter bank passing a list
        filters = []
        filters.append(self.get_iir_filter())
        filters.append(self.get_fir_filter(True))
        fb = dsp.FilterBank(
            filters=filters,
            same_sampling_rate=False,
            info={"Type of filter bank": "Test"},
        )

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs, self.fs // 2]

        with pytest.raises(AssertionError):
            fb = dsp.FilterBank(
                filters=filters,
                same_sampling_rate=True,
                info={"Type of filter bank": "Test"},
            )

    def test_plotting_multirate(self):
        # Should not fail but no plots are created
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb.add_filter(self.get_iir_filter())
        fb.add_filter(self.get_fir_filter(True))

        fb.plot_magnitude(dsp.FilterBankMode.Parallel)
        fb.plot_phase(dsp.FilterBankMode.Parallel)
        fb.plot_group_delay(dsp.FilterBankMode.Parallel)
        fb.get_ir(dsp.FilterBankMode.Parallel)
        with pytest.raises(AssertionError):
            fb.get_ir(mode=dsp.FilterBankMode.Summed)

    def test_filtering_multirate_multiband(self):
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb.add_filter(self.get_iir_filter())
        fb.add_filter(self.get_fir_filter(True))

        s1 = dsp.generators.noise(length_seconds=1, sampling_rate_hz=self.fs)
        s2 = dsp.generators.noise(
            length_seconds=2, sampling_rate_hz=self.fs // 2
        )

        mb = dsp.MultiBandSignal(bands=[s1, s2], same_sampling_rate=False)
        assert np.all(mb.sampling_rate_hz == [self.fs, self.fs // 2])

        mb_ = fb.filter_multiband_signal(
            mb, activate_zi=False, zero_phase=False
        )
        assert np.all(mb_.sampling_rate_hz == [self.fs, self.fs // 2])
        fb.filter_multiband_signal(mb, activate_zi=True, zero_phase=False)
        fb.filter_multiband_signal(mb, activate_zi=False, zero_phase=True)

    def test_iterator(self):
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb.add_filter(self.get_iir_filter())
        fb.add_filter(self.get_fir_filter(True))
        for n in fb:
            assert dsp.Filter == type(n)

    def test_transfer_function(self):
        # Create
        fb = dsp.FilterBank(same_sampling_rate=False)
        fb.add_filter(self.get_iir_filter())
        fb.add_filter(self.get_fir_filter())

        freqs = np.linspace(1, 2e3, 400)
        fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Parallel)
        fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Sequential)
        fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Summed)

        with pytest.raises(AssertionError):
            freqs = np.linspace(1, self.fs, 40)
            fb.get_transfer_function(freqs, mode=dsp.FilterBankMode.Parallel)


class TestMultiBandSignal:
    fs = 44100
    s = np.random.normal(0, 0.01, (fs * 3, 3))
    s = dsp.Signal(None, s, fs)
    fb = dsp.filterbanks.auditory_filters_gammatone(
        frequency_range_hz=[500, 1200], sampling_rate_hz=fs
    )

    def get_mb(self) -> dsp.MultiBandSignal:
        return self.fb.filter_signal(self.s, dsp.FilterBankMode.Parallel)

    def test_create_and_general_functionalities(self):
        # Test creating from two signals and other functionalities
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        assert mbs.number_of_bands == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        assert mbs.sampling_rate_hz == self.s.sampling_rate_hz

        mbs.add_band(self.s)
        assert mbs.number_of_bands == 3
        assert mbs.number_of_channels == self.s.number_of_channels
        assert mbs.sampling_rate_hz == self.s.sampling_rate_hz
        mbs.remove_band(0)
        assert mbs.number_of_bands == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        mbs.swap_bands([1, 0])
        mbs.show_info()
        print(mbs)
        mbs.copy()

        with pytest.raises(IndexError):
            mbs.remove_band(4)
        with pytest.raises(AssertionError):
            mbs.swap_bands([1, 1])
        with pytest.raises(AssertionError):
            mbs.swap_bands([5, 0])
        with pytest.raises(AssertionError):
            # Inconsistent data in regards to complex values
            s2 = self.s.copy()
            s2.time_data = s2.time_data + 1j
            mbs = dsp.MultiBandSignal(
                bands=[self.s, s2],
                same_sampling_rate=True,
                info=dict(information="test filter bank"),
            )

        # Create from filter bank
        mbs = self.fb.filter_signal(self.s, dsp.FilterBankMode.Parallel)
        assert type(mbs) is dsp.MultiBandSignal

    def test_collapse(self):
        td = self.s.time_data.copy()
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        mbs_ = mbs.collapse()

        assert np.all(mbs_.time_data == td + td)

    def test_get_all_bands(self):
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        mbs_ = mbs.get_all_bands(0)
        assert type(mbs_) is dsp.Signal
        # Number of channels has to match number of bands
        assert mbs_.number_of_channels == mbs.number_of_bands

    def test_get_all_time_data(self):
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        td, fs = mbs.get_all_time_data()

        td_s = self.s.time_data
        td_s = np.concatenate([td_s[:, None, :], td_s[:, None, :]], axis=1)

        assert np.all(td == td_s)
        assert fs == self.s.sampling_rate_hz

        # Complex time data
        s2 = self.s.copy()
        s2.time_data = s2.time_data + 1j
        mbs = dsp.MultiBandSignal(
            bands=[s2, s2],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        td, fs = mbs.get_all_time_data()

        td_s = s2.time_data + 1j * s2.time_data_imaginary
        td_s = np.concatenate([td_s[:, None, :], td_s[:, None, :]], axis=1)

        assert np.all(td == td_s)
        assert fs == self.s.sampling_rate_hz

        # Multirate
        s2 = dsp.resample(self.s, self.s.sampling_rate_hz // 2)
        mbs = dsp.MultiBandSignal(
            bands=[self.s, s2],
            same_sampling_rate=False,
            info=dict(information="test filter bank"),
        )
        tds = mbs.get_all_time_data()

        assert np.all(tds[0][0] == self.s.time_data)
        assert np.all(tds[1][0] == s2.time_data)

        assert np.all(tds[0][1] == self.s.sampling_rate_hz)
        assert np.all(tds[1][1] == s2.sampling_rate_hz)

    def test_multirate(self):
        s2 = dsp.resample(self.s, self.s.sampling_rate_hz // 2)

        # Parameter same sampling rate has to be False
        with pytest.raises(AssertionError):
            mbs = dsp.MultiBandSignal(
                bands=[self.s, s2],
                same_sampling_rate=True,
                info=dict(information="test filter bank"),
            )

        mbs = dsp.MultiBandSignal(
            bands=[self.s, s2],
            same_sampling_rate=False,
            info=dict(information="test filter bank"),
        )
        assert mbs.number_of_bands == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        assert np.all(
            mbs.sampling_rate_hz
            == [self.s.sampling_rate_hz, s2.sampling_rate_hz]
        )

        mbs.add_band(self.s)
        assert mbs.number_of_bands == 3
        assert mbs.number_of_channels == self.s.number_of_channels
        assert np.all(
            mbs.sampling_rate_hz
            == [
                self.s.sampling_rate_hz,
                s2.sampling_rate_hz,
                self.s.sampling_rate_hz,
            ]
        )

        mbs.remove_band(0)
        assert mbs.number_of_bands == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        assert np.all(
            mbs.sampling_rate_hz
            == [s2.sampling_rate_hz, self.s.sampling_rate_hz]
        )

        mbs.swap_bands([1, 0])
        assert mbs.number_of_bands == 2
        assert len(mbs) == 2
        assert mbs.number_of_channels == self.s.number_of_channels
        assert np.all(
            mbs.sampling_rate_hz
            == [self.s.sampling_rate_hz, s2.sampling_rate_hz]
        )
        mbs.show_info()

    def test_iterator(self):
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        for n in mbs:
            assert dsp.Signal == type(n)

    def test_multibandsignal_properties(self):
        mb = self.get_mb()

        # Get
        mb.length_seconds
        mb.number_of_bands
        mb.number_of_channels
        mb.length_samples

        mb.bands

        # Read-only properties
        with pytest.raises(AttributeError):
            mb.length_seconds = 1.0
        with pytest.raises(AttributeError):
            mb.number_of_bands = 1
        with pytest.raises(AttributeError):
            mb.number_of_channels = 1
        with pytest.raises(AttributeError):
            mb.length_samples = 1


class TestImpulseResponse:
    fs_hz = 10_000
    seconds = 2
    d = dsp.generators.dirac(seconds * fs_hz, sampling_rate_hz=fs_hz)

    path_rir = join(os.path.dirname(__file__), "..", "example_data", "rir.wav")

    def get_ir(self):
        return dsp.ImpulseResponse.from_file(self.path_rir)

    def test_constructors(self):
        rir = self.get_ir()
        dsp.ImpulseResponse.from_time_data(rir.time_data, rir.sampling_rate_hz)
        dsp.ImpulseResponse.from_signal(dsp.Signal.from_file(self.path_rir))

    def test_channel_handling_with_window(self):
        rir = self.get_ir()
        rir = dsp.transfer_functions.window_centered_ir(rir, len(rir))[0]

        # Add channel
        rir.add_channel(self.path_rir)
        assert not hasattr(rir, "window")

        # Window again
        rir = dsp.transfer_functions.window_centered_ir(rir, len(rir))[0]
        assert rir.window.shape == rir.time_data.shape
        np.testing.assert_array_equal(rir.window[:, 1], rir.window[:, 0])

        # Remove channel
        rir.remove_channel(1)

        # Swap channels
        rir.add_channel(self.path_rir)
        rir.add_channel(self.path_rir)
        rir.swap_channels([2, 1, 0])

    def test_plotting_with_window(self):
        rir = self.get_ir()
        rir = dsp.transfer_functions.window_centered_ir(rir, len(rir))[0]
        rir.plot_time()
        rir.plot_spl()
        rir.add_channel(self.path_rir)
        rir.plot_time()
        rir.plot_spl()
        # dsp.plots.show()

    def test_other_plotting(self):
        rir = self.get_ir()
        rir.plot_bode()
        rir.plot_bode(show_group_delay=True)
        # dsp.plots.show()


class TestFilterTopologies:
    fs_hz = 24_000

    def get_noise(self):
        return dsp.generators.noise(
            length_seconds=1, sampling_rate_hz=self.fs_hz
        )

    def test_svfilter(self):
        # Functionality
        PLOT = False
        sv_filt = dsp.filterbanks.StateVariableFilter(1000.0, 1.0, self.fs_hz)
        n = self.get_noise()

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = sv_filt.process_sample(td[ind], 0)[0]

        sv_filt.reset_state()
        mb = sv_filt.filter_signal(n)
        n2 = mb.get_all_bands(0)

        np.testing.assert_array_equal(td, n2.time_data[:, 0])

        if PLOT:
            n2.spectrum_method = dsp.SpectrumMethod.FFT
            _, ax = n2.plot_magnitude(normalize=None)
            ax.plot(
                np.fft.rfftfreq(len(td), 1 / self.fs_hz),
                dsp.tools.to_db(np.fft.rfft(td), True),
            )
            dsp.plots.show()

    def test_lattice_ladder_filter(self):
        PLOT = False
        n = self.get_noise()

        # IIR sos
        iir = dsp.Filter.iir_filter(
            4,
            1000.0,
            type_of_pass=dsp.FilterPassType.Lowpass,
            filter_design_method=dsp.IirDesignMethod.Butterworth,
            sampling_rate_hz=self.fs_hz,
        )
        llf = dsp.filterbanks.LatticeLadderFilter.from_filter(iir)

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = llf.process_sample(td[ind], 0)

        llf.reset_state()
        n2 = llf.filter_signal(n)
        np.testing.assert_array_equal(td, n2.time_data[:, 0])
        np.testing.assert_allclose(
            td,
            sig.sosfilt(
                iir.get_coefficients(dsp.FilterCoefficientsType.Sos),
                n.time_data.squeeze(),
            ),
        )

        # IIR ba
        iir = dsp.Filter.from_ba(
            *iir.get_coefficients(dsp.FilterCoefficientsType.Ba),
            sampling_rate_hz=self.fs_hz,
        )
        llf = dsp.filterbanks.LatticeLadderFilter.from_filter(iir)

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = llf.process_sample(td[ind], 0)

        llf.reset_state()
        n2 = llf.filter_signal(n)
        np.testing.assert_array_equal(td, n2.time_data[:, 0])
        np.testing.assert_allclose(
            td,
            sig.lfilter(
                *iir.get_coefficients(dsp.FilterCoefficientsType.Ba),
                n.time_data.squeeze(),
            ),
        )

        # FIR ba (this filter does not work due to the reflection coefficients,
        # maybe use another one ?)
        # fir = dsp.transfer_functions.ir_to_filter(iir.get_ir(1024))
        # llf = dsp.filterbanks.convert_into_lattice_filter(fir)

        # td = n.time_data.squeeze()
        # for ind in np.arange(len(td)):
        #     td[ind] = llf.process_sample(td[ind], 0)

        # llf.reset_state()
        # n2 = llf.filter_signal(n)
        # np.testing.assert_array_equal(td, n2.time_data[:, 0])
        # np.testing.assert_allclose(
        #     td,
        #     sig.lfilter(*fir.get_coefficients("ba"), n.time_data.squeeze()),
        # )

        if PLOT:
            n2.spectrum_method = dsp.SpectrumMethod.FFT
            _, ax = n2.plot_magnitude(normalize=None)
            ax.plot(
                np.fft.rfftfreq(len(td), 1 / self.fs_hz),
                dsp.tools.to_db(np.fft.rfft(td), True),
            )
            dsp.plots.show()

    def test_iir_filter(self):
        iir_original = dsp.Filter.iir_filter(
            4,
            1000.0,
            type_of_pass=dsp.FilterPassType.Highpass,
            filter_design_method=dsp.IirDesignMethod.Butterworth,
            sampling_rate_hz=self.fs_hz,
        )
        b, a = iir_original.get_coefficients(dsp.FilterCoefficientsType.Ba)
        iir = dsp.filterbanks.IIRFilter(b, a)
        n = self.get_noise()

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = iir.process_sample(td[ind], 0)

        np.testing.assert_allclose(td, sig.lfilter(b, a, n.time_data[:, 0]))

        # Check functionality of constructor
        dsp.filterbanks.IIRFilter.from_filter(iir_original)

    def test_fir_filter(self):
        fir_original = dsp.Filter.fir_filter(
            25,
            1000.0,
            type_of_pass=dsp.FilterPassType.Lowpass,
            window=dsp.Window.Blackman,
            sampling_rate_hz=self.fs_hz,
        )
        b, _ = fir_original.get_coefficients(dsp.FilterCoefficientsType.Ba)
        b = b[: len(b) // 2 + 3]  # some asymmetrical window
        fir = dsp.filterbanks.FIRFilter(b)
        n = self.get_noise()

        td = n.time_data.copy().squeeze()
        for ind in np.arange(len(td)):
            td[ind] = fir.process_sample(td[ind], 0)

        np.testing.assert_allclose(td, sig.lfilter(b, [1], n.time_data[:, 0]))

        # Check functionality of constructor
        dsp.filterbanks.FIRFilter.from_filter(fir_original)

    def test_kautz_filters(self):
        # Only functionality
        fs_hz = 48000

        # Define some poles for smoothing according to Bank, B. (2022). Warped,
        # Kautz, and Fixed-Pole Parallel Filters: A Review. Journal of the
        # Audio Engineering Society.
        fractional_octave_smoothing = 24  # beta
        K = int(10 * (fractional_octave_smoothing / 2) + 1)
        pole_freqs_hz = np.logspace(
            np.log10(20), np.log10(20480), K, endpoint=True
        )
        pole_freqs_rad = 2 * np.pi * pole_freqs_hz / fs_hz
        bandwidth = np.zeros_like(pole_freqs_rad)
        bandwidth[0] = pole_freqs_rad[1] - pole_freqs_rad[0]
        bandwidth[-1] = pole_freqs_rad[-1] - pole_freqs_rad[-2]
        bandwidth[1:-1] = (
            pole_freqs_rad[2:] - pole_freqs_rad[:-2]
        ) / 2  # Eq. 24
        poles = np.exp(-bandwidth / 2 + 1j * pole_freqs_rad)  # Eq. 25

        # Add two real poles just for testing
        poles = np.hstack([0.1, poles, -0.4])

        filter = dsp.filterbanks.KautzFilter(poles, fs_hz)

        # Process sample and complete signal, compare both are equal
        d = dsp.generators.dirac(2**11, sampling_rate_hz=fs_hz)
        d.constrain_amplitude = False
        td = d.time_data.squeeze()
        for ind in np.arange(len(td)):
            td[ind] = filter.process_sample(td[ind], 0)
        dd = filter.get_ir(2**11)

        # Normalize
        td /= np.max(np.abs(td))
        dd = dsp.normalize(dd, norm_dbfs=0.0)
        np.testing.assert_allclose(td, dd.time_data.squeeze(), rtol=1e-6)

        filter.fit_coefficients_to_ir(d)
        assert np.any(filter.coefficients_complex_poles != 1.0)
        assert np.any(filter.coefficients_real_poles != 1.0)

    def test_exponential_averager(self):
        # Only functionality
        n = np.random.normal(0, 0.1, 200)
        f = dsp.filterbanks.ExponentialAverageFilter(1e-3, 1e-3, self.fs_hz)
        for i in n:
            f.process_sample(i, 0)

    def test_parallel_filterbank(self):
        # Only functionality
        rir = dsp.ImpulseResponse(
            os.path.join(
                os.path.dirname(__file__), "..", "example_data", "rir.wav"
            )
        )
        poles = np.logspace(
            np.log10(1e-2), np.log10(np.pi * 0.95), 3, endpoint=True
        )
        poles = 0.5 * np.exp(1j * poles)

        # All cases
        fb = dsp.filterbanks.ParallelFilter(poles, 0, rir.sampling_rate_hz)
        fb.fit_to_ir(rir)
        fb.get_ir(256)
        fb.set_n_channels(3)

        for i in rir.time_data:
            fb.process_sample(i, 1)
        fb.reset_state()
        iir_coeffs = np.random.normal(0, 0.1, (len(poles), 2))
        fb.set_coefficients(iir_coeffs, np.random.normal(0, 0.01, 10))

        fb = dsp.filterbanks.ParallelFilter(poles, 1, rir.sampling_rate_hz)
        fb.set_parameters(4, 0.0)
        fb.fit_to_ir(rir)
        fb = dsp.filterbanks.ParallelFilter(poles, 3, rir.sampling_rate_hz)
        fb.set_parameters(10, 1e-3)
        fb.fit_to_ir(rir)

    def test_filter_chain(self):
        # Only functionality
        fc = dsp.filterbanks.FilterChain(
            [
                dsp.filterbanks.IIRFilter(
                    np.array([0.5]), np.array([0.5, 0.1])
                ),
                dsp.filterbanks.FIRFilter(np.array([0.5, 0.5])),
            ]
        )
        assert fc.n_filters == 2

        n = np.random.normal(0, 0.1, 50)

        for nn in n:
            fc.process_sample(nn, 0)

        fc.reset_state()
        fc.set_n_channels(1)

    def test_state_space_filtering(self):
        # Check filter's output against usual TDF2 implementation
        ff = dsp.Filter.biquad(
            dsp.BiquadEqType.Peaking, 100, 6, 0.7, self.fs_hz
        )
        b, a = ff.get_coefficients(dsp.FilterCoefficientsType.Ba)
        A, B, C, D = sig.tf2ss(b, a)
        noise = dsp.generators.noise(
            length_seconds=1.0,
            type_of_noise=-2.0,
            sampling_rate_hz=self.fs_hz,
            number_of_channels=2,
        )
        ff2 = dsp.filterbanks.StateSpaceFilter(A, B, C, D)
        ff2.set_n_channels(noise.number_of_channels)
        reference = ff.filter_signal(noise)

        channel = 0
        for ch_n in noise:
            output = np.zeros(len(ch_n))
            for ind in range(len(ch_n)):
                output[ind] = ff2.process_sample(ch_n[ind], channel)
            np.testing.assert_allclose(reference.time_data[:, channel], output)
            channel += 1

        # Constructors
        # TODO: check output
        iir = dsp.Filter.iir_filter(
            12, 500.0, dsp.FilterPassType.Lowpass, self.fs_hz
        )
        dsp.filterbanks.StateSpaceFilter.from_filter(iir)
        out = dsp.filterbanks.StateSpaceFilter.from_filter_as_sos_list(iir)
        assert len(out) == 6

    def test_fir_filter_overlap_save(self):
        rir = dsp.ImpulseResponse.from_file(RIR_PATH)
        noise = dsp.resample(self.get_noise(), rir.sampling_rate_hz)
        fir = dsp.filterbanks.FIRFilterOverlapSave.from_filter(
            dsp.transfer_functions.ir_to_filter(rir)
        )

        blocksize = 512
        fir.prepare(blocksize, 1)
        n_blocks = len(noise) // blocksize + 1
        noise = dsp.pad_trim(noise, n_blocks * blocksize)
        accumulator = np.zeros_like(noise.time_data)
        for n in range(n_blocks):
            stop = min((n + 1) * blocksize, len(accumulator))
            sl = slice(n * blocksize, stop)
            accumulator[sl, 0] = fir.process_block(noise.time_data[sl, 0], 0)

        reference = sig.oaconvolve(
            noise.time_data[:, 0], rir.time_data[:, 0], mode="full"
        )
        diff = accumulator.squeeze() - reference.squeeze()[: len(accumulator)]
        np.testing.assert_array_almost_equal(diff, 0.0)

    def test_warped_fir_filter(self):
        # Only functionality
        rir = dsp.pad_trim(dsp.ImpulseResponse.from_file(RIR_PATH), 300)
        fir = dsp.filterbanks.WarpedFIR(
            np.hanning(15), -0.6, rir.sampling_rate_hz
        )
        [fir.process_sample(x, 0) for x in rir.time_data[:, 0]]

        # Try out filtering multichannel
        rir.time_data = np.repeat(rir.time_data, 2, axis=1)
        fir.filter_signal(rir)

        # Constructor
        dsp.filterbanks.WarpedFIR.from_filter(
            dsp.Filter.from_ba(np.hanning(20), [1], rir.sampling_rate_hz), 0.1
        )

    def test_warped_iir_filter(self):
        # Only functionality
        rir = dsp.pad_trim(dsp.ImpulseResponse.from_file(RIR_PATH), 300)
        iir_coefficients = dsp.Filter.biquad(
            dsp.BiquadEqType.Peaking, 200.0, 4, 0.7, rir.sampling_rate_hz
        )

        iir_w = dsp.filterbanks.WarpedIIR(
            iir_coefficients.ba[0].copy(),
            iir_coefficients.ba[1].copy(),
            -0.6,
            rir.sampling_rate_hz,
        )
        [iir_w.process_sample(x, 0) for x in rir.time_data[:, 0]]

        # Try out filtering multichannel
        rir.time_data = np.repeat(rir.time_data, 2, axis=1)
        iir_w.filter_signal(rir)

        # With different orders of a and b coefficients
        iir_w = dsp.filterbanks.WarpedIIR(
            np.pad(iir_coefficients.ba[0], ((0, 4))),
            np.pad(iir_coefficients.ba[1], ((0, 10))),
            -0.6,
            rir.sampling_rate_hz,
        )
        [iir_w.process_sample(x, 0) for x in rir.time_data[:, 0]]

        # Constructor
        dsp.filterbanks.WarpedIIR.from_filter(iir_coefficients, 0.1)


class TestSpectrum:
    def get_spectrum_from_filter(self, freqs=None, complex=False):
        """Get some spectrum from a filter. If `freqs=None`, it is a
        logarithmic vector."""
        filt = dsp.Filter.biquad(
            dsp.BiquadEqType.Peaking, 500.0, 10.0, 1.0, 48000
        )
        return dsp.Spectrum.from_filter(
            (
                dsp.tools.log_frequency_vector([20, 20e3], 128)
                if freqs is None
                else freqs
            ),
            filt,
            complex,
        )

    rir_spec_complex = dsp.Spectrum.from_signal(
        dsp.ImpulseResponse.from_file(RIR_PATH), True
    )
    rir = dsp.ImpulseResponse.from_file(RIR_PATH)
    rir_spec_real = dsp.Spectrum.from_signal(
        dsp.ImpulseResponse.from_file(RIR_PATH), False
    )

    def get_spectrum_from_rir(self, complex=False):
        return (
            self.rir_spec_complex.copy()
            if complex
            else self.rir_spec_real.copy()
        )

    def test_properties(self):
        spec = self.get_spectrum_from_filter(complex=False)
        assert spec.frequency_vector_type == dsp.FrequencySpacing.Logarithmic
        assert spec.is_magnitude
        assert spec.number_of_channels == 1

        spec = self.get_spectrum_from_filter(complex=True)
        assert not spec.is_magnitude

        freqs = np.array([100.0, 200.0, 300.0])
        spec = self.get_spectrum_from_filter(freqs, complex=True)
        assert spec.frequency_vector_type == dsp.FrequencySpacing.Linear

        freqs = np.array([100.0, 200.0, 300.0, 504.0])
        spec = self.get_spectrum_from_filter(freqs, complex=True)
        assert spec.frequency_vector_type == dsp.FrequencySpacing.Other
        assert spec.number_frequency_bins == len(freqs)

    def test_constructor_and_setters(self):
        freqs = np.array([100.0, 200.0, 300.0])
        spec = dsp.Spectrum(freqs, [np.zeros(3) for _ in range(2)])
        assert len(spec) == len(freqs)
        assert spec.number_of_channels == 2

    def test_trim(self):
        freqs = np.array([100.0, 200.0, 300.0, 504.0])
        spec = self.get_spectrum_from_filter(freqs, complex=True)
        spec2 = spec.copy().trim(200.0, 300.0, True)
        np.testing.assert_array_equal(
            np.array([200.0, 300.0]), spec2.frequency_vector_hz
        )

        spec2 = spec.copy().trim(100.0, 300.0, False)
        np.testing.assert_array_equal(
            np.array([200.0]), spec2.frequency_vector_hz
        )

    def test_resample(self):
        # Only functionaltiy
        freqs = np.array([100.0, 200.0, 300.0])

        sp = self.get_spectrum_from_rir(False)
        sp.resample(freqs)
        sp = self.get_spectrum_from_rir(True)
        sp.resample(freqs)

    def test_interpolation_magnitude(self):
        sp_mag = self.get_spectrum_from_filter(None, False)
        f = np.array([200.0, 300.0])
        f_outside = np.array([sp_mag.frequency_vector_hz[-1] + 1.0])

        # Assertions
        # Complex and magnitude
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(dsp.InterpolationDomain.Complex)
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(dsp.InterpolationDomain.Complex)
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(
                dsp.InterpolationDomain.Magnitude
            )
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(dsp.InterpolationDomain.Power)
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(
                dsp.InterpolationDomain.MagnitudePhase
            )
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(
                dsp.InterpolationDomain.MagnitudePhase
            )
            sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)

        # Padding
        with pytest.raises(AssertionError):
            sp_mag.set_interpolator_parameters(
                dsp.InterpolationDomain.Power,
                edges_handling=dsp.InterpolationEdgeHandling.Error,
            )
            sp_mag.get_interpolated_spectrum(
                f_outside, dsp.SpectrumType.Magnitude
            )

        # Normal functionality magnitude (no checking results)
        #
        sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Power)
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.Extend,
        )
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_mag.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.OnePad,
        )
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Magnitude)
        sp_mag.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)

    def test_interpolation_complex(self):
        sp_comp = self.get_spectrum_from_filter(None, True)
        f = np.array([200.0, 300.0])
        f_outside = np.array([sp_comp.frequency_vector_hz[-1] + 1.0])

        # Normal functionality magnitude (no checking results)
        #
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        #
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        #
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Complex)
        #
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Cubic,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Magnitude)
        sp_comp.get_interpolated_spectrum(f, dsp.SpectrumType.Db)
        #
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Power,
            dsp.InterpolationScheme.Linear,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Power)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Magnitude,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.Extend,
        )
        sp_comp.get_interpolated_spectrum(
            f_outside, dsp.SpectrumType.Magnitude
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.Complex,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.Extend,
        )
        sp_comp.get_interpolated_spectrum(
            f_outside, dsp.SpectrumType.Magnitude
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Complex)
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.Extend,
        )
        sp_comp.get_interpolated_spectrum(
            f_outside, dsp.SpectrumType.Magnitude
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Complex)
        #
        sp_comp.set_interpolator_parameters(
            dsp.InterpolationDomain.MagnitudePhase,
            dsp.InterpolationScheme.Pchip,
            dsp.InterpolationEdgeHandling.ZeroPad,
        )
        sp_comp.get_interpolated_spectrum(
            f_outside, dsp.SpectrumType.Magnitude
        )
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Db)
        sp_comp.get_interpolated_spectrum(f_outside, dsp.SpectrumType.Complex)

    def test_get_energy(self):
        # Total energy
        sp = self.get_spectrum_from_rir()
        np.testing.assert_allclose(
            sp.get_energy(),
            np.sum(sp.spectral_data**2.0, axis=0)
            * (sp.frequency_vector_hz[1] - sp.frequency_vector_hz[0]),
            rtol=0.01,
        )

        # Functionality of boundaries
        sp.get_energy(10.0, 50.0)
        sp.get_energy(10.0, None)
        sp.get_energy(None, 50.0)

        with pytest.raises(AssertionError):
            sp.get_energy(200.0, 50.0)

    def test_apply_octave_smoothing(self):
        # Only functionality
        sp = self.get_spectrum_from_filter()
        sp.apply_octave_smoothing(12.0)

        sp = self.get_spectrum_from_filter(np.linspace(500, 2000))
        sp.apply_octave_smoothing(12.0)

    def test_coherence(self):
        sp = self.get_spectrum_from_rir()
        sp.set_coherence(np.zeros((len(sp), 1)))
        sp.plot_coherence()

    def test_plot_magnitude(self):
        sp = self.get_spectrum_from_filter()
        sp.plot_magnitude(
            True, dsp.MagnitudeNormalization.NoNormalization, None
        )
        sp.plot_magnitude(
            True, dsp.MagnitudeNormalization.NoNormalization, 10.0
        )
        sp.plot_magnitude(True, dsp.MagnitudeNormalization.OneKhz, 10.0)
        sp.plot_magnitude(True, dsp.MagnitudeNormalization.Max, 10.0)
        sp.plot_magnitude(True, dsp.MagnitudeNormalization.Energy, 10.0)
        sp.plot_magnitude(
            False, dsp.MagnitudeNormalization.NoNormalization, None
        )
        sp.plot_magnitude(
            False, dsp.MagnitudeNormalization.NoNormalization, None
        )
        sp.plot_magnitude(
            False, dsp.MagnitudeNormalization.NoNormalization, None
        )

        sp.plot_magnitude(
            False, dsp.MagnitudeNormalization.OneKhzFirstChannel, None
        )
        sp.plot_magnitude(
            False, dsp.MagnitudeNormalization.MaxFirstChannel, None
        )
        sp.plot_magnitude(
            False, dsp.MagnitudeNormalization.EnergyFirstChannel, None
        )

    def test_to_signal(self):
        # Only functionality
        spec = self.get_spectrum_from_rir(True)
        spec.to_signal(48000)
        spec.to_signal(96000)
        spec.to_signal(44100, 2.0)

        # Non-linear frequency
        spec.resample(
            dsp.tools.log_frequency_vector([1, 24e3], 512)
        ).to_signal(44100, 2.0)

        with pytest.raises(AssertionError):
            spec.to_signal(44100)

        # Non-complex spectrum
        spec = self.get_spectrum_from_rir(False)
        with pytest.raises(AssertionError):
            spec.to_signal(96000)

    def test_warp(self):
        # Only functionality
        spec = self.get_spectrum_from_rir(False)
        spec.warp(-0.7, self.rir.sampling_rate_hz)
        spec.warp(0.7, self.rir.sampling_rate_hz)
        with pytest.raises(AssertionError):
            spec.warp(1.1, self.rir.sampling_rate_hz)
        with pytest.raises(AssertionError):
            spec.warp(0.1, self.rir.sampling_rate_hz - 200)
