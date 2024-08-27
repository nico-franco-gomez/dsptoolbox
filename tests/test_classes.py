"""
Tests for basic functionalities of the classes in dsptoolbox
"""

import pytest
import dsptoolbox as dsp
import numpy as np
from os.path import join
import scipy.signal as sig
from matplotlib.pyplot import close


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
        path = join("examples", "data", "chirp.wav")
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

    def test_get_fft(self):
        sp = np.fft.rfft(self.time_vec, axis=0)

        # Check normal FFT
        s = dsp.Signal(None, self.time_vec, self.fs)
        s.set_spectrum_parameters(method="standard", scaling=None)
        _, sp_sig = s.get_spectrum()
        assert np.all(sp == sp_sig)

        # Check amplitude spectrum scaling for normal FFT
        s.set_spectrum_parameters(method="standard", scaling="power spectrum")
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
            method="standard", scaling="power spectral density"
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
            method="standard", scaling="amplitude spectrum", smoothe=3
        )
        s.get_spectrum()

    def test_managing_channels(self):
        # Add new channel
        new_ch = np.random.normal(0, 0.1, (self.length_samp, 1))
        t_vec = np.append(self.time_vec, new_ch, axis=1)
        s = dsp.Signal(None, self.time_vec, self.fs)
        s.add_channel(None, new_ch, s.sampling_rate_hz)
        assert np.all(t_vec == s.time_data)

        # Remove channel
        s.remove_channel(-1)
        assert np.all(self.time_vec == s.time_data)

        # Try to remove channel that does not exist
        with pytest.raises(AssertionError):
            s.remove_channel(self.channels + 10)

        # Get specific channel
        ch = s.get_channels(0)
        assert np.all(self.time_vec[:, 0][..., None] == ch.time_data)

        # Try to get a channel that does not exist
        with pytest.raises(AssertionError):
            s.get_channels(self.channels + 10)

        # Swap channels
        new_order = np.arange(0, self.channels)[::-1]
        s.swap_channels(new_order)
        assert np.all(self.time_vec[:, ::-1] == s.time_data)

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

        # Number of channels should not be changeable
        with pytest.raises(AssertionError):
            s.number_of_channels = 10

    def test_plot_generation(self):
        s = dsp.ImpulseResponse(
            time_data=self.time_vec, sampling_rate_hz=self.fs
        )
        # Test that all plots are generated without problems
        s.plot_magnitude()
        s.plot_magnitude(show_info_box=True)
        s.plot_time()
        s.plot_spectrogram(channel_number=0, logfreqs=True)
        s.plot_csm()
        s.plot_csm(with_phase=False)
        s.plot_spl(False)
        s.plot_spl(True)

        # Plot phase and group delay
        s.set_spectrum_parameters(method="standard")
        s.plot_phase()
        s.plot_group_delay()

        # Try to plot coherence
        with pytest.raises(AssertionError):
            s.plot_coherence()
        # Try to plot phase having welch's method for magnitude
        with pytest.raises(AssertionError):
            s.set_spectrum_parameters(method="welch", window_length_samples=32)
            s.plot_phase()

        # Plot signal with window and imaginary time data
        d = dsp.generators.dirac(1024, 512, sampling_rate_hz=self.fs)
        d, _ = dsp.transfer_functions.window_centered_ir(d, len(d))
        d = dsp.transforms.hilbert(d)
        d.plot_time()
        d.plot_spl()
        close("all")

    def test_get_power_spectrum_welch(self):
        # Try to get power spectrum
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        s.set_spectrum_parameters()
        f, sp = s.get_spectrum()

    def test_get_csm(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        f, csm = s.get_csm()

    def test_get_stft(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        # Use parameters just like librosa for validation
        s.set_spectrogram_parameters(
            window_length_samples=1024,
            window_type="hann",
            overlap_percent=50,
            fft_length_samples=4096,
            detrend=False,
            padding=False,
            scaling="power spectrum",
        )
        t, f, stft = s.get_spectrogram()
        s.set_spectrogram_parameters(
            window_length_samples=1024,
            window_type="hann",
            overlap_percent=50,
            fft_length_samples=None,
            detrend=False,
            padding=False,
            scaling=None,
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
        t_ = np.linspace(0, le / self.fs, le)
        assert np.all(t == t_)

    def test_length_signal(self):
        s = dsp.Signal(time_data=self.time_vec, sampling_rate_hz=self.fs)
        assert len(s) == s.time_data.shape[0]

    def test_constrain_amplitude(self):
        t = np.random.normal(0, 1, 200)
        s = dsp.Signal(None, t, sampling_rate_hz=100, constrain_amplitude=True)
        assert np.all(s.time_data <= 1)

        s = dsp.Signal(
            None, t, sampling_rate_hz=100, constrain_amplitude=False
        )
        assert np.all(t == s.time_data.squeeze())


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

    def test_create_from_coefficients(self):
        # Try creating a filter from the coefficients, recognizing filter
        # type and returning the coefficients in the right way

        # FIR
        f = dsp.Filter(
            filter_type="other",
            filter_configuration=dict(ba=[self.fir, 1]),
            sampling_rate_hz=self.fs,
        )
        condfir = f.filter_type == "fir"
        b, _ = f.ba
        condfir = condfir and np.all(b == self.fir)

        # IIR
        f = dsp.Filter(
            filter_type="other",
            filter_configuration=dict(sos=self.iir),
            sampling_rate_hz=self.fs,
        )
        condiir = f.filter_type == "iir"
        sos = f.sos
        condiir = condiir and np.all(sos == self.iir)

        assert condfir and condiir

    def test_standard_filtering(self):
        # Try filtering compared to scipy's functions
        t_vec = np.random.normal(0, 0.01, self.fs * 2)

        # FIR
        result_scipy = sig.lfilter(self.fir, [1], t_vec.copy())

        s = dsp.Signal(None, t_vec, self.fs)
        f = dsp.Filter(
            "other",
            filter_configuration=dict(ba=[self.fir, 1]),
            sampling_rate_hz=self.fs,
        )
        result_own = f.filter_signal(s).time_data.squeeze()
        condfir = np.all(np.isclose(result_scipy, result_own))

        # IIR
        result_scipy = sig.sosfilt(self.iir, t_vec.copy())
        f = dsp.Filter(
            "other",
            filter_configuration=dict(sos=self.iir),
            sampling_rate_hz=self.fs,
        )
        result_own = f.filter_signal(s).time_data.squeeze()
        condiir = np.all(np.isclose(result_scipy, result_own))

        assert condfir and condiir

    def test_other_filtering(self):
        # Try filtering using filtfilt compared to scipy's functions
        t_vec = np.random.normal(0, 0.01, self.fs * 2)

        # FIR
        result_scipy = sig.filtfilt(self.fir, [1], t_vec)

        s = dsp.Signal(None, t_vec, self.fs)
        f = dsp.Filter(
            "other",
            filter_configuration=dict(ba=[self.fir, 1]),
            sampling_rate_hz=self.fs,
        )
        result_own = f.filter_signal(s, zero_phase=True).time_data.squeeze()
        condfir = np.all(np.isclose(result_scipy, result_own))

        # IIR
        result_scipy = sig.sosfiltfilt(self.iir, t_vec)
        f = dsp.Filter(
            "other",
            filter_configuration=dict(sos=self.iir),
            sampling_rate_hz=self.fs,
        )
        result_own = f.filter_signal(s, zero_phase=True).time_data.squeeze()
        condiir = np.all(np.isclose(result_scipy, result_own))

        assert condfir and condiir

    def test_plots(self):
        f = dsp.Filter(
            "other",
            filter_configuration=dict(sos=self.iir),
            sampling_rate_hz=self.fs,
        )
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

        f.plot_magnitude(normalize="1k")
        f.plot_magnitude(normalize="max")
        close("all")

    def test_get_coefficients(self):
        f = dsp.Filter(
            "other",
            filter_configuration=dict(sos=self.iir),
            sampling_rate_hz=self.fs,
        )
        f.get_coefficients(mode="ba")
        f.get_coefficients(mode="sos")
        f.get_coefficients(mode="zpk")

    def test_get_ir(self):
        f = dsp.Filter(
            "other",
            filter_configuration=dict(sos=self.iir),
            sampling_rate_hz=self.fs,
        )
        f.get_ir()

    def test_other_functionalities(self):
        f = dsp.Filter(
            "other",
            filter_configuration=dict(sos=self.iir),
            sampling_rate_hz=self.fs,
        )
        f.get_filter_metadata()
        f._get_metadata_string()
        f.show_info()
        print(f)
        f.copy()
        f.initialize_zi(1)
        with pytest.raises(AssertionError):
            f.initialize_zi(0)

    def test_get_transfer_function(self):
        # Functionality
        f = dsp.Filter(
            "other",
            filter_configuration=dict(sos=self.iir),
            sampling_rate_hz=self.fs,
        )
        freqs = np.linspace(1, 4e3, 200)
        f.get_transfer_function(freqs)

        b = sig.firwin(
            1500,
            (self.fs // 2 // 2),
            pass_zero="lowpass",
            fs=self.fs,
            window="flattop",
        )
        f = dsp.Filter(
            "other",
            filter_configuration=dict(ba=[b, 1]),
            sampling_rate_hz=self.fs,
        )
        f.get_transfer_function(freqs)

        f = dsp.Filter(
            "biquad",
            filter_configuration={
                "eq_type": "peaking",
                "freqs": 200,
                "gain": 3,
                "q": 0.7,
            },
            sampling_rate_hz=self.fs,
        )
        f.get_transfer_function(freqs)

    def test_filter_and_resampling_IIR(self):
        f = dsp.Filter(
            "other",
            filter_configuration=dict(sos=self.iir),
            sampling_rate_hz=self.fs,
        )

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
            "other",
            filter_configuration=dict(ba=[b, 1]),
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
            "other",
            filter_configuration=dict(ba=[b, 1]),
            sampling_rate_hz=self.fs,
        )
        assert len(f) == len(b)


class TestFilterBankClass:
    fs = 44100

    def test_create_filter_bank(self):
        # Create filter bank sequentially
        fb = dsp.FilterBank()
        config = dict(
            order=5,
            freqs=[1510, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))

        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == self.fs

        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        fb.add_filter(dsp.Filter("fir", config, self.fs))

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Create filter bank passing a list
        filters = []
        config = dict(
            order=5,
            freqs=[1501, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        filters.append(dsp.Filter("iir", config, self.fs))
        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        filters.append(dsp.Filter("fir", config, self.fs))
        fb = dsp.FilterBank(
            filters=filters,
            same_sampling_rate=True,
            info={"Type of filter bank": "Test"},
        )
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

    def test_plots(self):
        # Create
        fb = dsp.FilterBank()
        config = dict(
            order=5,
            freqs=[1502, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))
        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        fb.add_filter(dsp.Filter("fir", config, self.fs))

        # Get plots
        fb.plot_magnitude(mode="parallel")
        fb.plot_magnitude(mode="sequential")
        fb.plot_magnitude(mode="summed")
        fb.plot_magnitude(mode="parallel", test_zi=True)

        fb.plot_phase(mode="parallel")
        fb.plot_phase(mode="sequential")
        fb.plot_phase(mode="summed")
        fb.plot_phase(mode="parallel", test_zi=True)

        fb.plot_group_delay(mode="parallel")
        fb.plot_group_delay(mode="sequential")
        fb.plot_group_delay(mode="summed")
        fb.plot_group_delay(mode="parallel", test_zi=True)

    def test_filterbank_functionalities(self):
        fb = dsp.FilterBank()

        config = dict(
            order=5,
            freqs=[1500, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))
        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        fb.add_filter(dsp.Filter("fir", config, self.fs))

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == self.fs

        # Remove
        fb.remove_filter(0)
        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == self.fs

        # Readd
        fb.add_filter(dsp.Filter("fir", config, self.fs))
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
        fb.get_ir()
        fb.copy()
        fb.show_info()
        print(fb)

    def test_filtering(self):
        # Create
        fb = dsp.FilterBank()
        config = dict(
            order=5,
            freqs=[1500, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))
        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        fb.add_filter(dsp.Filter("fir", config, self.fs))

        t_vec = np.random.normal(0, 0.01, (self.fs * 3, 2))
        s = dsp.Signal(None, t_vec, self.fs)

        # Type of output and filter results
        filt1 = fb.filters[0].get_coefficients(mode="sos")
        filt2, _ = fb.filters[1].get_coefficients(mode="ba")
        # Parallel
        s_ = fb.filter_signal(s, mode="parallel", activate_zi=False)
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
        s_ = fb.filter_signal(s, mode="sequential", activate_zi=False)
        assert type(s_) is dsp.Signal
        # Change order (just because they're linear systems)
        temp = sig.lfilter(filt2, [1], s.time_data[:, 1])
        temp = sig.sosfilt(filt1, temp)
        # Try second channel
        assert np.all(np.isclose(s_.time_data[:, 1], temp))

        # Summed mode
        s_ = fb.filter_signal(s, mode="summed", activate_zi=False)
        assert type(s_) is dsp.Signal
        # Add together
        temp = sig.lfilter(filt2, [1], s.time_data[:, 1])
        temp += sig.sosfilt(filt1, s.time_data[:, 1])
        assert np.all(np.isclose(s_.time_data[:, 1], temp))

        # Filter's zi
        s_ = fb.filter_signal(s, mode="parallel", activate_zi=True)
        s_ = fb.filter_signal(s, mode="sequential", activate_zi=True)
        s_ = fb.filter_signal(s, mode="summed", activate_zi=True)

        # Zero-phase filtering
        s_ = fb.filter_signal(s, mode="parallel", zero_phase=True)
        s_ = fb.filter_signal(s, mode="sequential", zero_phase=True)
        s_ = fb.filter_signal(s, mode="summed", zero_phase=True)

        # No zi and zero phase filtering at the same time!
        with pytest.raises(AssertionError):
            s_ = fb.filter_signal(
                s, mode="summed", activate_zi=True, zero_phase=True
            )

    def test_multirate(self):
        fb = dsp.FilterBank(same_sampling_rate=False)
        config = dict(
            order=5,
            freqs=[1500, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))

        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == [self.fs]

        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        fb.add_filter(dsp.Filter("fir", config, self.fs // 2))

        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs, self.fs // 2]

        # Remove
        fb.remove_filter(0)
        assert fb.number_of_filters == 1
        assert fb.sampling_rate_hz == [self.fs // 2]

        # Readd
        fb.add_filter(dsp.Filter("fir", config, self.fs))
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs // 2, self.fs]

        # Swap (and Assertions)
        fb.swap_filters([1, 0])
        assert fb.number_of_filters == 2
        assert fb.sampling_rate_hz == [self.fs, self.fs // 2]

        # Should not be possible to create
        with pytest.raises(AssertionError):
            fb = dsp.FilterBank(same_sampling_rate=True)
            config = dict(
                order=5,
                freqs=[1500, 2000],
                type_of_pass="bandpass",
                filter_design_method="bessel",
            )
            fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))
            config = dict(
                order=150, freqs=[1500, 2000], type_of_pass="bandpass"
            )
            fb.add_filter(dsp.Filter("fir", config, self.fs // 2))

        # Create filter bank passing a list
        filters = []
        config = dict(
            order=5,
            freqs=[1500, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        filters.append(dsp.Filter("iir", config, self.fs))
        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        filters.append(dsp.Filter("fir", config, self.fs // 2))
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

        config = dict(
            order=5,
            freqs=[1500, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))
        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        fb.add_filter(dsp.Filter("fir", config, self.fs // 2))

        fb.plot_magnitude()
        fb.plot_phase()
        fb.plot_group_delay()
        fb.get_ir()
        with pytest.raises(AssertionError):
            fb.get_ir(mode="summed")

    def test_filtering_multirate_multiband(self):
        fb = dsp.FilterBank(same_sampling_rate=False)
        config = dict(
            order=5,
            freqs=[1500, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))
        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        fb.add_filter(dsp.Filter("fir", config, self.fs // 2))

        s1 = dsp.generators.noise(
            "white", length_seconds=1, sampling_rate_hz=self.fs
        )
        s2 = dsp.generators.noise(
            "white", length_seconds=2, sampling_rate_hz=self.fs // 2
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
        config = dict(
            order=5,
            freqs=[1500, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))
        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        fb.add_filter(dsp.Filter("fir", config, self.fs // 2))
        for n in fb:
            assert dsp.Filter == type(n)

    def test_transfer_function(self):
        # Create
        fb = dsp.FilterBank()
        config = dict(
            order=5,
            freqs=[1500, 2000],
            type_of_pass="bandpass",
            filter_design_method="bessel",
        )
        fb.add_filter(dsp.Filter("iir", config, sampling_rate_hz=self.fs))
        config = dict(order=150, freqs=[1500, 2000], type_of_pass="bandpass")
        fb.add_filter(dsp.Filter("fir", config, self.fs))

        freqs = np.linspace(1, 2e3, 400)
        fb.get_transfer_function(freqs, mode="parallel")
        fb.get_transfer_function(freqs, mode="sequential")
        fb.get_transfer_function(freqs, mode="summed")

        with pytest.raises(AssertionError):
            freqs = np.linspace(1, self.fs, 40)
            fb.get_transfer_function(freqs, mode="parallel")


class TestMultiBandSignal:
    fs = 44100
    s = np.random.normal(0, 0.01, (fs * 3, 3))
    s = dsp.Signal(None, s, fs)
    fb = dsp.filterbanks.auditory_filters_gammatone(
        frequency_range_hz=[500, 1200], sampling_rate_hz=fs
    )

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
        mbs = self.fb.filter_signal(self.s)
        assert type(mbs) is dsp.MultiBandSignal

    def test_collapse(self):
        mbs = dsp.MultiBandSignal(
            bands=[self.s, self.s],
            same_sampling_rate=True,
            info=dict(information="test filter bank"),
        )
        mbs_ = mbs.collapse()

        td = self.s.time_data
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


class TestImpulseResponse:
    fs_hz = 10_000
    seconds = 2
    d = dsp.generators.dirac(seconds * fs_hz, sampling_rate_hz=fs_hz)

    path_rir = join("examples", "data", "rir.wav")

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
        window_previous = rir.window[:, 0]
        rir.add_channel(self.path_rir)
        assert rir.window.shape == rir.time_data.shape
        np.testing.assert_array_equal(rir.window[:, 0], window_previous)
        np.testing.assert_array_equal(rir.window[:, 1], 1.0)

        # Remove channel
        rir.remove_channel(1)
        assert rir.window.shape == rir.time_data.shape
        np.testing.assert_array_equal(rir.window[:, 0], window_previous)

        # Swap channels
        rir.add_channel(self.path_rir)
        rir.add_channel(self.path_rir)
        rir.swap_channels([2, 1, 0])
        assert rir.window.shape == rir.time_data.shape
        np.testing.assert_array_equal(rir.window[:, -1], window_previous)

    def test_plotting_with_window(self):
        rir = self.get_ir()
        rir = dsp.transfer_functions.window_centered_ir(rir, len(rir))[0]
        rir.plot_time()
        rir.plot_spl()

        # Expect no coherence saved
        with pytest.raises(AssertionError):
            rir.plot_coherence()

        rir.add_channel(self.path_rir)
        rir.plot_time()
        rir.plot_spl()
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

        td = n.time_data.squeeze()
        for ind in np.arange(len(td)):
            td[ind] = sv_filt.process_sample(td[ind], 0)[0]

        sv_filt.reset_state()
        mb = sv_filt.filter_signal(n)
        n2 = mb.get_all_bands(0)

        np.testing.assert_array_equal(td, n2.time_data[:, 0])

        if PLOT:
            n2.set_spectrum_parameters("standard")
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
        iir = dsp.Filter.iir_design(
            4, 1000.0, "lowpass", "butter", sampling_rate_hz=self.fs_hz
        )
        llf = dsp.filterbanks.convert_into_lattice_filter(iir)

        td = n.time_data.squeeze()
        for ind in np.arange(len(td)):
            td[ind] = llf.process_sample(td[ind], 0)

        llf.reset_state()
        n2 = llf.filter_signal(n)
        np.testing.assert_array_equal(td, n2.time_data[:, 0])
        np.testing.assert_allclose(
            td, sig.sosfilt(iir.get_coefficients("sos"), n.time_data.squeeze())
        )

        # IIR ba
        iir = dsp.Filter.from_ba(
            *iir.get_coefficients("ba"), sampling_rate_hz=self.fs_hz
        )
        llf = dsp.filterbanks.convert_into_lattice_filter(iir)

        td = n.time_data.squeeze()
        for ind in np.arange(len(td)):
            td[ind] = llf.process_sample(td[ind], 0)

        llf.reset_state()
        n2 = llf.filter_signal(n)
        np.testing.assert_array_equal(td, n2.time_data[:, 0])
        np.testing.assert_allclose(
            td, sig.lfilter(*iir.get_coefficients("ba"), n.time_data.squeeze())
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
            n2.set_spectrum_parameters("standard")
            _, ax = n2.plot_magnitude(normalize=None)
            ax.plot(
                np.fft.rfftfreq(len(td), 1 / self.fs_hz),
                dsp.tools.to_db(np.fft.rfft(td), True),
            )
            dsp.plots.show()

    def test_iir_filter(self):
        iir_original = dsp.Filter.iir_design(
            4, 1000.0, "highpass", "butter", sampling_rate_hz=self.fs_hz
        )
        b, a = iir_original.get_coefficients("ba")
        iir = dsp.filterbanks.IIRFilter(b, a)
        n = self.get_noise()

        td = n.time_data.squeeze()
        for ind in np.arange(len(td)):
            td[ind] = iir.process_sample(td[ind], 0)

        np.testing.assert_allclose(td, sig.lfilter(b, a, n.time_data[:, 0]))

    def test_fir_filter(self):
        fir_original = dsp.Filter.fir_design(
            25,
            1000.0,
            "lowpass",
            "blackman",
            sampling_rate_hz=self.fs_hz,
        )
        b, _ = fir_original.get_coefficients("ba")
        b = b[: len(b) // 2 + 3]  # some asymmetrical window
        fir = dsp.filterbanks.FIRFilter(b)
        n = self.get_noise()

        td = n.time_data.squeeze()
        for ind in np.arange(len(td)):
            td[ind] = fir.process_sample(td[ind], 0)

        np.testing.assert_allclose(td, sig.lfilter(b, [1], n.time_data[:, 0]))
