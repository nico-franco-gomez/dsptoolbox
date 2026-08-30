"""
Tests for the Filter class.
"""

import os
import pickle
import tempfile
from os.path import join

import numpy as np
import pytest
import scipy.signal as sig
from matplotlib.pyplot import close

import dsptoolbox as dsp

_rng = np.random.default_rng(6)

RIR_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "example_data",
    "rir.wav",
)
CHIRP_STEREO_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "example_data",
    "chirp_stereo.wav",
)


class TestFilterClass:
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
        # FIR
        f = dsp.Filter.from_ba(self.fir, 1, self.fs)
        assert f.is_fir
        b, _ = f.ba
        assert np.all(b == self.fir)

        # IIR
        f = dsp.Filter.from_sos(self.iir, self.fs)
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

        # Integer "a" coefficients should still be stored as float
        fir = dsp.Filter.from_ba(self.fir, [1], self.fs)
        assert fir.ba[1].dtype == np.float64
        assert not fir.has_sos

    def test_filtering_fir(self):
        t_vec = _rng.normal(0, 0.01, self.fs * 2)

        result_scipy = sig.lfilter(self.fir, [1], t_vec)
        s = dsp.Signal.from_time_data(t_vec, self.fs)
        f = self.get_fir()
        result_own = f.filter_signal(s).time_data.squeeze()
        np.testing.assert_allclose(result_scipy, result_own)

        # filtfilt
        result_scipy = sig.filtfilt(self.fir, [1], t_vec)
        result_own = f.filter_signal(s, zero_phase=True).time_data.squeeze()
        np.testing.assert_allclose(result_scipy, result_own)

        # filter_signal must not mutate the input signal
        np.testing.assert_array_equal(s.time_data.squeeze(), t_vec)

    def test_filtering_iir(self):
        t_vec = _rng.normal(0, 0.01, self.fs * 2)
        s = dsp.Signal(None, t_vec, self.fs)
        result_scipy = sig.sosfilt(self.iir, t_vec)
        f = self.get_iir()
        result_own = f.filter_signal(s).time_data.squeeze()
        np.testing.assert_allclose(result_scipy, result_own)

        # filtfilt
        result_scipy = sig.sosfiltfilt(self.iir, t_vec)
        result_own = f.filter_signal(s, zero_phase=True).time_data.squeeze()
        np.testing.assert_allclose(result_scipy, result_own)

        np.testing.assert_array_equal(s.time_data.squeeze(), t_vec)

    def test_plots(self):
        f = self.get_iir()
        f.plot_magnitude(length_samples=512)
        f.plot_phase(length_samples=512)
        f.plot_group_delay(length_samples=512)
        f.plot_zp()

        f.plot_magnitude(length_samples=512, show_info_box=True)
        f.plot_phase(length_samples=512, show_info_box=True)
        f.plot_group_delay(length_samples=512, show_info_box=True)
        f.plot_zp(show_info_box=True)

        f.plot_magnitude(
            length_samples=512, normalize=dsp.MagnitudeNormalization.OneKhz
        )
        f.plot_magnitude(length_samples=512, normalize=dsp.MagnitudeNormalization.Max)
        f.plot_magnitude(
            length_samples=512, normalize=dsp.MagnitudeNormalization.Energy
        )
        f.plot_magnitude(
            length_samples=512, normalize=dsp.MagnitudeNormalization.OneKhzFirstChannel
        )
        f.plot_magnitude(
            length_samples=512, normalize=dsp.MagnitudeNormalization.MaxFirstChannel
        )
        f.plot_magnitude(
            length_samples=512, normalize=dsp.MagnitudeNormalization.EnergyFirstChannel
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
        f.get_ir(128)

    def test_fir_from_file_matches_impulse_response(self):
        """Per the source, `fir_from_file` just reads the audio file via
        `ImpulseResponse.from_file` and takes one channel as the FIR taps
        -- content should match exactly, and the requested channel should
        be the one selected for stereo files.

        """
        ir = dsp.ImpulseResponse.from_file(RIR_PATH)
        f = dsp.Filter.fir_from_file(RIR_PATH)
        np.testing.assert_array_equal(f.ba[0], ir.time_data[:, 0])
        assert f.sampling_rate_hz == ir.sampling_rate_hz

        stereo_ir = dsp.ImpulseResponse.from_file(CHIRP_STEREO_PATH)
        f_ch0 = dsp.Filter.fir_from_file(CHIRP_STEREO_PATH, channel=0)
        f_ch1 = dsp.Filter.fir_from_file(CHIRP_STEREO_PATH, channel=1)
        np.testing.assert_array_equal(f_ch0.ba[0], stereo_ir.time_data[:, 0])
        np.testing.assert_array_equal(f_ch1.ba[0], stereo_ir.time_data[:, 1])

    def test_save_filter_round_trip_and_format_checking(self):
        """Every `save_*` method in the library takes the format from the
        path's extension, which has to be present and correct."""
        f = self.get_fir()
        with tempfile.TemporaryDirectory() as d:
            f.save_filter(join(d, "with_ext.pkl"))
            with open(join(d, "with_ext.pkl"), "rb") as fh:
                reloaded = pickle.load(fh)
            np.testing.assert_array_equal(reloaded.ba[0], f.ba[0])
            np.testing.assert_array_equal(reloaded.ba[1], f.ba[1])
            assert reloaded.sampling_rate_hz == f.sampling_rate_hz

            with pytest.raises(ValueError):
                f.save_filter(join(d, "no_ext"))
            with pytest.raises(ValueError):
                f.save_filter(join(d, "wrong_ext.txt"))

    def test_other_functionalities(self):
        dsp.Filter.fir_from_file(RIR_PATH)

        f = self.get_iir()
        f.show_info()
        print(f)
        f.copy()
        f.initialize_zi(1)
        with pytest.raises(AssertionError):
            f.initialize_zi(0)

    def test_get_transfer_function(self):
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

        t_vec = _rng.normal(0, 0.01, self.fs * 2)

        t_signal = dsp.Signal(None, t_vec, self.fs)
        t_res = f.filter_and_resample_signal(t_signal, self.fs // 2)
        t_res = t_res.time_data.squeeze()

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
        f = dsp.Filter.from_ba(b, 1, self.fs)
        t_vec = _rng.normal(0, 0.01, self.fs * 2)

        t_signal = dsp.Signal(None, t_vec, self.fs)
        t_res = f.filter_and_resample_signal(t_signal, self.fs // 2)
        t_res = t_res.time_data.squeeze()

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
        f = dsp.Filter.from_ba(b, 1, self.fs)
        assert len(f) == len(b)

    def test_order(self):
        b = sig.firwin(
            1500,
            (self.fs // 2 // 2),
            pass_zero="lowpass",
            fs=self.fs,
            window="flattop",
        )
        f = dsp.Filter.from_ba(b, 1, self.fs)
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
        ff, gg = dsp.transfer_functions.group_delay(bb.get_ir(length_samples=2**14))

        interpolated_gd = dsp.tools.interpolate_fr(
            ff,
            gg.squeeze(),
            f_log,
            interpolation_kind=dsp.InterpolationKind.Cubic,
        )
        np.testing.assert_allclose(interpolated_gd, gd, atol=1e-6)

        gd = bb.get_group_delay(f_log, False)

    def test_only_one_coefficient_type_accepted(self):
        """Passing several coefficient types at once must be rejected."""
        with pytest.raises(AssertionError):
            dsp.Filter(
                self.fs,
                {
                    dsp.FilterCoefficientsType.Zpk: (
                        np.array([0.1]),
                        np.array([0.5]),
                        1.0,
                    ),
                    dsp.FilterCoefficientsType.Sos: np.array(
                        [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
                    ),
                    dsp.FilterCoefficientsType.Ba: [np.array([1.0]), np.array([1.0])],
                },
            )

    def test_apply_gain_does_not_modify_original(self):
        f = dsp.Filter.iir_filter(4, 1000, dsp.FilterPassType.Lowpass, self.fs)
        sos_before = f.sos.copy()
        zpk_gain_before = f.zpk[-1]
        f.apply_gain(6.0)
        np.testing.assert_array_equal(sos_before, f.sos)
        assert f.zpk[-1] == zpk_gain_before

        fir = dsp.Filter.from_ba([1.0, 0.5], [1.0], self.fs)
        b_before = fir.ba[0].copy()
        fir.apply_gain(6.0)
        np.testing.assert_array_equal(b_before, fir.ba[0])

    def test_metadata_str_underline(self):
        f = dsp.Filter.from_ba([1.0, 0.5], [1.0], self.fs)
        lines = f.metadata_str.splitlines()
        assert lines[0] == "Filter:"
        assert lines[1] == "-" * len(lines[0])

    def test_activate_zi_carries_state_between_calls(self):
        """The packed zi must come back in the per-channel layout that
        `initialize_zi` produces, otherwise the channel-count check re-
        initializes the state on every call and nothing is carried over.

        """
        td = _rng.normal(0, 0.1, (2000, 3))
        whole = dsp.Signal.from_time_data(td, self.fs)
        first = dsp.Signal.from_time_data(td[:1000], self.fs)
        second = dsp.Signal.from_time_data(td[1000:], self.fs)

        for f in (
            dsp.Filter.iir_filter(4, 500, dsp.FilterPassType.Lowpass, self.fs),
            dsp.Filter.from_ba(*sig.butter(4, 500, fs=self.fs), self.fs),
        ):
            reference = f.filter_signal(whole).time_data
            f.initialize_zi(3)
            f.filter_signal(first, activate_zi=True)
            continuation = f.filter_signal(second, activate_zi=True).time_data
            assert len(f.zi) == 3
            np.testing.assert_allclose(continuation, reference[1000:], atol=1e-8)
