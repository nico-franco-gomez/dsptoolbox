import os

import numpy as np
import pytest

import dsptoolbox as dsp


class TestStandardModule:
    fs = 44100

    def test_load_pkl_object(self):
        f = dsp.Filter.fir_filter(
            order=216,
            frequency_hz=1000,
            type_of_pass=dsp.FilterPassType.Highpass,
            sampling_rate_hz=self.fs,
        )
        f.save_filter(os.path.join("tests", "f.pkl"))

        reloaded = dsp.load_pkl_object(os.path.join("tests", "f.pkl"))
        assert type(reloaded) is dsp.Filter
        np.testing.assert_array_equal(reloaded.ba[0], f.ba[0])
        np.testing.assert_array_equal(reloaded.ba[1], f.ba[1])

        # A missing or mismatched extension is rejected before even trying
        # to open the file
        with pytest.raises(ValueError):
            dsp.load_pkl_object(os.path.join("tests", "f"))
        with pytest.raises(ValueError):
            dsp.load_pkl_object(os.path.join("tests", "f.txt"))

        os.remove(os.path.join("tests", "f.pkl"))

    def test_resample_filter(self):
        fs_hz = 48000
        f = dsp.Filter.iir_filter(
            order=8,
            frequency_hz=[500, 2e3],
            type_of_pass=dsp.FilterPassType.Bandpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )
        f.resample_filter(24000)
        f = dsp.Filter.iir_filter(
            order=5,
            frequency_hz=500,
            type_of_pass=dsp.FilterPassType.Lowpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )
        f.resample_filter(24000)
        f = dsp.Filter.iir_filter(
            order=8,
            frequency_hz=500,
            type_of_pass=dsp.FilterPassType.Highpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )
        f.resample_filter(24000)
        f = dsp.Filter.iir_filter(
            order=7,
            frequency_hz=[500, 18e3],
            type_of_pass=dsp.FilterPassType.Bandpass,
            filter_design_method=dsp.IirDesignMethod.Bessel,
            sampling_rate_hz=fs_hz,
        )

    def test_merge_fir_filters(self):
        f1 = dsp.Filter.fir_filter(
            50,
            100.0,
            type_of_pass=dsp.FilterPassType.Lowpass,
            window=dsp.Window.Hamming,
            sampling_rate_hz=self.fs,
        )

        # Dirac with some delay
        dirac = np.zeros(30)
        delay = len(dirac) - 1
        dirac[-1] = 1.0
        f2 = dsp.Filter.from_ba(dirac, [1.0], self.fs)

        f3 = dsp.FilterBank([f1, f2]).merge_filters()
        np.testing.assert_array_equal(f3.ba[0][delay:], f1.ba[0])

        f3 = dsp.FilterBank([f1, f2]).merge_filters()
        np.testing.assert_array_equal(f3.ba[0][delay:], f1.ba[0])

        with pytest.raises(AssertionError):
            dsp.FilterBank([f1]).merge_filters()

        with pytest.raises(AssertionError):
            iir = dsp.Filter.biquad(
                dsp.BiquadEqType.LowpassFirstOrder, 50.0, -3.0, 0.7, self.fs
            )
            dsp.FilterBank([f1, iir]).merge_filters()

        with pytest.raises(AssertionError):
            f2 = dsp.Filter.from_ba(dirac, [1.0], self.fs * 2)
            dsp.FilterBank([f1, f2]).merge_filters()

    def test_merge_iir_filters(self):
        f1 = dsp.Filter.biquad(
            eq_type=dsp.BiquadEqType.Allpass,
            frequency_hz=500.0,
            gain_db=5.0,
            q=0.7,
            sampling_rate_hz=self.fs,
        )

        f3 = dsp.FilterBank([f1, f1.copy()]).merge_filters()
        assert f3.has_sos
        assert f3.sos.shape[0] == 2
