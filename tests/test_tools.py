from random import choice

import numpy as np
import pytest

import dsptoolbox as dsp


class TestTools:
    def test_functionality(self):
        # Only assess basic functionality, not results
        x = np.linspace(100, 150, 30)
        dsp.tools.log_frequency_vector([20, 200], 50)
        dsp.tools.frequency_crossover([100, 200], True)(x)
        dsp.tools.log_mean(x)
        dsp.tools.to_db(x, True, None, None)
        dsp.tools.from_db(x, True)
        dsp.tools.time_smoothing(x, 200, 0.1, None)
        dsp.tools.time_smoothing(x, 200, 0.1, 0.2)
        dsp.tools.fractional_octave_frequencies()
        dsp.tools.erb_frequencies()

    def test_log_frequency_vector_matches_closed_form(self):
        """Per the source, the k-th bin is `f0 * 2**(k/n_bins_per_octave)`
        for `k=0, 1, ...` up to (but not including) the octave count that
        would reach the stop frequency.

        """
        f0 = 100.0
        n_bins_per_octave = 12
        v = dsp.tools.log_frequency_vector([f0, 800.0], n_bins_per_octave)
        k = np.arange(len(v))
        expected = f0 * 2 ** (k / n_bins_per_octave)
        np.testing.assert_allclose(v, expected, rtol=1e-14)

    def test_log_frequency_vector_invalid_parameters_raise(self):
        with pytest.raises(AssertionError):
            dsp.tools.log_frequency_vector([0, 200], 10)
        with pytest.raises(AssertionError):
            dsp.tools.log_frequency_vector([-20, 200], 10)

    def test_fractional_octave_frequencies_match_iec_61260(self):
        """Published IEC 61260-1:2014 nominal band centers (Hz). 1/1-octave:
        31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000. 1/3-octave
        (subset): 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
        400, 500, 630.

        """
        nominal_1_1, _ = dsp.tools.fractional_octave_frequencies(1, (20, 20e3))
        expected_1_1 = np.array(
            [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        )
        np.testing.assert_allclose(nominal_1_1, expected_1_1)

        nominal_1_3, _ = dsp.tools.fractional_octave_frequencies(3, (20, 700))
        expected_1_3 = np.array(
            [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630]
        )
        np.testing.assert_allclose(nominal_1_3, expected_1_3)

    def test_fractional_octave_frequencies_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            dsp.tools.fractional_octave_frequencies(1, (20,))
        with pytest.raises(ValueError):
            dsp.tools.fractional_octave_frequencies(1, (2000, 20))

    def test_erb_frequencies_round_trip_and_closed_form(self):
        """Per the source (citing Hohmann 2002, Eq. 16), the ERB scale used
        here is `erb(f) = 9.2645*sign(f)*ln(1+|f|*0.00437)`, with inverse
        `f(erb) = sign(erb)/0.00437 * (exp(|erb|/9.2645) - 1)`. This is a
        distinct (natural-log-based) formula from the more commonly cited
        Glasberg & Moore `21.4*log10(...)` ERB-rate scale, so it is
        re-derived independently here from the source's own docstring/
        reference rather than the textbook formula.

        """

        def erb_of_hz(f_hz):
            return 9.2645 * np.sign(f_hz) * np.log(1 + np.abs(f_hz) * 0.00437)

        def hz_of_erb(erb):
            return np.sign(erb) / 0.00437 * (np.exp(np.abs(erb) / 9.2645) - 1)

        freqs_hz = np.array([50.0, 500.0, 1000.0, 4000.0, 15000.0])
        np.testing.assert_allclose(hz_of_erb(erb_of_hz(freqs_hz)), freqs_hz, rtol=1e-10)

        result = dsp.tools.erb_frequencies(
            freq_range_hz=(100, 4000), resolution=1.0, reference_frequency_hz=1000
        )
        # The vector must be linearly spaced by exactly `resolution` ERB
        # units and its own Hz<->ERB round trip must hold.
        erb_vals = erb_of_hz(result)
        np.testing.assert_allclose(np.diff(erb_vals), 1.0, atol=1e-10)
        np.testing.assert_allclose(hz_of_erb(erb_vals), result, rtol=1e-10)

    def test_frequency_crossover_boundary_values(self):
        """The crossover is a Hann-window fade-in interpolated over the
        frequency axis, with `fill_value=(0.0, 1.0)` for out-of-range
        queries (per the source). For the (default) logarithmic mode, the
        vector's first sample coincides exactly with the start frequency
        (where the underlying Hann window is exactly 0), while the last
        sample falls strictly short of the stop frequency -- so querying
        exactly at the stop frequency lands in the *extrapolated* region
        and returns the exact fill value 1.0, not a near-1 interpolated
        value (verified empirically). The non-logarithmic mode does NOT
        have this property (its vector's last sample lands exactly ON the
        stop frequency, an actually-interpolated near-1 value), so this
        checks the logarithmic (default) mode only.

        """
        crossover = dsp.tools.frequency_crossover([100.0, 200.0], logarithmic=True)
        assert crossover(100.0) == 0.0
        assert crossover(200.0) == 1.0
        assert crossover(50.0) == 0.0
        assert crossover(300.0) == 1.0

    def test_log_mean_matches_manual_log_resampling(self):
        """Per the source, `log_mean` treats `x` as sampled at linearly
        spaced integer positions `1..N`, resamples it at `N` log-spaced
        positions between 1 and N (`N**(k/(N-1))` for `k=0..N-1`) via
        linear interpolation, then takes the arithmetic mean. Re-derived
        here directly with `scipy.interpolate.interp1d`, independent of
        the internal helper.

        """
        from scipy.interpolate import interp1d

        x = np.linspace(100.0, 1000.0, 50)
        n = len(x)
        positions = np.arange(1, n + 1)
        log_positions = n ** (np.arange(n) / (n - 1))
        resampled = interp1d(positions, x, kind="linear", assume_sorted=True)(
            log_positions
        )
        expected = np.mean(resampled)

        result = dsp.tools.log_mean(x)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_framed_signal(self):
        # Only functionality, no results
        n = np.random.normal(0, 0.1, (100, 1))
        dsp.tools.framed_signal(n, 20, 10, True)
        nn1 = dsp.tools.framed_signal(n, 20, 10, False)

        n = np.random.normal(0, 0.1, (100, 2))
        dsp.tools.framed_signal(n, 20, 10, True)
        nn2 = dsp.tools.framed_signal(n, 20, 10, False)

        dsp.tools.reconstruct_from_framed_signal(nn1, 10, None, len(n))
        dsp.tools.reconstruct_from_framed_signal(nn2, 10, None, len(n))

    def test_convert_sample_conversion(self):
        v = np.array([0.0, 1.0, -1.0, 0.5])
        np.testing.assert_equal(
            v,
            dsp.tools.convert_sample_representation(v, "f64", "f32", True)[0],
        )
        with pytest.raises(AssertionError):
            dsp.tools.convert_sample_representation(v, "f64", "f64", True)

        # ------ Standard f64 input
        # With casting
        for t in ["u8", "u16", "u32", "i8", "i16", "i32"]:
            out, eq, max_val = dsp.tools.convert_sample_representation(
                v, "f64", t, True
            )
            np.testing.assert_equal(
                out,
                np.array([eq, eq + max_val, eq - max_val, eq + max_val // 2]),
            )

        # Without casting
        for t in ["i24", "u24"]:
            out, eq, max_val = dsp.tools.convert_sample_representation(
                v, "f64", t, False
            )
            np.testing.assert_equal(
                out,
                np.array([eq, eq + max_val, eq - max_val, eq + max_val // 2]),
            )

        # ------ Some different inputs to "f64" output
        for f in ["i8", "u8", "i16", "u16", "i24", "u24", "i32", "u32"]:
            bits = int(f[1:])
            signed = f[0] == "i"
            val = 2 ** (bits - 1) - 1
            eq = 0 if signed else val
            v = np.array([eq, eq + val, eq - val])
            np.testing.assert_equal(
                np.array([0, 1.0, -1.0]),
                dsp.tools.convert_sample_representation(v, f, "f64", False)[0],
            )

        # Random input and output
        formats = [
            "u8",
            "u16",
            "u32",
            "i8",
            "i16",
            "i32",
            "f32",
            "f64",
            "i24",
            "u24",
        ]
        for _ in range(4):
            inds = list(range(len(formats)))
            input_ind = choice(inds)
            inds.pop(input_ind)
            output_ind = choice(inds)
            dsp.tools.convert_sample_representation(
                v, formats[input_ind], formats[output_ind], False
            )

        # Bytes with 24-bits representations
        inp = np.array([0.0, 1.0, -1.0, 0.5])
        for t in ["i24", "u24", "i32", "f32"]:
            b = dsp.tools.convert_sample_representation(inp, "f64", t, True, True)[0]
            outp = dsp.tools.convert_sample_representation(b, t, "f64", True, True)[0]
            np.testing.assert_allclose(inp, outp, atol=1e-4)

    def test_fractional_octave_smoothing(self):
        fs_hz = 48000
        lin_freqs = np.fft.rfftfreq(10000, 1 / fs_hz)[:-1]
        filt = dsp.Filter.biquad(dsp.BiquadEqType.Peaking, 200.0, 1.0, 0.8, fs_hz)
        transfer_lin = np.abs(filt.get_transfer_function(lin_freqs))
        smoothed_lin = dsp.tools.fractional_octave_smoothing(transfer_lin, None, 8.0)

        log_freqs = dsp.tools.log_frequency_vector([10, 10e3], 128)
        transfer_log = np.abs(filt.get_transfer_function(log_freqs))
        smoothed_log = dsp.tools.fractional_octave_smoothing(transfer_log, None, 8.0)

        smoothed_lin_log = dsp.tools.interpolate_fr(
            lin_freqs, smoothed_lin, log_freqs, mode="amplitude2power"
        )
        np.testing.assert_allclose(
            dsp.tools.to_db(smoothed_lin_log, True),
            dsp.tools.to_db(smoothed_log, True),
            atol=0.01,
        )
