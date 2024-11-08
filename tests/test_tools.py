import dsptoolbox as dsp
import numpy as np


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
            v, dsp.tools.convert_sample_representation(v, "f32", True)[0]
        )

        # With casting
        for t in ["u8", "u16", "u32", "i8", "i16", "i32"]:
            out, eq, max_val = dsp.tools.convert_sample_representation(
                v, t, True
            )
            np.testing.assert_equal(
                out,
                np.array([eq, eq + max_val, eq - max_val, eq + max_val // 2]),
            )

        # Without casting
        for t in ["i24", "u24"]:
            out, eq, max_val = dsp.tools.convert_sample_representation(
                v, t, False
            )
            np.testing.assert_equal(
                out,
                np.array([eq, eq + max_val, eq - max_val, eq + max_val // 2]),
            )
