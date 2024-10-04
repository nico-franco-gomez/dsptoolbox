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
