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
