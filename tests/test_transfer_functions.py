import dsptoolbox as dsp
from os.path import join


class TestTransferFunctionsModule():
    y_m = dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    y_st = dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    x = dsp.Signal(join('examples', 'data', 'chirp.wav'))

    def test_deconvolve(self):
        # Only functionality is tested here
        # Regularized
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m, self.x, mode='regularized', start_stop_hz=None,
            threshold_db=-30, padding=False, keep_original_length=False)
        dsp.transfer_functions.spectral_deconvolve(
            self.y_st, self.x, mode='regularized', start_stop_hz=None,
            threshold_db=-30, padding=False, keep_original_length=False)
        dsp.transfer_functions.spectral_deconvolve(
            self.y_st, self.x, mode='regularized',
            start_stop_hz=[30, 15e3], threshold_db=None, padding=False,
            keep_original_length=False)
        # Window
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m, self.x, mode='window', start_stop_hz=None,
            threshold_db=-30, padding=False, keep_original_length=False)
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m, self.x, mode='window', start_stop_hz=[30, 15e3],
            threshold_db=-30, padding=False, keep_original_length=False)
        # Standard
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m, self.x, mode='standard', start_stop_hz=None,
            threshold_db=None, padding=False, keep_original_length=False)
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m, self.x, mode='standard', start_stop_hz=None,
            threshold_db=None, padding=True, keep_original_length=False)
        dsp.transfer_functions.spectral_deconvolve(
            self.y_m, self.x, mode='standard', start_stop_hz=None,
            threshold_db=None, padding=True, keep_original_length=True)

    def test_window_ir(self):
        # Only functionality
        h = dsp.transfer_functions.spectral_deconvolve(self.y_m, self.x)

        dsp.transfer_functions.window_ir(
            h, window_type='hann', at_start=True)
        dsp.transfer_functions.window_ir(
            h, window_type='hann', at_start=False)
        dsp.transfer_functions.window_ir(
            h, exp2_trim=None, window_type='hann', at_start=True)

        # Try window with extra parameters
        dsp.transfer_functions.window_ir(
            h, exp2_trim=None, window_type=('chebwin', 50), at_start=True)

    def test_compute_transfer_function(self):
        # Only functionality
        # Multi-channel
        dsp.transfer_functions.compute_transfer_function(
            self.y_st, self.x, mode='H1', window_length_samples=1024,
            spectrum_parameters=None)
        # Single-channel with other windows
        h = dsp.transfer_functions.compute_transfer_function(
            self.y_m, self.x, mode='H2', window_length_samples=1024,
            spectrum_parameters=dict(window_type=('chebwin', 40)))
        # Check that coherence is saved
        h.get_coherence()

    def test_spectral_average(self):
        # Only functionality is tested
        h = dsp.transfer_functions.spectral_deconvolve(self.y_st, self.x)
        dsp.transfer_functions.spectral_average(h)

    def test_min_phase_from_mag(self):
        # Only functionality is tested
        self.y_st.set_spectrum_parameters(method='standard')
        f, sp = self.y_st.get_spectrum()
        dsp.transfer_functions.min_phase_from_mag(
            sp, self.y_st.sampling_rate_hz)

    def test_lin_phase_from_mag(self):
        # Only functionality is tested here
        self.y_st.set_spectrum_parameters(method='standard')
        f, sp = self.y_st.get_spectrum()
        dsp.transfer_functions.lin_phase_from_mag(
            sp, self.y_st.sampling_rate_hz, group_delay_ms='minimum')
