from dsptoolbox.classes import Signal
from dsptoolbox._standard import (_get_framed_signal,
                                  _reconstruct_framed_signal)
from dsptoolbox._general_helpers import _get_next_power_2
from ._effects import AudioEffect

from scipy.signal.windows import get_window
import numpy as np
from warnings import warn


class SpectralSubtractor(AudioEffect):
    """This class implements a classical spectral subtraction for denoising
    or other purposes that can act adaptively (in blocking mode) or globally
    (non-blocking mode)

    """
    def __init__(self, blocking_mode: bool = True,
                 threshold_rms_dbfs: float = -40,
                 block_length_s: float = 0.1,
                 spectrum_to_subtract: np.ndarray = None):
        """Constructor for a spectral subtractor denoising effect.

        Parameters
        ----------
        blocking_mode : bool, optional
            When `True`, the spectrum to subtract is dynamic and gets updated
            during the signal's passing. Otherwise, the power spectrum of the
            noise in the whole signal is computed and applied statically.
            Default: `True`.
        threshold_rms_dbfs : float, optional
            Threshold for the RMS value of a block in dBFS. When below the
            threshold, the block is regarded as noise. Default: -40.
        spectrum_to_subtract : np.ndarray, optional
            If a spectrum is passed, it is used as the one to subtract and
            all other parameters are ignored. This should be passed in dB
            without any scaling after applying the FFT in order to avoid
            scaling discrepancies. Default: `None`.

        """
        super().__init__(True, True,
                         description='Spectral Subtraction (Denoiser)')
        self.__set_parameters(blocking_mode, threshold_rms_dbfs,
                              block_length_s)
        self.set_advanced_parameters()
        self.spectrum_to_subtract = spectrum_to_subtract

    def __set_parameters(self, blocking_mode, threshold_rms_dbfs,
                         block_length_s):
        """Internal method to set the parameters for the spectral subtraction.

        """
        assert type(blocking_mode) == bool, \
            'Blocking mode must be of boolean type'
        self.blocking_mode = blocking_mode

        assert type(threshold_rms_dbfs) in (int, float), \
            'Threshold must be of type int or float'
        if threshold_rms_dbfs >= 0:
            warn('Threshold is positive. This might be a wrong input')
        self.threshold_rms_dbfs = threshold_rms_dbfs

        assert type(block_length_s) in (int, float), \
            'Block length should be of type int or float'
        self.block_length_s = block_length_s

    def set_advanced_parameters(self, overlap_percent: float = 0.5,
                                window_type: str = 'hann',
                                noise_forgetting_factor: float = 0.95,
                                subtraction_factor: float = 1,
                                subtraction_exponent: float = 2):
        """This allows for setting the advanced parameters of the audio
        effect.

        Parameters
        ----------
        overlap_percent : float, optional
            Window overlap in percent. Default: 0.5.
        window_type : str, optional
            Window type to use. Default: `'hann'`.
        noise_forgetting_factor : float, optional
            This factor is used to average the noise spectrum in order to
            reduce distortions at the expense of responsiveness. It should
            be between 0 and 1. Default: 0.95.
        subtraction_factor : float, optional
            The subtraction factor defines how strongly noise is subtracted
            from the signal. It can take values larger than one leading to
            a strong noise subtraction. Default: 1.
        subtraction_exponent : float, optional
            The subtraction exponent defines the exponent to which the spectral
            are scaled during the subtraction. 2 means it is a power
            subtraction and 1 is an amplitude subtraction. Other values are
            also possible. Default: 2.

        """
        assert (0 <= overlap_percent) and (1 > overlap_percent), \
            'Overlap should be in [0, 1['
        self.overlap_percent = overlap_percent

        self.window_type = window_type

        assert (0 < noise_forgetting_factor) and \
            (noise_forgetting_factor <= 1), \
            'Noise forgetting factor must be in ]0, 1]'
        self.noise_forgetting_factor = noise_forgetting_factor

        assert subtraction_factor > 0, \
            'The subtraction factor must be positive'
        self.subtraction_factor = subtraction_factor

        assert subtraction_exponent > 0, \
            'Subtraction exponent should be above zero'
        self.subtraction_exponent = subtraction_exponent

    def set_parameters(self, blocking_mode: bool = True,
                       threshold_rms_dbfs: float = -40,
                       block_length_s: float = 0.1):
        """Sets the audio effects parameters.

        """
        self.__set_parameters(
            blocking_mode, threshold_rms_dbfs, block_length_s)

    def _get_lengths(self, sampling_rate_hz):
        """Internal method to compute the window and step size in samples.

        """
        window_length = _get_next_power_2(self.block_length_s*sampling_rate_hz)
        self.window = np.clip(get_window(
            self.window_type, window_length), a_min=1e-6, a_max=None)
        self.step_size = int(window_length * (1 - self.overlap_percent))

    def apply(self, signal: Signal):
        """Applies the spectral subtraction to a passed signal.

        Parameters
        ----------
        signal : `Signal`
            Signal to which to apply the denoising. If it contains multiple
            channels, the effect is applied independently for each one.

        Returns
        -------
        denoised_signal : `Signal`
            Denoised signal.

        """
        # Lengths and window
        self._get_lengths(signal.sampling_rate_hz)
        # Factor for Power Spectral Density (sum to integrate afterwards)
        factor = 2 / (self.window @ self.window) / len(self.window)

        # Framed signal
        td = _get_framed_signal(signal.time_data, len(self.window),
                                self.step_size)
        # Windowed signal
        td_windowed = td * self.window[:, np.newaxis, np.newaxis]
        td_spec = np.fft.rfft(td_windowed, axis=0)

        # Phase
        td_spec_phase = np.angle(td_spec)
        td_spec = np.abs(td_spec)

        # Power spectrum
        td_spec_power = td_spec ** self.subtraction_exponent

        # Get RMS values in dB for each time frame and channel
        td_rms_db = 20*np.log10(np.sum(td_spec, axis=0)) + \
            10*np.log10(factor)

        # Noise estimate
        noise_psd = np.zeros((len(self.window)//2+1))

        # Iterate over frames
        for n in range(signal.number_of_channels):
            print(f'Denoising channel {n+1} of {signal.number_of_channels}')
            for i in range(td_spec.shape[1]):
                if td_rms_db[i, n] < self.threshold_rms_dbfs:
                    noise_psd = noise_psd * self.noise_forgetting_factor + \
                        td_spec[:, i, n] * (1 - self.noise_forgetting_factor)
                temp = np.clip(
                    td_spec_power[:, i, n] - self.subtraction_factor *
                    noise_psd**self.subtraction_exponent,
                    a_min=0, a_max=None)
                td[:, i, n] = np.fft.irfft(
                    temp**(1/self.subtraction_exponent) *
                    np.exp(1j*td_spec_phase[:, i, n]))

        new_td = _reconstruct_framed_signal(
            td, self.step_size, self.window, signal.time_data.shape[0])
        denoised_signal = signal.copy()
        denoised_signal.time_data = new_td
        return denoised_signal
