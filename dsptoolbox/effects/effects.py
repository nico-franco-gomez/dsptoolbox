from dsptoolbox.classes import Signal
from dsptoolbox import activity_detector
from dsptoolbox._standard import (_get_framed_signal,
                                  _reconstruct_framed_signal,
                                  _pad_trim)
from dsptoolbox._general_helpers import _get_next_power_2
from ._effects import AudioEffect

from scipy.signal.windows import get_window
import numpy as np
from warnings import warn


class SpectralSubtractor(AudioEffect):
    """This class implements a classical spectral subtraction for denoising
    or other purposes that can act adaptively (in adaptive mode) or globally
    (static mode).

    """
    def __init__(self, adaptive_mode: bool = True,
                 threshold_rms_dbfs: float = -40,
                 block_length_s: float = 0.1,
                 spectrum_to_subtract: np.ndarray = None):
        """Constructor for a spectral subtractor denoising effect. More
        parameters can be passed using the method `set_advanced_parameters`.

        Parameters
        ----------
        adaptive_mode : bool, optional
            When `True`, the subtracted spectrum is dynamic and gets updated
            during the signal's passing. Otherwise, the spectrum of the
            noise in the whole signal is computed and applied statically. This
            could be advantageous when the noise in the signal is thought to be
            stationary and the least possible audible distortions are expected.
            In order to separate signal from noise,
            `dsptoolbox.activity_detector` is used. Default: `True`.
        threshold_rms_dbfs : float, optional
            Threshold for the RMS value of a signal segment in dBFS that has
            to separates signal from noise. This means, when below the
            threshold, the signal segment is regarded as noise. Default: -40.
        block_length_s : float, optional
            Block length in seconds. The spectral subtraction is done over
            blocks of the signal. The real block length in samples is always
            clipped to the closest power of 2 for efficiency of the FFT.
            Default: 0.1.
        spectrum_to_subtract : np.ndarray, optional
            If a spectrum is passed, it is used as the one to subtract and
            all other parameters are ignored. This should be the result of the
            squared magnitude of the FFT without any scaling in order to avoid
            scaling discrepancies. It should be only the spectrum corresponding
            to the positive frequencies (including 0). Default: `None`.

        Methods
        -------
        - `set_parameters()`: Basic parameters used.
        - `set_advanced_parameters()`: fine-tuning parameters for both adaptive
          and static mode.
        - `apply()`: Apply effect on a given signal.

        """
        super().__init__(True, True,
                         description='Spectral Subtraction (Denoiser)')
        self.__set_parameters(adaptive_mode, threshold_rms_dbfs,
                              block_length_s, spectrum_to_subtract)
        self.set_advanced_parameters()

    def __set_parameters(self, adaptive_mode, threshold_rms_dbfs,
                         block_length_s, spectrum_to_subtract):
        """Internal method to set the parameters for the spectral subtraction.

        """
        assert type(adaptive_mode) == bool, \
            'Adaptive mode must be of boolean type'
        self.adaptive_mode = adaptive_mode

        assert type(threshold_rms_dbfs) in (int, float), \
            'Threshold must be of type int or float'
        if threshold_rms_dbfs >= 0:
            warn('Threshold is positive. This might be a wrong input')
        self.threshold_rms_dbfs = threshold_rms_dbfs

        assert type(block_length_s) in (int, float), \
            'Block length should be of type int or float'
        self.block_length_s = block_length_s

        if spectrum_to_subtract is not None:
            assert type(spectrum_to_subtract) == np.ndarray, \
                'Spectrum to subtract must be of type numpy.ndarray'
            spectrum_to_subtract = np.squeeze(spectrum_to_subtract)
            assert spectrum_to_subtract.ndim == 1, \
                'Spectrum to subtract could not be broadcasted to a 1D-Array'
            if self.adaptive_mode:
                warn('A spectrum to subtract was passed but adaptive mode ' +
                     'was selected. This is unsupported. Setting adaptive ' +
                     'mode to False')
                self.adaptive_mode = False
        self.spectrum_to_subtract = spectrum_to_subtract

    def set_advanced_parameters(
            self, overlap_percent: int = 50,
            window_type: str = 'hann',
            noise_forgetting_factor: float = 0.9,
            subtraction_factor: float = 2,
            subtraction_exponent: float = 2,
            ad_attack_time_ms: float = 0.5,
            ad_release_time_ms: float = 30):
        """This allows for setting up the advanced parameters of the spectral
        subtraction.

        Parameters
        ----------
        overlap_percent : int, optional
            Window overlap in percent. Default: 50.
        window_type : str, optional
            Window type to use. Default: `'hann'`.
        noise_forgetting_factor : float, optional
            This factor is used to average the noise spectrum in order to
            reduce distortions at the expense of responsiveness. It should
            be between 0 and 1. The lower this value, the faster the algorithm
            responds to changes in the noise. Default: 0.5.
        subtraction_factor : float, optional
            The subtraction factor defines how strongly noise is subtracted
            from the signal. It can take values larger than one leading to
            a strong noise subtraction with possibly more distortion.
            Default: 2.
        subtraction_exponent : float, optional
            The subtraction exponent defines the exponent to which the spectral
            are scaled during the subtraction. 2 means it is a power
            subtraction and 1 is an amplitude subtraction. Other values are
            also possible. Default: 2.
        ad_attack_time_ms : float, optional
            Attack time in ms for the activity detector (static mode).
            Default: 0.9.
        ad_release_time_ms : float, optional
            Release time for the activity detector (static mode).
            Default: 30.
        maximum_amplification_db : float, optional
            Maximum sample amplification in dB. During signal reconstruction,
            some samples in the signal might be amplified by large values
            (depending on window and overlap). This parameter sets the maximum
            value to which this amplification is allowed. Pass `None` to ignore
            it. This might reconstruct the signal better but can lead sometimes
            to instabilities. Default: 60.

            It is also advisable to zero-pad a signal in the beginning to
            avoid instabilities due to a lack of window overlap on the edges.

        Notes
        -----
        Parameters in use according to mode:

            - Adaptive mode:
                - overlap_percent
                - window_type
                - noise_forgetting_factor
                - subtraction_factor
                - subtraction_exponent
                - maximum_amplification_db

            - Static Mode:
                - overlap_percent
                - window_type
                - subtraction_factor
                - subtraction_exponent
                - vad_attack_time_ms
                - vad_release_time_ms
                - maximum_amplification_db

        """
        assert (0 <= overlap_percent) and (100 > overlap_percent), \
            'Overlap should be in [0, 100['
        self.overlap = overlap_percent / 100

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

        # === Static Mode
        assert ad_attack_time_ms >= 0, \
            'Attack time for activity detector must be above 0'
        self.ad_attack_time_ms = ad_attack_time_ms

        assert ad_release_time_ms >= 0, \
            'Attack time for activity detector must be above 0'
        self.ad_release_time_ms = ad_release_time_ms

    def set_parameters(self, adaptive_mode: bool = True,
                       threshold_rms_dbfs: float = -40,
                       block_length_s: float = 0.1,
                       spectrum_to_subtract: np.ndarray = None):
        """Sets the audio effects parameters.

        """
        self.__set_parameters(
            adaptive_mode, threshold_rms_dbfs, block_length_s,
            spectrum_to_subtract)

    def _compute_window(self, sampling_rate_hz):
        """Internal method to compute the window and step size in samples.

        """
        if self.spectrum_to_subtract is None:
            self.window_length = _get_next_power_2(
                self.block_length_s*sampling_rate_hz)
        else:
            self.window_length = (len(self.spectrum_to_subtract)-1) * 2
        self.window = get_window(self.window_type, self.window_length)
        self.window = np.clip(get_window(
            self.window_type, self.window_length), a_min=1e-6, a_max=None)
        self.step_size = int(self.window_length * (1 - self.overlap))

    def apply(self, signal: Signal) -> Signal:
        """Applies the spectral subtraction to a passed signal. When
        `adaptive_mode=True`, the subtracted spectrum is dynamically updated.
        Otherwise, an activity detector is used and the subtracted noise is
        statically defined. If a spectrum was passed, it is statically used.

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
        if self.adaptive_mode:
            return self._apply_adaptive_mode(signal)
        else:
            return self._apply_offline(signal)

    def _apply_offline(self, signal: Signal) -> Signal:
        """Spectral Subtraction in static mode (offline).

        """
        # Lengths according to sampling rate
        self._compute_window(signal.sampling_rate_hz)

        # Pad zeros in beginning and end to avoid window instabilities
        td = signal.time_data
        td = _pad_trim(td, td.shape[0]+len(self.window), in_the_end=True)
        td = _pad_trim(td, td.shape[0]+len(self.window), in_the_end=False)
        original_length = td.shape[0]

        # Frame initial time data
        td_framed = _get_framed_signal(td, len(self.window), self.step_size)

        # Windowed signal
        td_windowed = td_framed * self.window[:, np.newaxis, np.newaxis]
        td_spec = np.fft.rfft(td_windowed, axis=0)

        # Phase
        td_spec_phase = np.angle(td_spec)
        td_spec_power = np.abs(td_spec) ** self.subtraction_exponent

        for n in range(signal.number_of_channels):
            if self.spectrum_to_subtract is None:
                # Obtain noise psd
                _, noise = activity_detector(
                    signal, channel=n, threshold_dbfs=self.threshold_rms_dbfs,
                    attack_time_ms=self.ad_attack_time_ms,
                    release_time_ms=self.ad_release_time_ms)
                noise['noise'].set_spectrum_parameters(
                    method='welch',
                    window_length_samples=len(self.window),
                    overlap_percent=self.overlap*100,
                    window_type=self.window_type,
                    scaling=None)
                _, noise_psd = noise['noise'].get_spectrum()
            else:
                noise_psd = self.spectrum_to_subtract.copy()
            # It is already raised to the power of 2!
            noise_psd = np.abs(noise_psd).squeeze() ** \
                (self.subtraction_exponent/2)
            for i in range(td_spec.shape[1]):
                temp = np.clip(
                    td_spec_power[:, i, n] - self.subtraction_factor *
                    noise_psd, a_min=0, a_max=None)
                td_framed[:, i, n] = np.fft.irfft(
                    temp**(1/self.subtraction_exponent) *
                    np.exp(1j*td_spec_phase[:, i, n]))

        # Reconstruct signal from time frames
        new_td = _reconstruct_framed_signal(
            td_framed, self.step_size, self.window, original_length,
            None)

        # Trim back to original length
        new_td = _pad_trim(new_td, new_td.shape[0]-len(self.window),
                           in_the_end=True)
        new_td = _pad_trim(new_td, new_td.shape[0]-len(self.window),
                           in_the_end=False)

        denoised_signal = signal.copy()
        denoised_signal.time_data = new_td
        return denoised_signal

    def _apply_adaptive_mode(self, signal: Signal) -> Signal:
        """Spectral Subtraction in adaptive mode.

        """
        # Lengths and window
        self._compute_window(signal.sampling_rate_hz)

        td = signal.time_data
        td = _pad_trim(td, td.shape[0]+len(self.window), in_the_end=True)
        td = _pad_trim(td, td.shape[0]+len(self.window), in_the_end=False)
        original_length = td.shape[0]

        # Framed signal
        td = _get_framed_signal(td, len(self.window), self.step_size)

        # Get RMS values in dB for each time frame and channel
        td_rms_db = 20*np.log10(np.var(td, axis=0))

        # Windowed signal
        td_windowed = td * self.window[:, np.newaxis, np.newaxis]
        td_spec = np.fft.rfft(td_windowed, axis=0)

        # Phase
        td_spec_phase = np.angle(td_spec)
        td_spec = np.abs(td_spec)

        # Power spectrum
        td_spec_power = td_spec ** self.subtraction_exponent

        # Iterate over frames
        for n in range(signal.number_of_channels):
            # Noise estimate
            noise_psd = np.zeros((len(self.window)//2+1))

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

        # Reconstruct signal from time frames
        new_td = _reconstruct_framed_signal(
            td, self.step_size, self.window, original_length)

        # Trim back to original length
        new_td = _pad_trim(new_td, new_td.shape[0]-len(self.window),
                           in_the_end=True)
        new_td = _pad_trim(new_td, new_td.shape[0]-len(self.window),
                           in_the_end=False)

        denoised_signal = signal.copy()
        denoised_signal.time_data = new_td
        return denoised_signal
