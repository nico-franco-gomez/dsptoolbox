from dsptoolbox.classes import Signal, MultiBandSignal
from dsptoolbox import activity_detector
from dsptoolbox._standard import (_get_framed_signal,
                                  _reconstruct_framed_signal,
                                  _pad_trim,
                                  _rms)
from dsptoolbox._general_helpers import _get_next_power_2
from ._effects import (
    _arctan_distortion, _clean_signal, _hard_clip_distortion,
    _soft_clip_distortion, _compressor, _get_knee_func)
from dsptoolbox.plots import general_plot

from scipy.signal.windows import get_window
import numpy as np
from warnings import warn


class AudioEffect():
    """Base class for audio effects.

    """
    def __init__(self, description: str = None):
        """Base constructor for an audio effect.

        Parameters
        ----------
        description : str, optional
            A string containing a general description about the audio effect.
            Default: `None`.

        """
        self.description = description

    def apply(self, signal: Signal | MultiBandSignal) \
            -> Signal | MultiBandSignal:
        """Apply audio effect on a given signal.

        Parameters
        ----------
        signal : `Signal` or `MultiBandSignal`
            Signal to which the effect should be applied.

        Returns
        -------
        modified_signal : `Signal` or `MultiBandSignal`
            Modified signal.

        """
        if type(signal) == Signal:
            return self._apply_this_effect(signal)
        elif type(signal) == MultiBandSignal:
            new_mbs = signal.copy()
            for i, b in enumerate(new_mbs.bands):
                new_mbs.bands[i] = self.apply(b)
            return new_mbs
        else:
            raise TypeError('Audio effect can only be applied to Signal ' +
                            'or MultiBandSignal')

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Abstract class method to apply the audio effect on a given signal.

        """
        return signal

    def _add_gain_in_db(self, time_data: np.ndarray, gain_db: float) \
            -> np.ndarray:
        """General gain stage.

        Parameters
        ----------
        time_data : `np.ndarray`
            Time samples of the signal.
        gain_db : float
            Gain in dB.

        Returns
        -------
        new_time_data : `np.ndarray`
            Time data with new gain.

        """
        if gain_db is None:
            return time_data
        return time_data * 10**(gain_db/20)

    def _save_peak_values(self, inp: np.ndarray):
        """Save the peak values of an input.

        """
        self._peak_values = np.max(np.abs(inp), axis=0)

    def _restore_peak_values(self, inp: np.ndarray) -> np.ndarray:
        """Restore saved peak values of a signal.

        """
        if not hasattr(self, '_peak_values'):
            return inp
        if len(self._peak_values) != inp.shape[1]:
            warn('Number of saved peak values does not match number of ' +
                 'channels. Restoring is ignored')
            return inp
        return inp * (self._peak_values/np.max(np.abs(inp), axis=0))

    def _save_rms_values(self, inp: np.ndarray):
        """Save the RMS values of a signal.

        """
        self._rms_values = _rms(inp)

    def _restore_rms_values(self, inp: np.ndarray) -> np.ndarray:
        """Restore the RMS values of a signal.

        """
        if not hasattr(self, '_rms_values'):
            return inp
        if len(self._rms_values) != inp.shape[1]:
            warn('Number of saved RMS values does not match number of ' +
                 'channels. Restoring is ignored')
            return inp
        return inp * (self._rms_values/_rms(inp))


class SpectralSubtractor(AudioEffect):
    """This class implements a classical spectral subtraction for denoising
    or other purposes that can act adaptively (in adaptive mode) or globally
    (static mode). It is possible to pass either a Signal or a MultiBandSignal
    to denoise in different frequency bands.

    """
    def __init__(self, adaptive_mode: bool = True,
                 threshold_rms_dbfs: float = -40,
                 block_length_s: float = 0.1,
                 spectrum_to_subtract: np.ndarray = False):
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
            to the positive frequencies (including 0). Default: `False`.

        Methods
        -------
        - `set_parameters()`: Basic parameters used.
        - `set_advanced_parameters()`: fine-tuning parameters for both adaptive
          and static mode.
        - `apply()`: Apply effect on a given signal.

        """
        super().__init__(description='Spectral Subtraction (Denoiser)')
        self.__set_parameters(adaptive_mode, threshold_rms_dbfs,
                              block_length_s, spectrum_to_subtract)
        self.set_advanced_parameters()

    def __set_parameters(self, adaptive_mode, threshold_rms_dbfs,
                         block_length_s, spectrum_to_subtract):
        """Internal method to set the parameters for the spectral subtraction.

        """
        if adaptive_mode is not None:
            assert type(adaptive_mode) == bool, \
                'Adaptive mode must be of boolean type'
            self.adaptive_mode = adaptive_mode

        if threshold_rms_dbfs is not None:
            assert type(threshold_rms_dbfs) in (int, float), \
                'Threshold must be of type int or float'
            if threshold_rms_dbfs >= 0:
                warn('Threshold is positive. This might be a wrong input')
            self.threshold_rms_dbfs = threshold_rms_dbfs

        if block_length_s is not None:
            assert type(block_length_s) in (int, float), \
                'Block length should be of type int or float'
            self.block_length_s = block_length_s

        if spectrum_to_subtract is not None:
            if spectrum_to_subtract:
                assert type(spectrum_to_subtract) == np.ndarray, \
                    'Spectrum to subtract must be of type numpy.ndarray'
                spectrum_to_subtract = np.squeeze(spectrum_to_subtract)
                assert spectrum_to_subtract.ndim == 1, \
                    'Spectrum to subtract could not be broadcasted to ' +\
                    'a 1D-Array'
                if self.adaptive_mode:
                    warn('A spectrum to subtract was passed but adaptive ' +
                         'mode was selected. This is unsupported. Setting ' +
                         'adaptive mode to False')
                    self.adaptive_mode = False
            self.spectrum_to_subtract = spectrum_to_subtract

    def set_advanced_parameters(
            self, overlap_percent: int = 50,
            window_type: str = 'hann',
            noise_forgetting_factor: float = 0.9,
            subtraction_factor: float = 2,
            subtraction_exponent: float = 2,
            ad_hold_time_ms: float = 0.5,
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
        ad_hold_time_ms : float, optional
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
                - maximum_amplification_db
                - ad_hold_time_ms
                - ad_release_time_ms

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
        assert ad_hold_time_ms >= 0, \
            'Hold time for activity detector must be 0 or above'
        self.ad_hold_time_ms = ad_hold_time_ms

        assert ad_release_time_ms >= 0, \
            'Release time for activity detector must be 0 or above'
        self.ad_release_time_ms = ad_release_time_ms

    def set_parameters(self, adaptive_mode: bool = None,
                       threshold_rms_dbfs: float = None,
                       block_length_s: float = None,
                       spectrum_to_subtract: np.ndarray = None):
        """Sets the audio effects parameters. Pass `None` to leave the
        previously selected value for each parameter unchanged.

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

        """
        self.__set_parameters(
            adaptive_mode, threshold_rms_dbfs, block_length_s,
            spectrum_to_subtract)
        assert self.adaptive_mode is not None, 'None is not a valid value'
        assert self.threshold_rms_dbfs is not None, 'None is not a valid value'
        assert self.block_length_s is not None, 'None is not a valid value'
        assert self.spectrum_to_subtract is not None, \
            'None is not a valid value'

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

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Internal method to trigger the effect on a given signal.

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
                    hold_time_ms=self.ad_hold_time_ms,
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
        td_rms_db = 20*np.log10(np.clip(np.var(td, axis=0), a_min=1e-25,
                                        a_max=None))

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


class Distortion(AudioEffect):
    """This implements a basic distortion effect that can be expanded by the
    user by passing a non-linear functions that can be applied to the waveform.
    Multiple distortions can be linearly combined.

    """
    def __init__(self, distortion_level: float = 20, post_gain_db: float = 0,
                 type_of_distortion: str = 'arctan'):
        """This effect adds non-linear distortion to an audio signal by
        clipping its waveform according to some specific function and
        parameters. Use `set_advanced_parameters` for more control.

        Parameters
        ----------
        distortion_level : float, optional
            This parameter defines the amount of distortion in the signal.
            Depending on the type of distortion, its usable range is between
            0 and 50, though any value can be passed, even below 0.
            Default: 20.
        post_gain_db : float, optional
            This is an additional gain stage in dB after the distortion has
            been applied. Default: 0.
        type_of_distortion : str {'arctan', 'hard clip', 'soft clip'}, optional
            This sets the type of non-linear distortion to be applied. The
            three available types are `'arctan'`, `'hard clip'` and
            `'soft clip'`. Default: `'arctan'`.

        References
        ----------
        - The distortion functions implemented here are partly taken from
          https://www.dsprelated.com/freebooks/pasp/Nonlinear_Distortion.html.

        Notes
        -----
        - The distortion_level is scale-invariant, meaning that the signal is
          always normalized to peak value before applying distortion. If it was
          not the case, the effect would largely depend on both the distortion
          level and the input gain.

        """
        super().__init__('Distortion')
        self.set_advanced_parameters(
            type_of_distortion=type_of_distortion,
            distortion_levels_db=distortion_level, post_gain_db=post_gain_db)

    def set_advanced_parameters(
            self, type_of_distortion='arctan',
            distortion_levels_db: np.ndarray = 20,
            mix_percent: np.ndarray = 100, offset_db: np.ndarray = -np.inf,
            post_gain_db: float = 0):
        r"""This sets the parameters of the distortion. Multiple
        non-linear distortions can be combined with the clean signal and among
        each other. In that case, `distortion_levels`, `mix_percent` and
        `offset_db` must be arrays. Furthermore, the original peak levels of
        each channel in the signal are kept after applying the distortion.

        Parameters
        ----------
        type_of_distortion : list or str or callable, optional
            Type of distortion to be applied. If it is a single string,
            it is applied to the signal and mixed with the clean signal
            according to the mixed parameter. If a list is passed, each entry
            must be either a string corresponding to the supported modes
            (`'arctan'`, `'hard clip'`, `'soft clip'`, `'clean'`) or a callable
            containing a user-defined distortion. Its signature must be::

                func(time_data: np.ndarray, distortion_level_db: float,
                     offset_db: float) -> np.ndarray

            The output data is assumed to have shape (time samples, channels)
            as the input data. If a list is passed, `distortion_levels_db`,
            `mix_percent` and `offset_db` must have the same length as the
            list. Default: `'arctan'`.
        distortion_levels : `np.ndarray`, optional
            This defines how strong the distortion effect is applied. It can
            vary according to the non-linear function. Usually, a range
            between 0 and 50 should be reasonable, though any value is
            possible. If multiple types of distortion are being used, this
            should be an array corresponding to each distortion. Default: 20.
        mix_percent : `np.ndarray`, optional
            This defines how much of each distortion is used in the final
            mix. If `type_of_distortion` is only one string or callable,
            mix_percent is its amount in the final mix with the clean signal.
            This means that 100 leads to only using the distorted signal while
            40 leads to 40% distorted, 60% clean. If multiple types of
            distortion are being used, this should be an array corresponding
            to each distortion and its sum must be 100. Default: 100.
        offset_db : `np.ndarray`, optional
            This offset corresponds to the offset shown in [1]. It must be a
            value between -np.inf and 0. The bigger this value, the more even
            harmonics are caused by the distortion. Pass -np.inf to avoid any
            offset If multiple types of distortion are being used, this should
            be an array corresponding to each distortion. Default: `-np.inf`.
        post_gain_db : float, optional
            This is an additional gain stage in dB after the distortion has
            been applied. Peak values of the original clean signal are always
            maintained after distortion. Default: 0.

        Returns
        -------
        distorted_signal : `Signal`
            Distorted signal.

        References
        ----------
        - [1]: https://tinyurl.com/Non-linear-distortions.

        """
        # Assert ranges
        mix_percent = np.atleast_1d(mix_percent)
        assert np.all(mix_percent <= 100), \
            'No value of mix_percent can be greater than 100'

        # Set distortions to use
        self.__select_distortions(type_of_distortion)
        n = len(self.__distortion_funcs)

        # Rearrange the other
        self.mix = mix_percent/100
        self.distortion_levels = np.atleast_1d(distortion_levels_db)
        self.offset_db = np.atleast_1d(offset_db)

        # Add extra 'clean' stage if only one distortion type is passed
        if n == 1:
            self.__distortion_funcs.append(_clean_signal)
            self.mix = np.append(self.mix, 1 - self.mix[0])
            self.distortion_levels = np.append(self.distortion_levels, 0)
            self.offset_db = np.append(self.offset_db, -np.inf)
            n += 1

        # Check that all parameters have right lengths
        assert n == len(self.mix), \
            'Length of mix_percent does not match distortions'
        assert np.isclose(np.sum(self.mix), 1), \
            'mix_percent does not sum up to 100'
        assert n == len(self.distortion_levels), \
            'Length of distortion_levels does not match distortions'
        assert n == len(self.offset_db), \
            'Length of offset_db does not match distortions'

        self.post_gain_db = post_gain_db

    def __select_distortions(self, type_of_distortion):
        """This sets `self.__distortion_funcs` which is a list containing the
        callables corresponding to the selected distortion functions.

        """
        if type(type_of_distortion) != list:
            type_of_distortion = [type_of_distortion]

        self.__distortion_funcs = []
        for dist in type_of_distortion:
            if type(dist) == str:
                dist = dist.lower()
                if dist == 'arctan':
                    self.__distortion_funcs.append(_arctan_distortion)
                elif dist == 'hard clip':
                    self.__distortion_funcs.append(_hard_clip_distortion)
                elif dist == 'soft clip':
                    self.__distortion_funcs.append(_soft_clip_distortion)
                elif dist == 'clean':
                    self.__distortion_funcs.append(_clean_signal)
                else:
                    raise ValueError(
                        f'The type of distortion {dist} is not implemented.' +
                        'Use either arctan, hard clip, soft clip or clean')
            else:
                try:
                    dist(np.zeros((100, 2)), 10, -np.inf)  # Some number
                except Exception as e:
                    raise ValueError(
                        'Distortion as callable has not been defined ' +
                        'right: ', e)
                self.__distortion_funcs.append(dist)

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Internal method which applies distortion to the passed signal.

        Parameters
        ----------
        signal : `Signal`
            Signal to apply distortion to.

        Returns
        -------
        distorted_signal : `Signal`
            Distorted signal.

        """
        td = signal.time_data
        self._save_peak_values(td)

        new_td = np.zeros_like(td)
        for i in range(len(self.__distortion_funcs)):
            new_td += self._restore_peak_values(
                self.__distortion_funcs[i](td, self.distortion_levels[i],
                                           self.offset_db[i])
                * self.mix[i])

        new_td = self._add_gain_in_db(new_td, self.post_gain_db)

        distorted_signal = signal.copy()
        distorted_signal.time_data = new_td
        return distorted_signal


class Compressor(AudioEffect):
    """This is a standard compressor that can also function as a multi-band
    compressor if the input is a MultiBandSignal.

    """
    def __init__(self, threshold_dbfs: float = -10,
                 attack_time_ms: float = 0.5, release_time_ms: float = 20,
                 ratio: float = 3, relative_to_peak_level: bool = True):
        """This effect compresses the dynamic range of a signal based on
        a threshold in dBFS.

        Parameters
        ----------
        threshold_dbfs : float, optional
            Threshold in dB above which compression is triggered. Default: -10.
        attack_time_ms : float, optional
            Attack time in milliseconds. Default: 0.5.
        release_time_ms : float, optional
            Release time in milliseconds. Default: 20.
        ratio : float, optional
            Compression ratio. When setting the compression to a value larger
            than 10, the compressor will start acting as a limiter. Values
            below 1 are not permitted since it would amplify the signal.
            Default: 3.
        relative_to_peak_level : bool, optional
            When `True`, the threshold is relative to the signal's peak level.
            Otherwise, it is an absolute value. Default: `True`.

        """
        super().__init__('Compressor')
        self.__set_parameters(threshold_dbfs, attack_time_ms, release_time_ms,
                              ratio, relative_to_peak_level)
        self.set_advanced_parameters()

    def __set_parameters(self, threshold_dbfs: float, attack_time_ms: float,
                         release_time_ms: float, ratio: float,
                         relative_to_peak_level: bool):
        """Internal method to set the parameters.

        """
        if threshold_dbfs is not None:
            if threshold_dbfs > 0:
                warn('Threshold is above 0 dBFS, this might lead to ' +
                     'unexpected results')
            self.threshold_dbfs = threshold_dbfs

        if attack_time_ms is not None:
            assert attack_time_ms >= 0, \
                'Attack time has to be 0 or above'
            self.attack_time_ms = attack_time_ms

        if release_time_ms is not None:
            assert release_time_ms >= 0, \
                'Release time has to be 0 or above'
            self.release_time_ms = release_time_ms

        if ratio is not None:
            assert ratio >= 1, \
                'Compression ratio must be above 1'
            self.ratio = ratio

        if relative_to_peak_level is not None:
            self.relative_to_peak_level = relative_to_peak_level

    def set_parameters(self, threshold_dbfs: float = None,
                       attack_time_ms: float = None,
                       release_time_ms: float = None, ratio: float = None,
                       relative_to_peak_level: bool = None):
        """This effect compresses the dynamic range of a signal based on
        a threshold in dBFS. Pass `None` to leave the previoulsy selected
        values unchanged.

        Parameters
        ----------
        threshold_dbfs : float
            Threshold in dB above which compression is triggered.
        attack_time_ms : float
            Attack time in milliseconds.
        release_time_ms : float
            Release time in milliseconds.
        ratio : float
            Compression ratio. When setting the compression to a value larger
            than 10, the compressor will start acting as a limiter. Values
            below 1 are not permitted since it would amplify the signal.
        relative_to_peak_level : bool
            When `True`, the threshold is relative to the signal's peak level.
            Otherwise, it is an absolute value.

        """
        self.__set_parameters(threshold_dbfs, attack_time_ms, release_time_ms,
                              ratio, relative_to_peak_level)
        assert self.threshold_dbfs is not None, 'None is not a valid value'
        assert self.attack_time_ms is not None, 'None is not a valid value'
        assert self.release_time_ms is not None, 'None is not a valid value'
        assert self.ratio is not None, 'None is not a valid value'
        assert self.relative_to_peak_level is not None, \
            'None is not a valid value'

    def set_advanced_parameters(self, knee_factor_db: float = 0,
                                hold_time_ms: float = 0,
                                pre_gain_db: float = 0,
                                post_gain_db: float = 0,
                                mix_percent: float = 100,
                                automatic_make_up_gain: bool = True,
                                side_chain_vector: np.ndarray = None):
        """The advanced parameters of the compressor.

        Parameters
        ----------
        knee_factor_db : float, optional
            The knee factor in dB changes the triggering of the compressor.
            A value of 0 is a hard knee while increasing it produces a smoother
            knee. Default: 0.
        hold_time_ms : float, optional
            Time to hold compression after signal level is again below
            threshold. Default: 0.
        pre_gain_db : float, optional
            Pre-compression gain in dB. Default: 0.
        post_gain_db : float, optional
            Post-compression gain in dB. Default: 0.
        mix_percent : float, optional
            Mix percent is the amount of the compressed signal that is mixed
            with the clean signal at the output. 100 means for instance that
            only compressed signal is returned. Values near 100 are advisable.
            Default: 100.
        automatic_make_up_gain : bool, optional
            When `True`, the RMS value of the signal is kept after compression.
            Default: `True`.
        side_chain_vector : `np.ndarray`, optional
            The side chain vector should be an array of boolean that triggers
            the compressor when its values are `True`. It can only be a
            1D-array. It can be retrieved from another signal by using
            `dsptoolbox.activity_detector`, for instance. If its length is
            different to that of the signal, it is padded or trimmed in the
            end to match the signal length. Attack and release time are always
            additioned to the vector. Default: `None`.

        Notes
        -----
        - The compression function with its threshold, ratio and knee can be
          plotted with the method `plot_knee()`.

        """
        assert knee_factor_db >= 0, \
            'Knee factor must be 0 or above'
        self.knee_factor_db = knee_factor_db

        assert mix_percent > 0 and mix_percent <= 100, \
            'Mix percent must be in ]0, 100]'
        self.mix = mix_percent / 100

        assert hold_time_ms >= 0, \
            'Hold time must be 0 or above'
        self.hold_time_ms = hold_time_ms

        self.pre_gain_db = pre_gain_db
        self.post_gain_db = post_gain_db
        self.automatic_make_up_gain = automatic_make_up_gain

        if side_chain_vector is not None:
            assert type(side_chain_vector) == np.ndarray, \
                'Side chain must be of type np.ndarray'
            assert side_chain_vector.ndim == 1, \
                'Side chain can only be a 1D-Array'
        self.side_chain = side_chain_vector

    def plot_knee(self):
        """Plot the compressor with the actual settings.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        gains_db = np.linspace(self.threshold_dbfs-20, 0, 2_000)
        func = _get_knee_func(self.threshold_dbfs, self.ratio,
                              self.knee_factor_db)
        gains_db_after = func(gains_db)
        gains_mixed = 10**(gains_db_after/20) * self.mix + \
            10**(gains_db/20) * (1 - self.mix)
        gains_mixed = 20*np.log10(gains_mixed)

        fig, ax = general_plot(gains_db, gains_db, log=False,
                               xlabel='Input Gain / dB',
                               ylabel='Output Gain / dB',
                               returns=True)
        ax.plot(gains_db, gains_mixed)
        ax.axvline(self.threshold_dbfs, alpha=0.5, color='xkcd:greenish',
                   linestyle='dashed')
        ax.axhline(self.threshold_dbfs, alpha=0.5, color='xkcd:greenish',
                   linestyle='dashed')
        ax.legend(['Input', 'Output', 'Threshold'])

        fig.tight_layout()
        return fig, ax

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Apply compression to a passed signal.

        """
        fs_hz = signal.sampling_rate_hz
        td = signal.time_data

        # Pre-compression gain
        td = self._add_gain_in_db(td, self.pre_gain_db)

        self._save_rms_values(td)
        self._save_peak_values(td)

        # If normalize or absolute
        if self.relative_to_peak_level:
            td /= self._peak_values

        attack_time_samples = int(self.attack_time_ms*1e-3 * fs_hz)
        release_time_samples = int(self.release_time_ms*1e-3 * fs_hz)
        hold_time_samples = int(self.hold_time_ms*1e-3 * fs_hz)

        td = _compressor(td, self.threshold_dbfs, self.ratio,
                         self.knee_factor_db, attack_time_samples,
                         hold_time_samples, release_time_samples, self.mix,
                         self.side_chain)

        # Restore original signal level
        if self.relative_to_peak_level:
            td *= self._peak_values

        # Restore RMS
        if self.automatic_make_up_gain:
            td = self._restore_rms_values(td)

        # Post-compression gain
        td = self._add_gain_in_db(td, self.pre_gain_db)

        compressed_sig = signal.copy()
        compressed_sig.time_data = td
        return compressed_sig
