from ..classes import Signal, MultiBandSignal
from .. import activity_detector
from ..standard._framed_signal_representation import (
    _get_framed_signal,
    _reconstruct_framed_signal,
)
from .._general_helpers import _get_next_power_2, _rms, _pad_trim
from ._effects import (
    _arctan_distortion,
    _clean_signal,
    _hard_clip_distortion,
    _soft_clip_distortion,
    _compressor,
    _get_knee_func,
    LFO,
    get_frequency_from_musical_rhythm,
    get_time_period_from_musical_rhythm,
)
from ..plots import general_plot
from ..tools import to_db
from ..standard.enums import SpectrumMethod, SpectrumScaling, Window
from .enums import DistortionType

from scipy.signal.windows import get_window
import numpy as np
from numpy.typing import NDArray
from warnings import warn

__all__ = [
    "get_frequency_from_musical_rhythm",
    "get_time_period_from_musical_rhythm",
]


class AudioEffect:
    """Base class for audio effects."""

    def __init__(self, description: str | None = None):
        """Base constructor for an audio effect.

        Parameters
        ----------
        description : str, optional
            A string containing a general description about the audio effect.
            Default: `None`.

        """
        self.description = description

    def apply(
        self, signal: Signal | MultiBandSignal
    ) -> Signal | MultiBandSignal:
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
        if isinstance(signal, Signal):
            return self._apply_this_effect(signal)
        elif type(signal) is MultiBandSignal:
            new_mbs = signal.copy()
            for i, b in enumerate(new_mbs.bands):
                new_mbs.bands[i] = self.apply(b)
            return new_mbs
        else:
            raise TypeError(
                "Audio effect can only be applied to Signal "
                + "or MultiBandSignal"
            )

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Abstract class method to apply the audio effect on a given
        signal.

        """
        return signal

    def _add_gain_in_db(
        self, time_data: NDArray[np.float64], gain_db: float
    ) -> NDArray[np.float64]:
        """General gain stage.

        Parameters
        ----------
        time_data : NDArray[np.float64]
            Time samples of the signal.
        gain_db : float
            Gain in dB.

        Returns
        -------
        new_time_data : NDArray[np.float64]
            Time data with new gain.

        """
        if gain_db is None:
            return time_data
        return time_data * 10 ** (gain_db / 20)

    def _save_peak_values(self, inp: NDArray[np.float64]):
        """Save the peak values of an input."""
        self._peak_values = np.max(np.abs(inp), axis=0)

    def _restore_peak_values(
        self, inp: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Restore saved peak values of a signal."""
        if not hasattr(self, "_peak_values"):
            return inp
        if len(self._peak_values) != inp.shape[1]:
            warn(
                "Number of saved peak values does not match number of "
                + "channels. Restoring is ignored"
            )
            return inp
        return inp * (self._peak_values / np.max(np.abs(inp), axis=0))

    def _save_rms_values(self, inp: NDArray[np.float64]):
        """Save the RMS values of a signal."""
        self._rms_values = _rms(inp)

    def _restore_rms_values(
        self, inp: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Restore the RMS values of a signal."""
        if not hasattr(self, "_rms_values"):
            return inp
        if len(self._rms_values) != inp.shape[1]:
            warn(
                "Number of saved RMS values does not match number of "
                + "channels. Restoring is ignored"
            )
            return inp
        return inp * (self._rms_values / _rms(inp))


class SpectralSubtractor(AudioEffect):
    """This class implements a classical spectral subtraction for denoising
    or other purposes that can act adaptively (in adaptive mode) or globally
    (static mode). It is possible to pass either a Signal or a MultiBandSignal
    to denoise in different frequency bands.

    """

    def __init__(
        self,
        adaptive_mode: bool = True,
        threshold_rms_dbfs: float = -40,
        block_length_s: float = 0.1,
        spectrum_to_subtract: NDArray[np.float64] | bool = False,
    ):
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
        spectrum_to_subtract : NDArray[np.float64] or `False`, optional
            If a spectrum is passed, it is used as the one to subtract and
            all other parameters are ignored. This should be the result of the
            squared magnitude of the FFT without any scaling in order to avoid
            scaling discrepancies. It should be only the spectrum corresponding
            to the positive frequencies (including 0). Pass `False` to ignore.
            Default: `False`.

        Methods
        -------
        - `set_parameters()`: Basic parameters used.
        - `set_advanced_parameters()`: fine-tuning parameters for both adaptive
          and static mode.
        - `apply()`: Apply effect on a given signal.

        """
        super().__init__(description="Spectral Subtraction (Denoiser)")
        self.__set_parameters(
            adaptive_mode,
            threshold_rms_dbfs,
            block_length_s,
            spectrum_to_subtract,
        )
        self.set_advanced_parameters()

    def __set_parameters(
        self,
        adaptive_mode,
        threshold_rms_dbfs,
        block_length_s,
        spectrum_to_subtract,
    ):
        """Internal method to set the parameters for the spectral
        subtraction.

        """
        if adaptive_mode is not None:
            assert (
                type(adaptive_mode) is bool
            ), "Adaptive mode must be of boolean type"
            self.adaptive_mode = adaptive_mode

        if threshold_rms_dbfs is not None:
            assert type(threshold_rms_dbfs) in (
                int,
                float,
            ), "Threshold must be of type int or float"
            if threshold_rms_dbfs >= 0:
                warn("Threshold is positive. This might be a wrong input")
            self.threshold_rms_dbfs = threshold_rms_dbfs

        if block_length_s is not None:
            assert type(block_length_s) in (
                int,
                float,
            ), "Block length should be of type int or float"
            self.block_length_s = block_length_s

        if spectrum_to_subtract is not None:
            if np.any(spectrum_to_subtract):
                assert (
                    type(spectrum_to_subtract) is np.ndarray
                ), "Spectrum to subtract must be of type numpy.ndarray"
                spectrum_to_subtract = np.squeeze(spectrum_to_subtract)
                assert spectrum_to_subtract.ndim == 1, (
                    "Spectrum to subtract could not be broadcasted to "
                    + "a 1D-Array"
                )
                if self.adaptive_mode:
                    warn(
                        "A spectrum to subtract was passed but adaptive "
                        + "mode was selected. This is unsupported. Setting "
                        + "adaptive mode to False"
                    )
                    self.adaptive_mode = False
            self.spectrum_to_subtract = spectrum_to_subtract

    def set_advanced_parameters(
        self,
        overlap_percent: int = 50,
        window_type: Window = Window.Hann,
        noise_forgetting_factor: float = 0.9,
        subtraction_factor: float = 2,
        subtraction_exponent: float = 2,
        ad_attack_time_ms: float = 0.5,
        ad_release_time_ms: float = 30,
    ):
        """This allows for setting up the advanced parameters of the spectral
        subtraction.

        Parameters
        ----------
        overlap_percent : int, optional
            Window overlap in percent. Default: 50.
        window_type : Window, optional
            Window type to use. Default: Hann.
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
                - maximum_amplification_db
                - ad_attack_time_ms
                - ad_release_time_ms

        """
        assert (0 <= overlap_percent) and (
            100 > overlap_percent
        ), "Overlap should be in [0, 100["
        self.overlap = overlap_percent / 100

        self.window_type = window_type

        assert (0 < noise_forgetting_factor) and (
            noise_forgetting_factor <= 1
        ), "Noise forgetting factor must be in ]0, 1]"
        self.noise_forgetting_factor = noise_forgetting_factor

        assert (
            subtraction_factor > 0
        ), "The subtraction factor must be positive"
        self.subtraction_factor = subtraction_factor

        assert (
            subtraction_exponent > 0
        ), "Subtraction exponent should be above zero"
        self.subtraction_exponent = subtraction_exponent

        # === Static Mode
        assert (
            ad_attack_time_ms >= 0
        ), "Attack time for activity detector must be 0 or above"
        self.ad_attack_time_ms = ad_attack_time_ms

        assert (
            ad_release_time_ms >= 0
        ), "Release time for activity detector must be 0 or above"
        self.ad_release_time_ms = ad_release_time_ms

    def set_parameters(
        self,
        adaptive_mode: bool | None = None,
        threshold_rms_dbfs: float | None = None,
        block_length_s: float | None = None,
        spectrum_to_subtract: NDArray[np.float64] = False,
    ):
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
        spectrum_to_subtract : NDArray[np.float64], optional
            If a spectrum is passed, it is used as the one to subtract and
            all other parameters are ignored. This should be the result of the
            squared magnitude of the FFT without any scaling in order to avoid
            scaling discrepancies. It should be only the spectrum corresponding
            to the positive frequencies (including 0). Pass `False` to ignore.
            Default: `False`.

        """
        self.__set_parameters(
            adaptive_mode,
            threshold_rms_dbfs,
            block_length_s,
            spectrum_to_subtract,
        )
        assert self.adaptive_mode is not None, "None is not a valid value"
        assert self.threshold_rms_dbfs is not None, "None is not a valid value"
        assert self.block_length_s is not None, "None is not a valid value"
        assert (
            self.spectrum_to_subtract is not None
        ), "None is not a valid value"

    def _compute_window(self, sampling_rate_hz):
        """Internal method to compute the window and step size in samples."""
        if not np.any(self.spectrum_to_subtract):
            self.window_length = _get_next_power_2(
                self.block_length_s * sampling_rate_hz
            )
        else:
            self.window_length = (len(self.spectrum_to_subtract) - 1) * 2
        self.window = get_window(
            self.window_type.to_scipy_format(), self.window_length
        )
        self.window = np.clip(
            get_window(self.window_type.to_scipy_format(), self.window_length),
            a_min=1e-6,
            a_max=None,
        )
        self.step_size = int(self.window_length * (1 - self.overlap))

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Internal method to trigger the effect on a given signal."""
        self._save_peak_values(signal.time_data)
        if self.adaptive_mode:
            out = self._apply_adaptive_mode(signal)
        else:
            out = self._apply_offline(signal)
        out.time_data = self._restore_peak_values(out.time_data)
        return out

    def _apply_offline(self, signal: Signal) -> Signal:
        """Spectral Subtraction in static mode (offline)."""
        # Lengths according to sampling rate
        self._compute_window(signal.sampling_rate_hz)

        # Pad zeros in beginning and end to avoid window instabilities
        td = signal.time_data
        td = _pad_trim(td, td.shape[0] + len(self.window), in_the_end=True)
        td = _pad_trim(td, td.shape[0] + len(self.window), in_the_end=False)
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
            if not np.any(self.spectrum_to_subtract):
                # Obtain noise psd
                _, noise = activity_detector(
                    signal,
                    channel=n,
                    threshold_dbfs=self.threshold_rms_dbfs,
                    attack_time_ms=self.ad_attack_time_ms,
                    release_time_ms=self.ad_release_time_ms,
                )
                noise["noise"].set_spectrum_parameters(
                    method=SpectrumMethod.WelchPeriodogram,
                    window_length_samples=len(self.window),
                    overlap_percent=self.overlap * 100,
                    window_type=self.window_type,
                    scaling=SpectrumScaling.FFTBackward,
                )
                _, noise_psd = noise["noise"].get_spectrum()
            else:
                noise_psd = self.spectrum_to_subtract.copy()
            # It is already raised to the power of 2!
            noise_psd = np.abs(noise_psd).squeeze() ** (
                self.subtraction_exponent / 2
            )
            for i in range(td_spec.shape[1]):
                temp = np.clip(
                    td_spec_power[:, i, n]
                    - self.subtraction_factor * noise_psd,
                    a_min=0,
                    a_max=None,
                )
                td_framed[:, i, n] = np.fft.irfft(
                    temp ** (1 / self.subtraction_exponent)
                    * np.exp(1j * td_spec_phase[:, i, n])
                )

        # Reconstruct signal from time frames
        new_td = _reconstruct_framed_signal(
            td_framed, self.step_size, self.window, original_length, None
        )

        # Trim back to original length
        new_td = _pad_trim(
            new_td, new_td.shape[0] - len(self.window), in_the_end=True
        )
        new_td = _pad_trim(
            new_td, new_td.shape[0] - len(self.window), in_the_end=False
        )

        denoised_signal = signal.copy()
        denoised_signal.time_data = new_td
        return denoised_signal

    def _apply_adaptive_mode(self, signal: Signal) -> Signal:
        """Spectral Subtraction in adaptive mode."""
        # Lengths and window
        self._compute_window(signal.sampling_rate_hz)

        td = signal.time_data
        td = _pad_trim(td, td.shape[0] + len(self.window), in_the_end=True)
        td = _pad_trim(td, td.shape[0] + len(self.window), in_the_end=False)
        original_length = td.shape[0]

        # Framed signal
        td = _get_framed_signal(td, len(self.window), self.step_size)

        # Get RMS values in dB for each time frame and channel
        td_rms_db = to_db(np.var(td, axis=0), False)

        # Windowed signal
        td_windowed = td * self.window[:, np.newaxis, np.newaxis]
        td_spec = np.fft.rfft(td_windowed, axis=0)

        # Phase
        td_spec_phase = np.angle(td_spec)
        td_spec = np.abs(td_spec)

        # Power spectrum
        td_spec_power = td_spec**self.subtraction_exponent

        # Iterate over frames
        for n in range(signal.number_of_channels):
            # Noise estimate
            noise_psd = np.zeros((len(self.window) // 2 + 1))

            print(f"Denoising channel {n + 1} of {signal.number_of_channels}")
            for i in range(td_spec.shape[1]):
                if td_rms_db[i, n] < self.threshold_rms_dbfs:
                    noise_psd = (
                        noise_psd * self.noise_forgetting_factor
                        + td_spec[:, i, n] * (1 - self.noise_forgetting_factor)
                    )
                temp = np.clip(
                    td_spec_power[:, i, n]
                    - self.subtraction_factor
                    * noise_psd**self.subtraction_exponent,
                    a_min=0,
                    a_max=None,
                )
                td[:, i, n] = np.fft.irfft(
                    temp ** (1 / self.subtraction_exponent)
                    * np.exp(1j * td_spec_phase[:, i, n])
                )

        # Reconstruct signal from time frames
        new_td = _reconstruct_framed_signal(
            td, self.step_size, self.window, original_length
        )

        # Trim back to original length
        new_td = _pad_trim(
            new_td, new_td.shape[0] - len(self.window), in_the_end=True
        )
        new_td = _pad_trim(
            new_td, new_td.shape[0] - len(self.window), in_the_end=False
        )

        denoised_signal = signal.copy()
        denoised_signal.time_data = new_td
        return denoised_signal


class Distortion(AudioEffect):
    """This implements a basic distortion effect that can be expanded by the
    user by passing a non-linear functions that can be applied to the waveform.
    Multiple distortions can be linearly combined.

    """

    def __init__(
        self,
        distortion_level: float = 20,
        post_gain_db: float = 0,
        type_of_distortion: DistortionType = DistortionType.Arctan,
    ):
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
        type_of_distortion : DistortionType, optional
            This sets the type of non-linear distortion to be applied. Default:
            Arctan.

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
        super().__init__("Distortion")
        self.set_advanced_parameters(
            type_of_distortion=type_of_distortion,
            distortion_levels_db=distortion_level,
            post_gain_db=post_gain_db,
        )

    def set_advanced_parameters(
        self,
        type_of_distortion: (
            DistortionType | list[DistortionType]
        ) = DistortionType.Arctan,
        distortion_levels_db: NDArray[np.float64] = 20,
        mix_percent: NDArray[np.float64] = 100,
        offset_db: NDArray[np.float64] = -np.inf,
        post_gain_db: float = 0,
    ):
        r"""This sets the parameters of the distortion. Multiple
        non-linear distortions can be combined with the clean signal and among
        each other. In that case, `distortion_levels`, `mix_percent` and
        `offset_db` must be arrays. Furthermore, the original peak levels of
        each channel in the signal are kept after applying the distortion.

        Parameters
        ----------
        type_of_distortion : DistortionType, list[DistortionType], optional
            Type of distortion to be applied. If it is a single DistortionType,
            it is applied to the signal and mixed with the clean signal
            according to the mixed parameter. If a list is passed, each entry
            must be either a string corresponding to the supported modes.
            Default: Arctan.
        distortion_levels : NDArray[np.float64], optional
            This defines how strong the distortion effect is applied. It can
            vary according to the non-linear function. Usually, a range
            between 0 and 50 should be reasonable, though any value is
            possible. If multiple types of distortion are being used, this
            should be an array corresponding to each distortion. Default: 20.
        mix_percent : NDArray[np.float64], optional
            This defines how much of each distortion is used in the final
            mix. If `type_of_distortion` is only one string or callable,
            mix_percent is its amount in the final mix with the clean signal.
            This means that 100 leads to only using the distorted signal while
            40 leads to 40% distorted, 60% clean. If multiple types of
            distortion are being used, this should be an array corresponding
            to each distortion and its sum must be 100. Default: 100.
        offset_db : NDArray[np.float64], optional
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
        assert np.all(
            mix_percent <= 100
        ), "No value of mix_percent can be greater than 100"

        # Set distortions to use
        self.__select_distortions(type_of_distortion)
        n = len(self.__distortion_funcs)

        # Rearrange the other
        self.mix = mix_percent / 100
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
        assert n == len(
            self.mix
        ), "Length of mix_percent does not match distortions"
        assert np.isclose(
            np.sum(self.mix), 1
        ), "mix_percent does not sum up to 100"
        assert n == len(
            self.distortion_levels
        ), "Length of distortion_levels does not match distortions"
        assert n == len(
            self.offset_db
        ), "Length of offset_db does not match distortions"

        self.post_gain_db = post_gain_db

    def __select_distortions(self, type_of_distortion):
        """This sets `self.__distortion_funcs` which is a list containing the
        callables corresponding to the selected distortion functions.

        """
        if type(type_of_distortion) is not list:
            type_of_distortion = [type_of_distortion]

        self.__distortion_funcs = []
        for dist in type_of_distortion:
            match dist:
                case DistortionType.Arctan:
                    self.__distortion_funcs.append(_arctan_distortion)
                case DistortionType.HardClip:
                    self.__distortion_funcs.append(_hard_clip_distortion)
                case DistortionType.SoftClip:
                    self.__distortion_funcs.append(_soft_clip_distortion)
                case DistortionType.NoDistortion:
                    self.__distortion_funcs.append(_clean_signal)
                case _:
                    raise ValueError(
                        "The type of distortion is not implemented."
                    )

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
            if self.mix[i] == 0.0:
                continue
            new_td += self._restore_peak_values(
                self.__distortion_funcs[i](
                    td, self.distortion_levels[i], self.offset_db[i]
                )
                * self.mix[i]
            )

        new_td = self._add_gain_in_db(new_td, self.post_gain_db)

        distorted_signal = signal.copy()
        distorted_signal.time_data = new_td
        return distorted_signal


class Compressor(AudioEffect):
    """This is a standard compressor that can also function as a multi-band
    compressor if the input is a MultiBandSignal.

    """

    def __init__(
        self,
        threshold_dbfs: float = -10,
        attack_time_ms: float = 0.5,
        release_time_ms: float = 20,
        ratio: float = 3,
        relative_to_peak_level: bool = True,
    ):
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
        super().__init__("Compressor")
        self.__set_parameters(
            threshold_dbfs,
            attack_time_ms,
            release_time_ms,
            ratio,
            relative_to_peak_level,
        )
        self.set_advanced_parameters()

    def __set_parameters(
        self,
        threshold_dbfs: float,
        attack_time_ms: float,
        release_time_ms: float,
        ratio: float,
        relative_to_peak_level: bool,
    ):
        """Internal method to set the parameters."""
        if threshold_dbfs is not None:
            if threshold_dbfs > 0:
                warn(
                    "Threshold is above 0 dBFS, this might lead to "
                    + "unexpected results"
                )
            self.threshold_dbfs = threshold_dbfs

        if attack_time_ms is not None:
            assert attack_time_ms >= 0, "Attack time has to be 0 or above"
            self.attack_time_ms = attack_time_ms

        if release_time_ms is not None:
            assert release_time_ms >= 0, "Release time has to be 0 or above"
            self.release_time_ms = release_time_ms

        if ratio is not None:
            assert ratio >= 1, "Compression ratio must be above 1"
            self.ratio = ratio

        if relative_to_peak_level is not None:
            self.relative_to_peak_level = relative_to_peak_level

    def set_parameters(
        self,
        threshold_dbfs: float | None = None,
        attack_time_ms: float | None = None,
        release_time_ms: float | None = None,
        ratio: float | None = None,
        relative_to_peak_level: bool | None = None,
    ):
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
        self.__set_parameters(
            threshold_dbfs,
            attack_time_ms,
            release_time_ms,
            ratio,
            relative_to_peak_level,
        )
        assert self.threshold_dbfs is not None, "None is not a valid value"
        assert self.attack_time_ms is not None, "None is not a valid value"
        assert self.release_time_ms is not None, "None is not a valid value"
        assert self.ratio is not None, "None is not a valid value"
        assert (
            self.relative_to_peak_level is not None
        ), "None is not a valid value"

    def set_advanced_parameters(
        self,
        knee_factor_db: float = 0,
        pre_gain_db: float = 0,
        post_gain_db: float = 0,
        mix_percent: float = 100,
        automatic_make_up_gain: bool = True,
        downward_compression: bool = True,
    ):
        """The advanced parameters of the compressor.

        Parameters
        ----------
        knee_factor_db : float, optional
            The knee factor in dB changes the triggering of the compressor.
            A value of 0 is a hard knee while increasing it produces a smoother
            knee. Default: 0.
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
        downward_compression : bool, optional
            When `True`, the compressor acts as a downward compressor where
            signal above the threshold level gets attenuated. If `False`,
            it acts as an upward compressor (expander) where the signal below
            the threshold gets amplified. Default: `True`.

        Notes
        -----
        - The compression function with its threshold, ratio and knee can be
          plotted with the method `show_compression()`.

        """
        assert knee_factor_db >= 0, "Knee factor must be 0 or above"
        self.knee_factor_db = knee_factor_db

        assert (
            mix_percent > 0 and mix_percent <= 100
        ), "Mix percent must be in ]0, 100]"
        self.mix = mix_percent / 100

        self.pre_gain_db = pre_gain_db
        self.post_gain_db = post_gain_db
        self.automatic_make_up_gain = automatic_make_up_gain

        self.downward_compression = downward_compression

    def show_compression(self):
        """Plot the compressor with the actual settings.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        gains_db = np.linspace(self.threshold_dbfs - 20, 0, 2_000)
        func = _get_knee_func(
            self.threshold_dbfs,
            self.ratio,
            self.knee_factor_db,
            self.downward_compression,
        )
        gains_db_after = func(gains_db)
        gains_mixed = 10 ** (gains_db_after / 20) * self.mix + 10 ** (
            gains_db / 20
        ) * (1 - self.mix)
        gains_mixed = 20 * np.log10(gains_mixed)

        fig, ax = general_plot(
            gains_db,
            gains_db,
            log=False,
            xlabel="Input Gain / dB",
            ylabel="Output Gain / dB",
        )
        ax.plot(gains_db, gains_mixed)
        ax.axvline(
            self.threshold_dbfs,
            alpha=0.5,
            color="xkcd:greenish",
            linestyle="dashed",
        )
        ax.axhline(
            self.threshold_dbfs,
            alpha=0.5,
            color="xkcd:greenish",
            linestyle="dashed",
        )
        ax.legend(["Input", "Output", "Threshold"])

        fig.tight_layout()
        return fig, ax

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Apply compression to a passed signal."""
        fs_hz = signal.sampling_rate_hz
        td = signal.time_data

        # Pre-compression gain
        td = self._add_gain_in_db(td, self.pre_gain_db)

        self._save_rms_values(td)
        self._save_peak_values(td)

        # If normalize or absolute
        if self.relative_to_peak_level:
            td /= self._peak_values

        attack_time_samples = int(self.attack_time_ms * 1e-3 * fs_hz)
        release_time_samples = int(self.release_time_ms * 1e-3 * fs_hz)

        td = _compressor(
            td,
            self.threshold_dbfs,
            self.ratio,
            self.knee_factor_db,
            attack_time_samples,
            release_time_samples,
            self.mix,
            self.downward_compression,
        )

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


class Tremolo(AudioEffect):
    """Tremolo effect that varies the amplitude of a signal according to a
    low-frequency oscillator or another modulation signal.

    """

    def __init__(
        self,
        depth: float = 0.5,
        modulator: LFO | NDArray[np.float64] | None = None,
    ):
        """Constructor for a tremolo effect.

        Parameters
        ----------
        depth : float, optional
            Depth of the amplitude variation. This must be a positive value.
            Default: 0.5.
        modulator : `LFO` or NDArray[np.float64]
            This is the modulator signal that modifies the amplitude of the
            carrier signal. It can either be a LFO or a numpy array. If the
            length of the numpy array is different to that of the carrier
            signal, it is zero-padded or trimmed in the end to match the
            length. Passing `None` in the constructor generates a harmonic
            LFO with frequency 1 Hz. Default: `None`.

        """
        super().__init__("Modulation effect: Tremolo")
        if modulator is None:
            modulator = LFO(1, "harmonic")
        self.__set_parameters(depth, modulator)

    def __set_parameters(
        self, depth: float, modulator: LFO | NDArray[np.float64]
    ):
        """Internal method to change parameters."""
        if modulator is not None:
            assert type(modulator) in (
                LFO,
                NDArray[np.float64],
            ), "Unsupported modulator type. Use LFO or numpy.ndarray"
            if type(modulator) is NDArray[np.float64]:
                modulator = modulator.squeeze()
                assert (
                    modulator.ndim == 1
                ), "Modulator signal can have only one channel"
            self.modulator = modulator

        if depth is not None:
            if type(self.modulator) is LFO:
                assert depth > 0 and depth <= 1, "Depth must be in ]0, 1]"
            self.depth = depth

    def set_parameters(
        self,
        depth: float | None = None,
        modulator: LFO | NDArray[np.float64] | None = None,
    ):
        """Set the parameters for the tremolo effect. Passing `None` in this
        function leaves them unchanged.

        Parameters
        ----------
        depth : float, optional
            Depth of the amplitude variation. This must be a positive value.
            Default: `None`.
        modulator : `LFO` or NDArray[np.float64], optional
            This is the modulator signal that modifies the amplitude of the
            carrier signal. It can either be a LFO or a numpy array. If the
            length of the numpy array is different to that of the carrier
            signal, it is zero-padded or trimmed in the end to match the
            length. Default: `None`.

        """
        self.__set_parameters(depth, modulator)
        assert self.depth is not None
        assert self.modulator is not None

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Apply tremolo effect."""
        if type(self.modulator) is LFO:
            modulation_signal = self.modulator.get_waveform(
                signal.sampling_rate_hz, len(signal)
            )
        else:
            modulation_signal = _pad_trim(self.modulator.copy(), len(signal))
        modulation_signal = np.abs(modulation_signal * self.depth + 1)

        modulated_signal = signal.copy()
        modulated_signal.time_data *= modulation_signal[..., None]
        return modulated_signal


class Chorus(AudioEffect):
    """Basic chorus effect."""

    def __init__(
        self,
        depths_ms: float | NDArray[np.float64] = 5,
        base_delays_ms: float | NDArray[np.float64] = 15,
        modulators: LFO | list | tuple | NDArray[np.float64] | None = None,
        mix_percent: float = 100,
    ):
        """Constructor for a chorus effect. Multiple voices with modulated
        delays are generated. The number of voices is inferred by the length
        of largest parameter.

        Parameters
        ----------
        depths_ms : float or array-like, optional
            This represents the amplitude of the delay variation in ms
            around the base delay. The bigger, the more dramatic the effect.
            Each voice can have a different depth. If a single value
            is passed, it is used for all voices. Default: 5.
        base_delays_ms : NDArray[np.float64], optional
            Base delays for each voice. By default, 15 ms are used for all
            voices but different values can be passed per voice.
            Default: 15.
        modulators : `LFO` or list or tuple or NDArray[np.float64], optional
            This is the modulators signal that modifies the delay of the
            carrier signal. It can either be an LFO, a list or tuple of LFOs or
            a numpy array with delay values in milliseconds. If the length of
            the numpy array is different to that of the carrier signal, it is
            zero-padded or trimmed in the end to match the length.
            Passing `None` in the constructor generates a harmonic LFO with
            frequency 2 Hz which uses a random phase for each voice to be
            delayed with. Frequency values between 0.1 Hz and 4 Hz are
            recommended. Default: `None`.
        mix_percent : float, optional
            Amount of signal (in percent) with effect in the final mix.
            Default: 100.

        Notes
        -----
        - The effect is equally applied to all channels.
        - The duration of the signal is always maintained.
        - Signal's peak values are always kept.
        - Setting a low base delay and low depth results in a flanger sound.

        """
        super().__init__("Modulation effect: Chorus/Flanger")
        if modulators is None:
            modulators = LFO(2, "harmonic", random_phase=True)
        self.__set_parameters(
            depths_ms, base_delays_ms, modulators, mix_percent
        )

    def __set_parameters(
        self,
        depths_ms: float | NDArray[np.float64],
        base_delays_ms: float | NDArray[np.float64],
        modulators: LFO | list | tuple | NDArray[np.float64],
        mix_percent: float,
    ):
        """Internal method to change parameters."""
        # Check lengths
        nv_base = nv_depths = nv_mod = 0

        if base_delays_ms is not None:
            base_delays_ms = np.atleast_1d(base_delays_ms)
            nv_base = len(base_delays_ms)
        else:
            nv_base = len(self.base_delays_ms)

        if depths_ms is not None:
            depths_ms = np.atleast_1d(depths_ms)
            nv_depths = len(depths_ms)
        else:
            nv_depths = len(self.depths_ms)

        if modulators is not None:
            if type(modulators) in (list, tuple):
                nv_mod = len(modulators)
            elif type(modulators) is NDArray[np.float64]:
                modulators = np.atleast_2d(modulators)
                nv_mod = modulators.shape[1]
            else:
                nv_mod = 1
        else:
            nv_mod = len(self.modulators)

        # Extract number of voices
        self.number_of_voices = max(nv_base, nv_depths, nv_mod)

        # Asserts for base delays
        if base_delays_ms is not None:
            assert np.all(base_delays_ms > 0), "Base delays must be above 0"
            assert len(base_delays_ms) in (
                1,
                self.number_of_voices,
            ), "Base delays can only be length 1 or number of voices"
            self.base_delays_ms = base_delays_ms
            if len(self.base_delays_ms) == 1:
                self.base_delays_ms = np.repeat(
                    self.base_delays_ms, self.number_of_voices
                )

        if modulators is not None:
            assert type(modulators) in (
                LFO,
                list,
                tuple,
                NDArray[np.float64],
            ), "Unsupported modulators type. Use LFO or numpy.ndarray"
            if type(modulators) is NDArray[np.float64]:
                modulators = np.atleast_2d(modulators)
                modulators.shape[1] == self.number_of_voices, (
                    "The modulators signal must "
                    + "have the same number of channels as there are "
                    + f"voices {self.number_of_voices}"
                )
                self.modulators = modulators
            elif type(modulators) is LFO:
                self.modulators = [modulators] * self.number_of_voices
            else:
                assert len(modulators) in (1, self.number_of_voices), (
                    "The number of modulators signals does not match the "
                    + f"number of voices {self.number_of_voices}"
                )
                assert all(
                    [type(i) is LFO for i in modulators]
                ), "All modulators signals have to be of type LFO"
                self.modulators = modulators
                if len(self.modulators) == 1:
                    self.modulators = [
                        self.modulators[0]
                    ] * self.number_of_voices

        if depths_ms is not None:
            if type(self.modulators) is LFO:
                assert depths_ms >= 0, "Depth must be above 0"
            self.depths_ms = np.atleast_1d(depths_ms)
            assert len(self.depths_ms) in (1, self.number_of_voices), (
                "Depth must be of length 1 or number of "
                + f"voices {self.number_of_voices}"
            )
            if len(self.depths_ms) == 1:
                self.depths_ms = np.repeat(
                    self.depths_ms, self.number_of_voices
                )

        if mix_percent is not None:
            mix_percent /= 100
            assert (
                mix_percent <= 1 and mix_percent > 0
            ), "Mix percent must be below 100 and above 0"
            self.mix = mix_percent

    def set_parameters(
        self,
        depths_ms: float | NDArray[np.float64] | None = None,
        base_delays_ms: float | NDArray[np.float64] | None = None,
        modulators: LFO | list | tuple | NDArray[np.float64] | None = None,
        mix_percent: float | None = None,
    ):
        """Sets the advanced parameters for the chorus effect. By passing
        multiple base delays, depths and LFOs, the effect can be fine-tuned.
        The number of voices is always extracted from the maximal length of
        the arrays. Pass `None` to leave a parameter unchanged.

        Parameters
        ----------
        depths_ms : float, optional
            Depth of the delay variation in ms. This must be a positive value.
            Default: `None`.
        modulators : LFO or list or tuple or NDArray[np.float64], optional
            This defines the modulators signal. It can be a single LFO object
            or a list containing an LFO for each voice. Alternatively, a
            numpy.ndarray with shape (time samples, voice) can be passed. If
            the length in the time axis does not match, it is zero-padded or
            trimmed in the end. Default: `None`.
        number_of_voices : int, optional
            Number of voices to use in the chorus effect. Default: `None`.

        """
        self.__set_parameters(
            depths_ms, base_delays_ms, modulators, mix_percent
        )
        assert self.depths_ms is not None
        assert self.modulators is not None
        assert self.number_of_voices is not None
        assert self.base_delays_ms is not None, "Base delay cannot be None"

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Apply chorus effect."""
        fs = signal.sampling_rate_hz
        le = len(signal)

        # Get valid modulation signals
        if type(self.modulators) is not NDArray[np.float64]:
            modulation = np.zeros((le, self.number_of_voices))
            for ind, m in enumerate(self.modulators):
                modulation[:, ind] = (
                    m.get_waveform(fs, le) * self.depths_ms[ind]
                    + self.base_delays_ms[ind]
                )
        else:
            modulation = _pad_trim(self.modulators.copy(), len(signal))

        # Delays in samples
        modulation = np.round(modulation * 1e-3 * fs).astype(int)
        max_delay_samples = np.abs(modulation).max()

        # Original time data
        td = _pad_trim(signal.time_data, le + max_delay_samples)
        self._save_peak_values(td)
        new_td = np.zeros_like(td)

        # Add modulated voices. Could be improved...
        for ind in np.arange(td.shape[0] - max_delay_samples):
            new_td[ind, :] = td[ind, :]
            for v in range(self.number_of_voices):
                new_td[ind, :] += td[ind + modulation[ind, v], :]

        # Mix with clean signal
        new_td = new_td * self.mix + td * (1 - self.mix)

        new_td = self._restore_peak_values(_pad_trim(new_td, le))

        modulated_signal = signal.copy()
        modulated_signal.time_data = new_td
        return modulated_signal


class DigitalDelay(AudioEffect):
    """This applies a basic digital delay to a signal."""

    def __init__(self, delay_time_ms: float = 300, feedback: float = 0.1):
        """Constructor for a digital delay effect.

        Parameters
        ----------
        delay_time_ms : float, optional
            Delay time in milliseconds.
        feedback : float, optional
            This controls the amount of repetitions to be generated.
            The bigger the feedback, the more extreme the effect. It is
            constrained to the range [0, 1[. Default: 0.1.

        Notes
        -----
        - Peak levels of each channel are always kept after applying the
          effect.
        - The resulting signal is always longer than the input.

        """
        super().__init__("Digital Delay")
        self.__set_parameters(delay_time_ms, feedback)
        self.set_advanced_parameters()

    def __set_parameters(self, delay_time_ms: float, feedback: int):
        """Internal method to change parameters."""
        assert delay_time_ms > 0, "Delay time must be larger than 0"
        self.delay_ms = delay_time_ms

        assert feedback > 0, "Feedback must be larger than one"
        self.feedback = feedback

    def set_parameters(
        self, delay_time_ms: float | None = None, feedback: float | None = None
    ):
        """Set the parameters for the tremolo effect. Passing `None` in this
        function leaves them unchanged.

        Parameters
        ----------
        delay_time_ms : float, optional
            Delay time in milliseconds.
        feedback : float, optional
            This controls the amount of repetitions to be generated.
            The bigger the feedback, the more extreme the effect. It is
            constrained to the range [0, 1[. Default: 0.1.

        """
        self.__set_parameters(delay_time_ms, feedback)
        assert self.delay_ms is not None
        assert self.feedback is not None

    def set_advanced_parameters(self, saturation: str | None = None):
        """This function sets the advanced parameters for the delay effect.

        Parameters
        ----------
        saturation : str, optional
            If `None`, a linear digital delay line is applied. If `'arctan'`,
            some arctan saturation is added to the delayed signal. Pass
            a callable if a custom saturation should be applied. It must
            take in 1 float and return 1 float in order to be valid.
            Default: `None`.

        """
        if saturation is None:
            saturation = "digital"
        saturation = saturation.lower()
        if saturation == "digital":

            def func(x):
                return x

        elif saturation == "arctan":

            def func(x):
                return 0.5 * np.arctan(2 * x)

        else:
            assert (
                type(saturation(1.0)) == float
            ), "Saturation function might not be valid"

            def func(x):
                return saturation(x)

        self.saturation_func = func

    def plot_delay(self):
        """Plots the delay decay with the selected parameters.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        fs = 2_000
        delay_samples = np.round(self.delay_ms * 1e-3 * fs).astype(int)

        imp = np.zeros(delay_samples * 10)
        imp[0] = 1

        for i in np.arange(delay_samples, len(imp)):
            imp[i] = imp[i] + self.feedback * self.saturation_func(
                imp[i - delay_samples]
            )

        imp = to_db(imp, True)

        x = np.arange(len(imp)) / fs * 1e3
        fig, ax = general_plot(
            x,
            imp[..., None],
            log=False,
            xlabel="Time / ms",
            ylabel="Amplitude [dB]",
        )
        ax.set_ylim([-100, 1])
        ax.set_title("Delay – Repetitions decay")
        fig.tight_layout()
        return fig, ax

    def _apply_this_effect(self, signal: Signal) -> Signal:
        """Apply delay effect."""
        delay_samples = np.round(
            self.delay_ms * 1e-3 * signal.sampling_rate_hz
        ).astype(int)

        td = signal.time_data
        self._save_peak_values(td)

        # Pad signal in the end so that some repetitions are added
        padding = int(delay_samples * (1 + self.feedback * 15))
        td = np.append(td, np.zeros((padding, td.shape[1])), axis=0)

        for i in np.arange(delay_samples, len(td)):
            td[i, :] = td[i, :] + self.feedback * self.saturation_func(
                td[i - delay_samples, :]
            )

        td = self._restore_peak_values(td)

        delayed_signal = signal.copy()
        delayed_signal.time_data = td
        return delayed_signal
