import numpy as np

from ..standard import append_signals, rms
from .signal import Signal
from .multibandsignal import MultiBandSignal


class CalibrationData:
    """This is a class that takes in a calibration recording and can be used
    to calibrate other signals.

    """

    def __init__(
        self,
        calibration_data,
        calibration_spl_db: float = 94,
        high_snr: bool = True,
    ):
        """Load a calibration sound file. It is expected that it contains
        a recorded harmonic tone of 1 kHz with the given dB(SPL) value, common
        values are 94 dB or 114 dB SPL according to [1]. This class can later
        be used to calibrate a signal.

        Parameters
        ----------
        calibration_data : str, tuple or `Signal`
            Calibration recording. It can be a path (str), a tuple with entries
            (time_data, sampling_rate) or a `Signal` object.
        calibration_spl_db : float, optional
            dB(SPL) of calibration data. Typical values are 94 dB (1 Pa) or 114
            dB (10 Pa). Default: 94.
        high_snr : bool, optional
            If the calibration is expected to have a high Signal-to-noise
            ratio, RMS value is computed directly through the time signal. This
            is done when set to `True`. If not, it might be more precise to
            take the spectrum of the signal and evaluate it at 1 kHz.
            This is recommended for systems where the SNR drops below 10 dB.
            Default: `True`.

        References
        ----------
        - [1]: DIN EN IEC 60942:2018-07.

        """
        if isinstance(calibration_data, str):
            calibration_data = Signal(calibration_data, None, None)
        elif isinstance(calibration_data, tuple):
            assert len(calibration_data) == 2, "Tuple must have length 2"
            calibration_data = Signal(
                None, calibration_data[0], calibration_data[1]
            )
        elif isinstance(calibration_data, Signal):
            pass
        else:
            raise TypeError(
                f"{type(calibration_data)} is not a valid type. Use "
                "either str, tuple or Signal"
            )
        self.calibration_signal = calibration_data
        self.calibration_spl_db = calibration_spl_db
        self.high_snr = high_snr
        # State tracker
        self.__update = True

    def add_calibration_channel(self, new_channel):
        """Adds a new calibration channel to the calibration signal.

        Parameters
        ----------
        new_channel : str, tuple or `Signal`
            New calibration channel. It can be either a path (str), a tuple
            with entries (time_data, sampling_rate) or a `Signal` object.
            If the lengths are different, padding or trimming is done
            at the end of the new channel. This is supported, but not
            recommended since zero-padding might distort the real RMS value
            of the recorded signal.

        """
        if isinstance(new_channel, str):
            new_channel = Signal(new_channel, None, None)
        elif isinstance(new_channel, tuple):
            assert len(new_channel) == 2, "Tuple must have length 2"
            new_channel = Signal(None, new_channel[0], new_channel[1])
        elif isinstance(new_channel, Signal):
            pass
        else:
            raise TypeError(
                f"{type(new_channel)} is not a valid type. Use "
                "either str, tuple or Signal"
            )
        self.calibration_signal = append_signals(
            [self.calibration_signal, new_channel]
        )
        self.__update = True

    def _compute_calibration_factors(self):
        """Computes the calibration factors for each channel."""
        if self.__update:
            if self.high_snr:
                rms_channels = rms(self.calibration_signal, in_dbfs=False)
            else:
                rms_channels = self._get_rms_from_spectrum()
            p0 = 20e-6
            p_analytical = 10 ** (self.calibration_spl_db / 20) * p0
            self.calibration_factors = p_analytical / rms_channels
            self.__update = False

    def _get_rms_from_spectrum(self):
        self.calibration_signal.set_spectrum_parameters(
            method="standard", scaling="amplitude spectrum"
        )
        f, sp = self.calibration_signal.get_spectrum()
        ind1k = np.argmin(np.abs(f - 1e3))
        return np.abs(sp[ind1k, :])

    def calibrate_signal(
        self, signal: Signal | MultiBandSignal, force_update: bool = False
    ) -> Signal | MultiBandSignal:
        """Calibrates the time data of a signal and returns it as a new object.
        It can also be a `MultiBandSignal`. If the calibration data only
        contains one channel, this factor is used for all channels of the
        signal. Otherwise, the number of channels must coincide.

        Parameters
        ----------
        signal : `Signal` or `MultiBandSignal`
            Signal to calibrate.
        force_update : bool, optional
            When `True`, an update of the calibration data is forced. This
            might be necessary if the calibration signal or the parameters
            of the object have been manually changed. Default: `False`.

        Returns
        -------
        calibrated_signal : `Signal` or `MultiBandSignal`
            Calibrated signal with time data in Pascal. These values
            are no longer constrained to the range [-1, 1].

        """
        if force_update:
            self.__update = True
        self._compute_calibration_factors()
        if len(self.calibration_factors) > 1:
            assert signal.number_of_channels == len(
                self.calibration_factors
            ), "Number of channels does not match"
            calibration_factors = self.calibration_factors
        else:
            calibration_factors = (
                np.ones(signal.number_of_channels) * self.calibration_factors
            )

        if isinstance(signal, Signal):
            calibrated_signal = signal.copy()
            calibrated_signal.constrain_amplitude = False
            calibrated_signal.time_data *= calibration_factors
            calibrated_signal.calibrated_signal = True
        elif isinstance(signal, MultiBandSignal):
            calibrated_signal = signal.copy()
            for b in calibrated_signal:
                b.constrain_amplitude = False
                b.time_data *= calibration_factors
                b.calibrated_signal = True
        else:
            raise TypeError(
                "signal has not a valid type. Use Signal or "
                + "MultiBandSignal"
            )
        return calibrated_signal
