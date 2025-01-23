"""
This contains signal generators that might be useful for measurements.
See measure.py for routines where the signals that are created here can be
used
"""

import numpy as np

from ..helpers.frequency_conversion import _frequency_weightning
from ..helpers.gain_and_level import _fade, _normalize
from ..helpers.other import _pad_trim
from ..classes.signal import Signal
from ..classes.impulse_response import ImpulseResponse
from ..classes.filter_helpers import _impulse
from ._generators import _sync_log_chirp
from ..standard.enums import FadeType
from .enums import NoiseType, ChirpType, WaveForm


def noise(
    length_seconds: float,
    sampling_rate_hz: int,
    type_of_noise: NoiseType | float = NoiseType.White,
    peak_level_dbfs: float = -10.0,
    number_of_channels: int = 1,
    fade: FadeType = FadeType.Logarithmic,
    padding_end_seconds: float = 0.0,
) -> Signal:
    """Creates a noise signal.

    Parameters
    ----------
    length_seconds : float
        Length of the generated signal in seconds.
    sampling_rate_hz : int
        Sampling rate in Hz.
    type_of_noise : NoiseType, float, optional
        Type of noise to generate. If a float is passed, it corresponds to
        `beta`, where `beta` is used to define the slope of the power spectral
        density (psd) with `psd * frequency**(-beta)`. See notes for details.
        Default: White.
    peak_level_dbfs : float, optional
        Peak level of the signal in dBFS. Default: -10.
    number_of_channels : int, optional
        Number of channels (with different noise signals) to be created.
        Default: 1.
    fade : FadeType, optional
        Type of fade done on the generated signal. By default, 10% of signal
        length (without the padding in the end) is faded at the beginning and
        end. Default: Logarithmic.
    padding_end_seconds : float, optional
        Padding at the end of signal. Default: 0.

    Returns
    -------
    noise_sig : `Signal`
        Noise Signal object.

    References
    ----------
    - https://en.wikipedia.org/wiki/Colors_of_noise

    Notes
    -----
    - Using the `beta` parameter to define noise is the most flexible approach.
      For instance, `beta=1.0` will deliver pink noise, `beta=-1.0` corresponds
      to blue.

    """
    assert sampling_rate_hz is not None, "Sampling rate can not be None"
    if type(type_of_noise) is not NoiseType:
        assert (
            type(type_of_noise) is float
        ), "type_of_noise must be either NoiseType or float"

    assert length_seconds > 0, "Length has to be positive"
    assert peak_level_dbfs <= 0, "Peak level cannot surpass 0 dBFS"
    assert number_of_channels >= 1, "At least one channel should be generated"

    l_samples = int(length_seconds * sampling_rate_hz + 0.5)
    f = np.fft.rfftfreq(l_samples, 1 / sampling_rate_hz)

    if padding_end_seconds != 0:
        assert padding_end_seconds > 0, "Padding has to be a positive time"
        p_samples = int(padding_end_seconds * sampling_rate_hz + 0.5)
    else:
        p_samples = 0
    time_data = np.zeros((l_samples + p_samples, number_of_channels))

    mag = np.random.normal(2, 0.0025, (len(f), number_of_channels))

    # Set to 15 Hz to cover whole audible spectrum but without
    # numerical instabilities because of large values in lower
    # frequencies
    id_low = np.argmin(np.abs(f - 15))
    mag[0] = 0
    if type_of_noise != NoiseType.White or type_of_noise != 0.0:
        mag[:id_low] *= 1e-20

    ph = np.random.uniform(-np.pi, np.pi, (len(f), number_of_channels))

    # Correct DC and Nyquist
    ph[0, :] = 0
    if l_samples % 2 == 0:
        ph[-1, :] = 0

    if type_of_noise == NoiseType.Pink:
        mag[id_low:, :] /= (f[id_low:] ** 0.5)[..., None]
    elif type_of_noise == NoiseType.Red:
        mag[id_low:, :] /= f[id_low:][..., None]
    elif type_of_noise == NoiseType.Blue:
        mag[id_low:, :] *= (f[id_low:] ** 0.5)[..., None]
    elif type_of_noise == NoiseType.Violet:
        mag[id_low:, :] *= f[id_low:][..., None]
    elif type_of_noise == NoiseType.Grey:
        w = _frequency_weightning(f, "a", db_output=False)
        mag[id_low:, :] /= w[id_low:][..., None]
    elif type(type_of_noise) is float:
        mag[id_low:, :] *= (f[id_low:] ** (-type_of_noise * 0.5))[..., None]

    vec = np.fft.irfft(mag * np.exp(1j * ph), n=l_samples, axis=0)
    vec = _normalize(
        vec, dbfs=peak_level_dbfs, peak_normalization=True, per_channel=True
    )
    if fade is not None:
        fade_length = 0.05 * length_seconds
        vec = _fade(
            s=vec,
            length_seconds=fade_length,
            mode=fade,
            sampling_rate_hz=sampling_rate_hz,
            at_start=True,
        )
        vec = _fade(
            s=vec,
            length_seconds=fade_length,
            mode=fade,
            sampling_rate_hz=sampling_rate_hz,
            at_start=False,
        )
    time_data[:l_samples, :] = vec

    noise_sig = Signal(None, time_data, sampling_rate_hz)
    return noise_sig


def chirp(
    sampling_rate_hz: int,
    type_of_chirp: ChirpType = ChirpType.Logarithmic,
    range_hz=None,
    length_seconds: float = 1.0,
    peak_level_dbfs: float = -10.0,
    number_of_channels: int = 1,
    fade: FadeType = FadeType.Logarithmic,
    phase_offset: float = 0.0,
    padding_end_seconds: float = 0.0,
) -> Signal | tuple[Signal, float]:
    """Creates a sine-sweep signal.

    Parameters
    ----------
    sampling_rate_hz : int
        Sampling rate in Hz. Default: `None`.
    type_of_chirp : ChirpType, optional
        Type of chirp. Default: Logarithmic.
    range_hz : array-like with length 2
        Define range of chirp in Hz. When `None`, all frequencies between
        15 Hz and nyquist are taken. Default: `None`.
    length_seconds : float, optional
        Length of the generated signal in seconds. Default: 1.
    peak_level_dbfs : float, optional
        Peak level of the signal in dBFS. Default: -10.
    number_of_channels : int, optional
        Number of channels (with the same chirp) to be created. Default: 1.
    fade : FadeType, optional
        Type of fade done on the generated signal. By default, 10% of signal
        length (without the padding in the end) is faded at the beginning and
        end. Default: Logarithmic.
    phase_offset : float, optional
        This is an offset in radians for the phase of the sine. Default: 0.
    padding_end_seconds : float, optional
        Padding at the end of signal. Default: 0.

    Returns
    -------
    chirp_sig : `Signal`
        Chirp Signal object.
    chirp_duration_seconds : float
        Effective chirp duration. This is only returned if the chirp is of
        type "sync-log" because the provided length might differ slightly.

    Notes
    -----
    - The "sync-log" chirp is defined according to [2] and ensures that the
      harmonic responses have coherent phase with the linear response.

    References
    ----------
    - https://de.wikipedia.org/wiki/Chirp
    - [2]: Antonin Novak, Laurent Simon, Pierrick Lotton. Synchronized
      Swept-Sine: Theory, Application and Implementation.

    """
    if range_hz is not None:
        assert (
            len(range_hz) == 2
        ), "range_hz has to contain exactly two frequencies"
        range_hz = sorted(range_hz)
        assert (
            range_hz[0] > 0
        ), "Range has to start with positive frequencies excluding 0"
        assert range_hz[1] <= sampling_rate_hz // 2, (
            "Upper limit for frequency range cannot be bigger than the "
            + "nyquist frequency"
        )
    else:
        range_hz = [15, sampling_rate_hz // 2]
    if padding_end_seconds != 0:
        assert padding_end_seconds > 0, "Padding has to be a positive time"
        p_samples = int(padding_end_seconds * sampling_rate_hz)
    else:
        p_samples = 0
    l_samples = int(sampling_rate_hz * length_seconds + 0.5)

    if type_of_chirp != ChirpType.SyncLog:
        t = np.linspace(0, length_seconds, l_samples)

    match type_of_chirp:
        case ChirpType.Linear:
            k = (range_hz[1] - range_hz[0]) / length_seconds
            freqs = (range_hz[0] + k / 2 * t) * 2 * np.pi
            chirp_td = np.sin(freqs * t + phase_offset)
        case ChirpType.Logarithmic:
            k = np.exp(
                (np.log(range_hz[1]) - np.log(range_hz[0])) / length_seconds
            )
            chirp_td = np.sin(
                2 * np.pi * range_hz[0] / np.log(k) * (k**t - 1) + phase_offset
            )
        case ChirpType.SyncLog:
            chirp_td, T = _sync_log_chirp(
                range_hz, length_seconds, sampling_rate_hz
            )
        case _:
            raise ValueError("Unsupported chirp type")

    chirp_td = _normalize(
        chirp_td, peak_level_dbfs, peak_normalization=True, per_channel=True
    )

    if fade is not None:
        fade_length = 0.05 * length_seconds
        chirp_td = _fade(
            s=chirp_td,
            length_seconds=fade_length,
            mode=fade,
            sampling_rate_hz=sampling_rate_hz,
            at_start=True,
        )
        chirp_td = _fade(
            s=chirp_td,
            length_seconds=fade_length,
            mode=fade,
            sampling_rate_hz=sampling_rate_hz,
            at_start=False,
        )

    chirp_td = _pad_trim(chirp_td, l_samples + p_samples)

    chirp_n = chirp_td[..., None]
    if number_of_channels != 1:
        chirp_n = np.repeat(chirp_n, repeats=number_of_channels, axis=1)

    chirp_sig = Signal(None, chirp_n, sampling_rate_hz)
    return (chirp_sig, T) if type_of_chirp == ChirpType.SyncLog else chirp_sig


def dirac(
    length_samples: int,
    sampling_rate_hz: int,
    delay_samples: int = 0,
    number_of_channels: int = 1,
) -> ImpulseResponse:
    """Generates a dirac impulse (ImpulseResponse) with the specified length
    and sampling rate.

    Parameters
    ----------
    length_samples : int
        Length in samples.
    sampling_rate_hz : int
        Sampling rate to be used.
    delay_samples : int, optional
        Delay of the impulse in samples. Default: 0.
    number_of_channels : int, optional
        Number of channels to be generated with the same impulse. Default: 1.

    Returns
    -------
    imp : `ImpulseResponse`
        Signal with dirac impulse.

    """
    assert sampling_rate_hz is not None, "Sampling rate can not be None"
    assert (
        type(length_samples) is int and length_samples > 0
    ), "Only positive lengths are valid"
    assert (
        type(delay_samples) is int and delay_samples >= 0
    ), "Only positive delay is supported"
    assert (
        delay_samples < length_samples
    ), "Delay is bigger than the samples of the signal"
    assert number_of_channels > 0, "At least one channel has to be created"
    assert sampling_rate_hz > 0, "Sampling rate can only be positive"
    td = np.zeros((length_samples, number_of_channels))
    for n in range(number_of_channels):
        td[:, n] = _impulse(
            length_samples=length_samples, delay_samples=delay_samples
        )
    imp = ImpulseResponse(None, td, sampling_rate_hz)
    return imp


def oscillator(
    frequency_hz: float,
    sampling_rate_hz: int,
    length_seconds: float = 1.0,
    mode: WaveForm = WaveForm.Harmonic,
    harmonic_cutoff_hz: float | None = None,
    peak_level_dbfs: float = -10.0,
    number_of_channels: int = 1,
    uncorrelated: bool = False,
    fade: FadeType = FadeType.Logarithmic,
    padding_end_seconds: float = 0.0,
) -> Signal:
    """Creates a non-aliased, multi-channel wave tone.

    Parameters
    ----------
    frequency_hz : float, optional
        Frequency (in Hz). Default: 1000.
    length_seconds : float, optional
        Length of the generated signal in seconds. Default: 1.
    mode : WaveForm, optional
        Type of wave to generate. Default: Harmonic.
    harmonic_cutoff_hz : float, optional
        It is possible to pass a cutoff frequency for the harmonics. If `None`,
        they are computed up until before the nyquist frequency.
        Default: `None`.
    sampling_rate_hz : int
        Sampling rate in Hz. Default: `None`.
    peak_level_dbfs : float, optional
        Peak level of the signal in dBFS. Default: -10.
    number_of_channels : int, optional
        Number of channels to be created. Default: 1.
    uncorrelated : bool, optional
        When `True`, each channel gets a random phase shift so that the signals
        are not perfectly correlated. Default: `False`.
    fade : FadeType, optional
        Type of fade done on the generated signal. By default, 5% of signal
        length (without the padding in the end) is faded at the beginning and
        end. Default: Logarithmic.
    padding_end_seconds : float, optional
        Padding at the end of signal. Default: 0.

    Returns
    -------
    wave_signal : `Signal`
        Wave signal.

    """
    assert (
        frequency_hz < sampling_rate_hz // 2
    ), "Frequency must be beneath nyquist frequency"
    assert frequency_hz > 0, "Frequency must be bigger than 0"

    if padding_end_seconds != 0:
        assert padding_end_seconds > 0, "Padding has to be a positive time"
        p_samples = int(padding_end_seconds * sampling_rate_hz)
    else:
        p_samples = 0
    l_samples = int(sampling_rate_hz * length_seconds + 0.5)
    n = np.arange(l_samples)[..., None]
    n = np.repeat(n, number_of_channels, axis=-1)

    if harmonic_cutoff_hz is None:
        harmonic_cutoff_hz = sampling_rate_hz // 2
    assert (
        harmonic_cutoff_hz > 0 and harmonic_cutoff_hz <= sampling_rate_hz // 2
    ), "Cutoff frequency must be between 0 and the nyquist frequency!"

    if uncorrelated:
        phase_shift = np.random.uniform(-np.pi, np.pi, (number_of_channels))[
            None, ...
        ]
    else:
        phase_shift = np.zeros((number_of_channels))[None, ...]

    # Get waveforms
    td = np.zeros((l_samples, number_of_channels))
    k = 1
    match mode:
        case WaveForm.Harmonic:
            td = np.sin(
                2 * np.pi * frequency_hz / sampling_rate_hz * n + phase_shift
            )
        case WaveForm.Square:
            while (2 * k - 1) * frequency_hz < harmonic_cutoff_hz:
                td += np.sin(
                    np.pi
                    * 2
                    * (2 * k - 1)
                    * frequency_hz
                    / sampling_rate_hz
                    * n
                    + phase_shift
                ) / (2 * k - 1)
                k += 1
            td *= 4 / np.pi
        case WaveForm.Sawtooth:
            while k * frequency_hz < harmonic_cutoff_hz:
                td += (
                    np.sin(
                        np.pi * 2 * k * frequency_hz / sampling_rate_hz * n
                        + phase_shift
                    )
                    / k
                    * (-1) ** k
                )
                k += 1
            td *= -(2 / np.pi)
        case WaveForm.Triangle:
            while (2 * k - 1) * frequency_hz < harmonic_cutoff_hz:
                td += (
                    np.sin(
                        np.pi
                        * 2
                        * (2 * k - 1)
                        * frequency_hz
                        / sampling_rate_hz
                        * n
                        + phase_shift
                    )
                    / (2 * k - 1) ** 2
                    * (-1) ** k
                )
                k += 1
            td *= -8 / np.pi**2
        case _:
            raise ValueError("Unsupported wave form")

    td = _normalize(
        td, peak_level_dbfs, peak_normalization=True, per_channel=True
    )

    if fade is not None:
        fade_length = 0.05 * length_seconds
        td = _fade(
            s=td,
            length_seconds=fade_length,
            mode=fade,
            sampling_rate_hz=sampling_rate_hz,
            at_start=True,
        )
        td = _fade(
            s=td,
            length_seconds=fade_length,
            mode=fade,
            sampling_rate_hz=sampling_rate_hz,
            at_start=False,
        )

    td = _pad_trim(td, l_samples + p_samples)

    # Signal
    harmonic_sig = Signal(None, td, sampling_rate_hz)
    return harmonic_sig
