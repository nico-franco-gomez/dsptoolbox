"""
High-level methods for room acoustics functions
"""

import numpy as np
from scipy.signal import find_peaks, convolve
from numpy.typing import NDArray

from ..classes import Signal, MultiBandSignal, Filter
from ..filterbanks import fractional_octave_bands, linkwitz_riley_crossovers
from ._room_acoustics import (
    _reverb,
    _complex_mode_identification,
    _find_ir_start,
    _generate_rir,
    ShoeboxRoom,
    _add_reverberant_tail_noise,
    _d50_from_rir,
    _c80_from_rir,
    _ts_from_rir,
)
from .._general_helpers import _find_nearest, _normalize, _pad_trim
from ..standard_functions import pad_trim


def reverb_time(
    signal: Signal | MultiBandSignal,
    mode: str = "T20",
    ir_start: int | NDArray[np.int_] | None = None,
    automatic_trimming: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes reverberation time. Topt, T20, T30, T60 and EDT.

    Parameters
    ----------
    signal : `Signal` or `MultiBandSignal`
        Signal for which to compute reverberation times. It must be type
        `'ir'` or `'rir'`.
    mode : str, optional
        Reverberation time mode. Options are `'Topt'`, `'T20'`, `'T30'`,
        `'T60'` or `'EDT'`. Default: `'Topt'`.
    ir_start : int or array-like, NDArray[np.int_], optional
        If it is an integer, it is assumed as the start of the IR for all
        channels (and all bands). For more specific cases, pass a 1d-array
        containing the start indices for each channel or a 2d-array with
        shape (band, channel) for a `MultiBandSignal`. Default: `None`.
    automatic_trimming : bool, optional
        When set to `True`, the IR is trimmed using `trim_rir` independently
        for each channel. This can influence significantly the energy decay
        curve and, therefore, the reverberation time. Refer to the
        documentation for more details. Default: `True`.

    Returns
    -------
    reverberation_times : NDArray[np.float64]
        Reverberation times for each channel. Shape is (band, channel)
        if `MultiBandSignal` object is passed.
    correlation_coefficient : NDArray[np.float64]
        Pearson correlation coefficient to determine the accuracy of the
        reverberation time estimation. It has shape (channels) or
        (band, channels) if `MultiBandSignal` object is passed. See notes
        for more details.

    References
    ----------
    - DIN EN ISO 3382-1:2009-10, Acoustics - Measurement of the reverberation
      time of rooms with reference to other acoustical parameters.
    - Room-EQ-Wizard for Topt.

    Notes
    -----
    - A correlation coefficient of -1 means there is a perfectly linear
      relation between time and energy decay, which is an optimal estimation.
      Coefficients larger than -0.9 might mean that the estimation is not
      valid.
    - In order to compare EDT to the other measures, it must be multiplied
      by 6.

    """
    if type(signal) is Signal:
        ir_start = _check_ir_start_reverb(signal, ir_start)
        assert signal.signal_type in ("ir", "rir"), (
            f"{signal.signal_type} is not a valid signal type for "
            + "reverb_time. It should be ir or rir"
        )
        mode = mode.upper()
        valid_modes = ("TOPT", "T20", "T30", "T60", "EDT")
        assert mode in valid_modes, (
            f"{mode} is not valid. Use either one of "
            + "these: Topt, T20, T30, T60 or EDT"
        )
        reverberation_times = np.zeros((signal.number_of_channels))
        correlation_coefficients = np.zeros((signal.number_of_channels))
        for n in range(signal.number_of_channels):
            reverberation_times[n], correlation_coefficients[n] = _reverb(
                signal.time_data[:, n].copy(),
                signal.sampling_rate_hz,
                mode,
                ir_start=ir_start[n],
                return_ir_start=False,
                automatic_trimming=automatic_trimming,
            )
    elif type(signal) is MultiBandSignal:
        ir_start = _check_ir_start_reverb(signal, ir_start)
        reverberation_times = np.zeros(
            (signal.number_of_bands, signal.bands[0].number_of_channels)
        )
        correlation_coefficients = np.zeros(
            (signal.number_of_bands, signal.bands[0].number_of_channels)
        )
        for ind in range(signal.number_of_bands):
            band_ir_start = None if ir_start is None else ir_start[ind, :]
            reverberation_times[ind, :], correlation_coefficients[ind, :] = (
                reverb_time(
                    signal.bands[ind],
                    mode,
                    ir_start=band_ir_start,
                    automatic_trimming=automatic_trimming,
                )
            )
    else:
        raise TypeError(
            "Passed signal should be of type Signal or MultiBandSignal"
        )
    return reverberation_times, correlation_coefficients


def find_modes(
    signal: Signal,
    f_range_hz=[50, 200],
    dist_hz: float = 5,
    prominence_db: float | None = None,
    antiresonances: bool = False,
) -> NDArray[np.float64]:
    """Finds the room modes of a set of RIR using the peaks of the complex
    mode indicator function (CMIF).

    Parameters
    ----------
    signal : `Signal`
        Signal containing the RIR'S from which to find the modes.
    f_range_hz : array-like, optional
        Vector setting range for mode search. Default: [50, 200].
    dist_hz : float, optional
        Minimum distance (in Hz) between modes. Default: 5.
    prominence_db : float, optional
        Prominence of the peaks in dB of the CMIF in order to be classified as
        modes. Pass `None` to avoid checking prominence. Default: `None`.
    antiresonances : bool, optional
        When `True`, the spectra are inverted so that antiresonances are
        found instead of resonances. Default: `False`.

    Returns
    -------
    f_modes : NDArray[np.float64]
        Vector containing frequencies where modes have been localized.

    References
    ----------
    - http://papers.vibetech.com/Paper17.pdf

    Notes
    -----
    - This function finds the resonant modes but not the antiresonants.

    """
    assert len(f_range_hz) == 2, (
        "Range of frequencies must have a " + "minimum and a maximum value"
    )

    assert signal.signal_type in ("rir", "ir"), (
        f"{signal.signal_type} is not a valid signal type. It should "
        + "be either rir or ir"
    )
    signal.set_spectrum_parameters("standard")
    # Pad signal to have a resolution of around 1 Hz
    length = signal.sampling_rate_hz
    signal = pad_trim(signal, length)
    f, sp = signal.get_spectrum()

    # Setting up frequency range
    ids = _find_nearest(f_range_hz, f)
    f = f[ids[0] : ids[1]]
    df = f[1] - f[0]

    # Compute CMIF
    sp = sp[ids[0] : ids[1], :]
    if antiresonances:
        sp = 1 / sp
    cmif = _complex_mode_identification(sp, True).squeeze()

    # Find peaks
    dist_samp = int(np.ceil(dist_hz / df))
    dist_samp = 1 if dist_samp < 1 else dist_samp

    id_cmif, _ = find_peaks(
        10 * np.log10(cmif),
        distance=dist_samp,
        # width=dist_samp,  # Is width here a good idea?
        prominence=prominence_db,
    )
    f_modes = f[id_cmif]

    return f_modes


def convolve_rir_on_signal(
    signal: Signal,
    rir: Signal,
    keep_peak_level: bool = True,
    keep_length: bool = True,
) -> Signal:
    """Applies an RIR to a given signal. The RIR should also be a signal object
    with a single channel containing the RIR time data. Signal type should
    also be set to IR or RIR. By default, all channels are convolved with
    the RIR.

    Parameters
    ----------
    signal : Signal
        Signal to which the RIR is applied. All channels are affected.
    rir : Signal
        Single-channel Signal object containing the RIR.
    keep_peak_level : bool, optional
        When `True`, output signal is normalized to the peak level of
        the original signal. Default: `True`.
    keep_length : bool, optional
        When `True`, the original length is kept after convolution, otherwise
        the output signal is longer than the input one. Default: `True`.

    Returns
    -------
    new_sig : `Signal`
        Convolved signal with RIR.

    """
    assert rir.signal_type in (
        "rir",
        "ir",
    ), f"{rir.signal_type} is not a valid signal type. Set it to rir or ir."
    assert (
        signal.time_data.shape[0] > rir.time_data.shape[0]
    ), "The RIR is longer than the signal to convolve it with."
    assert (
        rir.number_of_channels == 1
    ), "RIR should not contain more than one channel."
    assert (
        rir.sampling_rate_hz == signal.sampling_rate_hz
    ), "The sampling rates do not match"

    if keep_length:
        total_length_samples = signal.time_data.shape[0]
    else:
        total_length_samples = (
            signal.time_data.shape[0] + rir.time_data.shape[0] - 1
        )
    new_time_data = np.zeros((total_length_samples, signal.number_of_channels))

    for n in range(signal.number_of_channels):
        if keep_peak_level:
            old_peak = 20 * np.log10(np.max(np.abs(signal.time_data[:, n])))
        new_time_data[:, n] = convolve(
            signal.time_data[:, n], rir.time_data[:, 0], mode="full"
        )[:total_length_samples]
        if keep_peak_level:
            new_time_data[:, n] = _normalize(
                new_time_data[:, n], old_peak, mode="peak"
            )

    new_sig = signal.copy()
    new_sig.time_data = new_time_data
    new_sig.signal_id += " (convolved with RIR)"
    return new_sig


def find_ir_start(
    signal: Signal, threshold_dbfs: float = -20
) -> NDArray[np.int_]:
    """This function finds the start of an IR defined as the first sample
    before a certain threshold is surpassed. For room impulse responses, -20
    dB relative to peak level is recommended according to [1].

    Parameters
    ----------
    signal : `Signal`
        IR signal.
    threshold_dbfs : float, optional
        Threshold that should be passed (in dBFS). Default: -20.

    Returns
    -------
    start_index : NDArray[np.int_]
        Index of IR start for each channel.

    References
    ----------
    - [1]: ISO 3382-1:2009-10, Acoustics - Measurement of the reverberation
      time of rooms with reference to other acoustical parameters. pp. 22.

    """
    assert threshold_dbfs <= 0, "Threshold must be negative"
    start_index = np.empty(signal.number_of_channels, dtype=int)
    for n in range(signal.number_of_channels):
        start_index[n] = _find_ir_start(signal.time_data[:, n], threshold_dbfs)
    return start_index.astype(np.int_)


def generate_synthetic_rir(
    room: ShoeboxRoom,
    source_position,
    receiver_position,
    sampling_rate_hz: int,
    total_length_seconds: float = 0.5,
    add_noise_reverberant_tail: bool = False,
    apply_bandpass: bool = False,
    use_detailed_absorption: bool = False,
    max_order: int | None = None,
) -> Signal:
    """This function returns a synthetized RIR in a shoebox-room using the
    image source model. The implementation is based on Brinkmann,
    et al. See References for limitations and advantages of this method.

    Parameters
    ----------
    room : `ShoeboxRoom`
        Room object with the information about the room properties.
    source_position : array-like
        Vector with length 3 corresponding to the source's position (x, y, z)
        in meters.
    receiver_position : array-like
        Vector with length 3 corresponding to the receiver's position (x, y, z)
        in meters.
    sampling_rate_hz : int
        Sampling rate of the generated impulse (in Hz). Default: `None`.
    total_length_seconds : float, optional
        Total length of the output RIR in seconds. Default: 0.5.
    add_noise_reverberant_tail : bool, optional
        When `True`, decaying noise is added to the IR in order to model
        the late reflections of the room. Default: `True`.
    apply_bandpass : bool, optional
        When `True`, a bandpass filter is applied to signal in order to obtain
        a realistic audio representation of the RIR. Default: `True`.
    use_detailed_absorption : bool, optional
        When `True`, The detailed absorption data of the room is used to
        generate the impulse response. This allows a more realistic RIR but at
        the expense of a much higher computational cost. Default: `False`.
    max_order : int, optional
        This gives the option to limit the order of reflections computed for
        the method. This is specially useful when detailed absorption is used
        and the room has a long reverberation time, since this kind of setting
        will take a specially long time to run. Pass `None` to use an automatic
        estimation for the maximum order. Default: `None`.

    Returns
    -------
    rir : `Signal`
        Newly generated RIR.

    References
    ----------
    - Brinkmann, Fabian & Erbes, Vera & Weinzierl, Stefan. (2018). Extending
      the closed form image source model for source directivity.
    - pyroomacoustics: https://github.com/LCAV/pyroomacoustics

    Notes
    -----
    Depending on the computer and the given reverberation time, this function
    can take a relatively long runtime. If a faster or more flexible
    implementation is needed, please refer to the pyroomacoustics package
    (see references).

    """
    assert sampling_rate_hz is not None, "Sampling rate can not be None"
    assert type(room) is ShoeboxRoom, "Room must be of type ShoeboxRoom"
    source_position = np.asarray(source_position)
    receiver_position = np.asarray(receiver_position)
    assert room.check_if_in_room(
        source_position
    ), "Source is not located inside the room"
    assert room.check_if_in_room(
        receiver_position
    ), "Receiver is not located inside the room"

    total_length_samples = int(total_length_seconds * sampling_rate_hz)

    if not use_detailed_absorption:
        # ====== Frequency independent
        rir = _generate_rir(
            room_dim=room.dimensions_m,
            alpha=room.absorption_coefficient,
            s_pos=source_position,
            r_pos=receiver_position,
            rt=room.t60_s,
            mo=max_order,
            sr=sampling_rate_hz,
        )
        rir = _pad_trim(rir, total_length_samples)
        # Prune possible nan values
        np.nan_to_num(rir, copy=False, nan=0)
    else:
        # ====== Frequency dependent
        assert hasattr(
            room, "detailed_absorption"
        ), "Given room has no detailed absorption dictionary"
        # Create filter bank
        freqs = room.detailed_absorption["center_frequencies"][:-1] * np.sqrt(
            2
        )
        fb = linkwitz_riley_crossovers(
            crossover_frequencies_hz=freqs,
            order=12,
            sampling_rate_hz=sampling_rate_hz,
        )

        # Accumulator
        rir = np.zeros(total_length_samples)

        print("\nRIR Generator\n")
        for ind in range(fb.number_of_bands):
            print(
                f"Band {ind + 1} of {fb.number_of_bands} is being computed..."
            )
            alphas = room.detailed_absorption["absorption_matrix"][:, ind]
            rir_band = _generate_rir(
                room_dim=room.dimensions_m,
                alpha=alphas,
                s_pos=source_position,
                r_pos=receiver_position,
                rt=room.t60_s,
                mo=max_order,
                sr=sampling_rate_hz,
            )
            rir_band = _pad_trim(rir_band, total_length_samples)
            # Prune possible nan values
            np.nan_to_num(rir_band, copy=False, nan=0)
            rir0 = Signal(None, rir_band, sampling_rate_hz)
            rir_multi = fb.filter_signal(rir0, zero_phase=True)
            rir += rir_multi.bands[ind].time_data[:, 0]

    # Add decaying noise as reverberant tail
    if add_noise_reverberant_tail:
        if not hasattr(room, "mixing_time_s"):
            room.get_mixing_time("physical", n_reflections=1000)
        if room.mixing_time_s is None:
            room.get_mixing_time("physical", n_reflections=1000)
        rir = _add_reverberant_tail_noise(
            rir, room.mixing_time_s, room.t60_s, sr=sampling_rate_hz
        )

    rir_output = Signal(
        None,
        rir,
        sampling_rate_hz,
        signal_type="rir",
        signal_id="Synthetized RIR using the image source method",
    )

    # Bandpass signal in order to have a realistic audio signal representation
    if apply_bandpass:
        f = Filter(
            "iir",
            dict(
                order=12,
                filter_design_method="butter",
                type_of_pass="bandpass",
                freqs=[30, (sampling_rate_hz // 2) * 0.9],
            ),
            sampling_rate_hz=sampling_rate_hz,
        )
        rir_output = f.filter_signal(rir_output)

    return rir_output


def descriptors(
    rir: Signal | MultiBandSignal,
    mode: str = "d50",
    automatic_trimming_rir: bool = True,
):
    """Returns a desired room acoustics descriptor from an RIR.

    Parameters
    ----------
    rir : `Signal` or `MultiBandSignal`
        Room impulse response. If it is a multi-channel signal, the descriptor
        given back has the shape (channel). If it is a `MultiBandSignal`,
        the descriptor has shape (band, channel).
    mode : {'d50', 'c80', 'br', 'ts'} str, optional
        This defines the descriptor to be computed. Options are:
        - `'d50'`: Definition. It takes values between [0, 1] and should
          correlate (positively) with speech inteligibility.
        - `'c80'`: Clarity. It is a value in dB. The higher, the more energy
          arrives in the early part of the RIR compared to the later part.
        - `'br'`: Bass-ratio. It exposes the ratio of reverberation times
          of the lower-frequency octave bands (125, 250) to the higher ones
          (500, 1000). T20 is always used.
        - `'ts'`: Center time. It is the central time computed of the RIR.
    automatic_trimming_rir : bool, optional
        When `True`, the RIR is automatically trimmed after a certain energy
        threshold relative to the peak value has been surpassed. See notes
        for details on the algorithm. This parameter is ignored when computing
        the bass ratio. Default: `True`.

    Returns
    -------
    output_descriptor : NDArray[np.float64]
        Array containing the output descriptor. If RIR is a `Signal`,
        it has shape (channel). If RIR is a `MultiBandSignal`, the array has
        shape (band, channel).

    Notes
    -----
    - For defining the ending of the IR automatically, `trim_rir` is used.
      Refer to the documentation for more details.

    """
    mode = mode.lower()
    assert mode in (
        "d50",
        "c80",
        "br",
        "ts",
    ), "Given mode is not in the available descriptors"
    if type(rir) is Signal:
        if mode == "d50":
            func = _d50_from_rir
        elif mode == "c80":
            func = _c80_from_rir
        elif mode == "ts":
            func = _ts_from_rir
        else:
            # Bass ratio
            return _bass_ratio(rir)

        desc = np.zeros(rir.number_of_channels)
        for ch in range(rir.number_of_channels):
            desc[ch] = func(
                rir.time_data[:, ch],
                rir.sampling_rate_hz,
                automatic_trimming_rir,
            )
    elif type(rir) is MultiBandSignal:
        assert mode != "br", (
            "Bass-ratio is not a valid descriptor to be used on a "
            + "MultiBandSignal. Pass a RIR as Signal to compute it"
        )
        desc = np.zeros((rir.number_of_bands, rir.number_of_channels))
        for ind, b in enumerate(rir):
            desc[ind, :] = descriptors(b, mode=mode)
    else:
        raise TypeError("RIR must be of type Signal or MultiBandSignal")
    return desc


def _bass_ratio(rir: Signal) -> NDArray[np.float64]:
    """Core computation of bass ratio.

    Parameters
    ----------
    rir : `Signal`
        RIR.

    Returns
    -------
    br : NDArray[np.float64]
        Bass ratio per channel.

    """
    fb = fractional_octave_bands(
        [125, 1000], filter_order=10, sampling_rate_hz=rir.sampling_rate_hz
    )
    rir_multi = fb.filter_signal(rir, zero_phase=True)
    rt, _ = reverb_time(rir_multi)
    br = np.zeros(rir.number_of_channels)
    for ch in range(rir.number_of_channels):
        br[ch] = (rt[0, ch] + rt[1, ch]) / (rt[2, ch] + rt[3, ch])
    return br


def _check_ir_start_reverb(
    sig: Signal | MultiBandSignal,
    ir_start: int | NDArray[np.int_] | list | tuple | None,
) -> NDArray[np.float64] | list | None:
    """This method checks `ir_start` and parses it into the necessary form
    if relevant. For a `Signal`, it is a vector with the same number of
    elements as channels of `sig`. For `MultiBandSignal`, it is a 2d-array
    with shape (band, channel).

    `ir_start` must always have elements of type `int` or `intp`.

    For `None`, `None` is returned.

    """
    if ir_start is not None:
        if type(ir_start) in (list, tuple, NDArray[np.float64]):
            ir_start = np.atleast_1d(ir_start).astype(np.int_)
        assert (
            np.issubdtype(type(ir_start), np.integer)
            or type(ir_start) is np.ndarray
        ), "Unsupported type for ir_start"

    if type(sig) is Signal:
        if np.issubdtype(type(ir_start), np.integer):
            ir_start = (
                np.ones(sig.number_of_channels, dtype=np.int_) * ir_start
            )
        elif ir_start is None:
            return [None] * sig.number_of_channels
        assert (
            ir_start.ndim == 1 and len(ir_start) == sig.number_of_channels
        ), "Shape of ir_start is not valid"
    else:
        if np.issubdtype(type(ir_start), np.integer):
            ir_start = (
                np.ones(
                    (sig.number_of_bands, sig.number_of_channels),
                    dtype=np.int_,
                )
                * ir_start
            )
        if ir_start is None:
            return None
        if ir_start.ndim == 1:
            ir_start = np.repeat(
                ir_start[None, ...], sig.number_of_bands, axis=0
            )
        else:
            assert ir_start.shape == (
                sig.number_of_bands,
                sig.number_of_channels,
            ), "Shape of ir_start is not valid for the passed signal"
    if ir_start.dtype not in (int, np.intp):
        ir_start = ir_start.astype(np.int_)
    return ir_start
