import numpy as np
import pickle
from scipy.signal import (
    hilbert,
    oaconvolve,
    convolve,
)
from warnings import warn

from ..classes import (
    Signal,
    MultiBandSignal,
    FilterBank,
    Filter,
    Spectrum,
)
from ._standard_backend import (
    _indices_above_threshold_dbfs,
    _detrend,
)
from .._general_helpers import (
    _check_format_in_path,
    _get_smoothing_factor_ema,
)
from ..tools import from_db
from .enums import SpectrumType, InterpolationDomain


def load_pkl_object(path: str):
    """WARNING: This is not secure. Only unpickle data you know!
    Loads an object with all its attributes and methods.

    Parameters
    ----------
    path : str
        Path to the pickle object.

    Returns
    -------
    obj : object
        Unpacked pickle object.

    """
    obj = None
    path = _check_format_in_path(path, "pkl")
    with open(path, "rb") as inp:
        obj = pickle.load(inp)
    return obj


def activity_detector(
    signal: Signal,
    threshold_dbfs: float = -20,
    channel: int = 0,
    relative_to_peak: bool = True,
    pre_filter: Filter | None = None,
    attack_time_ms: float = 1,
    release_time_ms: float = 25,
) -> tuple[Signal, dict]:
    """This is a simple signal activity detector that uses a power threshold.
    It can be used relative to the signal's peak value or absolute. It is only
    applicable to one channel of the signal. This function returns the signal
    and a dictionary containing noise (as a signal) and
    the time indices corresponding to the bins that were found to surpass
    the threshold according to attack and release times.

    Prefiltering (for example with a bandpass filter) is possible when a
    `pre_filter` is passed.

    See Returns to gain insight into the returned dictionary and its keys.

    Parameters
    ----------
    signal : `Signal`
        Signal in which to detect activity.
    threshold_dbfs : float
        Threshold in dBFS to separate noise from activity.
    channel : int, optional
        Channel in which to perform the detection. Default: 0.
    relative_to_peak : bool, optional
        When `True`, the threshold value is relative to the signal's peak
        value. Otherwise, it is regarded as an absolute threshold.
        Default: `True`.
    pre_filter : `Filter`, optional
        Filter used for prefiltering the signal. It can be for instance a
        bandpass filter selecting the relevant frequencies in which the
        activity might be. Pass `None` to avoid any pre filtering. The filter
        is applied using zero-phase filtering. Default: `None`.
    attack_time_ms : float, optional
        Attack time (in ms). It corresponds to a lag time for detecting
        activity after surpassing the threshold. Default: 1.
    release_time_ms : float, optional
        Release time (in ms) for activity detector after signal has fallen
        below power threshold. Pass 0 to release immediately. Default: 25.

    Returns
    -------
    detected_sig : `Signal`
        Detected signal.
    others : dict
        Dictionary containing following keys:
        - `'noise'`: left-out noise in original signal (below threshold) as
          `Signal` object.
        - `'signal_indices'`: array of boolean that describes which indices
          of the original time series belong to signal and which to noise.
          `True` at index n means index n was passed to signal.
        - `'noise_indices'`: the inverse array to `'signal_indices'`.

    """
    assert isinstance(channel, int), (
        "Channel must be type integer. Function is not implemented for "
        + "multiple channels."
    )
    assert threshold_dbfs < 0, "Threshold must be below zero"
    assert release_time_ms >= 0, "Release time must be positive"
    assert attack_time_ms >= 0, "Attack time must be positive"

    # Get channel
    signal = signal.get_channels(channel)

    # Pre-filtering
    if pre_filter is not None:
        assert isinstance(
            pre_filter, Filter
        ), "pre_filter must be of type Filter"
        signal_filtered = pre_filter.filter_signal(signal, zero_phase=True)
    else:
        signal_filtered = signal

    # Release samples
    attack_coeff = _get_smoothing_factor_ema(
        attack_time_ms / 1e3, signal.sampling_rate_hz
    )
    release_coeff = _get_smoothing_factor_ema(
        release_time_ms / 1e3, signal.sampling_rate_hz
    )

    # Get indices
    signal_indices = _indices_above_threshold_dbfs(
        signal_filtered.time_data,
        threshold_dbfs=threshold_dbfs,
        attack_smoothing_coeff=attack_coeff,
        release_smoothing_coeff=release_coeff,
        normalize=relative_to_peak,
    )
    noise_indices = ~signal_indices

    # Separate signals
    detected_sig = signal.copy()
    noise = signal.copy()
    detected_sig.clear_time_window()
    noise.clear_time_window()

    try:
        detected_sig.time_data = signal.time_data[signal_indices, 0]
    except ValueError as e:
        warn(
            "No detected activity, threshold might be too high. Detected "
            + "signal will be a vector filled with zeroes"
        )
        print("Numpy error: ", e)
        detected_sig.time_data = np.zeros(500)

    try:
        noise.time_data = signal.time_data[noise_indices, 0]
    except ValueError as e:
        warn(
            "No detected noise, threshold might be too low. Noise will be "
            + "a vector filled with zeroes"
        )
        print("Numpy error: ", e)
        noise.time_data = np.zeros(500)

    others = dict(
        noise=noise, signal_indices=signal_indices, noise_indices=noise_indices
    )
    return detected_sig, others


def detrend(
    sig: Signal | MultiBandSignal, polynomial_order: int = 0
) -> Signal | MultiBandSignal:
    """Returns the detrended signal.

    Parameters
    ----------
    sig : Signal
        Signal to detrend.
    polynomial_order : int, optional
        Polynomial order of the fitted polynomial that will be removed
        from time data. 0 is equal to mean removal. Default: 0.

    Returns
    -------
    detrended_sig : Signal
        Detrended signal.

    """
    if isinstance(sig, Signal):
        assert polynomial_order >= 0, "Polynomial order should be positive"
        td = sig.time_data
        new_td = _detrend(td, polynomial_order)
        detrended_sig = sig.copy()
        detrended_sig.time_data = new_td
        return detrended_sig
    elif isinstance(sig, MultiBandSignal):
        detrended_sig = sig.copy()
        for n in range(sig.number_of_bands):
            detrended_sig.bands[n] = detrend(sig.bands[n], polynomial_order)
        return detrended_sig
    else:
        raise TypeError("Pass either a Signal or a MultiBandSignal")


def envelope(
    signal: Signal | MultiBandSignal,
    mode: str = "analytic",
    window_length_samples: int | None = None,
):
    """This function computes the envelope of a given signal by means of its
    hilbert transformation. It can also compute the RMS value over a certain
    window length (boxcar). The time signal is always detrended with a linear
    polynomial.

    Parameters
    ----------
    signal : `Signal` or `MultiBandSignal`
        Time series for which to find the envelope. If it is a
        `MultiBandSignal`, it must have the same sampling rate for all bands.
    mode : str {'analytic', 'rms'}, optional
        Type of envelope. It either uses the hilbert transform to obtain the
        analytic signal or RMS values. Default: `'analytic'`.
    window_length_samples : int, optional
        Window length (boxcar) to average the RMS values. Cannot be `None`
        if `mode = 'rms'`. Default: `None`.

    Returns
    -------
    NDArray[np.float64]
        Signal envelope. It has the shape (time sample, channel) or
        (time sample, band, channel) in case of `MultiBandSignal`.

    """
    mode = mode.lower()
    assert mode in (
        "analytic",
        "rms",
    ), "Invalid mode. Use either analytic or rms."

    if isinstance(signal, Signal):
        signal = detrend(signal, 1)

        if mode == "analytic":
            env = signal.time_data
            env = np.abs(hilbert(env, axis=0))
            return env

        assert (
            window_length_samples is not None
        ), "Some window length must be passed"
        assert (
            window_length_samples > 0
        ), "Window length must be more than 1 sample"
        rms_vec = signal.time_data
        rms_vec = oaconvolve(
            rms_vec**2,
            np.ones(window_length_samples)[..., None] / window_length_samples,
            mode="full",
            axes=0,
        )[: len(rms_vec), ...]
        rms_vec **= 0.5
        return rms_vec
    elif isinstance(signal, MultiBandSignal):
        assert (
            signal.same_sampling_rate
        ), "This is only available for constant sampling rate bands"
        rms_vec = np.zeros(
            (
                len(signal.bands[0]),
                signal.number_of_bands,
                signal.number_of_channels,
            ),
            float,
        )
        for ind, b in enumerate(signal.bands):
            rms_vec[:, ind, :] = envelope(
                b, mode=mode, window_length_samples=window_length_samples
            )
        return rms_vec
    else:
        raise TypeError("Signal must be type Signal or MultiBandSignal")


def dither(
    s: Signal,
    mode: str = "triangular",
    epsilon: float = float(np.finfo(np.float16).smallest_subnormal),
    noise_shaping_filterbank: FilterBank | None = None,
    truncate: bool = False,
) -> Signal:
    """
    This function applies dither to the signal and, optionally, truncates the
    time samples to 16-bits floating point representation.

    Parameters
    ----------
    s : `Signal`
        Signal to apply dither to.
    mode : str, optional
        Type of probability distribution to use noise from. Choose from
        `"rectangular"`, `"triangular"`. See notes and references for details.
        Default: `"triangular"`.
    epsilon : float, optional
        Value that represents the quantization step. The default value supposes
        quantization to 16-bit floating point. It is obtained through numpy's
        smallest subnormal for np.float16. See notes for the value concerning
        the 24-bit case. Default: 6e-08.
    noise_shaping_filterbank : `FilterBank`, `None`, optional
        Noise can be arbitrarily shaped using a filter bank (in sequential
        mode). Pass `None` to avoid any noise-shaping. Default: `None`.
    truncate : bool, optional
        When `True`, the time samples are truncated to np.float16 resolution.
        `False` only applies dither noise to the signal without truncating.
        Default: `False`.

    Returns
    -------
    new_s : `Signal`
        Signal with dither.

    Notes
    -----
    - The output signal has time samples with 16-bit precision, but the data
      type of the array is `np.float64` for consistency.
    - `"rectangular"` mode applies noise with samples coming from a uniform
      distribution [-epsilon/2, epsilon/2]. `"triangular"` has a triangle shape
      for the noise distribution with values between [-epsilon, epsilon]. See
      [1] for more details on this.
    - Dither might be only necessary when lowering the bit-depth down to 16
      bits, though the 24-bit case might be relevant if the there are signal
      components with very low volumes.
    - 24-bit signed integers range from -8388608 to 8388607. The quantization
      step is therefore `1/8388608=1.1920928955078125e-07`.

    References
    ----------
    - [1]: Lerch, Weinzierl. Handbuch der Audiotechnik: Chapter 14.

    """
    mode = mode.lower()
    shape = s.time_data.shape

    if mode == "rectangular":
        noise = np.random.uniform(-epsilon / 2, epsilon / 2, size=shape)
    elif mode == "triangular":
        noise = np.random.uniform(
            -epsilon / 2, epsilon / 2, size=shape
        ) + np.random.uniform(-epsilon / 2, epsilon / 2, size=shape)
    else:
        raise ValueError(f"{mode} is not supported.")

    if noise_shaping_filterbank is not None:
        noise_s = Signal(None, noise, s.sampling_rate_hz)
        noise_s = noise_shaping_filterbank.filter_signal(
            noise_s, mode="sequential"
        )
        noise = noise_s.time_data

    new_s = s.copy()

    if truncate:
        new_s.time_data = (
            (new_s.time_data + noise).astype(np.float16)
        ).astype(np.float64)
    else:
        new_s.time_data = new_s.time_data + noise
    return new_s


def merge_fir_filters(filters: list[Filter] | FilterBank) -> Filter:
    """This returns an FIR filter that results from convolving all passed FIR
    filters.

    Parameters
    ----------
    fir : list[Filter] or FilterBank
        List or FilterBank containing all FIR filters to combine. If any filter
        is not FIR, an assertion will be raised.

    Returns
    -------
    Filter
        Combined FIR filter.

    """
    fir = filters.filters if isinstance(filters, FilterBank) else filters
    assert len(fir) > 1, "There must be at least two filters to combine"
    assert all([not f.is_iir for f in fir]), "Some filter is not FIR"
    assert all(
        [fir[0].sampling_rate_hz == f.sampling_rate_hz for f in fir]
    ), "Sampling rates do not match"
    b_coefficients = fir[0].ba[0].copy()
    for ind in range(1, len(fir)):
        b_coefficients = convolve(
            b_coefficients, fir[ind].ba[0], mode="full", method="auto"
        )
    return Filter.from_ba(b_coefficients, [1.0], fir[0].sampling_rate_hz)


def spectral_difference(
    input_1: Signal | Spectrum,
    input_2: Signal | Spectrum,
    octave_fraction_smoothing: float = 0.0,
    energy_normalization: bool = True,
    complex: bool = False,
    dynamic_range_db: float | None = 100.0,
) -> Spectrum:
    """Compute the spectral difference between two signals or spectra. Their
    number of channels must match. It is computed as `input_1 / input_2`.

    Parameters
    ----------
    input_1 : Signal, Spectrum
    input_2 : Signal, Spectrum
    octave_fraction_smoothing : float, optional
        Smoothing can be applied prior to computing the difference.
        Default: 0 (no smoothing).
    energy_normalization : bool, optional
        When True, each channel is energy normalized before computing the
        difference. Default: True.
    complex : bool, optional
        When True, the output will be complex. This is only supported if the
        inputs are complex (for signals, the saved spectrum parameters must
        deliver a complex spectrum). Default: False.
    dynamic_range_db : float, None, optional
        Dynamic range in dB to regard when building the difference. Pass None
        to avoid limiting the range. Default: 100.

    Returns
    -------
    Spectrum
        Difference spectrum.

    """
    assert (
        input_1.number_of_channels == input_2.number_of_channels
    ), "Number of channels does not match"

    if isinstance(input_1, Signal):
        inp1 = Spectrum.from_signal(input_1, complex)
    else:
        if complex:
            assert not input_1.is_magnitude, "Input data should be complex"
        inp1 = input_1.copy()

    if isinstance(input_2, Signal):
        inp2 = Spectrum.from_signal(input_2, complex)
    else:
        if complex:
            assert not input_2.is_magnitude, "Input data should be complex"
        inp2 = input_2.copy()

    if energy_normalization:
        inp1.spectral_data /= inp1.get_energy() ** 0.5
        inp2.spectral_data /= inp2.get_energy() ** 0.5

    if octave_fraction_smoothing != 0:
        inp1.apply_octave_smoothing(octave_fraction_smoothing)
        inp2.apply_octave_smoothing(octave_fraction_smoothing)

    inp2.set_interpolator_parameters(
        InterpolationDomain.MagnitudePhase
        if complex
        else InterpolationDomain.Power
    )
    mag2 = inp2.get_interpolated_spectrum(
        inp1.frequency_vector_hz,
        SpectrumType.Complex if complex else SpectrumType.Magnitude,
    )

    if dynamic_range_db is not None:
        dynamic_range_factor = from_db(-abs(dynamic_range_db), True)
        mag2 = np.clip(mag2, np.max(mag2, axis=0) * dynamic_range_factor, None)

    inp1.spectral_data /= mag2
    return inp1
