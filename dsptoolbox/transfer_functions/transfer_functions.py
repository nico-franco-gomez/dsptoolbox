"""
Methods used for acquiring and windowing transfer functions
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import minimum_phase as min_phase_scipy
from scipy.fft import rfft as rfft_scipy, next_fast_len as next_fast_length_fft
from scipy.interpolate import interp1d

from ._transfer_functions import (
    _spectral_deconvolve,
    _window_this_ir_tukey,
    _window_this_ir,
    _get_harmonic_times,
    _trim_ir,
    _complex_smoothing_backend,
    _get_frequency_vector_with_frequency_resolution,
    _fdw_backend,
)
from ..helpers.spectrum_utilities import (
    _correct_for_real_phase_spectrum,
    _interpolate_fr,
)
from ..helpers.minimum_phase import (
    _get_minimum_phase_spectrum_from_real_cepstrum,
    _min_phase_ir_from_real_cepstrum,
    _remove_ir_latency_from_phase_min_phase,
)
from ..helpers.smoothing import _fractional_octave_smoothing
from ..helpers.latency import _get_fractional_impulse_peak_index
from ..helpers.other import find_frequencies_above_threshold, _pad_trim
from ..classes import Signal, Filter, ImpulseResponse, FilterBank, Spectrum
from ..classes.filter_helpers import _group_delay_filter
from ..standard._standard_backend import (
    _minimum_phase,
    _group_delay_direct,
)
from ..standard._spectral_methods import _welch
from ..standard import (
    fractional_delay,
    append_signals,
    normalize,
    latency,
)
from ..generators import dirac
from ..filterbanks import linkwitz_riley_crossovers
from ..helpers.gain_and_level import to_db, from_db
from ..standard.enums import (
    SpectrumMethod,
    SpectrumType,
    MagnitudeNormalization,
    Window,
)
from .enums import TransferFunctionType, SmoothingDomain


def spectral_deconvolve(
    num: Signal,
    denum: Signal,
    apply_regularization: bool = True,
    start_stop_hz=None,
    threshold_db: float = -30.0,
    padding: bool = False,
    keep_original_length: bool = False,
) -> ImpulseResponse:
    """Deconvolution by spectral division of two signals. If the denominator
    signal only has one channel, the deconvolution is done using that channel
    for all channels of the numerator.

    Parameters
    ----------
    num : `Signal`
        Signal to deconvolve from.
    denum : `Signal`
        Signal to deconvolve.
    apply_regularization : bool, optional
        When True, a regularization window is applied for avoiding noise
        outside the excitation frequency region. Default: True.
    start_stop_hz : array-like or `None`, optional
        This is a vector of length 2 or 4 with frequency values that define the
        area of the denominator that has some energy during the regularization.
        Pass `None` to use an automatic mode that recognizes the start and stop
        of the denominator (it assumes a chirp). If regularization is
        deactivated, `start_stop_hz` has to be set to `None`. Default: `None`.
    threshold_db : float, optional
        Threshold in dBFS for the automatic creation of the window.
        Default: -30.
    padding : bool, optional
        Pads the time data with 2 length. Done for separating distortion
        in negative time bins when deconvolving sweep measurements.
        Default: `False`.
    keep_original_length : bool, optional
        Only regarded when padding is `True`. It trims the newly deconvolved
        data to its original length. Default: `False`.

    Returns
    -------
    new_sig : `Signal`
        Deconvolved signal.

    """
    assert (
        num.time_data.shape[0] == denum.time_data.shape[0]
    ), "Lengths do not match for spectral deconvolution"
    if denum.number_of_channels != 1:
        assert (
            num.number_of_channels == denum.number_of_channels
        ), "The number of channels do not match."
        multichannel = False
    else:
        multichannel = True
    assert (
        num.sampling_rate_hz == denum.sampling_rate_hz
    ), "Sampling rates do not match"
    if not apply_regularization:
        assert (
            start_stop_hz is None
        ), "No start_stop_hz vector can be passed when using standard mode"

    num = num.copy()
    denum = denum.copy()
    original_length = num.time_data.shape[0]

    if padding:
        num.time_data = _pad_trim(num.time_data, original_length * 2)
        denum.time_data = _pad_trim(denum.time_data, original_length * 2)

    denum.spectrum_method = SpectrumMethod.FFT
    num.spectrum_method = SpectrumMethod.FFT
    _, denum_fft = denum.get_spectrum()
    freqs_hz, num_fft = num.get_spectrum()
    fs_hz = num.sampling_rate_hz

    new_time_data = np.zeros_like(num.time_data)

    for n in range(num.number_of_channels):
        n_denum = 0 if multichannel else n
        if apply_regularization:
            if start_stop_hz is None:
                start_stop_hz = find_frequencies_above_threshold(
                    denum_fft[:, n_denum], freqs_hz, threshold_db
                )
            if len(start_stop_hz) == 2:
                start_stop_hz = np.array(
                    [
                        start_stop_hz[0] / np.sqrt(2),
                        start_stop_hz[0],
                        start_stop_hz[1],
                        np.min([start_stop_hz[1] * np.sqrt(2), fs_hz / 2]),
                    ]
                )
            elif len(start_stop_hz) == 4:
                pass
            else:
                raise ValueError(
                    "start_stop_hz vector should have 2 or 4 values"
                )
        new_time_data[:, n] = _spectral_deconvolve(
            num_fft[:, n],
            denum_fft[:, n_denum],
            freqs_hz,
            original_length * 2 if padding else original_length,
            start_stop_hz=start_stop_hz,
            regularized=apply_regularization,
        )
    new_sig = ImpulseResponse(None, new_time_data, num.sampling_rate_hz)
    if padding:
        if keep_original_length:
            new_sig.time_data = _pad_trim(new_sig.time_data, original_length)
    return new_sig


def window_ir(
    signal: ImpulseResponse,
    total_length_samples: int,
    adaptive: bool = True,
    constant_percentage: float = 0.75,
    window_type: Window | list[Window] = Window.Hann,
    at_start: bool = True,
    offset_samples: int = 0,
    left_to_right_flank_length_ratio: float = 1.0,
) -> tuple[ImpulseResponse, NDArray]:
    """Windows an IR with trimming and selection of constant valued length.
    This is equivalent to a tukey window whose flanks can be selected to be
    any type. The peak of the impulse response is aligned to correspond to
    the first value with amplitude 1 of the window.

    Parameters
    ----------
    signal : `ImpulseResponse`
        Signal to window
    total_length_samples : int
        Total window length in samples.
    adaptive : bool, optional
        When `True`, some design parameters will modified in case that the
        IR does not have enough samples to accomodate them. See Notes for
        more details. Default: `True`.
    constant_percentage : float, optional
        Percentage (between 0 and 1) of the window's length that should be
        constant value. Default: 0.75.
    window_type : Window, list[Window], optional
        Window function to be used for the flanks. Pass a list containing two
        windows to use different windows for the left and right flanks
        respectively. Default: Hann.
    at_start : bool, optional
        Windows the start with a rising window as well as the end.
        Default: `True`.
    offset_samples : int, optional
        Passing an offset in samples delays the impulse w.r.t. the first window
        value with amplitude 1. The offset must be inside the constant region
        of the window. Default: 0.
    left_to_right_flank_length_ratio : float, optional
        This is the length ratio between left and right flanks. For instance,
        2 leads to a length of the left flank twice as long as the right one,
        while 0.1 would be a tenth of the length. Default: 1 (equal length).

    Returns
    -------
    new_sig : `ImpulseResponse`
        Windowed signal. The used window is also saved under `new_sig.window`.
    start_positions_samples : NDArray
        This array contains the position index of the start of the IR in
        each channel of the original IR (relative to the possibly padded
        windowed IR).

    Notes
    -----
    - With `adaptive=True`, following modifications are allowed:
        - Left flank length is variable to fit the first part of the IR. The
          offset is always maintained.
        - Constant amplitude part of the window is modified in order to fit
          the right flank into the IR. If the IR is too short, there might
          be only a couple samples with constant amplitude.

    - With `adaptive=False`, the desired window might not fit the given IR.
      In that case, the window values that will be multiplied with zero-padded
      parts of the window are set to 0 in order to make them visible.

    """
    assert (
        type(signal) is ImpulseResponse
    ), "This is only valid for an impulse response"
    assert (
        constant_percentage < 1 and constant_percentage >= 0
    ), "Constant percentage can not be larger than 1 or smaller than 0"
    assert offset_samples >= 0, "Offset must be positive"
    assert offset_samples <= constant_percentage * total_length_samples, (
        "Offset is too large for the constant part of the window and its "
        + "total length"
    )
    assert (
        left_to_right_flank_length_ratio >= 0
    ), "Ratio between window flanks must be a positive number"

    new_time_data = np.zeros((total_length_samples, signal.number_of_channels))
    start_positions_samples = np.zeros(signal.number_of_channels, dtype=int)
    window = np.zeros((total_length_samples, signal.number_of_channels))

    for n in range(signal.number_of_channels):
        (
            new_time_data[:, n],
            window[:, n],
            start_positions_samples[n],
        ) = _window_this_ir_tukey(
            signal.time_data[:, n],
            total_length_samples,
            window_type,
            constant_percentage,
            at_start,
            offset_samples,
            left_to_right_flank_length_ratio,
            adaptive,
        )

    new_sig = signal.copy_with_new_time_data(new_time_data)
    new_sig.set_window(window)
    return new_sig, start_positions_samples


def window_centered_ir(
    signal: ImpulseResponse,
    total_length_samples: int,
    window_type: Window = Window.Hann,
) -> tuple[ImpulseResponse, NDArray]:
    """This function windows an IR placing its peak in the middle. It trims
    it to the total length of the window or pads it to the desired length
    (padding in the end, window has `total_length`).

    Parameters
    ----------
    signal: `ImpulseResponse`
        Signal to window
    total_length_samples: int
        Total window length in samples.
    window_type: Window, optional
        Window function to be used. Default: Hann.

    Returns
    -------
    new_sig : `ImpulseResponse`
        Windowed signal. The used window is also saved under `new_sig.window`.
    start_positions_samples : NDArray
        This array contains the position index of the start of the IR in
        each channel of the original IR.

    Notes
    -----
    - If the window seems truncated, it is because the length and peak position
      were longer than the IR, so that it had to be zero-padded to match the
      given length.

    """
    assert (
        type(signal) is ImpulseResponse
    ), "This is only valid for an impulse response"

    new_time_data = np.zeros((total_length_samples, signal.number_of_channels))
    start_positions_samples = np.zeros(signal.number_of_channels, dtype=int)
    window = np.zeros((total_length_samples, signal.number_of_channels))

    for n in range(signal.number_of_channels):
        (
            new_time_data[:, n],
            window[:, n],
            start_positions_samples[n],
        ) = _window_this_ir(
            signal.time_data[:, n], total_length_samples, window_type
        )

    new_sig = signal.copy_with_new_time_data(new_time_data)
    new_sig.set_window(window)
    return new_sig, start_positions_samples


def compute_transfer_function(
    output: Signal,
    input: Signal,
    window_length_samples: int,
    mode: TransferFunctionType = TransferFunctionType.H2,
) -> Spectrum:
    """Gets transfer function H1, H2 or H3 (for stochastic signals). If the
    input signal only has one channel, it is assumed to be the input for all of
    the channels of the output.

    The spectrum parameters for the input will be used for the computation.

    Parameters
    ----------
    output : `Signal`
        Signal with output channels.
    input : `Signal`
        Signal with input channels.
    window_length_samples : int
        Window length for the IR. Spectrum has the length.
    mode : TransferFunction, optional
        Type of transfer function. Default: H2.

    Returns
    -------
    spec : `Spectrum`
        Transfer functions as Spectrum. Coherences are also computed and saved
        as `coherence` attribute.

    Notes
    -----
    - SNR can be gained from the coherence: `snr = coherence / (1 - coherence)`

    """
    assert (
        input.sampling_rate_hz == output.sampling_rate_hz
    ), "Sampling rates do not match"
    assert (
        input.time_data.shape[0] == output.time_data.shape[0]
    ), "Signal lengths do not match"
    if input.number_of_channels != 1:
        assert (
            input.number_of_channels == output.number_of_channels
        ), "Channel number does not match between signals"
        multichannel = False
    else:
        multichannel = True

    # Get rid of unnecessary spectrum parameters
    spectrum_parameters = input._spectrum_parameters.copy()
    assert (
        type(spectrum_parameters) is dict
    ), "Spectrum parameters should be passed as a dictionary"
    spectrum_parameters.pop("window_length_samples")
    spectrum_parameters.pop("method")
    spectrum_parameters.pop("smoothing")
    spectrum_parameters.pop("pad_to_fast_length")

    coherence = np.zeros(
        (window_length_samples // 2 + 1, output.number_of_channels)
    )
    tf = np.zeros(
        (window_length_samples // 2 + 1, output.number_of_channels),
        dtype=np.complex128,
    )
    if multichannel:
        G_xx = _welch(
            input.time_data[:, 0],
            None,
            input.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **spectrum_parameters,
        )
    for n in range(output.number_of_channels):
        G_yy = _welch(
            output.time_data[:, n],
            None,
            input.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **spectrum_parameters,
        )
        if multichannel:
            n_input = 0
        else:
            n_input = n
            G_xx = _welch(
                input.time_data[:, n_input],
                None,
                input.sampling_rate_hz,
                window_length_samples=window_length_samples,
                **spectrum_parameters,
            )
        if mode == TransferFunctionType.H2:
            G_yx = _welch(
                output.time_data[:, n],
                input.time_data[:, n_input],
                output.sampling_rate_hz,
                window_length_samples=window_length_samples,
                **spectrum_parameters,
            )
        G_xy = _welch(
            input.time_data[:, n_input],
            output.time_data[:, n],
            output.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **spectrum_parameters,
        )

        match mode:
            case TransferFunctionType.H1:
                tf[:, n] = G_xy / G_xx
            case TransferFunctionType.H2:
                tf[:, n] = G_yy / G_yx
            case TransferFunctionType.H3:
                tf[:, n] = G_xy / np.abs(G_xy) * (G_yy / G_xx) ** 0.5
            case _:
                raise ValueError("Unsupported transfer function type")
        coherence[:, n] = np.abs(G_xy) ** 2 / G_xx / G_yy
    spec = Spectrum(
        np.fft.rfftfreq(window_length_samples, 1 / input.sampling_rate_hz), tf
    )
    spec.set_coherence(coherence)
    return spec


def average_irs(
    signal: ImpulseResponse,
    time_average: bool = True,
    normalize_energy: bool = True,
) -> ImpulseResponse:
    """Averages all channels of a given IR. It can either use a time domain
    average while time-aligning all channels to the one with the longest
    latency, or average directly their magnitude and phase responses.

    Parameters
    ----------
    signal : `ImpulseResponse`
        Signal with channels to be averaged over.
    time_average : bool, optional
        When True, the IRs are time-aligned to the channel with the largest
        (minimum-phase) latency and then averaged in the time domain. False
        averages directly the magnitude and phase of each IR. Default: True.
    normalize_energy : bool, optional
        When `True`, the energy of all spectra is normalized to the first
        channel's energy and then averaged. Beware that normalization factors
        might be clipped if the impulses are already at or close to 0 dBFS.
        Default: `True`.

    Returns
    -------
    avg_sig : `ImpulseResponse`
        Averaged impulse response.

    """
    assert (
        type(signal) is ImpulseResponse
    ), "This is only valid for an impulse response"
    assert (
        signal.number_of_channels > 1
    ), "Signal has only one channel so no meaningful averaging can be done"
    avg_sig = signal.copy()

    if normalize_energy:
        energies = np.sum(signal.time_data**2, axis=0)
        energies /= energies[0]
        avg_sig.time_data *= energies

    if not time_average:
        # Obtain channel magnitude and phase spectra
        _, sp = signal.get_spectrum()
        mag = np.abs(sp)
        pha = np.unwrap(np.angle(sp), axis=0)

        # Build averages
        new_mag = np.mean(mag, axis=1)
        new_pha = np.mean(pha, axis=1)
        # New signal
        new_sp = new_mag * np.exp(1j * new_pha)

        # New time data and signal object
        new_time_data = np.fft.irfft(
            new_sp[..., None], n=signal.length_samples, axis=0
        )
    else:
        latencies = find_ir_latency(signal)
        channel_to_follow = np.argmax(latencies)
        for i in range(signal.number_of_channels):
            if channel_to_follow == i:
                continue
            latency_s = (
                latencies[channel_to_follow] - latencies[i]
            ) / signal.sampling_rate_hz
            new_channel = fractional_delay(
                signal.get_channels(i), latency_s, keep_length=True
            )
            avg_sig.time_data[:, i] = new_channel.time_data[:, 0]
        new_time_data = np.mean(avg_sig.time_data, axis=1)

    avg_sig.time_data = new_time_data
    return avg_sig


def min_phase_from_mag(
    spectrum: Spectrum,
    sampling_rate_hz: int,
    ir_length_samples: int | None = None,
) -> ImpulseResponse:
    """Returns a minimum-phase signal from a magnitude spectrum using
    the discrete hilbert transform.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum with only positive frequencies.
    sampling_rate_hz : int
        Sampling rate in Hz for output impulse response.
    ir_length_samples : int, None, optional
        Pass to define the frequency resolution during computation and the
        final length of the impulse response. Pass None to use some estimate
        which might suffice for most cases. Default: None.

    Returns
    -------
    `ImpulseResponse`
        Signal with same magnitude spectrum but minimum phase.

    References
    ----------
    - https://en.wikipedia.org/wiki/Minimum_phase

    """
    # Use very conservative estimate for most cases
    delta_f_hz = (
        0.5
        if ir_length_samples is None
        else sampling_rate_hz / ir_length_samples
    )

    # Frequency vector
    f_vec, delta_f_hz, original_length_time_data = (
        _get_frequency_vector_with_frequency_resolution(
            delta_f_hz, sampling_rate_hz
        )
    )

    # Get interpolated magnitude spectrum
    mag_spectrum = spectrum.get_interpolated_spectrum(
        f_vec, SpectrumType.Magnitude
    )

    phase = _minimum_phase(
        mag_spectrum, False, True, original_length_time_data % 2 == 1
    )
    time_data = np.fft.irfft(
        mag_spectrum * np.exp(1j * phase), axis=0, n=original_length_time_data
    )
    return ImpulseResponse.from_time_data(time_data, sampling_rate_hz)


def lin_phase_from_mag(
    spectrum: Spectrum,
    sampling_rate_hz: int,
    group_delay_ms: float | None = None,
    check_causality: bool = True,
    minimum_group_delay_factor: float = 1.0,
) -> ImpulseResponse:
    """Returns a linear phase signal from a magnitude spectrum. It is possible
    to return the smallest causal group delay by checking the minimum phase
    version of the signal and choosing a constant group delay that is never
    lower than minimum group delay (for each channel). A value for the group
    delay can be also passed directly and applied to all channels. If check
    causility is activated, it is assessed that the given group delay is not
    less than each minimum group delay. If deactivated, the generated phase
    could lead to a non-causal system!

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum with only positive frequencies and 0.
    sampling_rate_hz : int
        Sampling rate in Hz for the output impulse response.
    group_delay_ms : float, None, optional
        Constant group delay that the phase should have for all channels
        (in ms). Pass None to create a signal with the minimum linear
        phase possible by regarding the minimum-phase response (it is different
        for each channel). Default: None.
    check_causality : bool, optional
        When `True`, it is assessed for each channel that the given group
        delay is not lower than the minimum group delay. Default: `True`.
    minimum_group_delay_factor : float, optional
        When computing from the group delay from the minimum group delay, the
        magnitude response can be distorted for low frequencies. Increase this
        factor to add delay and correct this distortion. Default: 1.

    Returns
    -------
    `ImpulseResponse`
        Impulse response with same magnitude spectrum but linear phase. Its
        length is always twice the delay.

    """
    # Check group delay ms parameter
    minimum_group_delay = group_delay_ms is None
    # Only check causality when necessary and requested
    check_causality = not minimum_group_delay and check_causality
    if not minimum_group_delay:
        group_delay_s = group_delay_ms / 1000.0

    if minimum_group_delay:
        # Use very conservative estimate for most cases
        delta_f_hz = 0.5
    else:
        # Get minimum frequency resolution
        delta_f_hz = (
            1.0 / (group_delay_s * 2.0) * 0.9
        )  # Ensure good resolution by taking 0.9 of minimum

    # Frequency vector
    f_vec, delta_f_hz, original_length_time_data = (
        _get_frequency_vector_with_frequency_resolution(
            delta_f_hz, sampling_rate_hz
        )
    )

    # Get interpolated magnitude spectrum
    mag_spectrum = spectrum.get_interpolated_spectrum(
        f_vec, SpectrumType.Magnitude
    )

    if check_causality or minimum_group_delay:
        assert (
            minimum_group_delay_factor >= 1.0
        ), "Minimum group delay factor should at least be 1"
        min_phase = _minimum_phase(
            mag_spectrum,
            odd_length=original_length_time_data % 2 == 1,
        )
        min_gd = _group_delay_direct(min_phase, delta_f_hz)
        group_delay_to_use_s = minimum_group_delay_factor * (
            np.max(min_gd, axis=0) + 1e-3
        )  # add 1 ms as safety factor

        if check_causality:
            for n in range(len(group_delay_to_use_s)):
                assert group_delay_to_use_s[n] <= group_delay_s, (
                    f"Given group delay {group_delay_s * 1000} ms is lower "
                    + "than minimal group delay "
                    + f"{group_delay_to_use_s * 1000} ms for "
                    + f"channel {n}"
                )
            group_delay_to_use_s = (
                np.ones(spectrum.number_of_channels) * group_delay_s
            )

        if np.any(
            group_delay_to_use_s * 2
            > original_length_time_data / sampling_rate_hz
        ):
            # New resolution
            delta_f_hz = 1.0 / (max(group_delay_to_use_s) * 2) * 0.9
            f_vec, delta_f_hz, original_length_time_data = (
                _get_frequency_vector_with_frequency_resolution(
                    delta_f_hz, sampling_rate_hz
                )
            )
            mag_spectrum = spectrum.get_interpolated_spectrum(
                f_vec, SpectrumType.Magnitude
            )
    else:
        group_delay_to_use_s = (
            np.ones(spectrum.number_of_channels) * group_delay_s
        )

    # New spectrum
    time_data = np.fft.irfft(
        mag_spectrum
        * np.exp(
            1j
            * _correct_for_real_phase_spectrum(
                -2 * np.pi * f_vec[:, None] * group_delay_to_use_s[None, :]
            )
        ),
        axis=0,
        n=original_length_time_data,
    )
    time_data = _pad_trim(
        time_data, int(2 * max(group_delay_to_use_s) * sampling_rate_hz + 0.5)
    )
    return ImpulseResponse.from_time_data(time_data, sampling_rate_hz)


def min_phase_ir(
    sig: ImpulseResponse,
    use_real_cepstrum: bool = True,
    padding_factor: int = 8,
    alpha: float = 1.0,
) -> ImpulseResponse:
    """Returns same IR with minimum phase. Two methods are available for
    computing the minimum phase version of the IR: `'real cepstrum'` (using
    filtering the real-cepstral domain) and `'equiripple'` (for
    symmetric IR, uses `scipy.signal.minimum_phase`).

    Parameters
    ----------
    sig : `ImpulseResponse`
        IR for which to compute minimum phase IR.
    use_real_cepstrum : bool, optional
        Set to True for general cases. If the IR is symmetric (like a
        linear-phase filter), False is recommended. Default: True.
    padding_factor : int, optional
        Zero-padding to a length corresponding to
        `current_length * padding_factor` can be done, in order to avoid time
        aliasing errors. Default: 8.
    alpha : float, optional
        This value can be used to premultiply the IR with `alpha**n`, where `n`
        is the index of the time sample. This is done such that the zeroes
        of the transfer function are pushed towards the origin of the z-plane,
        thus ensuring minimum phase outputs. This value should be very close
        to 1. Useful values are around 1-1e-4 and 1-1e-8. See [1] and [2] for
        details. Default: 1. (No scaling is used).

    Returns
    -------
    `ImpulseResponse`
        Minimum-phase IR as time signal.

    References
    ----------
    - [1]: Adrian D. Smith, Robert J. Ferguson. Minimum-phase signal
      calculation using the real cepstrum.
    - [2]: Soo-Chang Pei, Huei-Shan Lin. MINIMUM-PHASE FIR FILTER DESIGN USING
      REAL CEPSTRUM.

    """
    assert (
        type(sig) is ImpulseResponse
    ), "This is only valid for an impulse response"
    assert padding_factor > 1, "Padding factor should be at least 1"
    assert alpha <= 1.0 and alpha > 0.0, "Alpha must be in the range ]0, 1]"
    new_time_data = sig.time_data.copy()

    if alpha != 1.0:
        new_time_data *= (alpha ** (np.arange(new_time_data.shape[0])))[
            :, None
        ]

    if use_real_cepstrum:
        new_time_data = _min_phase_ir_from_real_cepstrum(
            new_time_data, padding_factor
        )
    else:
        length_fft = next_fast_length_fft(
            max(
                new_time_data.shape[0] * padding_factor,
                new_time_data.shape[0],
            ),
            False,
        )
        for ch in range(new_time_data.shape[1]):
            new_time_data[:, ch] = min_phase_scipy(
                sig.time_data[:, ch], method="hilbert", n_fft=length_fft
            )[: new_time_data.shape[0]]

    if alpha != 1.0:
        new_time_data *= (alpha ** (-np.arange(new_time_data.shape[0])))[
            :, None
        ]

    return sig.copy_with_new_time_data(new_time_data[: len(sig)])


def group_delay(
    signal: Signal,
    analytic_computation: bool = True,
    smoothing: int = 0,
    remove_ir_latency: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes and returns group delay.

    Parameters
    ----------
    signal : `Signal`
        Signal for which to compute group delay.
    analytic_computation : bool, optional
        When True, this implementation is used: \
        https://www.dsprelated.com/freebooks/filters/Phase_Group_Delay.html.
        Otherwise, the numerical gradient of the unwrapped phase response is
        used. Default: True.
    smoothing : int, optional
        Octave fraction by which to apply smoothing. `0` avoids any smoothing
        of the group delay. Default: `0`.
    remove_ir_latency : bool, optional
        If the signal is of type `"ir"` or `"rir"`, the impulse delay can be
        removed by analyzing the minimum phase equivalent. This uses the
        padding factor 8 by default. Default: `False`.

    Returns
    -------
    freqs : NDArray[np.float64]
        Frequency vector in Hz.
    group_delays : NDArray[np.float64]
        Matrix containing group delays in seconds with shape (gd, channel).

    """
    length_time_signal = (
        next_fast_length_fft(signal.time_data.shape[0] * 8, True)
        if remove_ir_latency
        else signal.time_data.shape[0]
    )
    td = _pad_trim(signal.time_data, length_time_signal)
    f = np.fft.rfftfreq(td.shape[0], 1 / signal.sampling_rate_hz)

    if not analytic_computation:
        spec_parameters = signal._spectrum_parameters
        signal.spectrum_method = SpectrumMethod.FFT
        sp = rfft_scipy(td, axis=0)
        signal._spectrum_parameters = spec_parameters

        if remove_ir_latency:
            assert (
                type(signal) is ImpulseResponse
            ), "This is only valid for an impulse response"
            sp = _remove_ir_latency_from_phase_min_phase(
                f, np.angle(sp), signal.time_data, signal.sampling_rate_hz, 1
            )
        group_delays = _group_delay_direct(sp, f[1] - f[0])
    else:
        group_delays = np.zeros((length_time_signal // 2 + 1, td.shape[1]))
        for n in range(signal.number_of_channels):
            b = td[:, n]
            if remove_ir_latency:
                b = b[max(int(np.argmax(np.abs(b))) - 1, 0) :]
            a = [1]
            _, group_delays[:, n] = _group_delay_filter(
                [b, a], len(f), signal.sampling_rate_hz
            )

    if smoothing != 0:
        group_delays = _fractional_octave_smoothing(
            group_delays, None, smoothing
        )

    return f, group_delays


def minimum_phase(
    signal: ImpulseResponse,
    use_real_cepstrum: bool = True,
    padding_factor: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gives back a matrix containing the minimum phase signal for each
    channel. Two methods are available for computing the minimum phase of a
    system: `'real cepstrum'` (windowing in the cepstral domain) or
    `'equiripple'` (for symmetric IR's, uses `scipy.signal.minimum_phase`).

    Parameters
    ----------
    signal : `Signal`
        IR for which to compute the minimum phase.
    use_real_cepstrum : bool, optional
        Set to True for general cases. If the IR is symmetric (like a
        linear-phase filter), False is recommended. Default: True.
    padding_factor : int, optional
        Zero-padding to a length corresponding to at least
        `current_length * padding_factor` can be done in order to avoid time
        aliasing errors. Default: 8.

    Returns
    -------
    f : NDArray[np.float64]
        Frequency vector.
    min_phases : NDArray[np.float64]
        Minimum phases as matrix with shape (phase, channel).

    """
    assert (
        type(signal) is ImpulseResponse
    ), "This is only valid for an impulse response"

    if not use_real_cepstrum:
        f = np.fft.rfftfreq(
            signal.time_data.shape[0], d=1 / signal.sampling_rate_hz
        )
        min_phases = np.zeros(
            (len(f), signal.number_of_channels), dtype="float"
        )
        for n in range(signal.number_of_channels):
            temp = min_phase_scipy(
                signal.time_data[:, n],
                method="hilbert",
                n_fft=padding_factor * len(signal),
            )
            min_phases[:, n] = np.angle(
                np.fft.rfft(_pad_trim(temp, signal.time_data.shape[0]))
            )
    else:
        sp = _get_minimum_phase_spectrum_from_real_cepstrum(
            signal.time_data, padding_factor
        )
        f = np.fft.fftfreq(sp.shape[0], 1 / signal.sampling_rate_hz)
        if sp.shape[0] % 2 == 0:
            f[sp.shape[0] // 2] *= -1
        inds = f >= 0
        f = f[inds]
        min_phases = np.angle(sp[inds, ...])
    return f, min_phases


def minimum_group_delay(
    signal: ImpulseResponse,
    smoothing: int = 0,
    padding_factor: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes minimum group delay of given IR using the real cepstrum method.

    Parameters
    ----------
    signal : `ImpulseResponse`
        IR for which to compute minimal group delay.
    smoothing : int, optional
        Octave fraction by which to apply smoothing. `0` avoids any smoothing
        of the group delay. Default: `0`.
    padding_factor : int, optional
        Zero-padding to a length corresponding to at least
        `current_length * padding_factor` can be done in order to avoid time
        aliasing errors. Default: 8.

    Returns
    -------
    f : NDArray[np.float64]
        Frequency vector.
    min_gd : NDArray[np.float64]
        Minimum group delays in seconds as matrix with shape (gd, channel).

    References
    ----------
    - https://www.roomeqwizard.com/help/help_en-GB/html/minimumphase.html

    """
    assert (
        type(signal) is ImpulseResponse
    ), "This is only valid for an impulse response"
    f, min_phases = minimum_phase(signal, padding_factor=padding_factor)
    min_gd = _group_delay_direct(min_phases, f[1] - f[0])
    if smoothing != 0:
        min_gd = _fractional_octave_smoothing(min_gd, None, smoothing)
    return f, min_gd


def excess_group_delay(
    signal: ImpulseResponse,
    smoothing: int = 0,
    remove_ir_latency: bool = False,
    analytic_computation: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes excess group delay of an IR.

    Parameters
    ----------
    signal : `ImpulseResponse`
        IR for which to compute minimal group delay.
    smoothing : int, optional
        Octave fraction by which to apply smoothing. `0` avoids any smoothing
        of the group delay. Default: `0`.
    remove_ir_latency : bool, optional
        When True, the impulse delay will be removed by checking the peak
        latency. Default: `False`.
    analytic_computation : bool, optional
        When True, the analytic computation of group delay is performed instead
        of the numerical one. This is significantly slower. Default: False.

    Returns
    -------
    f : NDArray[np.float64]
        Frequency vector.
    ex_gd : NDArray[np.float64]
        Excess group delays in seconds with shape (excess_gd, channel).

    References
    ----------
    - https://www.roomeqwizard.com/help/help_en-GB/html/minimumphase.html
    - No zero-padding is performed when computing the minimum-phase equivalent

    """
    assert (
        type(signal) is ImpulseResponse
    ), "This is only valid for an impulse response"
    f_min, min_gd = minimum_group_delay(signal, smoothing=0, padding_factor=1)
    f, gd = group_delay(
        signal,
        smoothing=0,
        analytic_computation=analytic_computation,
        remove_ir_latency=remove_ir_latency,
    )

    # Min GD has fast FFT length, GD has fast RFFT length
    # => Interpolate if they do not match
    if len(f) != len(f_min):
        gd = _interpolate_fr(f, gd, f_min, None, "linear")

    ex_gd = gd - min_gd

    if smoothing != 0:
        ex_gd = _fractional_octave_smoothing(ex_gd, None, smoothing)

    return f_min, ex_gd


def combine_ir_with_dirac(
    ir: ImpulseResponse,
    crossover_frequency: float,
    take_lower_band: bool,
    order: int = 8,
    normalization: str | float | None = None,
) -> ImpulseResponse:
    """Combine an IR with a perfect impulse at a given crossover frequency
    using a linkwitz-riley crossover. Forward-Backward filtering is done so
    that no phase distortion occurs. They can optionally be energy matched
    using RMS or peak value.

    Parameters
    ----------
    ir : `dsp.Signal`
        Impulse Response.
    crossover_frequency : float
        Frequency at which to combine the impulse response with the perfect
        impulse.
    take_lower_band : bool
        When `True`, the part below the crossover frequency corresponds to the
        passed impulse response and above corresponds to the perfect impulse.
        `False` delivers the opposite result.
    order : int, optional
        Crossover order. Default: 8.
    normalization : str, float, optional
        `'energy'` means that the band of the perfect dirac impulse is
        normalized so that it matches the energy contained in the band of the
        impulse response. `'peak'` means that peak value is matched for both
        bands. `None` avoids any normalization (Impulse response is always
        normalized prior to computation). Alternatively, a value in dB can be
        passed in order to scale the dirac part of the resulting impulse.
        Default: `None`.

    Returns
    -------
    combined_ir : `dsp.Signal`
        New IR.

    Notes
    -----
    - The algorithm checks for the fractional delay of the IR and adds a
      fractional delayed dirac. For ensuring good results, it is
      recommended that the IR has some delay, so that the first part of the
      added dirac impulse has time to grow smoothly.

    """
    assert (
        type(ir) is ImpulseResponse
    ), "This is only valid for an impulse response"
    if normalization is not None and type(normalization) is str:
        normalization = normalization.lower()
        assert normalization in (
            "energy",
            "peak",
        ), "Invalid normalization parameter"
    ir = normalize(ir, 0.0)
    latencies_samples = _get_fractional_impulse_peak_index(ir.time_data)

    # Make impulse
    imp = dirac(
        len(ir.time_data),
        delay_samples=0,
        number_of_channels=1,
        sampling_rate_hz=ir.sampling_rate_hz,
    )

    # Regard polarity
    polarity = np.ones(ir.number_of_channels)

    for ch in range(ir.number_of_channels):
        delay_seconds = latencies_samples[ch] / ir.sampling_rate_hz
        imp_ch = imp.get_channels(ch)
        imp_ch = fractional_delay(
            imp_ch, delay_seconds=delay_seconds, keep_length=True
        )
        imp = append_signals([imp, imp_ch])

        # Save polarity for each channel using sample prior to peak
        polarity[ch] *= np.sign(
            ir.time_data[int(latencies_samples[ch] + 0.5), ch]
        )
    imp.remove_channel(0)

    # Filter crossover for both
    fb = linkwitz_riley_crossovers(
        [crossover_frequency], order, ir.sampling_rate_hz
    )
    ir_multi = fb.filter_signal(ir, zero_phase=True)
    imp_multi = fb.filter_signal(imp, zero_phase=True)
    if take_lower_band:
        band_ir = 0
        band_imp = 1
    else:
        band_ir = 1
        band_imp = 0
    td_ir = ir_multi.bands[band_ir].time_data
    td_imp = imp_multi.bands[band_imp].time_data

    if normalization == "energy":
        ir_rms = np.sqrt(np.mean(td_ir**2, axis=0))
        imp_rms = np.sqrt(np.mean(td_imp**2, axis=0))
        td_imp *= ir_rms / imp_rms
    elif normalization == "peak":
        ir_peak = np.max(np.abs(td_ir), axis=0)
        imp_peak = np.max(np.abs(td_imp), axis=0)
        td_imp *= ir_peak / imp_peak
    elif type(normalization) in (float, int, np.floating, np.int_):
        td_imp *= from_db(normalization, True)

    # Combine
    combined_ir = ir.copy_with_new_time_data(
        td_ir + td_imp * polarity[None, ...]
    )
    return normalize(combined_ir, 0.0)


def ir_to_filter(
    signal: ImpulseResponse,
    channel: int | None = 0,
    phase_mode: str = "direct",
) -> Filter | FilterBank:
    """This function takes in an impulse response and turns the selected
    channel into an FIR filter. With `phase_mode` it is possible
    to use minimum phase or minimum linear phase.

    Parameters
    ----------
    signal : `Signal`
        Signal to be converted into a filter.
    channel : int, optional
        Channel of the signal to be used. If None, all channels are used and
        the return is a FilterBank with each channel as an FIR filter. This
        also applies for a signal with a single channel. Default: 0.
    phase_mode : {"direct", "min", "lin"} str, optional
        Phase of the FIR filter. Choose from "direct" (no changes to phase),
        "min" (minimum phase) or "lin" (minimum linear phase).
        Default: "direct".

    Returns
    -------
    filt : Filter or FilterBank
        (FIR) Filter from a single channel or FilterBank with FIR filters from
        each channel.

    """
    assert (
        type(signal) is ImpulseResponse
    ), "This is only valid for an impulse response"
    phase_mode = phase_mode.lower()
    assert phase_mode in (
        "direct",
        "min",
        "lin",
    ), f"""{phase_mode} is not valid. Choose from ('direct', 'min', 'lin')"""

    # Choose channel
    signal = signal.get_channels(channel) if channel is not None else signal

    # Change phase
    if phase_mode == "min":
        signal = min_phase_from_mag(
            Spectrum.from_signal(signal), signal.sampling_rate_hz, len(signal)
        )
    elif phase_mode == "lin":
        signal = lin_phase_from_mag(
            Spectrum.from_signal(signal), signal.sampling_rate_hz, len(signal)
        )

    filters = []
    for ch in signal:
        filt = Filter.from_ba(ch, [1.0], signal.sampling_rate_hz)
        if channel is not None:
            return filt
        filters.append(filt)
    return FilterBank(filters)


def filter_to_ir(fir: Filter | FilterBank) -> ImpulseResponse:
    """Takes in an FIR filter or multiple filters in a filter bank and converts
    them into an IR by taking its b coefficients.

    Parameters
    ----------
    fir : Filter or FilterBank
        Filter containing an FIR filter. In case of a FilterBank, all filters
        should be FIR.

    Returns
    -------
    new_sig : ImpulseResponse
        New IR. If the input was a FilterBank, the ImpulseResponse has multiple
        channels. Its length always corresponds to the longest filter.

    """
    if isinstance(fir, Filter):
        assert not fir.is_iir, "This is only valid for FIR filters"
        return ImpulseResponse.from_time_data(
            fir.ba[0].copy(), sampling_rate_hz=fir.sampling_rate_hz
        )
    elif isinstance(fir, FilterBank):
        assert all([not f.is_iir for f in fir]), "Filter types must be fir"
        assert (
            fir.same_sampling_rate
        ), "Only valid for filter banks with consistent sampling rate"
        length_samples = max([len(f) for f in fir])
        td = np.zeros((length_samples, len(fir)), dtype=np.float64)
        for ind, f in enumerate(fir):
            td[: len(f), ind] = f.ba[0].copy()
        return ImpulseResponse.from_time_data(td, fir.sampling_rate_hz)
    else:
        raise TypeError("Unsupported type")


def window_frequency_dependent(
    ir: ImpulseResponse,
    cycles: int,
    end_window_value_db: float = -50.0,
) -> Spectrum:
    """A spectrum with frequency-dependent windowing defined by cycles is
    returned. To this end, a variable gaussian window is applied.

    A width of 5 cycles means that there are 5 periods of each frequency
    before the window values hit `end_window_value_db`.

    The output spectrum can be converted to a time series with a IRFFT. Its
    frequency vector has linear resolution.

    Parameters
    ----------
    ir : `ImpulseResponse`
        Impulse response from which to extract the spectrum.
    cycles : int
        Number of cycles to include for each frequency bin. It defines
        the window lengths.
    end_window_value_db : float, optional
        This is the value that the gaussian window should have at its given
        width in dB. It must be below 0 dB. Default: -50.

    Returns
    -------
    Spectrum
        Complex spectrum.

    Notes
    -----
    - It is recommended that the impulse response has been already left- and
      right-windowed using, for instance, a tukey window. However, its length
      should be somewhat larger than the longest window (this depends on the
      number of cycles and lowest frequency).
    - The length of the IR should be as short as possible for a fast
      computation. This implementation will use numba, if available, for
      parallelizing the computation.
    - The implemented method is a straight-forward windowing in the time domain
      for each respective frequency bin. Warping the IR is another valid
      approach, but its frequency resolution will not be linear.

    """
    assert (
        type(ir) is ImpulseResponse
    ), "This is only valid for an impulse response"
    assert end_window_value_db < 0.0, "Window ends must be less than 0 dB"

    end_window_value = from_db(end_window_value_db, True)
    fs = ir.sampling_rate_hz

    # Avoid 0. frequency
    f = np.fft.rfftfreq(ir.length_samples, 1 / fs)[1:]
    cycles_per_freq_samples = np.round(fs / f * cycles).astype(int)
    spec = np.zeros(
        (len(f), ir.number_of_channels), dtype=np.complex128, order="C"
    )

    # Alpha such that window is exactly end_window_value after the number of
    # required samples for each frequency
    half = (ir.length_samples - 1) / 2
    alpha_factor = np.log(1 / (end_window_value) ** 2) ** 0.5 * half

    # Construct window vectors
    ind_max = np.argmax(np.abs(ir.time_data), axis=0)
    n = np.zeros_like(ir.time_data)
    for ch in range(ir.number_of_channels):
        n[:, ch] = np.arange(-ind_max[ch], ir.length_samples - ind_max[ch])

    # Precompute some window factors
    n = (-0.5 * (n / half) ** 2.0).astype(np.complex128)
    alpha = ((alpha_factor / cycles_per_freq_samples) ** 2.0).astype(
        np.complex128, order="C"
    )

    freqs_normalized = (f * (ir.length_samples / fs)).astype(
        np.complex128, order="C"
    )
    dft_factor = np.repeat(
        -2j
        * np.pi
        * np.linspace(0.0, 1.0, ir.length_samples, endpoint=False)[..., None],
        repeats=ir.number_of_channels,
        axis=1,
    ).astype(np.complex128, order="C")

    spec = _fdw_backend(
        ir.time_data.astype(np.complex128, order="C"),
        freqs_normalized,
        dft_factor,
        spec,
        alpha,
        n,
    )
    return Spectrum(np.hstack([0.0, f]), np.pad(spec, ((1, 0), (0, 0))))


def find_ir_latency(
    ir: ImpulseResponse, compare_to_min_phase_ir: bool = True
) -> NDArray[np.float64]:
    """Find the subsample maximum of each channel of the IR.

    Parameters
    ----------
    ir : `ImpulseResponse`
        Impulse response to find the maximum.
    compare_to_min_phase_ir : bool, optional
        When True, the latency is found by comparing the latency of the IR in
        relation to its minimum phase equivalent. When False, the peak in the
        time data is searched. Both cases are done with subsample accuracy. For
        the former, the padding factor 8 is used. Default: True.

    Returns
    -------
    latency_samples : NDArray[np.float64]
        Array with the position of each channel's latency in samples.

    """
    assert (
        type(ir) is ImpulseResponse
    ), "This is only valid for an impulse response"
    if compare_to_min_phase_ir:
        min_ir = min_phase_ir(ir)
        return latency(ir, min_ir, 1)[0]

    return _get_fractional_impulse_peak_index(ir.time_data, 1)


def harmonics_from_chirp_ir(
    ir: ImpulseResponse,
    chirp_range_hz: list,
    chirp_length_s: float,
    n_harmonics: int = 5,
    offset_percentage: float = 0.05,
) -> list[ImpulseResponse]:
    """Get the individual harmonics (distortion) IRs of an IR computed with
    an exponential chirp.

    Parameters
    ----------
    ir : `ImpulseResponse`
        Impulse response obtained through deconvolution with an exponential
        chirp.
    chirp_range_hz : list of length 2
        The frequency range of the chirp.
    chirp_length_s : float
        Length of chirp in seconds (without zero-padding).
    n_harmonics : int, optional
        Number of harmonics to analyze. Default: 5.
    offset_percentage : float, optional
        When this is larger than zero, each IR will also contain some samples
        prior to the impulse. Their amount corresponds to a percentage of the
        time length between that harmonic and their adjacent ones.
        All samples are gathered in a mutually exclusive manner, such they are
        never passed to two different harmonics. Default: 0.05.

    Returns
    -------
    harmonics : list[ImpulseResponse]
        List containing the IRs of each harmonic in ascending order. The
        fundamental is not in the list.

    Notes
    -----
    - This will only work if the IR was gained utilizing an exponential
      chirp that has also been zero padded during the deconvolution. This will
      not be checked in this function.

    """
    assert (
        type(ir) is ImpulseResponse
    ), "This is only valid for an impulse response"
    assert (
        offset_percentage < 1 and offset_percentage >= 0
    ), "Offset must be smaller than one"
    assert (
        ir.number_of_channels == 1
    ), "Only an IR with a single channel is supported"

    # Get offsets
    td = ir.time_data
    offsets = -np.argmax(np.abs(td), axis=0) + 1
    td = np.roll(td, offsets, axis=0)

    # Get times of each harmonic
    ts = _get_harmonic_times(chirp_range_hz, chirp_length_s, n_harmonics + 1)
    time_harmonics_samples = len(td) + (ts * ir.sampling_rate_hz + 0.5).astype(
        int
    )

    time_harmonics_samples = np.insert(time_harmonics_samples, 0, len(td))

    # Dummy to obtain all metadata of the IR
    ir_dummy = ir.copy_with_new_time_data(ir.time_data[:10])

    harmonics = []
    for nh in range(n_harmonics):
        max_ind = int(
            time_harmonics_samples[nh]
            - (time_harmonics_samples[nh] - time_harmonics_samples[nh + 1])
            * offset_percentage
        )
        min_ind = int(
            time_harmonics_samples[nh + 1]
            - (time_harmonics_samples[nh + 1] - time_harmonics_samples[nh + 2])
            * offset_percentage
        )
        snippet = td[min_ind:max_ind, 0]
        harmonics.append(ir_dummy.copy_with_new_time_data(snippet))
    return harmonics


def harmonic_distortion_analysis(
    ir: ImpulseResponse | list[ImpulseResponse],
    chirp_range_hz: list | None = None,
    chirp_length_s: float | None = None,
    n_harmonics: int | None = 8,
    smoothing: int = 12,
    generate_plot: bool = True,
) -> dict:
    """Analyze non-linear distortion coming from an IR measured with an
    exponential chirp. The range of the chirp and its length must be known.
    The distortion spectra of each harmonic, as well as THD+N and THD, are
    returned. Optionally, a plot can be generated.

    Parameters
    ----------
    ir : `ImpulseResponse` or list[`ImpulseResponse`]
        Impulse response. It should only have one channel. Alternatively,
        a list containing the fundamental IR and all harmonics can be passed,
        in which case `chirp_range_hz`, `chirp_length_s` and `n_harmonics`
        will be ignored or inferred. In the second case, no windowing or
        trimming will be applied to either the fundamental or the harmonics.
    chirp_range_hz : list
        List with length 2 containing the lowest and highest frequency of the
        exponential chirp.
    chirp_length_s : float
        Length of the chirp (time from lowest to highest frequency) in seconds.
    n_harmonics : int, optional
        Number of harmonics to analyze. Default: 8.
    smoothing : int, optional
        Smoothing as fraction of an octave band to apply to all spectra.
        Default: 12.
    generate_plot : bool, optional
        When `True`, a plot with all the distortion spectra is generated.
        Default: `True`.

    Returns
    -------
    dict
        A dictionary containing each spectrum is returned. Each item is of type
        Spectrum. Its keys are:
            - "1": spectrum of the fundamental.
            - "2": spectrum of the second harmonic.
            - "3": ...
            - "thd": Total harmonic distortion. The spectrum is shifted to the
              frequency that caused the distortion.
            - "thd_percent": Total harmonic distortion normalized by the linear
              response. It is shifted as `thd` and returned in percent.
            - "thd_n": Total harmonic distortion + noise. This spectrum is not
              shifted to the frequency that caused the distortion.
            - "plot": a list with matplotlib's [figure, axes]. This is only
              returned if the plot was generated.

    Notes
    -----
    - The scaling of the spectrum is always done as set with
      `set_spectrum_parameters()` of the original IR.
    - THD in percent is usually defined in audio by the amplitude ratios
      instead of the power ratios, as is common for other fields. See
      https://de.wikipedia.org/wiki/Total_Harmonic_Distortion.
    - Passing `chirp_range_hz` with a list of IRs will still have an effect on
      the upper limit frequency of each harmonic.

    """
    if type(ir) is list:
        for each_ir in ir:
            assert isinstance(each_ir, ImpulseResponse), "Unsupported type"
            assert (
                each_ir.number_of_channels == 1
            ), "Only single-channel IRs are supported"

        ir2 = ir.pop(0)
        ir2._spectrum_parameters["smoothing"] = smoothing

        harm = ir
        n_harmonics = len(harm)
        if chirp_range_hz is None:
            chirp_range_hz = [0, ir2.sampling_rate_hz // 2]

        passed_harmonics = True
    elif isinstance(ir, ImpulseResponse):
        assert (
            chirp_length_s is not None
            and chirp_range_hz is not None
            and n_harmonics is not None
        ), "Chirp parameters and number of harmonics cannot be None"

        # Get different harmonics
        harm = harmonics_from_chirp_ir(
            ir, chirp_range_hz, chirp_length_s, n_harmonics, 0.01
        )

        # Trim and window IR
        ir2 = ir.copy()
        start, stop, _ = _trim_ir(
            ir2.time_data[:, 0], ir.sampling_rate_hz, 10e-3
        )
        ir2.time_data = ir2.time_data[start:stop]
        ir2 = window_ir(ir2, len(ir2), constant_percentage=0.9)[0]
        ir2._spectrum_parameters["smoothing"] = smoothing

        passed_harmonics = False
    else:
        raise TypeError("Type for ir is not supported")

    # At least 5 Hz frequency resolution for base spectrum
    pad_length = max(ir2.sampling_rate_hz // 5, len(ir2)) - len(ir2)
    ir2.time_data = np.pad(ir2.time_data, ((0, pad_length), (0, 0)))

    # Accumulator for THD time samples and dictionary
    thd = np.zeros(np.sum([len(h) for h in harm]))
    pos_thd = len(thd)
    d: dict = {}

    # Spectrum of fundamental
    quadratic_spectrum = not ir2.spectrum_scaling.is_amplitude_scaling()
    freqs, base_spectrum = ir2.get_spectrum()
    d["1"] = Spectrum(
        freqs, (base_spectrum**0.5 if quadratic_spectrum else base_spectrum)
    )

    # Accumulator for spectrum of harmonics
    sp_thd = np.zeros(len(freqs))

    if generate_plot:
        fig, ax = ir2.plot_magnitude(
            smoothing=smoothing,
            normalize=MagnitudeNormalization.NoNormalization,
        )

    for i in range(len(harm)):
        if not passed_harmonics:
            harm[i] = window_ir(
                harm[i], len(harm[i]), constant_percentage=0.9
            )[0]
        harm[i].set_spectrum_parameters(**ir2._spectrum_parameters)
        f, sp = harm[i].get_spectrum()

        # Select frequencies that really were excited by chirp and fftshift
        # for true excitation frequency
        inds = f < chirp_range_hz[-1]
        f = f[inds]
        sp = sp[inds]
        f /= i + 2

        # Get power
        sp_power = (
            sp.squeeze() if quadratic_spectrum else np.abs(sp.squeeze()) ** 2
        )

        # Save in dictionary
        d[f"{i + 2}"] = Spectrum(f, sp**0.5 if quadratic_spectrum else sp)

        # Make plot
        if generate_plot:
            ax.plot(f, to_db(sp_power, False))

        # Accumulate time samples for THD+N
        thd[pos_thd - len(harm[i]) : pos_thd] = harm[i].time_data.squeeze()
        pos_thd -= len(harm[i])

        # Sum power of each harmonic
        sp_thd += interp1d(
            f,
            sp_power,
            kind="linear",
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )(freqs)

    # THD
    ind_end = np.argmin(np.abs(freqs - chirp_range_hz[-1] / 2))
    sp_thd = sp_thd[:ind_end]
    freqs_thd = freqs[:ind_end]
    if generate_plot:
        sp_thd[sp_thd == 0] = np.nan
        ax.plot(freqs_thd, to_db(sp_thd, False), label="THD")
        np.nan_to_num(sp_thd, False, 0)

    # THD+N
    thd_n = Signal(None, thd, ir2.sampling_rate_hz)
    thd_n.set_spectrum_parameters(**ir2._spectrum_parameters)
    f_thd_n, sp_thd_n = thd_n.get_spectrum()
    if not quadratic_spectrum:
        sp_thd_n = np.abs(sp_thd_n) ** 2.0

    if generate_plot:
        ax.plot(
            f_thd_n,
            to_db(sp_thd_n, False),
            label="THD+N",
        )
        ax.legend(
            ["Fundamental"]
            + [f"{i + 2} Harmonic" for i in range(n_harmonics)]
            + ["THD", "THD+N"]
        )
        d["plot"] = [fig, ax]

    d["thd_n"] = Spectrum(f_thd_n, sp_thd_n**0.5)
    d["thd"] = Spectrum(freqs_thd, sp_thd**0.5)
    d["thd_percent"] = Spectrum(
        d["thd"].frequency_vector_hz,
        # (Magnitude) ratio of `THD / fundamental`
        d["thd"].spectral_data
        / d["1"].get_interpolated_spectrum(
            d["thd"].frequency_vector_hz, SpectrumType.Magnitude
        )
        * 100.0,
    )

    return d


def trim_ir(
    ir: ImpulseResponse,
    channel: int | None = None,
    start_offset_s: float | None = 20e-3,
) -> tuple[ImpulseResponse, int, int]:
    """Trim an IR in the beginning and end. This method acts only on one
    channel and returns it trimmed. For defining the ending, a smooth envelope
    of the energy time curve (ETC) is used, as well as the assumption that the
    energy should decay monotonically after the impulse arrives. See notes for
    details.

    Parameters
    ----------
    ir : `ImpulseResponse`
        Impulse response to trim.
    channel : int, None, optional
        Channel to take from `rir`. Pass None to apply to all channels and
        return a multichannel signal that has the largest time boundaries found
        across all channels. Default: None.
    start_offset_s : float, None, optional
        This is the time prior to the peak value that is left after trimming.
        Pass 0 to start the IR one sample prior to peak value or a very big
        offset (or None) to avoid any trimming at the beginning. Default: 20e-3
        (20 milliseconds).

    Returns
    -------
    trimmed_ir : `ImpulseResponse`
        IR with the new length.
    start : int
        Start index of the trimmed IR in the original vector.
    stop : int
        Stop index of the trimmed IR in the original vector.

    Notes
    -----
    - The method employed for finding the ending of the IR works as follows:
        - A (hilbert) envelope is computed in dB (energy time curve). This is
          smoothed by exponential averaging with 20 ms.
        - Non-overlapping windows with lengths 10, 30, 50 and 80 ms are
          checked starting from the impulse and going forwards. The first
          window to contain more energy than the previous one is regarded as
          the end of the IR.
        - Pearson correlation coefficients (cc) of the energy decay for the
          segments obtained with each window size are computed. The final end
          point is selected following criteria:
            - If a good linear fit is obtained (cc < -0.95), it is used as
              the final point.
            - Else, if there are acceptable fits (cc < -0.9), the ending
              point is the averaged from these.
            - Else, if there are any fits with cc < -0.7, they are all averaged
              but the best one is weighted significantly stronger.
            - If no fit has cc < -0.7, all are averaged together with the
              total length of the IR weighted stronger than the other values.

    """
    # Pass a large offset that won't trim the start
    start_offset_s = (
        len(ir) / ir.sampling_rate_hz
        if start_offset_s is None
        else start_offset_s
    )
    assert start_offset_s >= 0, "Offset must be at least 0"

    # Single-channel case
    if channel is not None:
        trimmed_rir = ir.get_channels(channel)
        td = trimmed_rir.time_data.squeeze()
        start, stop, _ = _trim_ir(
            td,
            ir.sampling_rate_hz,
            start_offset_s,
        )
        trimmed_rir.time_data = td[start:stop]
        return trimmed_rir, start, stop

    starts = np.zeros(ir.number_of_channels, dtype=np.int_)
    stops = starts.copy()

    for ch in range(ir.number_of_channels):
        starts[ch], stops[ch], _ = _trim_ir(
            ir.time_data[:, ch],
            ir.sampling_rate_hz,
            start_offset_s,
        )

    start = int(np.min(starts))
    stop = int(np.max(stops))
    return (
        ir.copy_with_new_time_data(ir.time_data[start:stop, ...]),
        start,
        stop,
    )


def complex_smoothing(
    ir: ImpulseResponse,
    octave_fraction: float,
    smoothing_domain: SmoothingDomain,
    window: Window = Window.Hann,
) -> Spectrum:
    """Complex smoothing of an impulse response using logarithmic spacing given
    in octaves. This is done according to [1].

    Parameters
    ----------
    ir : ImpulseResponse
        Impulse response to apply smoothing to.
    octave_fraction : float
        Width of smoothing range in fraction of octaves.
    smoothing_domain : SmoothingDomain
        Domain to use during the smoothing step.
    window : Window
        Type of window to use.

    Returns
    -------
    Spectrum

    References
    ----------
    - [1]: GENERALIZED FRACTIONAL OCTAVE SMOOTHING OF  AUDIO / ACOUSTIC
      RESPONSES. PANAGIOTIS D. HATZIANTONIOU AND JOHN N. MOURJOPOULOS.

    """
    assert octave_fraction > 0.0, "Octave fraction must be greater than 0"
    f, sp = ir.get_spectrum()
    sp = sp.astype(sp.dtype, order="C")

    # Get a window prototype – mapping to logarithmic space is done through
    # interpolation
    window_values = window(3000, True).astype(np.float64, order="C")
    output_sp = np.zeros_like(sp, order="C")

    match smoothing_domain:
        case SmoothingDomain.RealImaginary:
            output_sp = _complex_smoothing_backend(
                octave_fraction, sp, output_sp, f, window_values
            )
        case SmoothingDomain.MagnitudePhase:
            sp = np.abs(sp) + 1j * np.unwrap(np.angle(sp), axis=0)
            output_sp = _complex_smoothing_backend(
                octave_fraction, sp, output_sp, f, window_values
            )
            output_sp = np.real(output_sp) * np.exp(1j * np.imag(output_sp))
        case SmoothingDomain.PowerPhase:
            sp = np.abs(sp) ** 2.0 + 1j * np.unwrap(np.angle(sp), axis=0)
            output_sp = _complex_smoothing_backend(
                octave_fraction, sp, output_sp, f, window_values
            )
            output_sp = np.real(output_sp) ** 0.5 * np.exp(
                1j * np.imag(output_sp)
            )
        case SmoothingDomain.Power:
            sp_modified = (np.abs(sp) ** 2.0).astype(np.complex128)
            output_sp = _complex_smoothing_backend(
                octave_fraction, sp_modified, output_sp, f, window_values
            )
            output_sp = np.real(output_sp) ** 0.5 * np.exp(1j * np.angle(sp))
        case SmoothingDomain.Magnitude:
            sp_modified = np.abs(sp).astype(np.complex128)
            output_sp = _complex_smoothing_backend(
                octave_fraction, sp_modified, output_sp, f, window_values
            )
            output_sp = np.real(output_sp) * np.exp(1j * np.angle(sp))
        case SmoothingDomain.EquivalentComplex:
            # Apply complex smoothing
            output_sp = _complex_smoothing_backend(
                octave_fraction, sp, output_sp, f, window_values
            )

            # Apply power smoothing
            output2 = np.zeros_like(output_sp)
            output2 = _complex_smoothing_backend(
                octave_fraction,
                (np.abs(sp) ** 2.0).astype(np.complex128, order="C"),
                output2,
                f,
                window_values,
            )

            # Combine power with phase of complex smoothing
            output_sp = np.real(output2) ** 0.5 * np.exp(
                1j * np.angle(output_sp)
            )
        case _:
            raise ValueError("Invalid smoothing domain")
    return Spectrum(f, output_sp)
