"""
Methods used for acquiring and windowing transfer functions
"""
import numpy as np
from scipy.signal import minimum_phase as min_phase_scipy
from scipy.signal import hilbert

from ._transfer_functions import (
    _spectral_deconvolve,
    _window_this_ir_tukey,
    _window_this_ir,
    _min_phase_ir_from_real_cepstrum,
    _get_minimum_phase_spectrum_from_real_cepstrum,
    _warp_time_series,
)
from ..classes import Signal, Filter
from ..classes._filter import _group_delay_filter
from .._general_helpers import _find_frequencies_above_threshold
from .._standard import _welch, _minimum_phase, _group_delay_direct, _pad_trim
from ..standard_functions import fractional_delay, merge_signals, normalize
from ..generators import dirac
from ..filterbanks import linkwitz_riley_crossovers
from ..room_acoustics._room_acoustics import _find_ir_start


def spectral_deconvolve(
    num: Signal,
    denum: Signal,
    mode: str = "regularized",
    start_stop_hz=None,
    threshold_db=-30,
    padding: bool = False,
    keep_original_length: bool = False,
) -> Signal:
    """Deconvolution by spectral division of two signals. If the denominator
    signal only has one channel, the deconvolution is done using that channel
    for all channels of the numerator.

    Parameters
    ----------
    num : `Signal`
        Signal to deconvolve from.
    denum : `Signal`
        Signal to deconvolve.
    mode : str, optional
        `'window'` uses a spectral window in the numerator. `'regularized'`
        uses a regularized inversion. `'standard'` uses direct deconvolution.
        Default: `'regularized'`.
    start_stop_hz : array, None, optional
        `'automatic'` uses a threshold dBFS to create a spectral
        window for the numerator or regularized inversion. Array of 2 or
        4 frequency points can be also manually given. `None` uses no
        spectral window. If mode is standard, start_stop_hz has to be set
        to `None`. Default: `None`.
    threshold_db : int, optional
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
    num = num.copy()
    denum = denum.copy()
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
    mode = mode.lower()
    assert mode in (
        "regularized",
        "window",
        "standard",
    ), f"{mode} is not supported. Use regularized, window or None"
    if mode == "standard":
        assert (
            start_stop_hz is None
        ), "No start_stop_hz vector can be passed when using standard mode"

    original_length = num.time_data.shape[0]

    if padding:
        num.time_data = _pad_trim(num.time_data, original_length * 2)
        denum.time_data = _pad_trim(denum.time_data, original_length * 2)
    fft_length = original_length * 2 if padding else original_length

    denum.set_spectrum_parameters(method="standard")
    _, denum_fft = denum.get_spectrum()
    num.set_spectrum_parameters(method="standard")
    freqs_hz, num_fft = num.get_spectrum()
    fs_hz = num.sampling_rate_hz

    new_time_data = np.zeros_like(num.time_data)

    for n in range(num.number_of_channels):
        n_denum = 0 if multichannel else n
        if mode != "standard":
            if start_stop_hz is None:
                start_stop_hz = _find_frequencies_above_threshold(
                    denum_fft[:, n_denum], freqs_hz, threshold_db
                )
            if len(start_stop_hz) == 2:
                temp = []
                temp.append(start_stop_hz[0] / np.sqrt(2))
                temp.append(start_stop_hz[0])
                temp.append(start_stop_hz[1])
                temp.append(np.min([start_stop_hz[1] * np.sqrt(2), fs_hz / 2]))
                start_stop_hz = temp
            elif len(start_stop_hz) == 4:
                pass
            else:
                raise ValueError(
                    "start_stop_hz vector should have 2 or 4" + " values"
                )
        new_time_data[:, n] = _spectral_deconvolve(
            num_fft[:, n],
            denum_fft[:, n_denum],
            freqs_hz,
            fft_length,
            start_stop_hz=start_stop_hz,
            mode=mode,
        )
    new_sig = Signal(
        None, new_time_data, num.sampling_rate_hz, signal_type="ir"
    )
    if padding:
        if keep_original_length:
            new_sig.time_data = _pad_trim(new_sig.time_data, original_length)
    return new_sig


def window_ir(
    signal: Signal,
    constant_percentage=0.75,
    exp2_trim: int = 13,
    window_type="hann",
    at_start: bool = True,
) -> tuple[Signal, np.ndarray]:
    """Windows an IR with trimming and selection of constant valued length.
    This is equivalent to a tukey window whose flanks can be selected to be
    any type. The peak of the impulse response is aligned to correspond to
    the first value with amplitude 1 of the window.

    Parameters
    ----------
    signal: `Signal`
        Signal to window
    constant_percentage: float, optional
        Percentage (between 0 and 1) of the window's length that should be
        constant value. Default: 0.75.
    exp2_trim: int, optional
        Exponent of two defining the length to which the IR should be
        trimmed. For avoiding trimming set to `None`. Default: 13.
    window_type: str, optional
        Window function to be used. Available selection from
        scipy.signal.windows: `barthann`, `bartlett`, `blackman`,
        `boxcar`, `cosine`, `hamming`, `hann`, `flattop`, `nuttall` and
        others. Pass a tuple with window type and extra parameters if needed.
        Default: `hann`.
    at_start: bool, optional
        Windows the start with a rising window as well as the end.
        Default: `True`.

    Returns
    -------
    new_sig : `Signal`
        Windowed signal. The used window is also saved under `new_sig.window`.
    start_positions_samples : `np.ndarray`
        This array contains the position index of the start of the IR in
        each channel of the original IR.

    Notes
    -----
    - The window flanks are adapted in case that the distance between impulse
      and start is not enough for the selected flank lengths (flank lengths
      depend on `constant_percentage` and `exp2_trim`).

    """
    assert signal.signal_type in (
        "rir",
        "ir",
    ), f"{signal.signal_type} is not a valid signal type. Use rir or ir."
    if exp2_trim is not None:
        total_length = int(2**exp2_trim)
    else:
        total_length = len(signal.time_data)
    new_time_data = np.zeros((total_length, signal.number_of_channels))
    start_positions_samples = np.zeros(signal.number_of_channels, dtype=int)

    window = np.zeros((total_length, signal.number_of_channels))
    for n in range(signal.number_of_channels):
        (
            new_time_data[:, n],
            window[:, n],
            start_positions_samples[n],
        ) = _window_this_ir_tukey(
            signal.time_data[:, n],
            total_length,
            window_type,
            exp2_trim,
            constant_percentage,
            at_start,
        )

    new_sig = Signal(
        None,
        new_time_data,
        signal.sampling_rate_hz,
        signal_type=signal.signal_type,
    )
    new_sig.set_window(window)
    return new_sig, start_positions_samples


def window_centered_ir(
    signal: Signal, total_length: int, window_type="hann"
) -> tuple[Signal, np.ndarray]:
    """This function windows an IR placing its peak in the middle. It trims
    it to the total length of the window or pads it to the desired length
    (padding in the end, window has `total_length`).

    Parameters
    ----------
    signal: `Signal`
        Signal to window
    total_length: int
        Total window length.
    window_type: str, optional
        Window function to be used. Available selection from
        scipy.signal.windows: `barthann`, `bartlett`, `blackman`,
        `boxcar`, `cosine`, `hamming`, `hann`, `flattop`, `nuttall` and
        others. Pass a tuple with window type and extra parameters if needed,
        like `('gauss', 8)`. Default: `hann`.

    Returns
    -------
    new_sig : `Signal`
        Windowed signal. The used window is also saved under `new_sig.window`.
    start_positions_samples : `np.ndarray`
        This array contains the position index of the start of the IR in
        each channel of the original IR.

    Notes
    -----
    - If the window seems truncated, it is because the length and peak position
      were longer than the IR, so that it had to be zero-padded to match the
      given length.

    """
    assert signal.signal_type in (
        "rir",
        "ir",
    ), f"{signal.signal_type} is not a valid signal type. Use rir or ir."

    new_time_data = np.zeros((total_length, signal.number_of_channels))
    start_positions_samples = np.zeros(signal.number_of_channels, dtype=int)
    window = np.zeros((total_length, signal.number_of_channels))

    for n in range(signal.number_of_channels):
        (
            new_time_data[:, n],
            window[:, n],
            start_positions_samples[n],
        ) = _window_this_ir(signal.time_data[:, n], total_length, window_type)

    new_sig = Signal(
        None,
        new_time_data,
        signal.sampling_rate_hz,
        signal_type=signal.signal_type,
    )
    new_sig.set_window(window)
    return new_sig, start_positions_samples


def compute_transfer_function(
    output: Signal,
    input: Signal,
    mode="h2",
    window_length_samples: int = 1024,
    spectrum_parameters: dict | None = None,
) -> tuple[Signal, np.ndarray]:
    """Gets transfer function H1, H2 or H3 (for stochastic signals).
    H1: for noise in the output signal. `Gxy/Gxx`.
    H2: for noise in the input signal. `Gyy/Gyx`.
    H3: for noise in both signals. `G_xy / abs(G_xy) * (G_yy/G_xx)**0.5`.
    If the input signal only has one channel, it is assumed to be the input
    for all of the channels of the output.

    Parameters
    ----------
    output : `Signal`
        Signal with output channels.
    input : `Signal`
        Signal with input channels.
    mode : str, optional
        Type of transfer function. `'h1'`, `'h2'` and `'h3'` are available.
        Default: `'h2'`.
    window_length_samples : int, optional
        Window length for the IR. Spectrum has the length
        window_length_samples//2 + 1. Default: 1024.
    spectrum_parameters : dict, optional
        Extra parameters for the computation of the cross spectral densities
        using welch's method. See `Signal.set_spectrum_parameters()`
        for details. Default: empty dictionary.

    Returns
    -------
    tf_sig : `Signal`
        Transfer functions as `Signal` object. Coherences are also computed
        and saved in the `Signal` object.
    tf : `np.ndarray`
        Complex transfer function as type `np.ndarray`.

    """
    mode = mode.casefold()
    assert mode in (
        "h1".casefold(),
        "h2".casefold(),
        "h3".casefold(),
    ), f"{mode} is not a valid mode. Use H1, H2 or H3"
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
    if spectrum_parameters is None:
        spectrum_parameters = {}
    assert (
        type(spectrum_parameters) is dict
    ), "Spectrum parameters should be passed as a dictionary"

    coherence = np.zeros(
        (window_length_samples // 2 + 1, output.number_of_channels)
    )
    tf = np.zeros(
        (window_length_samples // 2 + 1, output.number_of_channels),
        dtype="cfloat",
    )
    if multichannel:
        G_xx = _welch(
            input.time_data[:, 0],
            input.time_data[:, 0],
            input.sampling_rate_hz,
            window_length_samples=window_length_samples,
            **spectrum_parameters,
        )
    for n in range(output.number_of_channels):
        G_yy = _welch(
            output.time_data[:, n],
            output.time_data[:, n],
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
                input.time_data[:, n_input],
                input.sampling_rate_hz,
                window_length_samples=window_length_samples,
                **spectrum_parameters,
            )
        if mode == "h2".casefold():
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

        if mode == "h1".casefold():
            tf[:, n] = G_xy / G_xx
        elif mode == "h2".casefold():
            tf[:, n] = G_yy / G_yx
        elif mode == "h3".casefold():
            tf[:, n] = G_xy / np.abs(G_xy) * (G_yy / G_xx) ** 0.5
        coherence[:, n] = np.abs(G_xy) ** 2 / G_xx / G_yy
    tf_sig = Signal(
        None,
        np.fft.irfft(tf, axis=0),
        output.sampling_rate_hz,
        signal_type=mode.lower(),
    )
    tf_sig.set_coherence(coherence)
    return tf_sig, tf


def spectral_average(signal: Signal, normalize_energy: bool = True) -> Signal:
    """Averages all channels of a given IR using their magnitude and
    phase spectra and returns the averaged IR.

    Parameters
    ----------
    signal : `Signal`
        Signal with channels to be averaged over.
    normalize_energy : bool, optional
        When `True`, the energy of all spectra is normalized to the first
        channel's energy and then averaged. Default: `True`.

    Returns
    -------
    avg_sig : `Signal`
        Averaged signal.

    """
    assert signal.signal_type in ("rir", "ir"), (
        "Averaging is valid for signal types rir or ir and not "
        + f"{signal.signal_type}"
    )
    assert (
        signal.number_of_channels > 1
    ), "Signal has only one channel so no meaningful averaging can be done"

    l_samples = signal.time_data.shape[0]

    # Obtain channel magnitude and phase spectra
    _, sp = signal.get_spectrum()
    mag = np.abs(sp)
    pha = np.unwrap(np.angle(sp), axis=0)

    # Build averages
    new_mag = np.mean(mag, axis=1)
    if normalize_energy:
        norm = np.sum(new_mag**2, axis=0, keepdims=True)
        new_mag *= norm[0] / norm
    new_pha = np.mean(pha, axis=1)
    # New signal
    new_sp = new_mag * np.exp(1j * new_pha)

    # New time data and signal object
    new_time_data = np.fft.irfft(new_sp[..., None], n=l_samples, axis=0)
    avg_sig = signal.copy()
    avg_sig.time_data = new_time_data
    if hasattr(avg_sig, "window"):
        del avg_sig.window
    return avg_sig


def min_phase_from_mag(
    spectrum: np.ndarray, sampling_rate_hz: int, signal_type: str = "ir"
):
    """Returns a minimum-phase signal from a magnitude spectrum using
    the discrete hilbert transform.

    Parameters
    ----------
    spectrum : `np.ndarray`
        Spectrum with only positive frequencies and 0.
    sampling_rate_hz : int
        Signal's sampling rate in Hz.
    signal_type : str, optional
        Type of signal to be returned. Default: `'ir'`.

    Returns
    -------
    sig_min_phase : `Signal`
        Signal with same magnitude spectrum but minimum phase.

    References
    ----------
    - https://en.wikipedia.org/wiki/Minimum_phase

    """
    if spectrum.ndim < 2:
        spectrum = spectrum[..., None]
    assert spectrum.ndim < 3, "Spectrum should have shape (bins, channels)"
    if spectrum.shape[0] < spectrum.shape[1]:
        spectrum = spectrum.T
    spectrum = np.abs(spectrum)
    min_spectrum = np.empty(spectrum.shape, dtype="cfloat")
    phase = _minimum_phase(spectrum, False)
    min_spectrum = spectrum * np.exp(1j * phase)
    time_data = np.fft.irfft(min_spectrum, axis=0)
    sig_min_phase = Signal(
        None,
        time_data=time_data,
        sampling_rate_hz=sampling_rate_hz,
        signal_type=signal_type,
    )
    return sig_min_phase


def lin_phase_from_mag(
    spectrum: np.ndarray,
    sampling_rate_hz: int,
    group_delay_ms: str | float = "minimum",
    check_causality: bool = True,
    signal_type: str = "ir",
) -> Signal:
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
    spectrum : `np.ndarray`
        Spectrum with only positive frequencies and 0.
    sampling_rate_hz : int
        Signal's sampling rate in Hz.
    group_delay_ms : str or float, optional
        Constant group delay that the phase should have for all channels
        (in ms). Pass `'minimum'` to create a signal with the minimum linear
        phase possible (that is different for each channel).
        Default: `'minimum'`.
    check_causality : bool, optional
        When `True`, it is assessed for each channel that the given group
        delay is not lower than the minimum group delay. Default: `True`.
    signal_type : str, optional
        Type of signal to be returned. Default: `'ir'`.

    Returns
    -------
    sig_lin_phase : `Signal`
        Signal with same magnitude spectrum but linear phase.

    """
    # Check spectrum
    if spectrum.ndim < 2:
        spectrum = spectrum[..., None]
    assert spectrum.ndim < 3, "Spectrum should have shape (bins, channels)"
    if spectrum.shape[0] < spectrum.shape[1]:
        spectrum = spectrum.T
    spectrum = np.abs(spectrum)

    # Check group delay ms parameter
    minimum_group_delay = False
    if type(group_delay_ms) is str:
        group_delay_ms = group_delay_ms.lower()
        assert (
            group_delay_ms == "minimum"
        ), "Group delay should be set to minimum"
        minimum_group_delay = True
    elif type(group_delay_ms) in (float, int):
        group_delay_ms /= 1000
    else:
        raise TypeError("group_delay_ms must be either str, float or int")

    # Frequency vector
    f_vec = np.fft.rfftfreq(spectrum.shape[0] * 2 - 1, 1 / sampling_rate_hz)
    delta_f = f_vec[1] - f_vec[0]

    # New spectrum
    lin_spectrum = np.empty(spectrum.shape, dtype="cfloat")
    for n in range(spectrum.shape[1]):
        if check_causality or minimum_group_delay:
            min_phase = _minimum_phase(spectrum[:, n], False)
            min_gd = _group_delay_direct(min_phase, delta_f)
            gd = np.max(min_gd) + 1e-3  # add 1 ms as safety factor
            if check_causality and type(group_delay_ms) is not str:
                assert gd <= group_delay_ms, (
                    f"Given group delay {group_delay_ms * 1000} ms is lower "
                    + f"than minimal group delay {gd * 1000} ms for "
                    + f"channel {n}"
                )
                gd = group_delay_ms
        else:
            gd = group_delay_ms
        lin_spectrum[:, n] = spectrum[:, n] * np.exp(
            -1j * 2 * np.pi * f_vec * gd
        )
    time_data = np.fft.irfft(lin_spectrum, axis=0)
    sig_lin_phase = Signal(
        None,
        time_data=time_data,
        sampling_rate_hz=sampling_rate_hz,
        signal_type=signal_type,
    )
    return sig_lin_phase


def min_phase_ir(sig: Signal, method: str = "real cepstrum") -> Signal:
    """Returns same IR with minimum phase. Three methods are available for
    computing the minimum phase version of the IR: `'real cepstrum'` (using
    filtering the real-cepstral domain), `'log hilbert'` (obtaining the phase
    from the hilbert transformed magnitude response) and `'equiripple'` (for
    symmetric IR, uses `scipy.signal.minimum_phase`).

    For general cases, `'real cepstrum'` and `'log hilbert'` deliver similar
    results.

    Parameters
    ----------
    sig : `Signal`
        IR for which to compute minimum phase IR.
    method : str, optional
        For general cases, `'real cepstrum'` and `'log hilbert'` can be used
        and render similar results. If the IR is symmetric (like a
        linear-phase filter), `'equiripple'` is recommended.
        Default: `'real cepstrum'`.

    Returns
    -------
    min_phase_sig : `Signal`
        Minimum-phase IR as time signal.

    """
    # Computation
    assert sig.signal_type in (
        "rir",
        "ir",
    ), "Signal type must be either rir or ir"
    method = method.lower()
    assert method in ("real cepstrum", "log hilbert", "equiripple"), (
        f"{method} is not valid. Use either real cepstrum, log hilbert or "
        + "equiripple"
    )
    new_time_data = np.zeros_like(sig.time_data)

    if method == "real cepstrum":
        new_time_data = _min_phase_ir_from_real_cepstrum(sig.time_data)
    else:
        _, min_phases = minimum_phase(sig, method=method)
        _, sp = sig.get_spectrum()
        new_time_data = np.fft.irfft(
            np.abs(sp) * np.exp(1j * min_phases), axis=0
        )

    min_phase_sig = sig.copy()
    min_phase_sig.time_data = new_time_data
    if hasattr(min_phase_sig, "window"):
        del min_phase_sig.window
    return min_phase_sig


def group_delay(
    signal: Signal, method="matlab"
) -> tuple[np.ndarray, np.ndarray]:
    """Computes and returns group delay.

    Parameters
    ----------
    signal : Signal
        Signal for which to compute group delay.
    method : str, optional
        `'direct'` uses gradient with unwrapped phase. `'matlab'` uses
        this implementation:
        https://www.dsprelated.com/freebooks/filters/Phase_Group_Delay.html.
        Default: `'matlab'`.

    Returns
    -------
    freqs : `np.ndarray`
        Frequency vector in Hz.
    group_delays : `np.ndarray`
        Matrix containing group delays in seconds with shape (gd, channel).

    """
    method = method.lower()
    assert method in (
        "direct",
        "matlab",
    ), f"{method} is not valid. Use direct or matlab"

    signal.set_spectrum_parameters("standard")
    f, sp = signal.get_spectrum()
    if method == "direct":
        group_delays = np.zeros((sp.shape[0], sp.shape[1]))
        for n in range(signal.number_of_channels):
            group_delays[:, n] = _group_delay_direct(sp[:, n], f[1] - f[0])
    else:
        group_delays = np.zeros(
            (signal.time_data.shape[0] // 2 + 1, signal.time_data.shape[1])
        )
        for n in range(signal.number_of_channels):
            b = signal.time_data[:, n].copy()
            a = [1]
            _, group_delays[:, n] = _group_delay_filter(
                [b, a], len(b) // 2 + 1, signal.sampling_rate_hz
            )
    return f, group_delays


def minimum_phase(
    signal: Signal, method: str = "real cepstrum"
) -> tuple[np.ndarray, np.ndarray]:
    """Gives back a matrix containing the minimum phase signal for each
    channel. Three methods are available for computing the minimum phase of a
    system: `'real cepstrum'` (windowing in the cepstral domain),
    `'log hilbert'` (from the magnitude response of a system), `'equiripple'`
    (for symmetric IR's, uses `scipy.signal.minimum_phase`).

    Parameters
    ----------
    signal : `Signal`
        IR for which to compute the minimum phase.
    method : str, optional
        Selects the method to use. `'real cepstrum'` and `'log hilbert'` are
        of general use and render similar results. `'equiripple'` is for
        symmetric IR (e.g. linear-phase FIR filters).

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    min_phases : `np.ndarray`
        Minimum phases as matrix with shape (phase, channel).

    """
    assert signal.signal_type in (
        "rir",
        "ir",
        "h1",
        "h2",
        "h3",
    ), "Signal type must be rir or ir"
    method = method.lower()
    assert method in (
        "real cepstrum",
        "log hilbert",
        "equiripple",
    ), f"{method} is not valid. Use real cepstrum, log hilbert or equiripple"

    if method == "equiripple":
        f = np.fft.rfftfreq(
            signal.time_data.shape[0], d=1 / signal.sampling_rate_hz
        )
        min_phases = np.zeros(
            (len(f), signal.number_of_channels), dtype="float"
        )
        for n in range(signal.number_of_channels):
            temp = min_phase_scipy(
                signal.time_data[:, n], method="hilbert", n_fft=None
            )
            min_phases[:, n] = np.angle(
                np.fft.rfft(_pad_trim(temp, signal.time_data.shape[0]))
            )
    elif method == "log hilbert":
        signal.set_spectrum_parameters("standard")
        f, sp = signal.get_spectrum()
        min_phases = _minimum_phase(np.abs(sp), unwrapped=False)
    else:
        sp = _get_minimum_phase_spectrum_from_real_cepstrum(signal.time_data)
        f = np.fft.fftfreq(
            signal.time_data.shape[0], 1 / signal.sampling_rate_hz
        )
        if sp.shape[0] % 2 == 0:
            f[sp.shape[0] // 2] *= -1
        inds = f >= 0
        f = f[inds]
        min_phases = np.angle(sp[inds, ...])
    return f, min_phases


def minimum_group_delay(
    signal: Signal, method: str = "real cepstrum"
) -> tuple[np.ndarray, np.ndarray]:
    """Computes minimum group delay of given IR.

    Parameters
    ----------
    signal : `Signal`
        IR for which to compute minimal group delay.
    method : str, optional
        Select method for computing the minimum phase. It might be either
        `'real cepstrum'` or `'log hilbert'`. Default: `'real cepstrum'`.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    min_gd : `np.ndarray`
        Minimum group delays in seconds as matrix with shape (gd, channel).

    References
    ----------
    - https://www.roomeqwizard.com/help/help_en-GB/html/minimumphase.html

    """
    assert signal.signal_type in ("rir", "ir"), "Only valid for rir or ir"
    f, min_phases = minimum_phase(signal, method=method)
    min_gd = np.zeros_like(min_phases)
    for n in range(signal.number_of_channels):
        min_gd[:, n] = _group_delay_direct(min_phases[:, n], f[1] - f[0])
    return f, min_gd


def excess_group_delay(
    signal: Signal, method: str = "real cepstrum"
) -> tuple[np.ndarray, np.ndarray]:
    """Computes excess group delay of an IR.

    Parameters
    ----------
    signal : `Signal`
        IR for which to compute minimal group delay.
    method : str, optional
        Select method for computing the minimum phase. It might be either
        `'real cepstrum'` or `'log hilbert'`. Default: `'real cepstrum'`.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    ex_gd : `np.ndarray`
        Excess group delays in seconds with shape (excess_gd, channel).

    References
    ----------
    - https://www.roomeqwizard.com/help/help_en-GB/html/minimumphase.html

    """
    assert signal.signal_type in ("rir", "ir"), "Only valid for rir or ir"
    f, min_gd = minimum_group_delay(signal, method)
    f, gd = group_delay(signal)
    ex_gd = gd - min_gd
    return f, ex_gd


def combine_ir_with_dirac(
    ir: Signal,
    crossover_frequency: float,
    take_lower_band: bool,
    order: int = 8,
    normalization: str | None = None,
) -> Signal:
    """Combine an IR with a perfect impulse at a given crossover frequency
    using a linkwitz-riley crossover. Forward-Backward filtering is done so
    that no phase distortion occurs and the given filter order is doubled.
    They can optionally be energy matched using RMS or peak value.

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
        Crossover order. This is doubled due to forward-backward filtering.
        Default: 8.
    normalization : str, optional
        `'energy'` means that the band of the perfect dirac impulse is
        normalized so that it matches the energy contained in the band of the
        impulse response. `'peak'` means that peak value is matched for both
        bands. `None` avoids any normalization (Impulse response is always
        normalized prior to computation). Default: `None`.

    Returns
    -------
    combined_ir : `dsp.Signal`
        New IR.

    """
    assert ir.signal_type in ("rir", "ir"), "Only valid for rir or ir"
    if normalization is not None:
        normalization = normalization.lower()
    assert normalization in (
        "energy",
        "peak",
        None,
    ), "Invalid normalization parameter"
    ir = normalize(ir)
    latencies_samples = find_ir_latency(ir)

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
        imp = merge_signals(imp, imp_ch)

        # Save polarity for each channel using sample prior to peak
        polarity[ch] *= np.sign(ir.time_data[int(latencies_samples[ch]), ch])
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

    # Combine
    combined_ir = ir.copy()
    combined_ir.time_data = td_ir + td_imp * polarity[None, ...]
    combined_ir = normalize(combined_ir)
    return combined_ir


def ir_to_filter(
    signal: Signal, channel: int = 0, phase_mode: str = "direct"
) -> Filter:
    """This function takes in a signal with type `'ir'` or `'rir'` and turns
    the selected channel into an FIR filter. With `phase_mode` it is possible
    to use minimum phase or minimum linear phase.

    Parameters
    ----------
    signal : `Signal`
        Signal to be converted into a filter.
    channel : int, optional
        Channel of the signal to be used. Default: 0.
    phase_mode : {'direct', 'min', 'lin'} str, optional
        Phase of the FIR filter. Choose from `'direct'` (no changes to phase),
        `'min'` (minimum phase) or `'lin'` (minimum linear phase).
        Default: `'direct'`.

    Returns
    -------
    filt : `Filter`
        FIR filter object.

    """
    assert signal.signal_type in ("ir", "rir", "h1", "h2", "h3"), (
        f"{signal.signal_type} is not valid. Use one of "
        + """('ir', 'rir', 'h1', 'h2', 'h3')"""
    )
    assert (
        channel < signal.number_of_channels
    ), f"Signal does not have a channel {channel}"
    phase_mode = phase_mode.lower()
    assert phase_mode in (
        "direct",
        "min",
        "lin",
    ), f"""{phase_mode} is not valid. Choose from ('direct', 'min', 'lin')"""

    # Choose channel
    signal = signal.get_channels(channel)

    # Change phase
    if phase_mode == "min":
        f, sp = signal.get_spectrum()
        signal = min_phase_from_mag(np.abs(sp), signal.sampling_rate_hz)
    elif phase_mode == "lin":
        f, sp = signal.get_spectrum()
        signal = lin_phase_from_mag(np.abs(sp), signal.sampling_rate_hz)
    b = signal.time_data[:, 0]
    a = [1]
    filt = Filter(
        "other", {"ba": [b, a]}, sampling_rate_hz=signal.sampling_rate_hz
    )
    return filt


def filter_to_ir(fir: Filter) -> Signal:
    """Takes in an FIR filter and converts it into an IR by taking its
    b coefficients.

    Parameters
    ----------
    fir : `Filter`
        Filter containing an FIR filter.

    Returns
    -------
    new_sig : `Signal`
        New IR signal.

    """
    assert (
        fir.filter_type == "fir"
    ), "This is only valid is only available for FIR filters"
    b, _ = fir.get_coefficients(mode="ba")
    new_sig = Signal(
        None,
        b,
        sampling_rate_hz=fir.sampling_rate_hz,
        signal_type="ir",
        signal_id="IR from FIR filter",
    )
    return new_sig


def window_frequency_dependent(
    ir: Signal,
    cycles: int,
    channel: int | None = None,
    frequency_range_hz: list | None = None,
):
    """A spectrum with frequency-dependent windowing defined by cycles is
    returned. To this end, a variable gaussian window is applied.

    A width of 5 cycles means that there are 5 periods of each frequency
    before the window values hit 0.5, i.e., -6 dB.

    This is computed only for real-valued signals (positive frequencies). No
    scaling is applied to the spectrum.

    Parameters
    ----------
    ir : `Signal`
        Impulse response from which to extract the spectrum.
    cycles : int
        Number of cycles to include for each frequency bin. It defines
        the window lengths.
    channel : int, optional
        Selected channel to compute the spectrum. Pass `None` to take all
        channels. Default: `None`.
    frequency_range_hz : list of length 2, optional
        Frequency range to extract spectrum. Use `None` to compute the whole
        spectrum. Default: `None`.

    Returns
    -------
    f : `np.ndarray`
        Frequency vector.
    spec : `np.ndarray`
        Spectrum with shape (frequency, channel).

    Notes
    -----
    - It is recommended that the impulse response has been already left- and
      right-windowed using, for instance, a tukey window. However, its length
      should be somewhat larger than the longest window (this depends on the
      number of cycles and lowest frequency).
    - The length of the IR should be a power of 2 and not very long in general
      to speed up the computation.
    - The implemented method is a straight-forward windowing in the time domain
      for each respective frequency bin. Warping the IR is a more flexible
      approach but not necessarily faster for IR with short lengths
      corresponding to powers of 2.

    """
    assert ir.signal_type in ("rir", "ir"), "Only valid for rir or ir"
    fs = ir.sampling_rate_hz
    if frequency_range_hz is not None:
        assert len(frequency_range_hz) == 2
        frequency_range_hz = np.sort(frequency_range_hz)
    else:
        frequency_range_hz = [0, fs // 2]

    if channel is None:
        channel = np.arange(ir.number_of_channels)
    else:
        channel = np.atleast_1d(channel)

    td = ir.time_data[:, channel]

    f = np.fft.rfftfreq(td.shape[0], 1 / fs)
    inds = (f > frequency_range_hz[0]) & (f < frequency_range_hz[1])
    inds_f = np.arange(len(f))[inds]
    f = f[inds]

    # Samples for each frequency according to number of cycles
    cycles_per_freq_samples = np.round(fs / f * cycles).astype(int)

    spec = np.zeros((len(f), td.shape[1]), dtype="cfloat")

    half = (td.shape[0] - 1) / 2
    alpha_factor = np.log(4) ** 0.5 * half
    for ind, ind_f in enumerate(inds_f):
        for ch in range(td.shape[1]):
            # Construct window centered around impulse
            ind_max = np.argmax(np.abs(td[:, ch]))
            n = np.arange(-ind_max, td.shape[0] - ind_max)

            # Alpha such that window is exactly 0.5 after the number of
            # required samples for each frequency
            alpha = alpha_factor / cycles_per_freq_samples[ind]

            w = np.exp(-0.5 * (alpha * n[: td.shape[0]] / half) ** 2)
            spec[ind, ch] = np.fft.rfft(w * td[:, ch])[ind_f]
    return f, spec


def warp_ir(
    ir: Signal,
    warping_factor: float,
    shift_ir: bool = True,
    total_length: int | None = None,
):
    """Compute the IR in the warped-domain as explained by [1].

    To warp a signal, pass a negative `warping_factor`. To unwarp it, use a the
    same positive `warping_factor`.

    Parameters
    ----------
    ir : `Signal`
        Impulse response to (un)warp.
    warping_factor : float
        Warping factor. It has to be in the range ]-1; 1[.
    shift_ir : bool, optional
        Since the warping of an IR is not shift-invariant (see [2]), it is
        recommended to place the start of the IR at the first index. When
        `True`, the first sample to surpass -20 dBFS (relative to peak) is
        shifted to the beginning and the previous samples are sent to the
        end of the signal. `False` avoids any manipulation. Default: `True`.
    total_length : int, optional
        Total length to use for the warped signal. If `None`, the original
        length is maintained. Default: `None`.

    Returns
    -------
    f_unwarped : float
        Frequency that remained unwarped after transformation.
    warped_ir : `Signal`
        The same IR with warped or dewarped time vector.

    Notes
    -----
    - Depending on the signal length, this might be a slow computation.
    - Frequency-dependent windowing can be easily done in the warped domain.
      This is not the approach used in `window_frequency_dependent()`, but
      it can be achieved with this function. See [2] for more details.

    References
    ----------
    - [1]: Härmä, Aki & Karjalainen, Matti & Avioja, Lauri & Välimäki, Vesa &
      Laine, Unto & Huopaniemi, Jyri. (2000). Frequency-Warped Signal
      Processing for Audio Applications. Journal of the Audio Engineering
      Society. 48. 1011-1031.
    - [2]: M. Karjalainen and T. Paatero, "Frequency-dependent signal
      windowing," Proceedings of the 2001 IEEE Workshop on the Applications of
      Signal Processing to Audio and Acoustics (Cat. No.01TH8575), New Platz,
      NY, USA, 2001, pp. 35-38, doi: 10.1109/ASPAA.2001.969536.

    """
    assert ir.signal_type in ("rir", "ir"), "Signal has to be an IR or a RIR"
    assert np.abs(warping_factor) < 1, "Warping factor has to be in ]-1; 1["

    td = ir.time_data
    if shift_ir:
        for ch in range(ir.number_of_channels):
            start = _find_ir_start(td[:, ch], -20)
            td[:, ch] = np.roll(td[:, ch], -start)

    if total_length is None:
        total_length = td.shape[0]

    td = _warp_time_series(td[:total_length, ...], warping_factor)
    warped_ir = ir.copy()
    warped_ir.time_data = td

    f_unwarped = ir.sampling_rate_hz / 2 / np.pi * np.arccos(warping_factor)

    return f_unwarped, warped_ir


def find_ir_latency(ir: Signal) -> np.ndarray:
    """Find the subsample maximum of each channel of the IR using the root of
    the analytical function. This value can be associated with the latency
    of the impulse response.

    Parameters
    ----------
    ir : `Signal`
        Impulse response to find the maximum.

    Returns
    -------
    latency_samples : `np.ndarray`
        Array with the position of each channel's maximum in samples.

    """
    assert ir.signal_type in ("rir", "ir"), "Only valid for rir or ir"
    # Get maximum with fractional precision by finding the root of the complex
    # part of the analytical signal
    delay_samples = np.argmax(np.abs(ir.time_data), axis=0).astype(int)
    h = hilbert(ir.time_data, axis=0)
    point_around = 1
    x = np.arange(-point_around, point_around)

    latency_samples = np.zeros(ir.number_of_channels)

    for ch in range(ir.number_of_channels):
        pol = np.polyfit(
            x,
            np.imag(
                h[
                    delay_samples[ch]
                    - point_around : delay_samples[ch]
                    + point_around,
                    ch,
                ]
            ),
            1,
        )
        fractional_delay_samples = np.roots(pol).squeeze()
        latency_samples[ch] = delay_samples[ch] + fractional_delay_samples
    return latency_samples
