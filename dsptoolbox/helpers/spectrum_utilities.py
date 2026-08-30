from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from ..standard.enums import (
    InterpolationConversion,
    InterpolationKind,
    MagnitudeNormalization,
    SpectrumScaling,
)
from .gain_and_level import from_db, to_db
from .other import find_nearest_points_index_in_vector
from .smoothing import _fractional_octave_smoothing


def _wrap_phase(phase_vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wraps phase between [-np.pi, np.pi[ after it has been unwrapped.
    This works for 1D and 2D arrays, more dimensions have not been tested.

    Parameters
    ----------
    phase_vector : NDArray[np.float64]
        Phase vector for which to wrap the phase.

    Returns
    -------
    NDArray[np.float64]
        Wrapped phase vector.

    """
    return (phase_vector + np.pi) % (2 * np.pi) - np.pi


def _get_normalization_offset_db(
    normalize: MagnitudeNormalization,
    f_hz: NDArray[np.float64],
    magnitude_db: NDArray[np.float64],
    energy_db: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Per-channel offset in dB to subtract from a magnitude spectrum in dB.

    Parameters
    ----------
    normalize : MagnitudeNormalization
        Type of normalization to apply.
    f_hz : NDArray[np.float64]
        Frequency vector, needed for the 1 kHz normalizations.
    magnitude_db : NDArray[np.float64]
        Magnitude spectrum in dB with shape (frequency, channel).
    energy_db : NDArray[np.float64], None, optional
        Mean energy per channel in dB, needed for the energy normalizations.
        Default: None.

    Returns
    -------
    NDArray[np.float64]
        Offset per channel, to be subtracted as `magnitude_db - offset`.

    """
    number_of_channels = magnitude_db.shape[1]

    match normalize:
        case MagnitudeNormalization.OneKhz:
            return _get_exact_gain_1khz(f_hz, magnitude_db)
        case MagnitudeNormalization.OneKhzFirstChannel:
            return np.ones(number_of_channels) * _get_exact_gain_1khz(
                f_hz, magnitude_db[:, 0]
            )
        case MagnitudeNormalization.Max:
            return np.max(magnitude_db, axis=0)
        case MagnitudeNormalization.MaxFirstChannel:
            return np.ones(number_of_channels) * np.max(magnitude_db[:, 0], axis=0)
        case MagnitudeNormalization.Energy:
            assert energy_db is not None, "Energy normalization needs energy_db"
            return energy_db
        case MagnitudeNormalization.EnergyFirstChannel:
            assert energy_db is not None, "Energy normalization needs energy_db"
            return np.ones(number_of_channels) * energy_db[0]
        case MagnitudeNormalization.NoNormalization:
            return np.zeros(number_of_channels)
        case _:
            raise ValueError("No valid normalization")


def _get_exact_gain_1khz(
    f: NDArray[np.float64], sp_db: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Uses linear interpolation in the power domain to get the exact gain
    value at 1 kHz.

    Parameters
    ----------
    f : NDArray[np.float64]
        Frequency vector.
    sp_db : NDArray[np.float64]
        Spectrum in dB. It can have multiple dimensions, but the first
        dimension is always used (it must be the frequency dimension).

    Returns
    -------
    NDArray[np.float64]
        Interpolated value in dB.

    Notes
    -----
    - The two neighbouring bins are converted to their power representation
      before interpolating, since averaging a logarithmic quantity does not
      preserve the underlying power.

    """
    assert np.min(f) < 1e3 and np.max(f) >= 1e3, (
        "No gain at 1 kHz can be obtained because it is outside the "
        + "given frequency vector"
    )
    # Get nearest value just before
    ind = find_nearest_points_index_in_vector(1e3, f).squeeze()
    if f[ind] > 1e3:
        ind -= 1
    sp_power = from_db(sp_db, False)
    return to_db(
        (sp_power[ind + 1] - sp_power[ind]) / (f[ind + 1] - f[ind]) * (1e3 - f[ind])
        + sp_power[ind],
        False,
    )


@overload
def _get_normalized_spectrum(
    f: NDArray[np.float64],
    spectra: NDArray[np.complex128 | np.float64],
    is_amplitude_scaling: bool,
    f_range_hz: tuple[float, float] | None,
    normalize: MagnitudeNormalization,
    smoothing: float,
    phase: Literal[False],
    calibrated_data: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def _get_normalized_spectrum(
    f: NDArray[np.float64],
    spectra: NDArray[np.complex128 | np.float64],
    is_amplitude_scaling: bool,
    f_range_hz: tuple[float, float] | None,
    normalize: MagnitudeNormalization,
    smoothing: float,
    phase: Literal[True],
    calibrated_data: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...


def _get_normalized_spectrum(
    f: NDArray[np.float64],
    spectra: NDArray[np.complex128 | np.float64],
    is_amplitude_scaling: bool,
    f_range_hz: tuple[float, float] | None,
    normalize: MagnitudeNormalization,
    smoothing: float,
    phase: bool,
    calibrated_data: bool,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """This function gives a normalized magnitude spectrum in dB with frequency
    vector for a given range. It is also smoothed. Use `None` for the
    spectrum without f_range_hz.

    Parameters
    ----------
    f : NDArray[np.float64]
        Frequency vector.
    spectra : NDArray[np.complex128 | np.complex128]
        Spectrum matrix. It can be the power or amplitude representation.
        Complex spectrum is assumed to have amplitude scaling.
    is_amplitude_scaling : bool
        Information about whether the spectrum is scaled as an amplitude or
        power.
    f_range_hz : tuple[float, float], None
        Range of frequencies to get the normalized spectrum back.
    normalize : MagnitudeNormalization
        Normalize spectrum (per channel).
    smoothing : float
        1/smoothing-fractional octave band smoothing for magnitude spectra.
        Pass `0` for no smoothing.
    phase : bool
        When `True`, phase spectra are also returned. Smoothing is also
        applied to the unwrapped phase.
    calibrated_data : bool
        When `True`, it is assumed that the time data has been calibrated
        to be in Pascal so that it is scaled by p0=20e-6 Pa.

    Returns
    -------
    f : NDArray[np.float64]
        Frequency vector.
    mag_spectra : NDArray[np.float64]
        Magnitude spectrum matrix.
    phase_spectra : NDArray[np.float64]
        Phase spectrum matrix, only returned when `phase=True`.

    Notes
    -----
    - The spectrum is clipped according to `tools.to_db()`.

    """
    # Shaping
    one_dimensional = False
    if spectra.ndim < 2:
        spectra = spectra[..., None]
        one_dimensional = True
    # Check for complex spectrum if phase is required
    if phase:
        assert np.iscomplexobj(spectra), (
            "Phase computation is not " + "possible since the spectra are not complex"
        )
    # Factor
    if is_amplitude_scaling:
        scale_factor = (
            20e-6
            if calibrated_data and normalize == MagnitudeNormalization.NoNormalization
            else 1
        )
    else:
        scale_factor = (
            4e-10
            if calibrated_data and normalize == MagnitudeNormalization.NoNormalization
            else 1
        )

    if f_range_hz is not None:
        assert len(f_range_hz) == 2, (
            "Frequency range must have only " + "a lower and an upper bound"
        )
        f_range_hz = np.sort(f_range_hz)
        ids = find_nearest_points_index_in_vector(f_range_hz, f)
        id1 = ids[0]
        id2 = ids[1] + 1  # Contains endpoint
    else:
        id1 = 0
        id2 = len(f)

    spectra = spectra[id1:id2]
    mag_spectra = np.abs(spectra)
    f = f[id1:id2]

    if smoothing != 0:
        if is_amplitude_scaling:
            mag_spectra = (
                _fractional_octave_smoothing(mag_spectra, None, smoothing)
                if is_amplitude_scaling
                # Smoothing always in amplitude representation
                else (
                    _fractional_octave_smoothing(mag_spectra**0.5, None, smoothing) ** 2
                )
            )

    mag_spectra_db = to_db(mag_spectra / scale_factor, is_amplitude_scaling, 500)

    mag_spectra_db -= _get_normalization_offset_db(
        normalize,
        f,
        mag_spectra_db,
        to_db(
            np.mean(
                mag_spectra**2.0 if is_amplitude_scaling else mag_spectra,
                axis=0,
            ),
            False,
        ),
    )[None, :]

    if phase:
        phase_spectra = np.angle(spectra)
        if smoothing != 0:
            phase_spectra = _wrap_phase(
                _fractional_octave_smoothing(
                    np.unwrap(phase_spectra, axis=0), None, smoothing
                )
            )

    if one_dimensional:
        mag_spectra_db = np.squeeze(mag_spectra_db)
        if phase:
            phase_spectra = np.squeeze(phase_spectra)

    if phase:
        return f, mag_spectra_db, phase_spectra

    return f, mag_spectra_db


def _correct_for_real_phase_spectrum(
    phase_spectrum: NDArray[np.float64],
) -> NDArray[np.float64]:
    """This function takes in a phase spectrum and corrects it to be for a real
    signal (assuming the last frequency bin corresponds to nyquist, i.e., time
    data had an even length). This effectively adds a small linear phase offset
    so that the phase at nyquist is either 0 or np.pi.

    Parameters
    ----------
    phase_spectrum : NDArray[np.float64]
        Phase to be corrected. It is assumed that its last element
        corresponds to the nyquist frequency.

    Returns
    -------
    NDArray[np.float64]
        Phase spectrum that can correspond to a real signal.

    """
    factor = phase_spectrum[-1] % np.pi

    # Single dimension
    if phase_spectrum.ndim == 1:
        return (
            phase_spectrum
            - np.linspace(0, 1, len(phase_spectrum), endpoint=True) * factor
        )

    # Two dims
    assert phase_spectrum.ndim == 2, "More than 2 dimensions are not supported"
    return phase_spectrum - (
        np.repeat(
            np.linspace(0, 1, len(phase_spectrum), endpoint=True)[..., None],
            phase_spectrum.shape[1],
            axis=1,
        )
        * factor[None, ...]
    )


def _scale_spectrum(
    spectrum: NDArray[np.float64] | NDArray[np.complex128],
    scaling: SpectrumScaling,
    time_length_samples: int,
    sampling_rate_hz: int,
    window: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Scale the spectrum directly from the unscaled ("backward" normalization)
    (R)FFT. If a window was applied, it is necessary to compute the right
    scaling factor.

    Parameters
    ----------
    spectrum : NDArray[np.float64] | NDArray[np.complex128]
        Spectrum to scale. It is assumed that the frequency bins are along
        the first dimension. No FFT normalization should have been applied to
        it.
    scaling : SpectrumScaling
        Type of scaling to use. Using a power representation will returned the
        squared spectrum.
    time_length_samples : int
        Original length of the time data.
    sampling_rate_hz : int
        Sampling rate.
    window : NDArray[np.float64], None, optional
        Applied window when obtaining the spectrum. It is necessary to compute
        the correct scaling factor. In case of None, "boxcar" window is
        assumed. Default: None.

    Returns
    -------
    NDArray[np.float64] | NDArray[np.complex128]
        Scaled spectrum

    Notes
    -----
    - The amplitude spectrum shows the RMS value of each frequency in the
      signal.
    - Integrating the power spectral density over the frequency spectrum
      delivers the total energy contained in the signal (parseval's theorem).

    """
    assert time_length_samples in (
        (spectrum.shape[0] - 1) * 2,
        spectrum.shape[0] * 2 - 1,
    ), "Time length does not match"

    factor = scaling.get_scaling_factor(time_length_samples, sampling_rate_hz, window)

    # One-sided fix for DC and Nyquist (assuming input was linear)
    spectrum[0] /= 2**0.5
    if time_length_samples % 2 == 0:
        spectrum[-1] /= 2**0.5

    # Amplitude vs. Power
    if not scaling.is_amplitude_scaling():
        spectrum = np.abs(spectrum) ** 2

    spectrum *= factor

    return spectrum


def _interpolate_fr(
    f_interp: NDArray[np.float64],
    fr_interp: NDArray[np.float64],
    f_target: NDArray[np.float64],
    conversion: InterpolationConversion | None = None,
    interpolation_kind: InterpolationKind = InterpolationKind.Linear,
) -> NDArray[np.float64]:
    """Interpolate one frequency response to a new frequency vector.

    Parameters
    ----------
    f_interp : NDArray[np.float64]
        Frequency vector of the frequency response that should be interpolated.
    fr_interp : NDArray[np.float64]
        Frequency response to be interpolated.
    f_target : NDArray[np.float64]
        Target frequency vector.
    conversion : InterpolationConversion, None, optional
        Convert between amplitude, power or dB representation during the
        interpolation step. For instance, `DbToPower` means input in dB,
        interpolation in power spectrum, output in dB. Pass None to avoid any
        conversion. Default: None.
    interpolation_kind : InterpolationKind, optional
        Type of interpolation to use. Default: Linear.

    Returns
    -------
    NDArray[np.float64]
        New interpolated frequency response corresponding to `f_target` vector.

    Notes
    -----
    - The input is always assumed to be already sorted.
    - In case `f_target` has values outside the boundaries of `f_interp`,
      0 is used as the fill value. For interpolation in dB, fill values are
      the vector's edges.
    - The interpolation is always done along the first (outer) axis or the
      vector.
    - When converting to dB, the default clipping value of `to_db` is used.
    - Theoretical thoughts on interpolating an amplitude or power
      frequency response:

        - Using complex and dB values during interpolation are not very precise
          when comparing the results in terms of the amplitude or power
          spectrum.
        - Interpolation can be done with amplitude or power representation with
          similar precision.
        - Changing the frequency resolution in a linear scale means zero-
          padding or trimming the underlying time series. For an amplitude
          representation , i.e. spectrum or spectral density, the values must
          be scaled using the factor `old_length/new_length`. This ensures that
          the RMS values (amplitude spectrum) are still correct, and that
          integrating the new power spectral density still renders the total
          signal's energy truthfully, i.e. parseval's theorem would still hold.
          For the power representation, it also applies with the same squared
          factor.
        - A direct FFT-result which is not in physical units needs rescaling
          depending on the normalization scheme used during the FFT -> IFFT (in
          the complex/amplitude representation):

              - Forward: scaling factor `old_length/new_length`.
              - Backward: no rescaling.
              - Orthogonal: scaling factor `(old_length/new_length)**0.5`

        - Interpolating the (amplitude or power) spectrum to a logarithmic-
          spaced frequency vector can be done without rescaling (the underlying
          transformation in the time domain would be warping). Doing so for the
          (amplitude or power) spectral density only retains its validity if
          the new spectrum is weighted exponentially with increasing frequency
          since each bin contains the energy of a larger “frequency band”
          (this changes the physical units of the spectral density). Doing so
          ensures that integrating the power spectral density over frequency
          still retains the energy of the signal (parseval).
        - Assuming a different time window in each frequency resolution would
          require knowing the specific windows in order to rescale correctly.
          Assuming the same time window while zero-padding in the time domain
          would mean that no rescaling has to be applied.

    """

    fill_value = (0.0, 0.0)
    y = fr_interp.copy()

    # Conversion if necessary
    if conversion is not None:
        if conversion == InterpolationConversion.PowerToAmplitude:
            y **= 0.5
        elif conversion == InterpolationConversion.AmplitudeToPower:
            y **= 2.0
        elif conversion.input_is_db():
            y = from_db(y, conversion == InterpolationConversion.DbToAmplitude)
        else:
            y = to_db(y, conversion == InterpolationConversion.AmplitudeToDb)
            fill_value = (y[0], y[-1])

    interpolated = interp1d(
        f_interp,
        y,
        kind=interpolation_kind.to_scipy_str(),
        copy=False,
        bounds_error=False,
        assume_sorted=True,
        fill_value=fill_value,
        axis=0,
    )(f_target)

    # Back conversion if activated
    if conversion is not None:
        if conversion == InterpolationConversion.PowerToAmplitude:
            interpolated **= 2.0
        elif conversion == InterpolationConversion.AmplitudeToPower:
            interpolated **= 0.5
        elif conversion.input_is_db():
            interpolated = to_db(
                interpolated, conversion == InterpolationConversion.DbToAmplitude
            )
        else:
            interpolated = from_db(
                interpolated, conversion == InterpolationConversion.AmplitudeToDb
            )

    return interpolated


def _warp_frequency_vector(
    freqs_hz: NDArray[np.float64], sampling_rate_hz: int, warping_factor: float
) -> NDArray[np.float64]:
    """Warp a frequency vector as shown in [1].

    Parameters
    ----------
    freqs_hz : NDArray[np.float64]
        Frequency vector to warp.
    sampling_rate_hz : int
        Sampling rate to assume during warping.
    warping_factor : float
        Warping factor. It must be between ]-1;1[.

    References
    ----------
    - [1]: Germán Ramos, José J. López, Basilio Pueo. Cascaded warped-FIR and
      FIR filter structure for loudspeaker equalization with low computational
      cost requirements. Digital Signal Processing, Volume 19, Issue 3, 2009,
      Pages 393-409, ISSN 1051-2004, https://doi.org/10.1016/j.dsp.2008.01.003.

    Notes
    -----
    - The formula presented in [1] has been modified with a negative sign for
      lambda in order to match the warping formulation used in this python package.
    - Negative lambda values increase the resolution for lower frequencies, while
      positive values expand higher frequencies.

    """
    assert np.abs(warping_factor) < 1.0, "Warping factor must be between ]-1;1["
    omega = 2 * np.pi * freqs_hz / sampling_rate_hz
    return freqs_hz + sampling_rate_hz / np.pi * np.arctan(
        -warping_factor * np.sin(omega) / (1 + warping_factor * np.cos(omega))
    )
