import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .gain_and_level import from_db, to_db
from .other import find_nearest_points_index_in_vector
from .smoothing import _fractional_octave_smoothing
from ..standard.enums import MagnitudeNormalization, SpectrumScaling


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


def _get_exact_gain_1khz(
    f: NDArray[np.float64], sp_db: NDArray[np.float64]
) -> float:
    """Uses linear interpolation to get the exact gain value at 1 kHz.

    Parameters
    ----------
    f : NDArray[np.float64]
        Frequency vector.
    sp : NDArray[np.float64]
        Spectrum. It can be in dB or not. It can have multiple dimensions, but
        the first dimension is always used (it must be the frequency
        dimension).

    Returns
    -------
    float
        Interpolated value.

    """
    assert np.min(f) < 1e3 and np.max(f) >= 1e3, (
        "No gain at 1 kHz can be obtained because it is outside the "
        + "given frequency vector"
    )
    # Get nearest value just before
    ind = find_nearest_points_index_in_vector(1e3, f).squeeze()
    if f[ind] > 1e3:
        ind -= 1
    return (sp_db[ind + 1] - sp_db[ind]) / (f[ind + 1] - f[ind]) * (
        1e3 - f[ind]
    ) + sp_db[ind]


def _get_normalized_spectrum(
    f,
    spectra: NDArray[np.complex128 | np.float64],
    is_amplitude_scaling: bool,
    f_range_hz: list[float] | None,
    normalize: MagnitudeNormalization,
    smoothing: int,
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
    f_range_hz : array-like with length 2
        Range of frequencies to get the normalized spectrum back.
    normalize : MagnitudeNormalization
        Normalize spectrum (per channel).
    smoothing : int
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
            "Phase computation is not "
            + "possible since the spectra are not complex"
        )
    # Factor
    if is_amplitude_scaling:
        scale_factor = (
            20e-6
            if calibrated_data
            and normalize == MagnitudeNormalization.NoNormalization
            else 1
        )
    else:
        scale_factor = (
            4e-10
            if calibrated_data
            and normalize == MagnitudeNormalization.NoNormalization
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
                else (
                    # Smoothing always in amplitude representation
                    _fractional_octave_smoothing(
                        mag_spectra**0.5, None, smoothing
                    )
                    ** 2
                )
            )

    mag_spectra_db = to_db(
        mag_spectra / scale_factor, is_amplitude_scaling, 500
    )

    match normalize:
        case MagnitudeNormalization.OneKhz:
            normalization_db = np.array(
                [
                    _get_exact_gain_1khz(f, mag_spectra_db[:, i])
                    for i in range(spectra.shape[1])
                ]
            )
        case MagnitudeNormalization.OneKhzFirstChannel:
            normalization_db = np.ones(
                spectra.shape[1]
            ) * _get_exact_gain_1khz(f, mag_spectra_db[:, 0])
        case MagnitudeNormalization.Max:
            normalization_db = np.max(mag_spectra_db, axis=0)
        case MagnitudeNormalization.MaxFirstChannel:
            normalization_db = np.max(
                mag_spectra_db[:, 0], axis=0, keepdims=True
            )
        case MagnitudeNormalization.Energy:
            normalization_db = to_db(
                np.mean(
                    mag_spectra**2.0 if is_amplitude_scaling else mag_spectra,
                    axis=0,
                ),
                False,
            )
        case MagnitudeNormalization.EnergyFirstChannel:
            normalization_db = to_db(
                np.mean(
                    (
                        mag_spectra[:, 0] ** 2.0
                        if is_amplitude_scaling
                        else mag_spectra
                    ),
                    axis=0,
                    keepdims=True,
                ),
                False,
            )
        case MagnitudeNormalization.NoNormalization:
            normalization_db = np.zeros(mag_spectra_db.shape[1])
        case _:
            raise ValueError("No valid normalization")

    mag_spectra_db -= normalization_db[None, :]

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


def _correct_for_real_phase_spectrum(phase_spectrum: NDArray[np.float64]):
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

    factor = scaling.get_scaling_factor(
        time_length_samples, sampling_rate_hz, window
    )

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
    mode: str | None = None,
    interpolation_scheme: str = "linear",
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
    mode : str {"db2amplitude", "amplitude2db", "power2db",\
            "power2amplitude", "amplitude2power"}, None, optional
        Convert between amplitude, power or dB representation during the\
        interpolation step. For instance, using the modes "db2power" means\
        input in dB, interpolation in power spectrum, output in dB. Available\
        modes are "db2amplitude", "amplitude2db", "power2db",\
        "power2amplitude", "amplitude2power". Pass None to avoid any\
        conversion. Default: None.
    interpolation_scheme : str {"linear", "quadratic", "cubic"}, optional
        Type of interpolation to use. See `scipy.interpolation.interp1d` for\
        details. Choose from "quadratic" or "cubic" splines, or "linear".\
        Default: "linear".

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
    if mode is not None:
        mode = mode.lower()
        if mode == "power2amplitude":
            y **= 0.5
        elif mode == "amplitude2power":
            y **= 2.0
        elif mode[:3] == "db2":
            y = from_db(y, "amplitude" in mode)
        elif mode[-3:] == "2db":
            y = to_db(y, "amplitude" in mode)
            fill_value = (y[0], y[-1])
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

    interpolated = interp1d(
        f_interp,
        y,
        kind=interpolation_scheme,
        copy=False,
        bounds_error=False,
        assume_sorted=True,
        fill_value=fill_value,
        axis=0,
    )(f_target)

    # Back conversion if activated
    if mode is not None:
        if mode == "power2amplitude":
            interpolated **= 2.0
        elif mode == "amplitude2power":
            interpolated **= 0.5
        elif mode[:3] == "db2":
            interpolated = to_db(interpolated, "amplitude" in mode)
        elif mode[-3:] == "2db":
            interpolated = from_db(interpolated, "amplitude" in mode)

    return interpolated
