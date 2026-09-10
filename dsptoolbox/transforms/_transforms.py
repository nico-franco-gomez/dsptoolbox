"""Backend for special module."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import get_window, lfilter

_warp_time_series_rust: Callable | None
try:
    from .._rust import warp_time_series as _warp_time_series_rust  # noqa: I001
except ImportError:
    _warp_time_series_rust = None

_squeeze_scalogram_rust: Callable | None
try:
    from .._rust import squeeze_scalogram as _squeeze_scalogram_rust  # noqa: I001
except ImportError:
    _squeeze_scalogram_rust = None


def _pitch2frequency(tuning_a_hz: float = 440.0) -> NDArray[np.float64]:
    """This function returns a vector having frequencies for pitches
    0 to 127 (Midi compatible), where 0 is C0.

    Parameters
    ----------
    tuning_a_hz : float, optional
        Tuning for A4 in Hz. Default: 440.

    Returns
    -------
    freqs : NDArray[np.float64]
        Frequencies for each pitch. It always has length 128.

    """
    N = 128
    return tuning_a_hz * 2 ** ((np.arange(N) - 69) / 12)


class Wavelet:
    """Base class for a wavelet function."""

    def __init__(self) -> None:
        """Constructor for the base wavelet class. It's not supposed to be
        used directly.

        """
        pass

    def get_base_wavelet(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Abstract method to get the mother wavelet. It must be implemented
        in each Wavelet class.

        """
        raise NotImplementedError("Wavelet function has not been implemented")

    def get_wavelet(
        self, f: float | NDArray[np.float64], fs: int
    ) -> NDArray[np.float64] | list[NDArray[np.float64]]:
        """Abstract method to get the sampled wavelet. It must be implemented
        in each Wavelet class.

        """
        raise NotImplementedError("Wavelet function has not been implemented")

    def get_center_frequency(self) -> float:
        """Returns the center frequency of the wavelet (normalized,
        i.e., with fs=1).

        """
        x, func = self.get_base_wavelet()
        ind = np.argmax(np.abs(np.fft.fft(func)))
        # Maybe for some wavelets it might be necessary to mirror around nyquist
        domain = x[-1] - x[0]
        return ind / domain

    def get_scale_lengths(
        self, frequencies: NDArray[np.float64], fs: int
    ) -> NDArray[np.int_]:
        """Returns the lengths of the queried frequencies.

        Parameters
        ----------
        frequencies : NDArray[np.float64]
            Frequencies for which to scale the wavelet.
        fs : int
            Sampling rate in Hz.

        Returns
        -------
        NDArray[np.float64]
            Lengths of wavelets in samples.

        """
        scales = np.atleast_1d(self.get_center_frequency() / frequencies * fs)
        x, _ = self.get_base_wavelet()
        return (scales * (x[-1] - x[0]) + 1).astype(int)


class MorletWavelet(Wavelet):
    """Complex morlet wavelet."""

    def __init__(
        self,
        b: float | None = None,
        h: float | None = None,
        scale: float = 1.0,
        precision_bounds: float = 1e-5,
        step: float = 5e-3,
        interpolation: bool = True,
    ) -> None:
        """Instantiate a complex morlet wavelet based on the given parameters.
        Bandwidth can be defined through `b` or `h` (see Notes for the
        difference).

        Parameters
        ----------
        b : float, optional
            Bandwidth for the wavelet. Defines effectively the opening of the
            gaussian. Default: `None`.
        h : float, optional
            Alternative definition for `b`, see Notes. It overwrites `b` when
            different than `None`. Default: `None`.
        scale : float, optional
            Scale for the base wavelet. Default: 1.
        precision_bounds : float, optional
            Precision for the bounds of the mother wavelet. This is the
            smallest value that must be reached at the bounds,
            approaching zero. Default: 1e-5.
        step : float, optional
            Step x for the mother wavelet. Small values are recommended
            if there are no memory constraints. Default: 5e-3.
        interpolation : bool, optional
            When `True`, linear interpolation is activated when sampling
            scales from the motherwavelet. This improves the result but also
            increases the computational load. Default: `True`.

        Notes
        -----
        - The relation between `b` and `h` is b=h**2/(4*np.log(2)). `h` is
          the full-width at half-maximum (FWHM) as defined by [1].

        References
        ----------
        - [1]: Michael X Cohen, A better way to define and describe Morlet
          wavelets for time-frequency analysis.

        """
        assert b is not None or h is not None, "Either b or h must be passed"
        if h is not None:
            self.b = h**2 / np.log(2) / 4
        else:
            self.b = b
        self.scale = scale

        t = np.sqrt(self.b * np.log(1 / precision_bounds))
        self.bounds = [-t, t]

        self.step = step
        self.interpolation = interpolation

    def _get_x(self) -> NDArray[np.float64]:
        """Returns x vector for the mother wavelet."""
        return np.arange(self.bounds[0], self.bounds[1] + self.step, self.step)

    def get_base_wavelet(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return complex morlet wavelet."""
        x = self._get_x()
        return x, 1 / np.sqrt(np.pi * self.b) * np.exp(
            2j * np.pi / self.scale * x
        ) * np.exp(-(x**2) / self.b)

    def get_center_frequency(self) -> float:
        """Return center frequency for the complex morlet wavelet."""
        return 1 / self.scale

    def get_wavelet(
        self, f: float | NDArray[np.float64], fs: int
    ) -> NDArray[np.float64] | list[NDArray[np.float64]]:
        """Return wavelet scaled for a specific frequency and sampling rate.
        The wavelet values can also be linearly interpolated for a higher
        accuracy at the expense of computation time.

        Parameters
        ----------
        f : float or NDArray[np.float64]
            Queried frequency or array of frequencies.
        fs : int
            Sampling rate in Hz.

        Returns
        -------
        wave : NDArray[np.float64] or list of NDArray[np.float64]
            Wavelet function. It is either a 1d-array for a single frequency
            or a list of arrays for multiple frequencies.

        """
        scales = np.atleast_1d(self.get_center_frequency() / f * fs)
        x, base = self.get_base_wavelet()
        wave = []

        for scale in scales:
            inds = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * self.step)
            if self.interpolation:
                wavef = self._get_interpolated_wave(base, inds)
            else:
                # 0-th interpolation
                inds = inds.astype(int)
                inds = inds[inds < len(base)]
                wavef = base[inds]

            # Accumulate or return directly
            if len(scales) == 1:
                return wavef
            else:
                wave.append(wavef)
        return wave

    def _get_interpolated_wave(
        self, base: NDArray[np.float64], inds: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Return the wavelet function for a selection of index using
        linear interpolation.

        """
        # Truncate indices and select only valid ones
        trunc = inds.astype(int)
        trunc = trunc[trunc < len(base)]

        accumulator = np.zeros(len(trunc), dtype=np.complex128)

        for i in range(len(trunc) - 1):
            if trunc[i] + 1 >= len(base):
                accumulator[i] = 0.0
                continue

            accumulator[i] = base[trunc[i]] + (base[trunc[i] + 1] - base[trunc[i]]) * (
                inds[i] - trunc[i]
            )
        accumulator[-1] = base[trunc[-1]]
        return accumulator


def _squeeze_scalogram_python(
    scalogram: NDArray[np.complex128],
    freqs: NDArray[np.float64],
    fs: int,
    delta_w: float = 0.05,
    apply_frequency_normalization: bool = False,
    gradient: NDArray[np.complex128] | None = None,
) -> NDArray[np.complex128]:
    """Synchrosqueeze a scalogram.

    Parameters
    ----------
    scalogram : NDArray[np.float64]
        Complex scalogram from the CWT with shape (frequency, time sample,
        channel).
    freqs : NDArray[np.float64]
        Frequency vector.
    fs : int
        Sampling rate in Hz.
    delta_w : float, optional
        Maximum relative difference in frequency allowed in the phase
        transform for taking summing the result of the scalogram. If it's
        too small, it might lead to significant energy leaks. Default: 0.05.
    apply_frequency_normalization : bool, optional
        When `True`, each scale is scaled by taking into account the
        normalization as shown in Eq. (2.4) of [1]. `False` does not apply
        any normalization. Default: `False`.

    Returns
    -------
    sync : NDArray[np.float64]
        Synchrosqueezed scalogram.

    References
    ----------
    - https://dsp.stackexchange.com/questions/71398/synchrosqueezing-wavelet
      -transform-explanation
    - [1]: Ingrid Daubechies, Jianfeng Lu, Hau-Tieng Wu. Synchrosqueezed
      wavelet transforms: An empirical mode decomposition-like tool. 2011.

    """
    scalpow = np.abs(scalogram) ** 2
    inds = scalpow > 1e-40

    # Phase Transform
    if gradient is None:
        gradient = np.gradient(scalogram, axis=1, edge_order=2)
    ph = gradient.copy()
    ph[~inds] = 0
    # Since only imaginary part needed -> computation could be improved
    ph[inds] = (ph[inds] / scalogram[inds]).imag / 2 / np.pi
    # Prune imaginary part
    ph = np.abs(ph.real)
    ph *= fs  # Scale to represent physical frequencies

    # Normalization factor
    if apply_frequency_normalization:
        normalizations = 1 / (freqs / fs)  # Scales
        normalizations **= -3 / 2

    # Thresholds
    delta_f = delta_w * freqs

    sync = np.zeros_like(scalogram)
    for ch in range(ph.shape[2]):
        for t in np.arange(ph.shape[1]):
            for f in range(ph.shape[0]):
                diff = np.abs(freqs - ph[f, t, ch])
                ind = np.argmin(diff)
                if diff[ind] > delta_f[f]:
                    continue
                if apply_frequency_normalization:
                    sync[ind, t, ch] += scalogram[f, t, ch] * normalizations[f]
                    continue

                sync[ind, t, ch] += scalogram[f, t, ch]
    return sync


def _squeeze_scalogram(
    scalogram: NDArray[np.complex128],
    freqs: NDArray[np.float64],
    fs: int,
    delta_w: float = 0.05,
    apply_frequency_normalization: bool = False,
) -> NDArray[np.complex128]:
    """Synchrosqueeze a scalogram, preferring the Rust implementation."""
    gradient = np.gradient(scalogram, axis=1, edge_order=2)
    if _squeeze_scalogram_rust is not None:
        return _squeeze_scalogram_rust(
            scalogram,
            freqs,
            fs,
            delta_w,
            apply_frequency_normalization,
            gradient,
        )
    return _squeeze_scalogram_python(
        scalogram,
        freqs,
        fs,
        delta_w,
        apply_frequency_normalization,
        gradient,
    )


def _get_length_longest_wavelet(
    wave: Wavelet | MorletWavelet, f: NDArray[np.float64], fs: int
) -> int:
    """Get longest wavelet for a frequency vector. This is useful information
    for zero-padding to avoid boundary effects.

    Parameters
    ----------
    wave : `Wavelet` or `MorletWavelet`
        Wavelet object.
    f : NDArray[np.float64]
        Frequency vector.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    length : int
        Length of longest wavelet in samples.

    """
    return len(wave.get_wavelet(np.min(f), fs))


def _get_kernels_vqt(
    q: float,
    highest_f: float,
    bins_per_octave: int,
    sampling_rate_hz: int,
    window_type: str | tuple,
    gamma: float,
) -> list[NDArray[np.complex128]]:
    """Compute the complex kernels for the VQT from the highest frequency
    and the sampling rate.

    Parameters
    ----------
    q : float
        Q factor.
    highest_f : float
        Highest frequency for which to compute the kernel.
    bins_per_octave : int
        Number of bins contained in each octave.
    sampling_rate_hz : int
        Sampling rate in Hz.
    window_type : str or tuple
        Window specification to pass to `scipy.signal.get_window()`.
    gamma : float
        Factor for variable Q.

    Returns
    -------
    kernels : list
        List containing the complex kernels arranged from high frequency to
        lower frequency.

    """
    freqs = highest_f * 2 ** (-1 / bins_per_octave * np.arange(bins_per_octave))
    factor = 2 ** (1 / bins_per_octave) - 1
    lengths = np.round(q * sampling_rate_hz / ((freqs * factor) + gamma)).astype(int)

    kernels = []

    for ind in range(len(lengths)):
        w = get_window(window_type, lengths[ind], fftbins=False)
        # Normalize window
        w /= w.sum()
        # Generate kernel centered in window
        kernels.append(
            w
            * np.exp(
                1j
                * freqs[ind]
                * 2
                * np.pi
                / sampling_rate_hz
                * np.arange(-lengths[ind] // 2, lengths[ind] // 2)
            )
        )

    return kernels


def _warp_time_series_python(
    td: NDArray[np.float64], warping_factor: float
) -> NDArray[np.float64]:
    """Warp or unwarp a time series using the Python implementation."""
    warped_td = np.zeros_like(td)

    dirac = np.zeros(td.shape[0])
    dirac[0] = 1

    b = np.array([-warping_factor, 1])
    a = np.array([1, -warping_factor])

    warped_td = dirac[..., None] * td[0, :]

    for n in np.arange(1, td.shape[0]):
        dirac = lfilter(b, a, dirac)
        warped_td += dirac[..., None] * td[n, :]
    return warped_td


def _warp_time_series(
    td: NDArray[np.float64], warping_factor: float
) -> NDArray[np.float64]:
    """Warp or unwarp a time series, preferring the Rust implementation."""
    if _warp_time_series_rust is not None:
        return _warp_time_series_rust(td, warping_factor)
    return _warp_time_series_python(td, warping_factor)


try:
    import numba as nb

    @nb.jit(
        nb.types.Array(nb.complex128, 2, "C")(
            nb.types.Array(nb.complex128, 2, "C"),
            nb.types.Array(nb.complex128, 1, "C"),
            nb.types.Array(nb.complex128, 1, "C"),
            nb.types.Array(nb.complex128, 2, "C"),
        ),
        parallel=True,
        cache=True,
    )
    def _dft_backend(
        time_data: NDArray[np.complex128],
        freqs_normalized: NDArray[np.complex128],
        dft_factor: NDArray[np.complex128],
        spectrum: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        for ind in nb.prange(len(freqs_normalized)):
            spectrum[ind, :] = np.exp(dft_factor * freqs_normalized[ind]) @ time_data
        return spectrum

except ModuleNotFoundError as e:
    str(e)

    def _dft_backend(
        time_data: NDArray[np.complex128],
        freqs_normalized: NDArray[np.complex128],
        dft_factor: NDArray[np.complex128],
        spectrum: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        for ind in range(len(freqs_normalized)):
            spectrum[ind, :] = np.exp(dft_factor * freqs_normalized[ind]) @ time_data
        return spectrum
