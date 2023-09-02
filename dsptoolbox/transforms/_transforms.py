"""
Backend for special module
"""
import numpy as np


def _pitch2frequency(tuning_a_hz: float = 440):
    """This function returns a vector having frequencies for pitches
    0 to 127 (Midi compatible), where 0 is C0.

    Parameters
    ----------
    tuning_a_hz : float, optional
        Tuning for A4 in Hz. Default: 440.

    Returns
    -------
    freqs : `np.ndarray`
        Frequencies for each pitch. It always has length 128.

    """
    N = 128
    return tuning_a_hz * 2**((np.arange(N)-69)/12)


class Wavelet():
    """Base class for a wavelet function.

    """
    def __init__(self):
        """Constructor for the base wavelet class. It's not supposed to be
        used directly.

        """
        pass

    def get_base_wavelet(self):
        """Abstract method to get the mother wavelet. It must be implemented
        in each Wavelet class.

        """
        raise NotImplementedError('Wavelet function has not been implemented')

    def get_center_frequency(self):
        """Returns the center frequency of the wavelet (normalized,
        i.e., with fs=1).

        """
        x, func = self.get_base_wavelet()
        ind = np.argmax(np.abs(np.fft.fft(func)))
        # Maybe for some wavelets it might be necessary to miror around nyquist
        domain = x[-1] - x[0]
        return ind / domain

    def get_scale_lengths(self, frequencies: np.ndarray, fs: int):
        """Returns the lengths of the queried frequencies.

        Parameters
        ----------
        frequencies : `np.ndarray`
            Frequencies for which to scale the wavelet.
        fs : int
            Sampling rate in Hz.

        Returns
        -------
        `np.ndarray`
            Lengths of wavelets in samples.

        """
        scales = np.atleast_1d(self.get_center_frequency() / frequencies * fs)
        x, _ = self.get_base_wavelet()
        return (scales * (x[-1] - x[0]) + 1).astype(int)


class MorletWavelet(Wavelet):
    """Complex morlet wavelet.

    """
    def __init__(self, b: float = None, h: float = None, scale: float = 1.,
                 precision_bounds: float = 1e-5, step: float = 5e-3):
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

        Notes
        -----
        - The relation between `b` and `h` is b=h**2/(4*np.log(2)). `h` is
          the full-width at half-maximum (FWHM) as defined by [1].

        References
        ----------
        - [1]: Michael X Cohen, A better way to define and describe Morlet
          wavelets for time-frequency analysis.

        """
        assert b is not None or h is not None, 'Either b or h must be passed'
        if h is not None:
            b = h**2 / np.log(2) / 4
        self.b = b
        self.scale = scale

        t = np.sqrt(b*np.log(1/precision_bounds))
        self.bounds = [-t, t]

        self.step = step

    def _get_x(self) -> np.ndarray:
        """Returns x vector for the mother wavelet.

        """
        return np.arange(self.bounds[0], self.bounds[1]+self.step, self.step)

    def get_base_wavelet(self) -> tuple[np.ndarray, np.ndarray]:
        """Return complex morlet wavelet.

        """
        x = self._get_x()
        return x, 1/np.sqrt(np.pi*self.b) * \
            np.exp(2j*np.pi/self.scale*x)*np.exp(-x**2/self.b)

    def get_center_frequency(self) -> float:
        """Return center frequency for the complex morlet wavelet.

        """
        return 1/self.scale

    def get_wavelet(self, f: float | np.ndarray, fs: int,
                    interpolation: bool = False) -> np.ndarray | list:
        """Return wavelet scaled for a specific frequency and sampling rate.
        The wavelet values can also be linearly interpolated for a higher
        accuracy at the expense of computation time.

        Parameters
        ----------
        f : float or `np.ndarray`
            Queried frequency or array of frequencies.
        fs : int
            Sampling rate in Hz.
        interpolation : bool, optional
            When `True`, the wavelet function values are linearly interpolated
            based on the mother wavelet. This increases accuracy but it's more
            computationally intense. Otherwise, floor is used on the indices
            to get the values. Default: `False`.

        Returns
        -------
        wave : `np.ndarray` or list of `np.ndarray`
            Wavelet function. It is either a 1d-array for a single frequency
            or a list of arrays for multiple frequencies.

        """
        scales = np.atleast_1d(self.get_center_frequency() / f * fs)
        x, base = self.get_base_wavelet()
        wave = []

        for scale in scales:
            inds = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * self.step)
            if interpolation:
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

    def _get_interpolated_wave(self, base: np.ndarray, inds: np.ndarray):
        """Return the wavelet function for a selection of index using
        linear interpolation.

        """
        # Truncate indices and select only valid ones
        trunc = inds.astype(int)
        trunc = trunc[trunc < len(base)]

        accumulator = np.zeros(len(trunc), dtype='cfloat')

        for i in range(len(trunc)-1):
            accumulator[i] = base[trunc[i]] + \
                (base[trunc[i]+1] - base[trunc[i]])*(inds[i] - trunc[i])
        accumulator[-1] = base[trunc[-1]]
        return accumulator


def _squeeze_scalogram(scalogram: np.ndarray, freqs: np.ndarray, fs: int,
                       delta_w: float = 0.05) \
        -> np.ndarray:
    """Synchrosqueeze a scalogram.

    Parameters
    ----------
    scalogram : `np.ndarray`
        Complex scalogram from the CWT with shape (frequency, time sample,
        channel).
    freqs : `np.ndarray`
        Frequency vector.
    fs : int
        Sampling rate in Hz.
    delta_w : float, optional
        Maximum relative difference in frequency allowed in the phase
        transform for taking summing the result of the scalogram. If it's
        too small, it might lead to significant energy leaks. Default: 0.05.

    Returns
    -------
    sync : `np.ndarray`
        Synchrosqueezed scalogram.

    References
    ----------
    - https://dsp.stackexchange.com/questions/71398/synchrosqueezing-wavelet
      -transform-explanation

    """
    scalpow = np.abs(scalogram)**2
    inds = scalpow > 1e-40

    # Phase Transform
    ph = np.gradient(scalogram, axis=1)
    ph[~inds] = 0
    # Since only imaginary part needed -> computation could be improved
    ph[inds] = (ph[inds] / scalogram[inds]).imag / 2 / np.pi
    # Prune imaginary part
    ph = np.abs(ph.real)
    ph *= fs  # Scale to represent physical frequencies

    # Normalization factor
    normalizations = 1 / (freqs / fs)  # Scales
    normalizations **= (-3/2)

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
                sync[ind, t, ch] += scalogram[f, t, ch] * normalizations[f]
    return sync


def _get_length_longest_wavelet(wave: Wavelet | MorletWavelet, f: np.ndarray,
                                fs: float):
    """Get longest wavelet for a frequency vector. This is useful information
    for zero-padding to avoid boundary effects.

    Parameters
    ----------
    wave : `Wavelet` or `MorletWavelet`
        Wavelet object.
    f : `np.ndarray`
        Frequency vector.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    length : int
        Length of longest wavelet in samples.

    """
    return len(wave.get_wavelet(np.min(f), fs, False))
