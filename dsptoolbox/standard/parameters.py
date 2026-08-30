"""
Frozen parameter sets shared by the spectrum, spectrogram and the functions
that consume them.
"""

from dataclasses import dataclass, replace
from typing import Any

from .enums import (
    SpectrumAverageMethod,
    SpectrumMethod,
    SpectrumScaling,
    Window,
    WindowType,
)


@dataclass(frozen=True)
class SpectrumParameters:
    """Configuration for `Signal.get_spectrum()`.

    Attributes
    ----------
    method : SpectrumMethod
        Method used to acquire the spectrum.
    smoothing : float
        Smoothing across (1/smoothing) octave bands. 0 means no smoothing.
    pad_to_fast_length : bool
        When True, the time data is zero-padded to a fast FFT length.
    window_length_samples : int
        Window length for Welch's method.
    window_type : WindowType
        Window applied to each frame in Welch's method.
    overlap_percent : float
        Overlap between frames in percent.
    detrend : bool
        When True, the mean of each frame is removed.
    average : SpectrumAverageMethod
        Statistic used to average the periodograms.
    scaling : SpectrumScaling
        Scaling of the spectrum.

    """

    method: SpectrumMethod = SpectrumMethod.WelchPeriodogram
    smoothing: float = 0.0
    pad_to_fast_length: bool = True
    window_length_samples: int = 1024
    window_type: WindowType = Window.Hann
    overlap_percent: float = 50.0
    detrend: bool = True
    average: SpectrumAverageMethod = SpectrumAverageMethod.Mean
    scaling: SpectrumScaling = SpectrumScaling.FFTBackward

    def replace(self, **changes: Any) -> "SpectrumParameters":
        """Return a copy with the given attributes replaced."""
        return replace(self, **changes)


@dataclass(frozen=True)
class SpectrogramParameters:
    """Configuration for `Signal.get_spectrogram()` and `transforms.istft()`.

    Attributes
    ----------
    window_length_samples : int
        Length of each time frame in samples.
    window_type : WindowType
        Window applied to each frame.
    overlap_percent : float
        Overlap between frames in percent.
    fft_length_samples : int, None
        Length of the FFT applied to each frame. None uses the window length.
    detrend : bool
        When True, the mean of each frame is removed.
    padding : bool
        When True, the signal is zero-padded at both edges so that no energy
        is lost to the windowing.
    scaling : SpectrumScaling
        Scaling of the spectra.

    """

    window_length_samples: int = 1024
    window_type: WindowType = Window.Hann
    overlap_percent: float = 50.0
    fft_length_samples: int | None = None
    detrend: bool = False
    padding: bool = True
    scaling: SpectrumScaling = SpectrumScaling.FFTBackward

    def replace(self, **changes: Any) -> "SpectrogramParameters":
        """Return a copy with the given attributes replaced."""
        return replace(self, **changes)
