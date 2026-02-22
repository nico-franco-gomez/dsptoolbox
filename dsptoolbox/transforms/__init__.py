"""
Transforms
----------
This module contains signal transforms.

- `cepstrum()`
- `from_cepstrum()`
- `log_mel_spectrogram()` (returns a logarithmic magnitude spectrogram
  with mel frequency axis)
- `mel_filterbank()` (returns matrix with triangular mel filters used to
  convert spectrograms' frequency axis from Hz into mel)
- `plot_waterfall()` (creates and returns a waterfall plot)
- `mfcc()` (mel-frequency cepstral coefficients)
- `istft()` (inverse STFT)
- `MorletWavelet` (class for a complex morlet wavelet)
- `cwt()` (continuous wavelet transform)
- `chroma_stft()` (STFT adapted to the chroma scale)
- `hilbert()` (Hilbert Transform)
- `vqt()` (Variable-Q Transform)
- `stereo_mid_side()` (Mid-Side representation of stereo signal)
- `laguerre()` (Frequency-warping by means of the Laguerre transform)
- `warp()` (Convert signal to/from warped domain)
- `warp_filter()` (Warp a filter transforming its poles and zeros)
- `lpc()` (linear-predictive coding)
- `dft()` (discrete fourier transform for arbitrary frequency resolution)
- `spectrum_via_filterbank()` (magnitude spectrum via filterbank)

"""

from .transforms import (
    cepstrum,
    from_complex_cepstrum,
    log_mel_spectrogram,
    mel_filterbank,
    plot_waterfall,
    mfcc,
    istft,
    MorletWavelet,
    cwt,
    chroma_stft,
    hilbert,
    vqt,
    stereo_mid_side,
    laguerre,
    warp,
    warp_filter,
    lpc,
    dft,
    spectrum_via_filterbank,
)

__all__ = [
    "cepstrum",
    "from_complex_cepstrum",
    "log_mel_spectrogram",
    "mel_filterbank",
    "plot_waterfall",
    "mfcc",
    "istft",
    "MorletWavelet",
    "cwt",
    "chroma_stft",
    "hilbert",
    "vqt",
    "stereo_mid_side",
    "laguerre",
    "warp",
    "warp_filter",
    "lpc",
    "dft",
    "spectrum_via_filterbank",
]
