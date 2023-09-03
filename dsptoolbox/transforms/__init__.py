"""
Transforms
----------
This module contains signal transforms.

- `cepstrum()`
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

"""
from .transforms import (cepstrum, log_mel_spectrogram, mel_filterbank,
                         plot_waterfall, mfcc, istft, MorletWavelet, cwt,
                         chroma_stft, hilbert, vqt)

__all__ = [
    'cepstrum',
    'log_mel_spectrogram',
    'mel_filterbank',
    'plot_waterfall',
    'mfcc',
    'istft',
    'MorletWavelet',
    'cwt',
    'chroma_stft',
    'hilbert',
    'vqt',
]
