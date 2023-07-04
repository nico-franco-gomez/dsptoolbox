"""
Special
-------
This module contains functions that might be regarded as not very common use.

- `cepstrum()`
- `log_mel_spectrogram()` (returns a logarithmic magnitude spectrogram
  with mel frequency axis)
- `mel_filterbank()` (returns matrix with triangular mel filters used to
  convert spectrograms' frequency axis from Hz into mel)
- `plot_waterfall()` (creates and returns a waterfall plot)

"""
from .special import (cepstrum, log_mel_spectrogram, mel_filterbank,
                      plot_waterfall, mfcc, istft)

__all__ = [
    'cepstrum',
    'log_mel_spectrogram',
    'mel_filterbank',
    'plot_waterfall',
    'mfcc',
    'istft',
]
