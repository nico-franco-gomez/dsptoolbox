"""
Special
-------
This module contains functions that might be regarded as not very common use.

- `cepstrum()`
- `min_phase_from_mag()` (generate a signal with minimum phase from a
  magnitude spectrum)
- `lin_phase_from_mag()` (generate a signal with linear phase from a
  magnitude spectrum)
- `log_mel_spectrogram()` (returns a logarithmic magnitude spectrogram
  with mel frequency axis)
- `mel_filterbank()` (returns matrix with triangular mel filters used to
  convert spectrograms' frequency axis from Hz into mel)

"""
from .special import (cepstrum, min_phase_from_mag, lin_phase_from_mag,
                      log_mel_spectrogram, mel_filterbank)

__all__ = [
    'cepstrum',
    'min_phase_from_mag',
    'lin_phase_from_mag',
    'log_mel_spectrogram',
    'mel_filterbank',
]
