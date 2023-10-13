"""
Distances
---------
This module contains distance measurements between signals. Even though the
results seem plausible, these implementations need validation from external
tools.

Frequency domain:

- `log_spectral()`
- `itakura_saito()`

Time domain:

- `snr()`
- `si_sdr()`

Mixed:

- `fw_snr_seg()`

"""
from .distances import log_spectral, itakura_saito, snr, si_sdr, fw_snr_seg

__all__ = [
    "log_spectral",
    "itakura_saito",
    "snr",
    "si_sdr",
    "fw_snr_seg",
]
