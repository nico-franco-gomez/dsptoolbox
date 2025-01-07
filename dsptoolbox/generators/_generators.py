import numpy as np
from numpy.typing import NDArray


def _sync_log_chirp(
    chirp_range_hz: list[float], length_seconds: float, sampling_rate_hz: int
) -> tuple[NDArray[np.float64], float]:
    """Generate a synchronized exponential chirp according to [1].

    Parameters
    ----------
    chirp_range_hz : list[float]
        Frequency range of the chirp in Hz.
    length_seconds : float
        Length of the chirp. This will not be the exact length due to the
        necessity of ensuring that the last frequency in the chirp with
        `phase=0` ends.

    Returns
    -------
    td : NDArray[np.float64]
        Synchronized chirp (single channel).
    T : float
        Effective chirp duration.

    References
    ----------
    - [1]: Antonin Novak, Laurent Simon, Pierrick Lotton. Synchronized
      Swept-Sine: Theory, Application and Implementation.

    """
    f1, f2 = chirp_range_hz[0], chirp_range_hz[1]
    f2f1 = np.log(f2 / f1)

    # Get k from estimated length
    k = int(f1 * length_seconds / f2f1 + 0.5)

    # Get real duration
    T = k / f1 * f2f1

    # Sweep-rate
    L = int(0.5 + T * f1 / f2f1) / f1
    t = np.linspace(0.0, T, int(T * sampling_rate_hz + 0.5))
    return np.sin(2.0 * np.pi * f1 * L * (np.exp(t / L) - 1.0)), T


if __name__ == "__main__":
    fs_hz = 48000
    td, t = _sync_log_chirp([20, 21e3], 2.1, fs_hz)
    print(td[-1])
    import matplotlib.pyplot as plt

    plt.plot(td)
    plt.xlim([len(td) - 50, len(td) + 20])
    plt.show()
