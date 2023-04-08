"""
Backend for the effects module
"""
from dsptoolbox._general_helpers import _pad_trim
import numpy as np
# import matplotlib.pyplot as plt


def _arctan_distortion(inp: np.ndarray,
                       distortion_level_db: float,
                       offset_db: float) -> np.ndarray:
    """Applies arctan distortion.

    """
    offset_linear = 10**(offset_db/20)
    distortion_level_linear = 10**(distortion_level_db/20)
    peak_level = np.max(np.abs(inp), axis=0)
    normalized = inp / peak_level
    return np.arctan(normalized * distortion_level_linear
                     + offset_linear) * (2/np.pi)


def _hard_clip_distortion(inp: np.ndarray,
                          distortion_level_db: float,
                          offset_db: float) -> np.ndarray:
    """Applies hard clipping distortion.

    """
    offset_linear = 10**(offset_db/20)
    distortion_level_linear = 10**(distortion_level_db/20)
    peak_level = np.max(np.abs(inp), axis=0)
    normalized = inp / peak_level
    return np.clip(normalized * distortion_level_linear + offset_linear,
                   a_min=-1, a_max=1)


def _soft_clip_distortion(inp: np.ndarray,
                          distortion_level_db: float,
                          offset_db: float) -> np.ndarray:
    """Applies non-linear cubic distortion.

    """
    offset_linear = 10**(offset_db/20)
    distortion_level_linear = 10**(distortion_level_db/20)
    peak_level = np.max(np.abs(inp), axis=0)
    normalized = inp / peak_level * (2/3)
    normalized += offset_linear
    normalized = (normalized - normalized**3 / 3) * distortion_level_linear
    return np.clip(normalized, a_min=-2/3, a_max=2/3)


def _clean_signal(inp: np.ndarray,
                  distortion_level_db: float,
                  offset_db: float) -> np.ndarray:
    """Returns the unchanged clean signal.

    """
    return inp


def _compressor(x: np.ndarray, threshold_db: float, ratio: float,
                knee_factor_db: float, attack_samples: int, hold_samples: int,
                release_samples: int, mix_compressed: float,
                side_chain: np.ndarray) \
        -> np.ndarray:
    """Compresses the dynamic range of a signal. Vector should be passed in
    linear space.

    """
    if mix_compressed > 1:
        mix_compressed = 1
    x_ = x.copy()
    single_channel = False
    if x_.ndim == 1:
        x_ = x_[..., None]
        single_channel = True

    # Save signs
    signs = np.sign(x_)

    # Absolute values
    x_abs = np.abs(x_)

    # Save tiny values
    tiny_values = x_abs < 1e-25

    # Get function
    compression_func = _get_knee_func(threshold_db, ratio, knee_factor_db)

    # Side-chain handling
    if side_chain is not None:
        if len(side_chain) != len(x):
            side_chain = _pad_trim(side_chain.astype(int), len(x)).astype(bool)

    # Iterate process over channels
    for n in range(x_.shape[1]):
        # Convert to logarithmic space
        x_db = 20*np.log10(np.clip(x_abs[:, n], 1e-25, None))
        # Get samples
        activated, released = _find_triggered_and_release_samples(
            x_db, threshold_db, attack_samples, hold_samples, release_samples,
            side_chain)
        # Compress in logarithmic space
        x_db_compressed = x_db.copy()
        x_db_compressed[activated] = compression_func(x_db[activated])
        # Cross-fade during release times
        x_compressed = _cross_fade_release_samples(
            released, release_samples, x_db, x_db_compressed)
        x_[:, n] = x_compressed * mix_compressed + \
            x_[:, n] * (1 - mix_compressed)

    # Restore signs
    x_ *= signs

    # Restore tiny values
    x_[tiny_values] = x[tiny_values].copy()

    if single_channel:
        x_ = x_.squeeze()
    return x_


def _get_knee_func(threshold_db: float, ratio: float, knee_factor_db: float):
    """This function returns a callable that acts as the compression function
    in logarithmic space.

    """
    T = threshold_db
    R = ratio
    W = knee_factor_db

    def compress_in_db(x: np.ndarray):
        y = np.zeros_like(x)
        first_section = x - T < -W/2
        y[first_section] = x[first_section]
        second_section = np.abs(x - T) <= W / 2
        y[second_section] = x[second_section] + \
            (1/R - 1)*(x[second_section]-T+W/2)**2 / 2 / W
        third_section = x - T > W / 2
        y[third_section] = T + (x[third_section] - T) / R
        return y

    return compress_in_db


def _find_triggered_and_release_samples(
        x: np.ndarray, threshold_db: float, attack_samples: int,
        hold_samples: int, release_samples: int, side_chain: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray]:
    """This function finds the triggered and release samples. It returns
    two boolean arrays. It can only handle 1D-arrays!

    """
    # Number of samples that have to be above the threshold to trigger the
    # effect – Should data be smoothed or just set to a couple samples?
    surpass_samples = 2

    hold_samples = max(1, hold_samples)
    release_samples = max(1, release_samples)

    # Smoothed
    # x = np.convolve(x, np.ones(2)/2)

    if side_chain is None:
        # Activation
        triggered = np.zeros_like(x).astype(bool)
        # Iterate over samples
        for i in np.arange(1, len(x)):
            ind = max(0, i - surpass_samples)
            if np.all(x[ind:i] > threshold_db):
                start = min(i+attack_samples, len(x)-1)
                end = min(i+attack_samples+hold_samples+release_samples,
                          len(x))
                triggered[start:end] = True
        # Find points of deactivation
        release = np.zeros_like(x).astype(bool)
        release[:-1] = np.bitwise_and(
            triggered[:-1], np.bitwise_not(triggered[1:]))
        for i in np.arange(release_samples-1, len(x)):
            if release[i]:
                release[i-release_samples:i] = True
    else:
        triggered = side_chain
        release = np.zeros_like(x).astype(bool)
        temp = np.zeros_like(x).astype(bool)
        release[:-1] = np.bitwise_and(
            triggered[:-1], np.bitwise_not(triggered[1:]))
        for i in np.arange(release_samples-1, len(x)):
            if release[i]:
                temp[i:i+release_samples] = True
        release = temp
        triggered = np.bitwise_or(triggered, release)
    return triggered, release


def _cross_fade_release_samples(release_vector: np.ndarray,
                                release_samples: int, x: np.ndarray,
                                x_compressed: np.ndarray,
                                type_of_cross: str = 'log') -> np.ndarray:
    """Cross fades two signals: a compressed version and the original assuming
    they are in logarithmic space.

    """
    if type_of_cross == 'lin':
        mix_in = np.linspace(0, 1, release_samples)
    elif type_of_cross == 'log':
        mix_in = np.exp(np.linspace(-10, 0, release_samples))

    x_lin = 10**(x/20)
    x_compressed_lin = 10**(x_compressed/20)

    for i in np.arange(len(x)):
        if release_vector[i]:
            length = np.sum(release_vector[i:i+release_samples])
            x_compressed_lin[i:i+length] = \
                x_lin[i:i+length] * mix_in[:length] + \
                x_compressed_lin[i:i+length] * (1 - mix_in[:length])

    return x_compressed_lin


if __name__ == '__main__':
    # Check functions
    import matplotlib.pyplot as plt

    x = np.zeros(1000) + 0.001
    n = np.random.normal(2, 0.1, 200)
    x[200:400] += n
    x[600:800] += n
    x += np.random.normal(0, 0.01, 1000)
    # x = 20*np.log10(np.abs(x))

    # triggered, release = _find_triggered_and_release_samples(
    #     x, 0, attack_samples=5, hold_samples=10, release_samples=20)

    # # plt.plot(x**(x/20))
    # plt.plot(triggered.astype(int))
    # plt.plot(release.astype(int)*0.5)
    # plt.show()
    # exit()

    # x = np.exp(np.linspace(-7, 0, 10000))
    y = _compressor(
        x, -20, ratio=4, knee_factor_db=10, attack_samples=1,
        hold_samples=1, release_samples=4, mix_compressed=1)
    # plt.plot(20*np.log10(np.abs(x)), 20*np.log10(np.abs(y)))
    # plt.plot(x, label='original')
    # plt.plot(y, label='compressed')
    # plt.legend()
    # plt.plot(act.astype(int))
    # plt.plot(rel.astype(int)*0.5)

    y = _compressor(x, -20, ratio=4, knee_factor_db=0, attack_samples=0,
                    hold_samples=0, release_samples=1, mix_compressed=100)
    # plt.plot(20*np.log10(np.abs(x)), 20*np.log10(np.abs(y)))
    plt.show()
