"""
Backend for the effects module
"""
from .._general_helpers import _get_smoothing_factor_ema
from ..plots import general_plot
import numpy as np
# import matplotlib.pyplot as plt


# ========= Distortion ========================================================
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
    normalized *= distortion_level_linear
    normalized = (normalized - normalized**3 / 3)
    return np.clip(normalized, a_min=-2/3, a_max=2/3)


def _clean_signal(inp: np.ndarray,
                  distortion_level_db: float,
                  offset_db: float) -> np.ndarray:
    """Returns the unchanged clean signal.

    """
    return inp


# ========= Compressor ========================================================
def _compressor(x: np.ndarray, threshold_db: float, ratio: float,
                knee_factor_db: float, attack_samples: int,
                release_samples: int, mix_compressed: float,
                downward_compression: bool) \
        -> np.ndarray:
    """Compresses the dynamic range of a signal.

    Parameters
    ----------
    x : `np.ndarray`
        Signal to compress.
    threshold_db : float
        Threshold level.
    ratio : float
        Compression ratio.
    knee_factor_db : float
        Knee width in dB.
    attack_samples : int
        Time of attack in samples.
    release_samples : int
        Time of release in samples.
    mix_compressed : float
        Amount of compressed signal in the output. Must be between 0 and 1
        where 1 means there is only compressed signal in the output.
    downward_compression : bool
        When `True`, downward compression is applied. Otherwise, upward
        compression is applied.

    Returns
    -------
    x_ : `np.ndarray`
        Compressed signal.

    """
    if mix_compressed > 1:
        mix_compressed = 1
    x_ = x.copy()
    single_channel = False
    if x_.ndim == 1:
        x_ = x_[..., None]
        single_channel = True

    # Get function
    compression_func = _get_knee_func(threshold_db, ratio, knee_factor_db,
                                      downward_compression)

    # RMS detector
    attack_coeff = _get_smoothing_factor_ema(attack_samples, 1)
    release_coeff = _get_smoothing_factor_ema(release_samples, 1)

    # Iterate process over channels
    for n in range(x_.shape[1]):
        momentary_rms = 0
        momentary_gain = 1
        for i in np.arange(len(x)):
            # RMS Detection – if peaks, directly take rms
            samp = x[i]**2
            if samp < momentary_rms:
                coeff = attack_coeff
            else:
                coeff = 1
            coeff = 0.01
            momentary_rms = coeff*samp + (1-coeff)*momentary_rms

            # Amount of required compression
            samp_db = 10*np.log10(samp)
            samp_db_comp = compression_func(samp_db)
            np.nan_to_num(samp_db, False, 0)
            np.nan_to_num(samp_db_comp, False, 0)
            gain_factor = 10**((samp_db_comp-samp_db)/20)

            # Gain depending on attack and release
            if gain_factor > momentary_gain:
                coeff = attack_coeff
            else:
                coeff = release_coeff
            momentary_gain = coeff*gain_factor + (1-coeff)*momentary_gain

            # Apply gain
            x_[i, n] *= momentary_gain

    if single_channel:
        x_ = x_.squeeze()
    return x_


def _get_knee_func(threshold_db: float, ratio: float, knee_factor_db: float,
                   downward_compression: bool):
    """This function returns a callable that acts as the compression function
    in logarithmic space.

    https://tinyurl.com/2uh9rjs7

    """
    T = threshold_db
    R = ratio
    W = knee_factor_db

    if downward_compression:
        def compress_in_db(x: np.ndarray | float):
            if type(x) == float:
                if (x - T < - W / 2):
                    return x
                elif (np.abs(x - T) <= W / 2):
                    return x - (1/R - 1)*(x-T-W/2)**2 / 2 / W
                elif (x - T > W / 2):
                    return T + (x - T) / R

            y = np.zeros_like(x)
            first_section = x - T < -W/2
            y[first_section] = x[first_section]

            second_section = np.abs(x - T) <= W / 2
            y[second_section] = x[second_section] + \
                (1/R - 1)*(x[second_section]-T+W/2)**2 / 2 / W

            third_section = x - T > W / 2
            y[third_section] = T + (x[third_section] - T) / R
            return y
    else:
        def compress_in_db(x: np.ndarray | float):
            if type(x) == float:
                if (x - T < - W / 2):
                    return T + (x - T) / R
                elif (np.abs(x - T) <= W / 2):
                    return x - (1/R - 1)*(x-T-W/2)**2 / 2 / W
                elif (x - T > W / 2):
                    return x

            y = np.zeros_like(x)
            first_section = x - T < - W / 2
            y[first_section] = T + (x[first_section] - T) / R

            second_section = np.abs(x - T) <= W / 2
            y[second_section] = x[second_section] - \
                (1/R - 1)*(x[second_section]-T-W/2)**2 / 2 / W

            third_section = x - T > W / 2
            y[third_section] = x[third_section]
            return y

    return compress_in_db


def _find_attack_hold_release(
        x: np.ndarray, threshold_db: float, attack_samples: int,
        hold_samples: int, release_samples: int, side_chain: np.ndarray,
        indices_above: bool) \
            -> tuple[np.ndarray, np.ndarray]:
    """This function finds the indices corresponding to attack, hold and
    release. It returns boolean arrays. It can only handle 1D-arrays as input!

    """
    # Number of samples that have to be above the threshold to trigger the
    # effect – Should data be smoothed or just set to a couple samples?
    surpass_samples = 2

    hold_samples = max(1, hold_samples)
    release_samples = max(1, release_samples)

    # Smoothed
    # x = np.convolve(x, np.ones(2)/2)

    # Select if above or below threshold (depending on upward or downward
    # compression)
    if indices_above:
        def trigger(x, ind1, ind2, y) -> bool:
            return np.all(x[ind1:ind2] > y)
    else:
        def trigger(x, ind1, ind2, y) -> bool:
            return np.all(x[ind1:ind2] < y)

    if side_chain is None:
        # Accumulate global activations
        global_activation = np.zeros_like(x).astype(bool)
        for i in np.arange(1, len(x)):
            ind = max(0, i - surpass_samples)
            if trigger(x, ind, i, threshold_db):
                global_activation[i:i + attack_samples + hold_samples +
                                  release_samples] = True
    else:
        global_activation = side_chain

    # Extract release, attack and hold
    attack = np.zeros_like(x).astype(bool)
    release = np.zeros_like(x).astype(bool)
    temp_attack = np.zeros_like(x).astype(bool)
    release[:-1] = np.bitwise_and(
        global_activation[:-1], np.bitwise_not(global_activation[1:]))
    temp_attack[1:] = np.bitwise_and(
        np.bitwise_not(global_activation[:-1]), global_activation[1:])
    for i in np.arange(len(x)):
        if release[i]:
            release[i-release_samples:i] = True
        if temp_attack[i]:
            attack[i:i+attack_samples] = True
    hold = (global_activation.astype(int) - attack.astype(int) -
            release.astype(int)).astype(bool)
    return attack, hold, release


def _cross_fade_samples(x_output: np.ndarray,
                        x_fade_in: np.ndarray, x_fade_out: np.ndarray,
                        indices: np.ndarray, length_of_fade: int,
                        type_of_cross: str = 'log') -> np.ndarray:
    """Cross fades two signals: at certain indices.

    """
    if type_of_cross == 'lin':
        mix_in = np.linspace(0, 1, length_of_fade)
    elif type_of_cross == 'log':
        mix_in = np.exp(np.linspace(-10, 0, length_of_fade))

    for i in np.arange(len(x_output)):
        if indices[i]:
            length = np.sum(indices[i:i+length_of_fade])
            x_output[i:i+length] = \
                x_fade_in[i:i+length] * mix_in[:length] + \
                x_fade_out[i:i+length] * (1 - mix_in[:length])


# ========= LFO ===============================================================
class LFO():
    """Low-frequency oscillator.

    """
    def __init__(self, frequency_hz: float | tuple,
                 waveform: str = 'harmonic', random_phase: bool = False,
                 smooth: float = 0):
        """Constructor for a low-frequency oscillator.

        Parameters
        ----------
        frequency_hz : float
            Frequency for the oscillator. If a tuple is passed, it should have
            a string representing the type of note and a number with
            corresponding to bpm. An example would be
            `frequency_hz=('quarter', 60)`, meaning that the frequency should
            be set to a quarter note at 60 bpm. See Notes for more details.
        waveform : str, optional
            Type of waveform to use. Choose from `'harmonic'`, `'sawtooth'`,
            `'square'`, `'triangle'`. Default: `'harmonic'`.
        random_phase : bool, optional
            When `True`, a random phase shift is applied everytime the LFO
            is called. Default: `False`.
        smooth : float, optional
            For the non-differentiable waveforms, it is possible to generate
            smooth, i.e. differentiable, equivalents. The higher the smooth
            parameter, the rounder the resulting waveform. The usable range
            is set between 0 and 10 though any value can be passed. Scaling
            might be different depending on the waveform. If 0 is passed,
            the standard, non-differentiable waveform is produced. Default: 0.

        Notes
        -----
        The frequency of the LFO can be set to a musical rhythm. The time
        signature is always assumed to be 4/4. Choose from:

        - `'quarter'`, `'half'`, `'whole'`, `'eighth'`, `'sixteenth'`.
        - `'eighth 3'` means triplets. It can be used for getting triplets of
          any other duration.
        - `'half dotted'` refers to a dotted duration. It can be added to a
          string of any duration.

        """
        self.__set_parameters(frequency_hz, waveform, random_phase, smooth)

    def __set_parameters(self, frequency_hz, waveform: str, random_phase,
                         smooth):
        """Internal method to set parameters.

        """
        if frequency_hz is not None:
            if type(frequency_hz) in (float, int):
                self.frequency_hz = np.abs(frequency_hz)
            elif type(frequency_hz) in (tuple, list):
                assert len(frequency_hz) == 2, \
                    'frequency_hz as tuple must have length 2'
                self.frequency_hz = get_frequency_from_musical_rhythm(
                    frequency_hz[0], frequency_hz[1])
            else:
                raise TypeError('frequency_hz does not have a valid type')

        if waveform is not None:
            waveform = waveform.lower()
            if waveform == 'harmonic':
                self.oscillator = _harmonic_oscillator
            elif waveform == 'sawtooth':
                self.oscillator = _sawtooth_oscillator
            elif waveform == 'square':
                self.oscillator = _square_oscillator
            elif waveform == 'triangle':
                self.oscillator = _triangle_oscillator
            else:
                raise ValueError('Selected waveform is not valid')

        if smooth is not None:
            self.smooth = smooth

        if random_phase is not None:
            self.random_phase = random_phase

    def set_parameters(self, frequency_hz: float | tuple = None,
                       waveform: str = None, random_phase: bool = None,
                       smooth: float = None):
        """Set the parameters of the LFO.

        """
        self.__set_parameters(frequency_hz, waveform, random_phase, smooth)

    def get_waveform(self, sampling_rate_hz: int, length_samples: int = None):
        """Get the waveform of the oscillator for a sampling frequency and a
        specified duration. If `length_samples` is `None`, only one oscillation
        is returned.

        """
        if length_samples is None:
            length_samples = int(sampling_rate_hz / self.frequency_hz)
        return self.oscillator(self.frequency_hz, sampling_rate_hz,
                               length_samples, self.random_phase, self.smooth)

    def plot_waveform(self):
        """Plot the waveform (2 periods).

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        osc = self.oscillator(2, 1000, 1000, self.random_phase, self.smooth)
        fig, ax = general_plot(None, osc, log=False, returns=True, xlabel=None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Waveform')
        return fig, ax


def _harmonic_oscillator(freq, fs, length, random_phase, smooth):
    if length is None:
        length = int(fs / freq)
    norm_freq = freq / fs
    phase_shift = np.random.uniform(-np.pi, np.pi) if random_phase else 0
    return np.sin(norm_freq * 2 * np.pi * np.arange(length) + phase_shift)


def _square_oscillator(freq, fs, length, random_phase, smooth):
    # https://tinyurl.com/4d634xnk
    if length is None:
        length = int(fs / freq)
    phase_shift = np.random.uniform(-np.pi, np.pi) if random_phase else 0
    x = freq / fs * 2 * np.pi * np.arange(length) + phase_shift
    x = np.sin(x)
    if smooth == 0:
        waveform = np.sign(x)
    else:
        smooth *= (0.25/10)  # Adapt to some useful range
        waveform = np.arctan(x / smooth)
    return waveform


def _sawtooth_oscillator(freq, fs, length, random_phase, smooth):
    # https://tinyurl.com/5e8actzp
    if length is None:
        length = int(fs / freq)
    norm_freq = freq / fs
    if smooth == 0:
        phase_shift = np.random.uniform(0, 1) if random_phase else 0
        x = norm_freq * np.arange(length) + phase_shift
        waveform = (x % 1 - 0.5)*2
    else:
        phase_shift = np.random.uniform(-np.pi, np.pi) if random_phase else 0
        x = np.pi * norm_freq * np.arange(length) + phase_shift
        # Adapt range
        smooth = (12 - smooth)**1.5
        smooth = max(1, smooth)
        waveform = np.arcsin(np.tanh(np.cos(x)*smooth)*np.sin(x))
        waveform /= np.abs(np.max(waveform))
    return waveform


def _triangle_oscillator(freq, fs, length, random_phase, smooth):
    # https://tinyurl.com/4d634xnk
    if length is None:
        length = int(fs / freq)
    phase_shift = np.random.uniform(-np.pi, np.pi) if random_phase else 0
    x = freq / fs * 2 * np.pi * np.arange(length) + phase_shift
    x = np.sin(x)
    if smooth == 0:
        waveform = 2 / np.pi * np.arcsin(x)
    else:
        smooth *= (0.08/10)  # Adapt to some useful range
        waveform = 1 - 2/np.pi * np.arccos((1 - smooth)*x)
    waveform /= np.max(np.abs(waveform))
    return waveform


def get_frequency_from_musical_rhythm(note, bpm):
    """Method to compute frequency from a musical rhythm notation. The time
    signature is always assumed to be 4/4. Choose from:

    - `'quarter'`, `'half'`, `'whole'`, `'eighth'`, `'sixteenth'`,
        `'32th'`, `'quintuplet'`.
    - `'eighth 3'` means eight note triplets. It can be used for getting
        triplets of any other duration.
    - `'half dotted'` refers to a dotted duration. It can be added to a
        string of any duration.

    Parameters
    ----------
    note : str
        String
    bpm : float
        Beats per minute to define rhythm.

    Returns
    -------
    float
        Frequency in Hz corresponding to the musical rhythm.

    """
    assert type(note) == str and \
        type(bpm) in (float, int), \
        'Wrong data types for note duration and bpm'
    factor = 0
    if 'quarter' in note:
        factor = 1
    if 'half' in note:
        factor = 2
    if 'whole' in note:
        factor = 4
    if 'eighth' in note:
        factor = 1/2
    if 'sixteenth' in note:
        factor = 1/4
    if '32th' in note:
        factor = 1/8
    if 'quintuplet' in note:
        factor = 1/5
    if '3' in note:
        factor *= 2/3
    if 'dotted' in note:
        factor *= 1.5
    if factor == 0:
        raise ValueError('No valid note description was passed')
    return 60/bpm/factor


def get_time_period_from_musical_rhythm(note, bpm):
    """Method to compute time period from a musical rhythm notation. The time
    signature is always assumed to be 4/4. Choose from:

    - `'quarter'`, `'half'`, `'whole'`, `'eighth'`, `'sixteenth'`,
        `'32th'`, `'quintuplet'`.
    - `'eighth 3'` means eight note triplets. It can be used for getting
        triplets of any other duration.
    - `'half dotted'` refers to a dotted duration. It can be added to a
        string of any duration.

    Parameters
    ----------
    note : str
        String
    bpm : float
        Beats per minute to define rhythm.

    Returns
    -------
    float
        Time period in s corresponding to the musical rhythm.

    """
    return 1/get_frequency_from_musical_rhythm(note, bpm)


if __name__ == '__main__':
    # Check functions
    import matplotlib.pyplot as plt

    x = np.zeros(1000)
    n = np.random.normal(0, 0.3, 200)
    x[200:400] += n
    x[600:800] += n
    x += np.random.normal(0, 0.01, 1000)
    # plt.plot(x)

    x = _harmonic_oscillator(1, 50, 50, True, 0)
    # x = _square_oscillator(2, 20, 21, True, 0)
    # plt.plot(x)
    # for i in np.linspace(1, 10, 9):
    # x = _triangle_oscillator(10, 20, 21, True, 0)

    # plt.plot(x)
    # x = _sawtooth_oscillator(1, 200, 2001, True, 10)
    plt.plot(x)
    # plt.plot(y)
    plt.show()
