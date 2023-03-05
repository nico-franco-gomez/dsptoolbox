"""
Low-level methods for room acoustics
"""
import numpy as np


def _reverb(h, fs_hz, mode, ir_start: int = None,
            return_ir_start: bool = False):
    """Computes reverberation time of signal.

    Parameters
    ----------
    h : `np.ndarray`
        Time series.
    fs_hz : int
        Sampling rate in Hz.
    mode : str
        Parameter for the reverberation time.
    ir_start : int, optional
        When not `None`, the index is used as the start of the impulse
        response. Default: `None`.
    return_ir_start : bool, optional
        When `True`, it returns not only reverberation time but also the
        index of the sample with the start of the impulse response.
        Default: `False`.

    Returns
    -------
    float
        Reverberation time in seconds.
    int
        Index of the start of the RIR. Only returned when
        `return_ir_start = True`.

    References
    ----------
    Regarding start of RIR: ISO 3382-1:2009-10, Acoustics - Measurement of
    the reverberation time of rooms with reference to other
    acoustical parameters. pp. 22.

    """
    # Energy decay curve
    energy_curve = h**2
    epsilon = 1e-20
    if ir_start is None:
        max_ind = _find_ir_start(h, threshold_dbfs=-20)
    else:
        max_ind = ir_start
    edc = np.sum(energy_curve) - np.cumsum(energy_curve)
    edc[edc <= 0] = epsilon
    edc = 10*np.log10(edc / edc[max_ind])
    # Reverb
    i1 = np.where(edc < -5)[0][0]
    mode = mode.upper()
    if mode == 'T20':
        i2 = np.where(edc < -25)[0][0]
    elif mode == 'T30':
        i2 = np.where(edc < -35)[0][0]
    elif mode == 'T60':
        i2 = np.where(edc < -65)[0][0]
    elif mode == 'EDT':
        i1 = np.where(edc < 0)[0][0]
        i2 = np.where(edc < -10)[0][0]
    else:
        raise ValueError('Supported modes are only T20, T30, T60 and EDT')
    # Time
    length_samp = i2 - i1
    time = np.linspace(0, length_samp/fs_hz, length_samp)
    reg = np.polyfit(time, edc[i1:i2], 1)
    if return_ir_start:
        return (60 / np.abs(reg[0])), ir_start
    return (60 / np.abs(reg[0]))


def _find_ir_start(ir, threshold_dbfs=-20):
    """Find start of an IR using a threshold. Done for 1D-arrays.

    """
    energy_curve = ir**2
    energy_curve_db = 10*np.log10(
        np.clip(energy_curve/np.max(energy_curve), a_min=1e-30, a_max=None))
    ind = np.where(energy_curve_db > threshold_dbfs)[0][0] - 1
    if ind < 0:
        ind = 0
    return ind


def _complex_mode_identification(spectra: np.ndarray, n_functions: int = 1) ->\
        np.ndarray:
    """Complex transfer matrix and CMIF from:
    http://papers.vibetech.com/Paper17-CMIF.pdf

    Parameters
    ----------
    spectra : `np.ndarray`
        Matrix containing spectra of the necessary IR.
    n_functions : int, optional
        Number of singular value vectors to be returned. Default: 1.

    Returns
    -------
    cmif : `np.ndarray`
        Complex mode identificator function (matrix).

    References
    ----------
    http://papers.vibetech.com/Paper17-CMIF.pdf

    """
    assert n_functions <= spectra.shape[1], f'{n_functions} is too many ' +\
        f'functions for spectra of shape {spectra.shape}'

    n_rir = spectra.shape[1]
    H = np.zeros((n_rir, n_rir, spectra.shape[0]), dtype='cfloat')
    for n in range(n_rir):
        H[0, n, :] = spectra[:, n]
        H[n, 0, :] = spectra[:, n]  # Conjugate?!
    cmif = np.empty((spectra.shape[0], n_functions))
    for ind in range(cmif.shape[0]):
        v, s, u = np.linalg.svd(H[:, :, ind])
        for nf in range(n_functions):
            cmif[ind, nf] = s[nf]
    return cmif


def _sum_magnitude_spectra(magnitudes: np.ndarray) -> np.ndarray:
    """np.sum of all magnitude spectra

    Parameters
    ----------
    magnitudes : `np.ndarray`
        The magnitude spectra. If complex, it is assumed to be the spectra.

    Returns
    -------
    summed : `np.ndarray`
        np.sum of magnitude spectra.

    """
    if np.iscomplexobj(magnitudes):
        magnitudes = abs(magnitudes)
    summed = np.sum(magnitudes, axis=1)
    return summed


def _generate_rir(room_dim, alpha, s_pos, r_pos, rt, sr) -> np.ndarray:
    """Generate RIR using image source model according to Brinkmann, et al.

    Parameters
    ----------
    room_dim : `np.ndarray`
        Room dimensions in meters.
    alpha : float
        Mean absorption coefficient of the room.
    s_pos : `np.ndarray`
        Source position.
    r_pos : `np.ndarray`
        Receiver position.
    rt : float
        Desired reverberation time to achieve in RIR.
    sr : int
        Sampling rate in Hz.

    Returns
    -------
    rir : `np.ndarray`
        Time vector of the RIR.

    References
    ----------
    - Brinkmann, Fabian & Erbes, Vera & Weinzierl, Stefan. (2018). Extending
      the closed form image source model for source directivity.

    """
    # Beta coefficient same for all walls – could be easily expanded to be
    # different for all walls
    beta = np.sqrt(1 - alpha)

    # Speed of sound
    c = 343
    # Estimated maximum order for computation based on reverberation time
    t_max = rt*1.1
    l_max = c*t_max/2/room_dim
    LIMIT = np.ceil(np.sqrt(l_max @ l_max)).astype(int)

    # Initialize empty vector
    rir_vec = np.zeros(int(t_max*5 * sr))

    def seconds2samples(t):
        return np.asarray(t*sr).astype(int)

    # Vectorized computation of nested sums U (Eq. 2)
    u_vectors = np.array([
        [0, 0, 0],
        [0, 0, 1], [0, 1, 0], [1, 0, 0],
        [0, 1, 1], [1, 0, 1], [1, 1, 0],
        [1, 1, 1]
    ])
    # Helper matrix for vectorized computation
    helper_matrix = np.zeros((u_vectors.shape[0]*u_vectors.shape[1], 1))
    helper_matrix[:u_vectors.shape[1], 0] = 1
    for _ in range(1, u_vectors.shape[0]):
        helper_matrix = np.append(
            helper_matrix,
            np.roll(helper_matrix[:, -1], u_vectors.shape[1])[..., None],
            axis=-1)

    # Distance (according to Eq. 6)
    def get_distance(lvec):
        pos = (((1 - 2*u_vectors)*s_pos) +
               (2*lvec*room_dim) - r_pos).flatten()**2
        return (pos @ helper_matrix)**0.5

    # Damping term (Numerator in Eq. 8)
    def get_damping(lvec):
        diff = np.abs(lvec - u_vectors)
        return np.prod(beta**diff, axis=1)*np.prod(beta**np.abs(lvec))

    # Core computation (Eq. 1) – could be further optimized by vectorizing
    # the outer loops
    for lind in np.arange(-LIMIT, LIMIT+1):
        for mind in np.arange(-LIMIT, LIMIT+1):
            for nind in np.arange(-LIMIT, LIMIT+1):
                l0 = np.array([lind, mind, nind])
                # Distances
                ds = get_distance(l0)
                # Write into RIR
                rir_vec[seconds2samples(ds/c)] += \
                    get_damping(l0) / (4*np.pi*ds)
    return rir_vec


class Room():
    """This class contains a room with its parameters and metadata.

    """
    def __init__(self, volume_m3: float, area_m2: float,
                 t60_s: float = None, absorption_coefficient: float = None):
        """Constructor for a generic Room. The passed reverberation time
        is checked for the volume and area.

        Parameters
        ----------
        volume_m3 : float
            Room volume in cubic meters.
        area_m2 : float
            Room area in square meters.
        t60_s : float, optional
            Reverberation time T60 in seconds. Pass `None` to define it from
            a mean absorption coefficient. Default: `None`.
        absorption_coefficient : float, optional
            Mean absorption coefficient for the room. It should be between 0
            and 1. Pass `None` to compute automatically from the reverberation
            time. Default: `None`.

        Attributes
        ----------
        - volume, area, t60_s (reverberation time in seconds),
          schroeders_frequency (in Hz), critical_distance_m (in meters, for an
          omnidirectional source).

        """
        assert area_m2 > 0, \
            'Room surface area has to be positive'
        self.volume = volume_m3
        self.area = area_m2

        assert (t60_s is None) ^ (absorption_coefficient is None), \
            'Either reverberation time or absorption coefficient should ' +\
            'not be None'
        if t60_s is None:
            assert absorption_coefficient > 0 and absorption_coefficient <= 1,\
                'Absorption coefficient should be ]0, 1]'
            self.absorption_coefficient = absorption_coefficient
            self.t60_s = 0.161 * self.volume / self.area / \
                self.absorption_coefficient
        if absorption_coefficient is None:
            absorption_coefficient = 0.161 * self.volume / self.area / t60_s
            assert absorption_coefficient > 0 and absorption_coefficient <= 1,\
                'Given reverberation time is not valid. Absorption ' +\
                'coefficient should be ]0, 1] and not ' +\
                f'{absorption_coefficient}'
            self.t60_s = t60_s
            self.absorption_coefficient = absorption_coefficient

        # Derived values
        self.schroeders_frequency = 2000 * np.sqrt(self.t60_s / self.volume)

        # Critical distance
        self.critical_distance_m = 0.057*np.sqrt(self.volume/self.t60_s)

    # ============== Properties ===============================================
    @property
    def volume(self):
        return self.__volume

    @volume.setter
    def volume(self, new_volume):
        assert new_volume > 0, \
            'Room volume has to be positive'
        self.__volume = new_volume

    @property
    def area(self):
        return self.__area

    @area.setter
    def area(self, new_area):
        assert new_area > 0, \
            'Room volume has to be positive'
        self.__area = new_area

    def modal_density(self, f_hz: float | np.ndarray, c: float = 343) -> \
            float | np.ndarray:
        """Compute and return the modal density for a given cut-off frequency
        and speed of sound.

        Parameters
        ----------
        f_hz : float or `np.ndarray`
            Frequency or array of frequencies.
        c : float, optional
            Speed of sound in m/s. Default: 343.

        Returns
        -------
        float or `np.ndarray`
            Modal density.

        """
        return 4*np.pi*f_hz**2 * self.volume / c**3 +\
            np.pi*f_hz*self.area / 2 / c**2


class ShoeboxRoom(Room):
    """Class for a shoebox room.

    """
    def __init__(self, dimensions_m, t60_s: float = None,
                 absorption_coefficient: float = None):
        """Constructor for a generic shoebox room.

        Parameters
        ----------
        dimensions_m : array-like
            Dimensions in meters. It should be a vector containing x, y and z
            dimensions. It is assumed that the room starts at the origin
            and a right-hand cartesian system is used.
        t60_s : float, optional
            Reverberation time T60 in seconds. Pass `None` to compute it
            through the absorption coefficient using Sabine's formula.
            Default: `None`.
        absorption_coefficient : float, optional
            Mean absorption coefficient for the room. Pass `None` to compute
            it from the passed reverberation time using Sabine's formula. An
            assertion error is raised if the computed value for the absorption
            coefficient is larger than 1. Default: `None`.

        Attributes
        ----------
        - volume, area, t60_s (reverberation time in seconds),
          schroeders_frequency (in Hz), critical_distance_m (for an
          omnidirectional source), mixing_time_s (mixing time in seconds),
          modes_hz.

        """
        dimensions_m = np.atleast_1d(np.squeeze(dimensions_m))
        assert len(dimensions_m) == 3, \
            'Dimensions for a shoebox room should have length 3 (x, y, z)'
        assert np.all(dimensions_m > 0), \
            'Room dimensions must be positive'
        self.dimensions_m = dimensions_m
        volume = np.prod(dimensions_m)
        area = np.roll(dimensions_m, 1) @ dimensions_m * 2
        super().__init__(volume, area, t60_s, absorption_coefficient)

    def check_if_in_room(self, coordinates_m) -> bool:
        """Checks if a given point is inside the room.

        Parameters
        ----------
        coordinates_m : array-like
            Coordinates of point in meters. It is assumed that the order is
            x, y, z.

        Returns
        -------
        bool
            `True` if point is in the room, `False` otherwise.

        """
        coordinates_m = np.squeeze(coordinates_m)
        return np.all(coordinates_m <= self.dimensions_m)

    def get_mixing_time(self, mode: str = 'perceptual',
                        n_reflections: int = 400,
                        c: float = 343) -> float:
        """Computes and returns mixing time defined as the time where early
        reflections end and late reflections start. For this, two options are
        implemented: either a perceptual estimation presented in [1]
        (eq. 13) or a physical model that takes into account the reflections
        density after which the late reverberant tail of the IR starts
        (corresponds to eq.1 in [1]).

        The result will be saved in the `mixing_time_s` object's property.

        Parameters
        ----------
        mode : str, optional
            Choose from `'perceptual'` or `'physical'`.
            Default: `'perceptual'`.
        n_reflections : int, optional
            Necessary only when `mode='physical'`. This is the reflections
            density that is reached when the late reverberation starts.
            Default: 400.
        c : float, optional
            Necessary only when `mode='physical'`. Speed of sound.
            Default: 343.

        Returns
        -------
        mixing_time_s : float
            Mixing time in seconds.

        References
        ----------
        - [1]: Lindau A.; Kosanke, L.; Weinzierl S. (2012): "Perceptual
          evaluation of model- and signalbased predictors of the mixing time
          in binaural room impulse responses", In: J. Audio Eng. Soc. 60
          (11), pp. 887-898.

        """
        mode = mode.lower()
        assert mode in ('perceptual', 'physical'), \
            f'{mode} is not supported. Use perceptual or physical'
        mixing_time_s = 0
        if mode == 'perceptual':
            mixing_time_s = (np.sqrt(self.volume) * 0.58 + 21.2)*1e-3
        else:
            assert n_reflections > 0,\
                'n_reflections must be positive'
            mixing_time_s = np.sqrt(n_reflections*self.volume/(4*np.pi*c**3))
        self.mixing_time_s = mixing_time_s
        return self.mixing_time_s

    def get_room_modes(self, max_order: int = 6, c: float = 343) -> np.ndarray:
        """Computes and returns room modes for a shoebox room assuming
        hard reflecting walls.

        The result is returned and saved in the `modes_hz` property of this
        ShoeboxRoom.

        Parameters
        ----------
        max_order : int, optional
            Maximum mode order to compute. Default: 6.
        c : float, optional
            Speed of sound in meters/seconds. Default: 343.

        Returns
        -------
        modes : np.ndarray
            Array containing the frequencies of the room modes as well as
            their characteristics (orders in each room dimension. This is
            necessary to know if it is an axial, a tangential or oblique mode).
            Its shape is (frequency, order x, order y, order z)

        """
        max_order += 1
        modes = np.zeros((max_order**3, 4))
        counter = 0
        for nx in range(max_order):
            for ny in range(max_order):
                for nz in range(max_order):
                    freq = c/2*np.sqrt(
                        (nx/self.dimensions_m[0])**2 +
                        (ny/self.dimensions_m[1])**2 +
                        (nz/self.dimensions_m[2])**2)
                    modes[counter, :] = np.array([freq, nx, ny, nz])
                    counter += 1
        modes = modes[1:]  # Prune first (trivial) entry
        self.modes_hz = modes[modes[:, 0].argsort()]
        return self.modes_hz


def _add_reverberant_tail_noise(rir: np.ndarray, mixing_time_s: int,
                                t60: float, sr: int) -> np.ndarray:
    """Adds a reverberant tail as noise to an IR.

    Parameters
    ----------
    rir : `np.ndarray`
        Impulse response as 1D-array.
    mixing_time_s : int
        Mixing time in samples.
    t60 : float
        Reverberation time in seconds.
    sr : int
        Sampling rate in Hz.

    Returns
    -------
    rir_late : `np.ndarray`
        RIR with added decaying noise as late reverberant tail.

    """
    # Find first sample
    ind_direct = np.squeeze(np.where(rir != 0))[0]

    # Define noise length
    mixing_time_samples = int(mixing_time_s * sr)
    noise_length = len(rir) - ind_direct - mixing_time_samples

    # Generate decaying noise (normalized)
    noise = np.abs(np.random.normal(0, 1, noise_length))
    delta = 0.02*343/t60
    noise *= np.exp(-delta*np.arange(noise_length)/sr)
    noise /= np.max(noise)

    # Find right amplitude by looking at a window around start of noise
    window = 100
    window = rir[-noise_length-window//2:-noise_length+window//2]
    gain = np.median(window[window != 0])*0.5
    noise *= gain

    # Apply noise
    indexes = rir[-noise_length:] == 0
    rir[-noise_length:][indexes] += noise[indexes]
    return rir


if __name__ == '__main__':
    print()
    # r = Room(200, 100, 0.35, None)
    # print(r.absorption_coefficient)
    # print(r.modal_density(100, c=343))
    # r = Room(200, 100, None, 0.1)
    # print(r.t60_s)
    r = ShoeboxRoom([3, 4, 5], absorption_coefficient=0.9)
    bla = r.get_room_modes(8)
    print(bla.shape)
    print(r.critical_distance_m)
    print(r.t60_s)
    print(r.get_mixing_time())
    # print(bla)
