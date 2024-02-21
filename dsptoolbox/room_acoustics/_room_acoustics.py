"""
Low-level methods for room acoustics
"""

import numpy as np
from scipy.stats import pearsonr
from warnings import warn
from scipy.signal import hilbert, lfilter
from ..plots import general_plot
from .._general_helpers import _get_smoothing_factor_ema


def _reverb(
    h,
    fs_hz,
    mode,
    ir_start: int | None = None,
    return_ir_start: bool = False,
    automatic_trimming: bool = True,
):
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
    trim_ending : bool, optional
        When `True`, signal's power is trimmed to the first point falling below
        a threshold before computing the energy decay curve.

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
    if ir_start is None:
        max_ind = _find_ir_start(h, threshold_dbfs=-20)
    else:
        max_ind = ir_start

    edc = _compute_energy_decay_curve(h, max_ind, automatic_trimming, fs_hz)
    time_vector = np.linspace(0, len(edc) / fs_hz, len(edc))

    # Reverb
    if mode == "TOPT":
        if return_ir_start:
            return _obtain_optimal_reverb_time(time_vector, edc)[0], ir_start
        return _obtain_optimal_reverb_time(time_vector, edc)[0]

    mode = mode.upper()
    if mode == "T20":
        p, _ = _get_polynomial_coeffs_from_edc(time_vector, edc, -5, -25)
    elif mode == "T30":
        p, _ = _get_polynomial_coeffs_from_edc(time_vector, edc, -5, -35)
    elif mode == "T60":
        p, _ = _get_polynomial_coeffs_from_edc(time_vector, edc, -5, -65)
    elif mode == "EDT":
        p, _ = _get_polynomial_coeffs_from_edc(time_vector, edc, 0, -10)
    else:
        raise ValueError("Supported modes are only T20, T30, T60 and EDT")

    factor = 60 if mode != "EDT" else 10

    if return_ir_start:
        return (factor / np.abs(p[0])), ir_start
    return factor / np.abs(p[0])


def _find_ir_start(ir, threshold_dbfs: float = -20) -> int:
    """Find start of an IR using a threshold. Done for 1D-arrays.

    Parameters
    ----------
    ir : `np.ndarray`
        IR as a 1D-array.
    threshold_dbfs : float, optional
        Threshold that should be surpassed at the start of the IR in dBFS.
        The signal is always normalized. Default: -20.

    Returns
    -------
    ind : int
        Index of the start of the IR. It is the sample before the given
        threshold is surpassed for the first time.

    """
    energy_curve = ir**2
    energy_curve_db = 10 * np.log10(
        np.clip(energy_curve / np.max(energy_curve), a_min=1e-30, a_max=None)
    )
    ind = int(np.where(energy_curve_db > threshold_dbfs)[0][0] - 1)
    if ind < 0:
        ind = 0
    return ind


def _complex_mode_identification(
    spectra: np.ndarray, maximum_singular_value: bool = True
) -> np.ndarray:
    """Complex transfer matrix and CMIF from:
    http://papers.vibetech.com/Paper17-CMIF.pdf

    Parameters
    ----------
    spectra : `np.ndarray`
        Matrix containing spectra of the necessary IR.
    maximum_singular_value : bool, optional
        When `True`, the maximum singular value at each frequency line is
        returned instead of the first. Default: `True`.

    Returns
    -------
    cmif : `np.ndarray`
        Complex mode identificator function.

    References
    ----------
    http://papers.vibetech.com/Paper17-CMIF.pdf

    """
    n_rir = spectra.shape[1]

    # If only one RIR is provided, then there is no need to compute the SVD
    if n_rir == 1:
        return np.abs(spectra.squeeze()) ** 2

    H = np.zeros((n_rir, n_rir, spectra.shape[0]), dtype="cfloat")
    for n in range(n_rir):
        H[0, n, :] = spectra[:, n]
        H[n, 0, :] = spectra[:, n]
    cmif = np.zeros(spectra.shape[0])
    for ind in range(cmif.shape[0]):
        s = np.linalg.svd(H[:, :, ind], compute_uv=False, hermitian=False)
        if maximum_singular_value:
            cmif[ind] = s.max()
        else:
            cmif[ind] = s[0]
    return cmif


def _generate_rir(room_dim, alpha, s_pos, r_pos, rt, mo, sr) -> np.ndarray:
    """Generate RIR using image source model according to Brinkmann, et al.

    Parameters
    ----------
    room_dim : `np.ndarray`
        Room dimensions in meters.
    alpha : float or `np.ndarray`
        Mean absorption coefficient of the room or array with the absorption
        coefficient for each wall (length 6. Ordered as north, south, east,
        west, floor, ceiling).
    s_pos : `np.ndarray`
        Source position.
    r_pos : `np.ndarray`
        Receiver position.
    rt : float
        Desired reverberation time to achieve in RIR.
    mo : int
        Maximum order of reflections.
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
    # Beta coefficient for all walls
    beta = np.atleast_1d(np.sqrt(1 - alpha))
    if len(beta) == 1:
        beta_1 = np.ones(3) * beta
        beta_2 = np.ones(3) * beta
    elif len(beta) == 6:
        beta_1 = np.array(
            [beta[1], beta[3], beta[4]]
        )  # South  # West  # Floor
        beta_2 = np.array(
            [beta[0], beta[2], beta[5]]
        )  # North  # East  # Ceiling
    else:
        raise ValueError("Wrong length for absorption coefficients")

    # Speed of sound
    c = 343
    # Estimated maximum order for computation based on reverberation time
    t_max = rt * 1.1
    l_max = c * t_max / 2 / room_dim
    LIMIT = np.ceil(np.sqrt(l_max @ l_max)).astype(int)
    if mo is not None:
        LIMIT = LIMIT if mo > LIMIT else mo

    # Initialize empty vector
    rir_vec = np.zeros(int(t_max * 5 * sr))

    def seconds2samples(t):
        return np.asarray(t * sr + 0.5).astype(int)

    # Vectorized computation of nested sums U (Eq. 2)
    u_vectors = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )  # Shape (8, 3)

    # Helper matrix for vectorized computation
    helper_matrix = np.zeros((u_vectors.shape[0] * u_vectors.shape[1], 1))
    helper_matrix[: u_vectors.shape[1], 0] = 1
    for _ in range(1, u_vectors.shape[0]):
        helper_matrix = np.append(
            helper_matrix,
            np.roll(helper_matrix[:, -1], u_vectors.shape[1])[..., None],
            axis=-1,
        )

    # Distance (according to Eq. 6)
    # Using scipy's norm (scipy.linalg.norm) was somewhat slower...
    def get_distance(lvec):
        pos = (
            ((1 - 2 * u_vectors) * s_pos) + (2 * lvec * room_dim) - r_pos
        ).flatten() ** 2
        return (pos @ helper_matrix) ** 0.5

    # Damping term (Numerator in Eq. 8)
    def get_damping(lvec):
        diff = np.abs(lvec - u_vectors)
        return np.prod(beta_1**diff, axis=1) * np.prod(beta_2 ** np.abs(lvec))

    # Core computation (Eq. 1) – could be further optimized by vectorizing
    # the outer loops
    limit_loop = np.arange(-LIMIT, LIMIT + 1)
    for lind in limit_loop:
        for mind in limit_loop:
            for nind in limit_loop:
                l0 = np.array([lind, mind, nind])
                # Distances
                ds = get_distance(l0)
                # Write into RIR
                rir_vec[seconds2samples(ds / c)] += get_damping(l0) / (
                    4 * np.pi * ds
                )
    return rir_vec


class Room:
    """This class contains a room with its parameters and metadata."""

    def __init__(
        self,
        volume_m3: float,
        area_m2: float,
        t60_s: float | None = None,
        absorption_coefficient: float | None = None,
    ):
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
        assert area_m2 > 0, "Room surface area has to be positive"
        self.volume = volume_m3
        self.area = area_m2

        assert (t60_s is None) ^ (absorption_coefficient is None), (
            "Either reverberation time or absorption coefficient should "
            + "not be None"
        )
        if t60_s is None:
            assert (
                absorption_coefficient > 0 and absorption_coefficient <= 1
            ), "Absorption coefficient should be ]0, 1]"
            self.absorption_coefficient = absorption_coefficient
            self.t60_s = (
                0.161 * self.volume / self.area / self.absorption_coefficient
            )
        if absorption_coefficient is None:
            absorption_coefficient = 0.161 * self.volume / self.area / t60_s
            assert (
                absorption_coefficient > 0 and absorption_coefficient <= 1
            ), (
                "Given reverberation time is not valid. Absorption "
                + "coefficient should be ]0, 1] and not "
                + f"{absorption_coefficient}"
            )
            self.t60_s = t60_s
            self.absorption_coefficient = absorption_coefficient

        # Derived values
        self.schroeders_frequency = 2000 * np.sqrt(self.t60_s / self.volume)

        # Critical distance
        self.critical_distance_m = 0.057 * np.sqrt(self.volume / self.t60_s)

    # ============== Properties ===============================================
    @property
    def volume(self):
        return self.__volume

    @volume.setter
    def volume(self, new_volume):
        assert new_volume > 0, "Room volume has to be positive"
        self.__volume = new_volume

    @property
    def area(self):
        return self.__area

    @area.setter
    def area(self, new_area):
        assert new_area > 0, "Room volume has to be positive"
        self.__area = new_area

    def modal_density(
        self, f_hz: float | np.ndarray, c: float = 343
    ) -> float | np.ndarray:
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
        return (
            4 * np.pi * f_hz**2 * self.volume / c**3
            + np.pi * f_hz * self.area / 2 / c**2
        )


class ShoeboxRoom(Room):
    """Class for a shoebox room."""

    def __init__(
        self,
        dimensions_m,
        t60_s: float | None = None,
        absorption_coefficient: float | None = None,
    ):
        """Constructor for a shoebox-type room.

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
            Mean absorption coefficient for the room. Here it is frequency-
            and wall-independent. Pass `None` to compute it from the passed
            reverberation time using Sabine's formula. An assertion error is
            raised if the computed value for the absorption coefficient is
            larger than 1. Default: `None`.

        Attributes
        ----------
        - General: volume, area.
        - Acoustics: t60_s (reverberation time in seconds),
          absorption_coefficient (mean absorption, frequency- and
          wall-independent), schroeders_frequency (in Hz), critical_distance_m
          (for an omnidirectional source), mixing_time_s
          (mixing time in seconds), modes_hz.

        Methods
        -------
        - check_if_in_room, get_mixing_time, get_room_modes,
          get_analytical_transfer_function.

        """
        dimensions_m = np.atleast_1d(np.squeeze(dimensions_m))
        assert (
            len(dimensions_m) == 3
        ), "Dimensions for a shoebox room should have length 3 (x, y, z)"
        assert np.all(dimensions_m > 0), "Room dimensions must be positive"
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

    def get_mixing_time(
        self,
        mode: str = "perceptual",
        n_reflections: int = 400,
        c: float = 343,
    ) -> float:
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
        assert mode in (
            "perceptual",
            "physical",
        ), f"{mode} is not supported. Use perceptual or physical"
        mixing_time_s = 0
        if mode == "perceptual":
            mixing_time_s = (np.sqrt(self.volume) * 0.58 + 21.2) * 1e-3
        else:
            assert n_reflections > 0, "n_reflections must be positive"
            mixing_time_s = np.sqrt(
                n_reflections * self.volume / (4 * np.pi * c**3)
            )
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
            Its shape is (mode, frequency and order xyz).

        """
        max_order += 1
        modes = np.zeros((max_order**3, 4))
        counter = 0

        max_order_loop = np.arange(max_order)
        for nx in max_order_loop:
            for ny in max_order_loop:
                for nz in max_order_loop:
                    freq = (
                        c
                        / 2
                        * np.sqrt(
                            (nx / self.dimensions_m[0]) ** 2
                            + (ny / self.dimensions_m[1]) ** 2
                            + (nz / self.dimensions_m[2]) ** 2
                        )
                    )
                    modes[counter, :] = np.array([freq, nx, ny, nz])
                    counter += 1
        modes = modes[1:]  # Prune first (trivial) entry
        self.modes_hz = modes[modes[:, 0].argsort()]
        return self.modes_hz

    def get_analytical_transfer_function(
        self,
        source_pos,
        receiver_pos,
        freqs,
        max_mode_order: int = 10,
        generate_plot: bool = True,
        c: float = 343,
    ):
        """Compute and return the analytical transfer function for the room.

        Parameters
        ----------
        source_pos : array-like
            Source position in meters. It must be inside the room, otherwise
            an assertion error is raised.
        receiver_pos : array-like
            Receiver position in meters. It must be inside the room, otherwise
            an assertion error is raised.
        freqs : `np.ndarray`
            Frequency vector for which to compute the transfer function.
        max_mode_order : int, optional
            Maximum mode order to be regarded. It should be high enough to
            represent the frequency response at the relevant frequencies.
            Default: 10.
        generate_plot : bool, optional
            Generates and returns a plot showing the transfer function
            (normalized by peak value). Default: `True`.
        c : float, optional
            Speed of sound in meters/seconds. Default: 343.

        Returns
        -------
        p : `np.ndarray`
            Complex transfer function, non-normalized.
        modes : `np.ndarray`
            Modes for which the transfer function was computed. It has shape
            (mode, frequency and order xyz) and it is sorted by
            frequency.
        plot : tuple
            When `generate_plot=True`, this is a tuple containing two entries:
            `(matplotlib.figure.Figure, matplotlib.axes.Axes)`. When `False`,
            `None` is returned as the content of the tuple

        """
        source_pos = np.asarray(source_pos).squeeze()
        receiver_pos = np.asarray(receiver_pos).squeeze()
        assert self.check_if_in_room(
            source_pos
        ), "Given source position is not in the room"
        assert self.check_if_in_room(
            receiver_pos
        ), "Given receiver position is not in the room"

        if hasattr(self, "detailed_absorption"):
            # Absorption for each mode taken from the respective octave band
            mode_damping = (
                np.log(1e3) / self.detailed_absorption["t60_s_per_frequency"]
            )
            alpha_freq_dep = True
            octave_bands = self.detailed_absorption["center_frequencies"]
        else:
            # Damping for all modes is assumed to be equal
            alpha_freq_dep = False
            mode_damping = np.log(1e3) / self.t60_s

        # Frequency vectors
        f = np.asarray(freqs).squeeze()
        omega = 2 * np.pi * f
        omega_2 = omega**2

        # Lookup table for some values
        cn_vals = np.array([4, 2, 1])

        # Maximum order
        max_mode_order += 1

        p = np.zeros(len(omega), dtype="cfloat")
        counter = 0
        modes = np.zeros((max_mode_order**3, 4))

        # Compute response
        max_order_loop = np.arange(max_mode_order)
        for nx in max_order_loop:
            for ny in max_order_loop:
                for nz in max_order_loop:
                    if counter == 0:
                        counter += 1
                        continue
                    ks = np.array(
                        [
                            nx / self.dimensions_m[0] * np.pi,
                            ny / self.dimensions_m[1] * np.pi,
                            nz / self.dimensions_m[2] * np.pi,
                        ]
                    )
                    # Frequency of mode
                    omega_n = c * np.sqrt(ks @ ks)
                    mode_freq = omega_n / 2 / np.pi

                    # Mode damping
                    if alpha_freq_dep:
                        temp_ind = np.argmin(np.abs(mode_freq - octave_bands))
                        eta = mode_damping[temp_ind]
                    else:
                        eta = mode_damping

                    # Coefficient based on type of mode
                    tom = np.sum(np.array([nx, ny, nz]).astype(bool)) - 1
                    cn = cn_vals[tom]

                    # Compute
                    p += np.prod(
                        np.cos(ks * source_pos) * np.cos(ks * receiver_pos)
                    ) / (cn * (omega_n**2 + 2j * eta * omega_n - omega_2))

                    # Save mode
                    modes[counter] = np.array([mode_freq, nx, ny, nz])
                    counter += 1
        # Factor
        p *= 8 * c**2 / np.prod(self.dimensions_m)

        # Modes
        modes = modes[1:]
        modes = modes[modes[:, 0].argsort()]

        if generate_plot:
            ind_norm = np.argmax(np.abs(p))
            plot = general_plot(
                f,
                20 * np.log10(np.abs(p)) - 20 * np.log10(np.abs(p[ind_norm])),
                range_x=[f[0], f[-1]],
                tight_layout=True,
                returns=True,
            )
            plot[1].set_ylabel("Magnitude / dBFS (norm @ Peak)")
        else:
            plot = None
        return p, modes, plot

    def add_detailed_absorption(self, detailed_absorption: dict):
        """This method allows for the room to take in a more complex
        description of the absorption in each wall. This updates the
        attributes `t60_s` and `absorption_coefficient`.

        The dictionary `detailed_absorption` must have keys `'north'`,
        `'south'`, `'east'`, `'west'`, `'floor'`, `'ceiling'`. These represent
        the six walls of the room (south, west and floor start at origin).
        For each key, absorption coefficients for up to 8 octave bands
        (with center frequencies 125, 250, 500, 1000, 2000, 4000, 8000, 16000)
        must be passed as an array. The total number of bands is derived from
        the longest coefficients array passed. Alternatively, only one
        absorption coefficient can be passed and it is then regarded as a
        frequency-independent absorption for that particular wall. If one wall
        has more than one value but less than another band, its last value is
        used for the rest of the frequency bands.

        The method `get_analytical_transfer_function` uses the updated
        coefficients and `generate_synthetic_rir` can also use them.

        Parameters
        ----------
        detailed_absorption : dict
            Dictionary containing the absorption coefficients for 8 octave
            bands of each wall of the room. A valid dictionary would be::

                detailed_absorption = {
                    'north': [0.5, 0.2, 0.3, 0.4],
                    'south': 0.47,
                    'east': [0.3, 0.1],  # This wall would be completed as
                                         # [0.3, 0.1, 0.1, 0.1]
                    'west': ... ,}

            and so forth for the remaining bands.

        Notes
        -----
        All computed parameters get saved in the detailed_absorption dictionary
        of the `ShoeboxRoom`. This dictionary also has a 'README' key with
        an information string about all other keys.

        """
        # Assertions of input
        assert len(detailed_absorption) == 6, (
            "The detailed absorption dictionary must have 6 entries (for "
            + "each wall)"
        )
        walls = set(["north", "south", "east", "west", "floor", "ceiling"])
        assert walls == set(detailed_absorption.keys()), (
            f"Keys of dictionary: {set(detailed_absorption.keys())}\ndo not"
            + f" match with the necessary keys: {walls}"
        )

        # Check absorption values and derive number of bands
        number_of_bands = 1
        for i in detailed_absorption:
            ab = np.atleast_1d(detailed_absorption[i])
            if len(ab) == 1:
                detailed_absorption[i] = ab * np.ones(8)
            elif len(ab) <= 8:
                detailed_absorption[i] = ab
                number_of_bands = (
                    number_of_bands if len(ab) < number_of_bands else len(ab)
                )
            else:
                raise ValueError(
                    "The absorption coefficient must be passed "
                    "with either 1 or less than 8 coefficients"
                )
            assert np.all(ab < 1) and np.all(
                ab > 0
            ), "Absorption must be between 0 and 1 (exclusively)"
        # Trim or pad for every wall
        for i in detailed_absorption:
            if len(detailed_absorption[i]) >= number_of_bands:
                detailed_absorption[i] = detailed_absorption[i][
                    :number_of_bands
                ]
            else:
                detailed_absorption[i] = np.pad(
                    detailed_absorption[i],
                    (0, number_of_bands - len(detailed_absorption[i])),
                    "edge",
                )

        # Get coefficients and generate mean absorption etc.
        walls_dict = {
            "north": 0,
            "south": 1,
            "east": 2,
            "west": 3,
            "floor": 4,
            "ceiling": 5,
        }
        # Matrix with shape (wall, absorption)
        absorption_matrix = np.zeros((6, number_of_bands))
        for wall in detailed_absorption:
            absorption_matrix[walls_dict[wall], :] = detailed_absorption[wall]

        # Equivalent absorption area (per frequency band)
        absorption_area = np.zeros(number_of_bands)
        xy = self.dimensions_m[0] * self.dimensions_m[1]
        absorption_area += xy * (
            absorption_matrix[walls_dict["ceiling"], :]
            + absorption_matrix[walls_dict["floor"], :]
        )
        xz = self.dimensions_m[0] * self.dimensions_m[2]
        absorption_area += xz * (
            absorption_matrix[walls_dict["south"], :]
            + absorption_matrix[walls_dict["north"], :]
        )
        yz = self.dimensions_m[1] * self.dimensions_m[2]
        absorption_area += yz * (
            absorption_matrix[walls_dict["east"], :]
            + absorption_matrix[walls_dict["west"], :]
        )

        # Get all parameters into one dictionary
        self.detailed_absorption = detailed_absorption
        self.detailed_absorption["absorption_matrix"] = absorption_matrix
        self.detailed_absorption["absorption_area"] = absorption_area
        self.detailed_absorption[
            "mean_absorption_coefficients_per_frequency"
        ] = acpf = (absorption_area / self.area)
        self.detailed_absorption["center_frequencies"] = 125 * 2 ** np.arange(
            number_of_bands
        )
        self.detailed_absorption["t60_s_per_frequency"] = (
            0.161 * self.volume / absorption_area
        )
        self.detailed_absorption["index_wall_dictionary"] = walls_dict
        self.detailed_absorption[
            "README"
        ] = """This dictionary contains all information about the room's
absorption properties. Its keys are:

- absorption_matrix : array containing absorption with shape
(wall, frequency band) = (6, number of bands). Frequencies are in increasing
order and walls indices can be retrieved with the
index_wall_dictionary.
- index_wall_dictionary : dictionary with indices for each wall.
- center_frequencies : band center frequencies.
- mean_absorption_coefficients_per_frequency.
- t60_s_per_frequency.
"""

        # Get mean absorption coefficient by a weighted average with
        # logarithmic weights
        weights = 2.0 ** np.arange(number_of_bands)
        weights /= np.sum(weights)
        self.absorption_coefficient = np.sum(acpf * weights)
        # Get new T60
        self.t60_s = (
            0.161 * self.volume / (self.absorption_coefficient * self.area)
        )


def _add_reverberant_tail_noise(
    rir: np.ndarray, mixing_time_s: int, t60: float, sr: int
) -> np.ndarray:
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
    delta = 0.02 * 343 / t60
    noise *= np.exp(-delta * np.arange(noise_length) / sr)
    noise /= np.max(noise)

    # Find right amplitude by looking at a window around start of noise
    window = 100
    window = rir[-noise_length - window // 2 : -noise_length + window // 2]
    gain = np.median(window[window != 0]) * 0.5
    noise *= gain

    # Apply noise
    indexes = rir[-noise_length:] == 0
    rir[-noise_length:][indexes] += noise[indexes]
    return rir


def _d50_from_rir(td: np.ndarray, fs: int, automatic_trimming: bool) -> float:
    """Compute definition D50 from a given RIR (1D-Array).

    Parameters
    ----------
    td : `np.ndarray`
        IR.
    fs : int
        Sampling rate in Hz.
    automatic_trimming : bool
        When `True`, the RIR is trimmed before computing.

    Returns
    -------
    float
        Definition D50 (no unit).

    """
    assert td.ndim == 1, "Only supported for 1D-Arrays"
    ind = _find_ir_start(td)
    td = td[ind:]
    window = int(50e-3 * fs)
    if automatic_trimming:
        stop = _get_stop_index_for_energy_decay_curve(td, 0, fs)
        stop = np.max([window, stop])
    else:
        stop = len(td)
    td **= 2
    return np.sum(td[:window]) / np.sum(td[:stop])


def _c80_from_rir(td: np.ndarray, fs: int, automatic_trimming: bool) -> float:
    """Compute clarity C80 from a given RIR (1D-Array).

    Parameters
    ----------
    td : `np.ndarray`
        IR.
    fs : int
        Sampling rate in Hz.
    automatic_trimming : bool
        When `True`, the RIR is trimmed before computing.

    Returns
    -------
    float
        Clarity C80 in dB.

    """
    assert td.ndim == 1, "Only supported for 1D-Arrays"
    # Trim IR
    ind = _find_ir_start(td)
    td = td[ind:]
    # Time window
    window = int(80e-3 * fs)
    if automatic_trimming:
        stop = _get_stop_index_for_energy_decay_curve(td, 0, fs)
        stop = np.max([window, stop])
    else:
        stop = len(td)
    td **= 2
    return 10 * np.log10(np.sum(td[:window]) / np.sum(td[window:stop]))


def _ts_from_rir(td: np.ndarray, fs: int, automatic_trimming: bool) -> float:
    """Compute center time from a given RIR (1D-Array).

    Parameters
    ----------
    td : `np.ndarray`
        IR.
    fs : int
        Sampling rate in Hz.
    automatic_trimming : bool
        When `True`, the RIR is trimmed before computing.

    Returns
    -------
    float
        Center time (in seconds).

    """
    assert td.ndim == 1, "Only supported for 1D-Arrays"
    # Trim IR
    ind = _find_ir_start(td)
    td = td[ind:]

    if automatic_trimming:
        stop = _get_stop_index_for_energy_decay_curve(td, 0, fs)
    else:
        stop = len(td)

    td = td[:stop] ** 2

    time_vec = np.linspace(0, len(td) / fs, len(td))
    return np.sum(td * time_vec) / np.sum(td)


def _obtain_optimal_reverb_time(
    time_vector: np.ndarray, edc: np.ndarray
) -> float:
    """Compute the optimal reverberation time by analyzing the best linear
    fit (with the smallest least-squares error) from T10 until T60. If EDT
    is much smaller than T30, the intersection of the two regression lines is
    taken as the start. Otherwise, -5 dB is the start.

    Parameters
    ----------
    time_vector : `np.ndarray`
        Time vector corresponding to the edc.
    edc : `np.ndarray`
        Energy decay curve in dB and normalized so that 0 dB corresponds to
        the impulse.

    Returns
    -------
    float
        Optimal reverberation time.
    r : float
        Pearson (linear) correlation coefficient for the regression fit.
        The closer it is to -1, the better.

    References
    ----------
    - Algorithm based on Room-EQ-Wizard's Topt. Same idea, though results might
      differ slightly.

    """
    # Invert edc for speed while using searchsorted

    coeff_edt = _get_polynomial_coeffs_from_edc(time_vector, edc, 0, -10)[0]
    coeff_t30 = _get_polynomial_coeffs_from_edc(time_vector, edc, -5, -35)[0]

    # Check EDT*60 is still smaller than T30
    very_short_edt = (-6 * 10 / coeff_edt[0]) * 10 < -60 / coeff_t30[0]

    if very_short_edt:
        x_intersection = (coeff_edt[1] - coeff_t30[1]) / (
            coeff_t30[0] - coeff_edt[0]
        )
        start: float = float(np.polyval(coeff_edt, [x_intersection]).squeeze())
    else:
        start = -5.0

    steps: np.ndarray = np.arange(start - 20, start - 60, -1)
    end, r = _get_best_linear_fit_for_edc(time_vector, edc, start, steps)
    if r > -0.95:
        warn(
            f"Correlation coefficient for reverb computation is {r} "
            + "(larger than -0.95). Computation might be invalid. "
            + "-1 is the ideal value."
        )
    coefficients = _get_polynomial_coeffs_from_edc(
        time_vector, edc, start, end
    )[0]
    return 60 / np.abs(coefficients[0]), r


def _get_best_linear_fit_for_edc(
    time_vector: np.ndarray,
    edc: np.ndarray,
    start_value: float,
    steps: np.ndarray,
):
    """Obtain the best end value for a linear regression of the EDC based on
    the lowest pearson correlation coefficient, i.e., with the maximum of
    linear correlation.

    Parameters
    ----------
    time_vector : `np.ndarray`
        Time vector.
    edc : `np.ndarray`
        Energy decay curve.
    start_value : float
        Start value of the EDC in dB for the regression.
    steps : `np.ndarray`
        Array of all ending values of the EDC in dB to take into account.

    Returns
    -------
    float
        Best end value for the linear regression in dB.
    float
        Corresponding pearson correlation coefficient.

    """
    # Invert for using searchsorted which is faster than other alternatives
    edc_inverted = edc[::-1]
    i1 = len(edc) - np.searchsorted(edc_inverted, start_value)
    rs = np.zeros(len(steps))

    for ind, step in enumerate(steps):
        i2 = len(edc) - np.searchsorted(edc_inverted, step)
        rs[ind] = pearsonr(time_vector[i1:i2], edc[i1:i2])[0]

    return steps[np.argmin(rs)], np.min(rs)


def _get_polynomial_coeffs_from_edc(
    time_vector: np.ndarray,
    edc: np.ndarray,
    start_value: float,
    end_value: float,
) -> tuple[np.ndarray, float]:
    """Return the polynomial coefficients from the energy decay curve for
    given starting and ending values. This can be used for all reverberation
    time computations.

    Parameters
    ----------
    time_vector : `np.ndarray`
        Time vector in seconds corresponding to the energy decay curve.
    edc : `np.ndarray`
        Energy decay curve in dB normalized to 0 dB at the point of the
        impulse.
    start_value : float
        Value in dB from which to start the polynomial fit.
    end_value : float
        Value in dB at which to end the polynomial fit.

    Returns
    -------
    coeff : `np.ndarray`
        Polynomial coefficients for x^1 and x^0, respectively.
    r_coefficient : float
        Pearson's correlation coefficient r. It takes values between [-1, 1]
        and the closest it is to -1, the better the fit.

    """
    L = len(edc)

    edc_inverted = edc[::-1]
    i1 = L - np.searchsorted(edc_inverted, start_value)
    i2 = L - np.searchsorted(edc_inverted, end_value)

    coeff = np.polyfit(time_vector[i1:i2], edc[i1:i2], 1)
    r_coefficient = pearsonr(time_vector[i1:i2], edc[i1:i2])[0]

    return coeff, r_coefficient


def _compute_energy_decay_curve(
    time_data: np.ndarray,
    impulse_index: int,
    trim_automatically: bool,
    fs_hz: int,
) -> np.ndarray:
    """Get the energy decay curve from an energy time curve."""
    if trim_automatically:
        # Trimming prior to impulse
        time_before_impulse = int(20e-3 * fs_hz + 0.5)
        if impulse_index - time_before_impulse > 0:
            start_index = impulse_index - time_before_impulse
            impulse_index = time_before_impulse
        else:
            start_index = 0
        # Trimming in the end
        stopping_index = (
            _get_stop_index_for_energy_decay_curve(
                time_data[start_index:], impulse_index, fs_hz
            )
            + start_index
        )
    else:
        start_index = 0
        stopping_index = len(time_data)

    signal_power = time_data[start_index:stopping_index] ** 2
    edc = np.sum(signal_power) - np.cumsum(signal_power)
    epsilon = 1e-50
    edc = 10 * np.log10(
        np.clip(edc / edc[impulse_index], a_min=epsilon, a_max=None)
    )
    return edc


def _get_stop_index_for_energy_decay_curve(
    time_data: np.ndarray, impulse_index: int, fs_hz: int
) -> int:
    """Obtain the stopping index for an energy decay curve using the smooth
    (exponential, 10 ms) envelope of the energy time curve. The threshold is
    the median of the last third of the RIR."""
    # Envelope (ETC)
    envelope = np.abs(hilbert(time_data, axis=0))
    etc = 20 * np.log10(np.clip(envelope, a_min=1e-50, a_max=None))

    # Smoothing (10 ms)
    factor = _get_smoothing_factor_ema(10e-3, fs_hz)
    envelope = lfilter([factor], [1, -(1 - factor)], etc)

    # Threshold
    threshold = np.median(envelope[int(len(envelope) * 0.66) :])
    try:
        stop = (
            np.where(envelope[impulse_index:] < threshold)[0][0]
            + impulse_index
        )
    except Exception as e:
        print(e)
        stop = 0

    if stop - impulse_index < 10:
        warn(
            "Passed impulse index or RIR might be wrong, no meaningful "
            + "trimming could be done"
        )
        stop = len(envelope)
    return stop


if __name__ == "__main__":
    print()
    # r = Room(200, 100, 0.35, None)
    # print(r.absorption_coefficient)
    # print(r.modal_density(100, c=343))
    # r = Room(200, 100, None, 0.1)
    # print(r.t60_s)
    r = ShoeboxRoom([3, 4, 5], absorption_coefficient=0.9)

    f = np.linspace(50, 1000, 2000)
    p1 = r.get_analytical_transfer_function(
        [1, 1, 1], [2, 2, 2], freqs=f, max_mode_order=15, generate_plot=False
    )[0]

    # Detailed absorption
    d = {}
    for i in ["north", "south", "east", "west", "floor", "ceiling"]:
        # d[i] = 0.1
        # d[i] = np.random.normal(0.5, 0.01, size=6)
        d[i] = np.random.uniform(0.1, 0.9, size=3)
    d["north"] = 0.6
    d["south"] = [0.1, 0.1, 0.1, 0.1, 0.1]
    r.add_detailed_absorption(d)
    print(r.detailed_absorption["center_frequencies"])
    p2 = r.get_analytical_transfer_function(
        [1, 1, 1], [2, 2, 2], freqs=f, max_mode_order=15, generate_plot=False
    )[0]
    import matplotlib.pyplot as plt

    plt.semilogx(f, 20 * np.log10(np.abs(p1)), label="mean alpha")
    plt.semilogx(f, 20 * np.log10(np.abs(p2)), label="detailed alpha")
    plt.legend()
    plt.show()
