"""
Backend for beamforming module
"""
import numpy as np
from .._general_helpers import _euclidean_distance_matrix
import matplotlib.pyplot as plt
from seaborn import set_style

set_style("whitegrid")


class BasePoints:
    """This is a base class for saving
    point data (like grids or mic arrays)."""

    # ======== Constructor ====================================================
    def __init__(self, positions: dict):
        """Initiate a grid based on the positions dictionary that contains
        all vectors.

        Parameters
        ----------
        positions : dict
            Dictionary containing point positions. Use `'x'`, `'y'` and `'z'`
            as keys to pass array-like objects with the positions.

        Attributes
        ----------
        - `coordinates`: Coordinates of the grid points as numpy.ndarray with
          shape (point, coordinate xyz).
        - `number_of_points`: Number of points contained in the array.
        - `ndim`: Number of dimensions of grid.
        - `dimensions`: Dimensions in which the points are extended.
        - `extent`: dictionary with keys `'x'=(min_x, max_x)` and analogous for
          `'y'` and `'z'`.

        """
        for i in ("x", "y", "z"):
            assert i in positions, f"{i} values are missing"
        x = np.asarray(positions["x"]).squeeze()[None, ...]
        y = np.asarray(positions["y"]).squeeze()[None, ...]
        z = np.asarray(positions["z"]).squeeze()[None, ...]
        assert (
            x.shape == y.shape and x.shape == z.shape
        ), "Shapes of x, y or z are not compatible"
        new_r = np.append(x, y, axis=0)
        new_r = np.append(new_r, z, axis=0)
        self.coordinates = new_r.T

    # ======== Properties =====================================================
    @property
    def number_of_points(self):
        return self.coordinates.shape[0]

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates.copy()

    @coordinates.setter
    def coordinates(self, new_r):
        assert isinstance(
            new_r, np.ndarray
        ), "R vectors array should be of type numpy.ndarray"
        # Check if grid is 1, 2 or 3D
        ndimensions = 3
        dimensions = ["x", "y", "z"]
        base_dimensions = ["x", "y", "z"]
        for i in range(new_r.shape[1]):
            if len(np.unique(new_r[:, i])) == 1:
                ndimensions -= 1
                dimensions.remove(base_dimensions[i])
        self.dim = dimensions
        self.ndim = ndimensions
        self._coordinates = new_r

    @property
    def extent(self):
        extent = {}
        dims = ["x", "y", "z"]
        for i, d in enumerate(dims):
            min_val = np.min(self.coordinates[:, i])
            max_val = np.max(self.coordinates[:, i])
            extent[d] = [min_val, max_val]
        return extent

    # ======== distances ======================================================
    def get_distances_to_point(self, point: np.ndarray) -> np.ndarray:
        """Compute distances (euclidean) from given point to all points of the
        object efficiently.

        Parameters
        ----------
        point : `np.ndarray`
            Point or points to which to compute the distances from all other
            points. Its shape should be (point, coordinate).

        Returns
        -------
        distances : `np.ndarray`
            Distances with shape (points, new_points).

        """
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        if point.ndim == 1:
            point = point[None, ...]
        assert (
            point.shape[1] == self.coordinates.shape[1]
        ), f"Invalid shapes: {point.shape}, {self.coordinates.shape}"
        return _euclidean_distance_matrix(self.coordinates, point).squeeze()

    # ======== Plotting =======================================================
    def plot_points(self, projection: str = None):
        """Plot points in 2D or 3D plot depending on the actual points.

        Parameters
        ----------
        projection : str, optional
            Projection for the plot. Choose from `'3d'` or `'2d'` or `None`
            to set it automatically. For 3D points, the projection will always
            be 3d. Default: `None`.

        """
        if projection is not None:
            projection = projection.lower()
        if self.ndim == 3 or projection == "3d":
            projection = "3d"
            threed = True
        elif projection in (None, "2d"):
            threed = False
            projection = None
        else:
            raise ValueError("projection must be 2d, 3d or None")

        fig, ax = plt.subplots(
            1, 1, figsize=(7, 5), subplot_kw={"projection": projection}
        )
        if threed:
            ax.scatter(
                xs=self.coordinates[:, 0],
                ys=self.coordinates[:, 1],
                zs=self.coordinates[:, 2],
            )
            ax.set_xlabel("$x$ / m")
            ax.set_ylabel("$y$ / m")
            ax.set_zlabel("$z$ / m")
        else:
            # Get right coordinates
            helper = dict(x=0, y=1, z=2)
            dim1 = helper[self.dim[0]]
            if self.ndim == 1:
                dim2 = dim1 - 1
            else:
                dim2 = helper[self.dim[1]]
            ax.scatter(
                x=self.coordinates[:, dim1],
                y=self.coordinates[:, dim2],
            )
            ax.set_xlabel(f"${self.dim[0]}$ / m")
            ax.set_ylabel(f"""${['x', 'y', 'z'][dim2]}$ / m""")
        fig.tight_layout()
        return fig, ax

    def find_nearest_point(self, point) -> tuple[int, np.ndarray]:
        """This method returns the coordinates and index of the nearest point
        to a given point using euclidean distance.

        Parameters
        ----------
        point : array-like
            Point coordinates (x, y, z) in an ordered array.

        Returns
        -------
        index : int
            Index of the nearest point.
        coord : `np.ndarray`
            Position vector with shape (x, y, z) of the nearest point.

        """
        point = np.asarray(point).squeeze()
        assert (
            point.ndim == 1
        ), "Passed vector is not broadcastable to a 1D-array"
        assert (
            len(point) == 3
        ), "Point must have exactly 3 dimensions (x, y, z)"
        dist = self.get_distances_to_point(point)
        index = np.argmin(dist)
        coord = self.coordinates[index, :]
        return index, coord


def _clean_sc_deconvolve(
    map: np.ndarray,
    csm: np.ndarray,
    h: np.ndarray,
    h_H: np.ndarray,
    maximum_iterations: int,
    remove_diagonal_csm: bool,
    safety_factor: float,
) -> np.ndarray:
    """Computes and returns the degraded csm.

    Parameters
    ----------
    map : `np.ndarray`
        Initial beamforming map to be deconvolved for a single frequency
        with shape (point).
    csm : `np.ndarray`
        Cross-spectral matrix for a single frequency with shape (mic, mic).
    h : `np.ndarray`
        Steering vector for a single frequency with shape (mic, grid point).
    h_H : `np.ndarray`
        Steering vector (hermitian transposed) for a single frequency with
        shape (grid point, mic).
    maximum_iterations : int
        Maximum number of iterations to deconvolve.
    remove_diagonal_csm : bool
        When `True`, the main diagonal of the csm is removed in each
        computation.
    safety_factor : float
        Also called loop gain, the safety factor dampens the result from
        each iteration. Should be between 0 and 1.

    Returns
    -------
    `np.ndarray`
        Deconvolved beamforming map.

    References
    ----------
    - [1]: Sijtsma P. CLEAN Based on Spatial Source Coherence. International
      Journal of Aeroacoustics. 2007;6(4):357-374.
      doi:10.1260/147547207783359459.

    """
    # CSM without diagonal
    D = csm

    # Save last CSM to check stopping criterion given in [1]
    D = np.append(D[None, ...] * 2, D[None, ...], axis=0)

    # Save powers for stopping criterion – Alternative
    # powers = np.zeros(maximum_iterations)

    second_map = np.zeros_like(map)

    # Deconvolve
    for itr in range(maximum_iterations):
        # Find maximum in map
        maximum_power_ind = np.argmax(map)
        maximum_power = map[maximum_power_ind]

        # Store maximum value
        second_map[maximum_power_ind] += maximum_power * safety_factor

        # Stopping criterion
        if np.linalg.norm(D[1, :, :], ord=1) >= np.linalg.norm(
            D[0, :, :], ord=1
        ):
            break

        # Alternatively...
        # powers[itr] = maximum_power.real
        # if np.all(maximum_power > powers[itr-3:itr-1]):
        #     break

        # Steering vector to maximum point
        w_max = h[:, maximum_power_ind]
        h_ = w_max.copy()

        # For saving computations later in loop
        w_max_squared = w_max.conjugate() * w_max
        D_ = D[1, :, :] @ w_max / maximum_power

        # Computation of G, according to [1], only a couple iterations
        # are needed; following acoular, 20 are used here
        for _ in range(20):
            H = h_.conjugate() * h_
            h_ = (D_ + H * w_max) / np.sqrt(1 + H @ w_max_squared)

        G = np.outer(h_, h_.conjugate()) * maximum_power

        if remove_diagonal_csm:
            np.fill_diagonal(G, 0)

        for gind in range(len(map)):
            # Clean map
            map[gind] -= (
                np.linalg.multi_dot([h_H[gind, :], G, h[:, gind]]).real
                * safety_factor
            )

        # Swap degraded CSM
        temp = D[1, :, :].copy()
        D[1, :, :] = D[1, :, :] - safety_factor * G
        D[0, :, :] = temp

    return second_map
