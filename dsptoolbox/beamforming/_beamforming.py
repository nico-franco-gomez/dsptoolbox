"""
Backend for beamforming module
"""
import numpy as np
import matplotlib.pyplot as plt
from seaborn import set_style
set_style('whitegrid')


class BasePoints():
    """This is a base class for saving point data (like grids or mic arrays).

    """
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
        for i in ('x', 'y', 'z'):
            assert i in positions, \
                f'{i} values are missing'
        x = np.asarray(positions['x']).squeeze()[None, ...]
        y = np.asarray(positions['y']).squeeze()[None, ...]
        z = np.asarray(positions['z']).squeeze()[None, ...]
        assert x.shape == y.shape and x.shape == z.shape, \
            'Shapes of x, y or z are not compatible'
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
        assert type(new_r) == np.ndarray,\
            'R vectors array should be of type numpy.ndarray'
        # Check if grid is 1, 2 or 3D
        ndimensions = 3
        dimensions = ['x', 'y', 'z']
        base_dimensions = ['x', 'y', 'z']
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
        dims = ['x', 'y', 'z']
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
            Point to which to compute the distances from all points saved in.

        Returns
        -------
        distances : `np.ndarray`
            Distances with shape (points).

        """
        assert len(point) == self.coordinates.shape[1], \
            f'Invalid shapes: {point.shape}, {self.coordinates.shape}'
        # Make helper matrix if needed
        if not hasattr(self, 'distance_helper_matrix'):
            self._compute_distance_helper_matrix()
        if self._distance_helper_matrix is None:
            self._compute_distance_helper_matrix()
        distances = self.coordinates - point
        return (distances.flatten()**2 @ self._distance_helper_matrix)**0.5

    # Helper matrix for vectorized computation
    def _compute_distance_helper_matrix(self):
        """Compute helper matrix for euclidean distances. Only intended as an
        internal method.

        """
        helper_matrix = \
            np.zeros((self.coordinates.shape[0]*self.coordinates.shape[1], 1))
        helper_matrix[:self.coordinates.shape[1], 0] = 1
        for _ in range(1, self.coordinates.shape[0]):
            helper_matrix = np.append(
                helper_matrix,
                np.roll(helper_matrix[:, -1],
                        self.coordinates.shape[1])[..., None],
                axis=-1)
        self._distance_helper_matrix = helper_matrix

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
        if self.ndim == 3 or projection == '3d':
            projection = '3d'
            threed = True
        elif projection in (None, '2d'):
            threed = False
            projection = None
        else:
            raise ValueError('projection must be 2d, 3d or None')

        fig, ax = plt.subplots(1, 1, figsize=(7, 5),
                               subplot_kw={'projection': projection})
        if threed:
            ax.scatter(
                xs=self.coordinates[:, 0], ys=self.coordinates[:, 1],
                zs=self.coordinates[:, 2],
                )
            ax.set_xlabel('$x$ / m')
            ax.set_ylabel('$y$ / m')
            ax.set_zlabel('$z$ / m')
        else:
            # Get right coordinates
            helper = dict(x=0, y=1, z=2)
            dim1 = helper[self.dim[0]]
            if self.ndim == 1:
                dim2 = dim1-1
            else:
                dim2 = helper[self.dim[1]]
            ax.scatter(
                x=self.coordinates[:, dim1], y=self.coordinates[:, dim2],
            )
            ax.set_xlabel(f'${self.dim[0]}$ / m')
            ax.set_ylabel(f'''${['x', 'y', 'z'][dim2]}$ / m''')
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
        assert point.ndim == 1, \
            'Passed vector is not broadcastable to a 1D-array'
        assert len(point) == 3, \
            'Point must have exactly 3 dimensions (x, y, z)'
        dist = self.get_distances_to_point(point)
        index = np.argmin(dist)
        coord = self.coordinates[index, :]
        return index, coord
