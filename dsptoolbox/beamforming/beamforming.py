"""
Beamforming classes and functions
"""
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from seaborn import set_style
from scipy.integrate import simpson
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from dsptoolbox.classes import Signal
from dsptoolbox import fractional_delay, merge_signals, pad_trim
from dsptoolbox._general_helpers import (
    _get_fractional_octave_bandwidth, _find_nearest, _pad_trim)
from ._beamforming import BasePoints, _clean_sc_deconvolve
from dsptoolbox.plots import general_matrix_plot

set_style('whitegrid')
nxs = np.newaxis


class Grid(BasePoints):
    """This class contains a grid to use for beamforming and all its metadata.
    This class is implemented only for right-hand systems with cartesian
    coordinates (in meters).

    """
    def __init__(self, positions: dict):
        """Construct a grid for beamforming by passing positions for the point
        coordinates in meters. Additionally, there is a class method
        (reconstruct_map_shape) that you can manually add to the object in
        order to obtain beamformer maps with an specific shape immediately
        instead of vector shape.

        Parameters
        ----------
        positions : dict, Pandas.DataFrame
            Dictionary or pandas dataframe containing point positions in
            meters. Use `'x'`, `'y'` and `'z'` as keys to pass array-like
            objects with the positions.

        Attributes and Methods
        ----------------------
        - `coordinates`: Coordinates of the grid points as numpy.ndarray with
          shape (point, coordinate xyz).
        - `number_of_points`: Number of points contained in the array.
        - `ndim`: Number of dimensions of grid.
        - `dimensions`: Dimensions in which the points are extended.
        - `extent`: Dictionary with the extent of points in each coordinate.
          As an example for the x direction `'x': [min(x), max(x)]`.
        - `get_distances_to_point()`: Gets all point distances to a
          given point.
        - `find_nearest_point()`: Finds the index and coordinates of
          the nearest point to a given one.

        """
        super().__init__(positions)

    def reconstruct_map_shape(self, map: np.ndarray) -> np.ndarray:
        """Placeholder for a user-defined map reconstruction. Here, it returns
        same given map. Use inheritance from the `Grid` class to overwrite this
        with an own implementation.

        Parameters
        ----------
        map : `np.ndarray`
            Map to be reshaped.

        Returns
        -------
        map : `np.ndarray`
            Reshaped map. Here with same passed shape as before.

        """
        return map


class Regular2DGrid(Grid):
    """This class creates a Grid object with a 2D, rectangular shape.

    """
    def __init__(self, line1, line2, dimensions, value3):
        """Creates a rectangular 2d grid on a coincident plane with coordinate
        system. If you wish to create a non-coincident grid do it manually and
        pass positions to Grid.

        Parameters
        ----------
        line1 : array-like
            First line with values to define grid.
        line2 : array-like
            Second line that defines grid.
        dimensions : array-like with length 2
            Array of length 2 with strings specifying in which coordinates the
            grid expands. For instance: ('x', 'z') means that `line1`
            corresponds to the x direction and `line2` corresponds to the z
            direction.
        value3: float
            Value for the third coordinate.

        Attributes and Methods
        ----------------------
        - `coordinates`: Coordinates of the grid points as numpy.ndarray with
          shape (point, coordinate xyz).
        - `number_of_points`: Number of points contained in the array.
        - `ndim`: Number of dimensions of grid.
        - `dimensions`: Dimensions in which the points are extended.
        - `extent`: Dictionary with the extent of points in each coordinate.
          As an example for the x direction `'x': [min(x), max(x)]`.
        - `get_distances_to_point()`: Gets all point distances to a
          given point.
        - `find_nearest_point()`: Finds the index and coordinates of
          the nearest point to a given one.
        - `reconstruct_map_shape()`: Reshapes flattened map according to grid's
          shape.
        - `plot_map()`: Plots a given map (reconstructed or flattened).

        """
        assert len(dimensions) == 2, \
            'dimensions must contain exactly two strings specifying to ' +\
            'which directions line1 and line2 correspond'
        assert len(np.unique(dimensions)) == len(dimensions), \
            'There are repeated dimensions'
        dimensions = [n.lower() for n in dimensions]
        self.extent_dimensions = dimensions
        value3 = np.asarray(value3).squeeze()
        assert value3.ndim == 0, \
            'value3 can only be a single value'

        line1 = np.asarray(line1).squeeze()
        line2 = np.asarray(line2).squeeze()

        # For reconstructing the matrix later
        self.original_lengths = (len(line1), len(line2))
        dim1, dim2 = np.meshgrid(line1, line2, indexing='ij')

        dim1 = dim1.flatten()
        dim2 = dim2.flatten()
        positions = np.append(dim1[..., None], dim2[..., None], axis=1)
        positions = np.append(
            positions, np.ones((len(dim1), 1))*value3, axis=1)

        # Convert to the positions dictionary
        base_dimensions = ['x', 'y', 'z']
        base_dimensions.remove(dimensions[0])
        base_dimensions.remove(dimensions[1])
        positions = {
            f'{dimensions[0]}': positions[:, 0],
            f'{dimensions[1]}': positions[:, 1],
            f'{base_dimensions[0]}': positions[:, 2],
            }
        super().__init__(positions)

    def reconstruct_map_shape(self, map_vector: np.ndarray) -> np.ndarray:
        """Reshapes the map to be a matrix that fits the grid.

        Parameters
        ----------
        map_vector : `np.ndarray`
            Map (as a vector) to be reshaped.

        Returns
        -------
        map : `np.ndarray`
            Reshaped map.

        """
        assert map_vector.ndim == 1, \
            'The passed map should be a vector (flattened)'
        assert len(map_vector) == self.number_of_points, \
            'Length of passed vector does not match the number of points'
        return map_vector.reshape(self.original_lengths)

    def plot_map(self, map: np.ndarray, range_db: float = 20) ->\
            tuple[Figure, Axes]:
        """Plot a map done with this type of grid.

        Parameters
        ----------
        map : `np.ndarray`
            Beamformer map.
        range_db : float, optional
            Range in dB to plot.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        # If map has not been reshaped by now...
        if len(map) == self.number_of_points:
            map = self.reconstruct_map_shape(map)
        assert map.shape == self.original_lengths, \
            'Map shape does not match grid shape'
        # Get right extent
        ex = self.extent
        map = 20*np.log10(np.clip(np.abs(map), a_min=1e-25, a_max=None))
        fig, ax = general_matrix_plot(
            map,
            # First dimension vertical and second dimension horizontal
            range_x=ex[self.extent_dimensions[1]],
            range_y=ex[self.extent_dimensions[0]],
            range_z=range_db,
            xlabel=self.extent_dimensions[1]+' / m',
            ylabel=self.extent_dimensions[0]+' / m',
            zlabel='dBFS', colorbar=True, lower_origin=True, returns=True)
        return fig, ax


class Regular3DGrid(Grid):
    """Class for 3D regular Grids.

    """
    def __init__(self, line_x, line_y, line_z):
        """Constructor for a regular 3D grid.

        Parameters
        ----------
        line_x : array-like
            Line to define points along the x coordinate.
        line_y : array-like
            Line to define points along the y coordinate.
        line_z : array-like
            Line to define points along the z coordinate.

        Attributes and Methods
        ----------------------
        - `coordinates`: Coordinates of the grid points in meters as
          numpy.ndarray with shape (point, coordinate xyz).
        - `number_of_points`: Number of points contained in the array.
        - `ndim`: Number of dimensions of grid.
        - `dimensions`: Dimensions in which the points are extended.
        - `extent`: Dictionary with the extent of points in each coordinate.
          As an example for the x direction `'x': [min(x), max(x)]`.
        - `get_distances_to_point()`: Gets all point distances to a
          given point.
        - `find_nearest_point()`: Finds the index and coordinates of
          the nearest point to a given one.
        - `reconstruct_map_shape()`: Reshapes flattened map according to grid's
          shape.
        - `plot_map()`: Plots a given map (reconstructed or flattened).

        """
        line_x = np.asarray(line_x).squeeze()
        line_y = np.asarray(line_y).squeeze()
        line_z = np.asarray(line_z).squeeze()
        self.lines = (line_x, line_y, line_z)
        assert all([n.ndim == 1 for n in self.lines]), \
            'Shape of lines is invalid'

        # For reconstructing the matrix later
        self.original_lengths = (len(line_x), len(line_y), len(line_z))
        xx, yy, zz = np.meshgrid(line_x, line_y, line_z, indexing='ij')
        xx = xx.flatten()
        yy = yy.flatten()
        zz = zz.flatten()

        positions = np.append(xx[..., None], yy[..., None], axis=1)
        positions = np.append(positions, zz[..., None], axis=1)

        # Convert to the positions dictionary
        positions = {'x': positions[:, 0], 'y': positions[:, 1],
                     'z': positions[:, 2]}
        super().__init__(positions)

    def reconstruct_map_shape(self, map_vector: np.ndarray) -> np.ndarray:
        """Reshapes the map to be a matrix that fits the grid.

        Parameters
        ----------
        map_vector : `np.ndarray`
            Map (as a vector) to be reshaped.

        Returns
        -------
        map : `np.ndarray`
            Reshaped map.

        """
        assert map_vector.ndim == 1, \
            'The passed map should be a vector (flattened)'
        assert len(map_vector) == self.number_of_points, \
            'Length of passed vector does not match the number of points'
        return map_vector.reshape(self.original_lengths)

    def plot_map(self, map: np.ndarray, third_dimension: str,
                 value_third_dimension: float, range_db: float = 20) ->\
            tuple[Figure, Axes]:
        """Plot a map done with this type of grid.

        Parameters
        ----------
        map : `np.ndarray`
            Beamformer map.
        third_dimension : str
            Choose the dimension that is normal to plane. Choose from `'x'`,
            `'y'` or `'z'`.
        value_third_dimension : float
            Value for third dimension that should be plotted. The nearest
            possible value will be taken if it is not exact.
        range_db : float, optional
            Range in dB to plot.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        # If map has not been reshaped by now...
        if len(map) == self.number_of_points:
            map = self.reconstruct_map_shape(map)
        assert map.shape == self.original_lengths, \
            'Map shape does not match grid shape'

        # Normal dimension to plane
        if third_dimension == 'x':
            ind_plane = np.argmin(np.abs(
                value_third_dimension - self.lines[0]))
            map = map[ind_plane, :, :]
            extent_dimensions = ['y', 'z']
        elif third_dimension == 'y':
            ind_plane = np.argmin(np.abs(
                value_third_dimension - self.lines[1]))
            map = map[:, ind_plane, :]
            extent_dimensions = ['x', 'z']
        elif third_dimension == 'z':
            ind_plane = np.argmin(np.abs(
                value_third_dimension - self.lines[2]))
            map = map[:, :, ind_plane]
            extent_dimensions = ['x', 'y']
        else:
            raise ValueError(f'{third_dimension} is not a valid dimension')

        # Get right extent
        ex = self.extent
        map = 20*np.log10(np.clip(np.abs(map), a_min=1e-25, a_max=None))
        fig, ax = general_matrix_plot(
            map, range_x=ex[extent_dimensions[1]],
            range_y=ex[extent_dimensions[0]], range_z=range_db,
            xlabel=extent_dimensions[1]+' / m',
            ylabel=extent_dimensions[0]+' / m',
            zlabel='dBFS', colorbar=True, lower_origin=True, returns=True)
        return fig, ax


class LineGrid(Grid):
    """Class for a line grid.

    """
    def __init__(self, line, dimension: str, value2: float, value3: float):
        """Constructor for a line grid. It is a line that goes in the
        direction of one of the coordinates. For a non-coincident line, create
        it manually using the Grid class.

        Parameters
        ----------
        line : array-like
            Position for the grid points (in meters) along the extended
            dimension.
        dimension : str
            Dimension along which line is extended. Choose from `'x'`, `'y'`
            or `'z'`.
        value2 : float
            Value for the second dimension. First dimension is the one along
            which the line is extended. Order goes x -> y -> z -> x -> etc.
        value3 :float
            Value for the third dimension.

        Attributes and Methods
        ----------------------
        - `coordinates`: Coordinates of the grid points in meters as
          numpy.ndarray with shape (point, coordinate xyz).
        - `number_of_points`: Number of points contained in the array.
        - `ndim`: Number of dimensions of grid.
        - `dimensions`: Dimensions in which the points are extended.
        - `extent`: Dictionary with the extent of points in each coordinate.
          As an example for the x direction `'x': [min(x), max(x)]`.
        - `get_distances_to_point()`: Method to get all point distances to a
          given point.
        - `find_nearest_point()`: Method to find the index and coordinates of
          the nearest point to a given one.

        """
        line = np.atleast_1d(np.squeeze(line))
        assert line.ndim == 1,\
            'Line has an invalid shape'
        dimension = dimension.lower()
        # Initialize with 4 values to later find second
        base_dimensions = ['x', 'y', 'z', 'x']
        assert dimension in base_dimensions, \
            'Dimension should be x, y or z'
        # Get dimensions
        ind = base_dimensions.index(dimension)
        base_dimensions.pop(ind)
        dim2 = base_dimensions[ind]
        dim3 = list(set(['x', 'y', 'z']) - set([dimension, dim2]))[0]

        self.extent_dimension = dimension
        # Initialize positions
        pos = {dimension: line,
               dim2: np.ones(len(line))*value2,
               dim3: np.ones(len(line))*value3}
        super().__init__(pos)


class MicArray(BasePoints):
    """This class contains a microphone array with all its metadata.

    """
    # ======== Constructor ====================================================
    def __init__(self, positions: dict):
        """Initiate a MicArray based on the positions dictionary that contains
        all vectors.

        NOTE: It is assumed that the order of the positions coincides
        with the channel number of the signal that is later passed to the
        `Beamformer` class!

        Parameters
        ----------
        positions : dict
            Dictionary containing point positions. Use `'x'`, `'y'` and `'z'`
            as keys to pass array-like objects with the positions.

        Attributes and Methods
        ----------------------
        - `coordinates`: Coordinates of the grid points in meters as
          numpy.ndarray with shape (point, coordinate xyz).
        - `number_of_points`: Number of points contained in the array.
        - `ndim`: Number of dimensions of grid.
        - `dimensions`: Dimensions in which the points are extended.
        - `extent`: Dictionary with the extent of points in each coordinate.
          As an example for the x direction `'x': [min(x), max(x)]`.
        - `array_center_coordinates`: Coordinates of microphone at the center
          of array.
        - `array_center_channel_number`: Channel number corresponding to center
          microphone.
        - `aperture`: Total array's aperture based on largest microphones'
          distance to each other.
        - `min_distance`: Minimum microphone distance in array.
        - `get_distances_to_point()`: Method to get all point distances to a
          given point.
        - `find_nearest_point()`: Method to find the index and coordinates of
          the nearest point to a given one.
        - `hz_to_he()`: Converts frequency of Hz to dimensionless Helmholtz
          number (He) using array's aperture and the speed of sound.
        - `he_to_hz()`: Converts He to Hz.
        - `get_maximum_frequency_range()`: Returns the recommended frequency
          range (in Hz) to analyze using this array based on array's aperture
          and minimum distance between two microphones.

        """
        super().__init__(positions)
        # Initialize aperture and minimum distance between microphones as None
        # (computation is only done on demand)
        self.__array_center_coordinates = None
        self.__array_center_channel_number = None
        # self.array_center_coord, self.array_center_mic = \
        #     _get_array_center(self.coordinates)
        self.__aperture = None
        self.__min_distance = None

    # ======== Properties =====================================================
    @property
    def aperture(self):
        if self.__aperture is None:
            self.__compute_aperture_min_distance()
        return self.__aperture

    @property
    def min_distance(self):
        if self.__min_distance is None:
            self.__compute_aperture_min_distance()
        return self.__min_distance

    @property
    def array_center_coordinates(self):
        if self.__array_center_coordinates is None:
            self.__compute_array_center()
        return self.__array_center_coordinates

    @property
    def array_center_channel_number(self):
        if self.__array_center_channel_number is None:
            self.__compute_array_center()
        return self.__array_center_channel_number

    def __compute_aperture_min_distance(self):
        """Method to trigger the computation for the array's aperture and
        minimum distance between microphones.

        """
        # Initialize values
        min_value = 1e20
        max_value = -1
        for i1 in range(self.coordinates.shape[0]):
            # Get distances from point i1 to all other points
            distances = self.get_distances_to_point(self.coordinates[i1, :])
            # Prune 0 value (distance for point with itself)
            distances = distances[distances != 0]
            # Get min and max values
            max_value = max(max_value, np.max(distances))
            min_value = min(min_value, np.min(distances))
        self.__min_distance = min_value
        self.__aperture = max_value

    def __compute_array_center(self):
        """Returns array center mic's coordinates and number.

        Parameters
        ----------
        coord : `np.ndarray`
            Coordinates of array with shape (points, xyz).

        Returns
        -------
        `np.ndarray`
            Array with coordinates for mic closest to center with
            shape (x, y, z).
        ind : int
            Index for mic closest to array center.

        """
        # Array center by averaging all mic positions
        center = np.mean(self.coordinates, axis=0)
        # Getting all distances
        distances = self.get_distances_to_point(center)
        # Get smallest distance
        ind = np.argmin(distances)
        self.__array_center_coordinates = self.coordinates[ind, :]
        self.__array_center_channel_number = ind

    # ======== Helmholtz number and Frequency =================================
    def he_to_hz(self, he: float, c: float = 343) -> float:
        """This method returns the frequency in Hz that corresponds to a given
        Helmholtz number based on array's aperture.

        Parameters
        ----------
        he : float
            Helmholtz number.
        c : float
            Speed of sound in m/s. Default: 343.

        Returns
        -------
        f_hz : float
            Frequency in Hz that corresponds to the passed Helmholtz number.

        """
        return he*c / self.aperture

    def hz_to_he(self, f_hz: float, c: float = 343) -> float:
        """This method returns the Helmholtz number corresponds to a given
        frequency based on array's aperture.

        Parameters
        ----------
        f_hz : float
            Frequency in Hz.
        c : float
            Speed of sound in m/s. Default: 343.

        Returns
        -------
        he : float
            Helmholtz number corresponding to passed frequency.

        """
        return f_hz * self.aperture / c

    # ======== Maximum frequency range ========================================
    def get_maximum_frequency_range(self, lowest_he: float = 4,
                                    c: float = 343) -> list:
        """Computes maximum recommended frequency range in Hz for this
        microphone array based on lowest Helmholtz number and the criterion
        `min_distance = wavelength/2` (to avoid spatial aliasing).

        Parameters
        ----------
        lowest_he : float, optional
            Lowest Helmholtz number to be regarded as valid. Default: 4.
        c : float, optional
            Speed of sound in m/s. Default: 343.

        Returns
        -------
        f_range_hz : list
            Array with frequency range for this microphone array.

        """
        f_range_hz = [self.he_to_hz(lowest_he, c=c), c / self.min_distance / 2]
        return f_range_hz


class SteeringVector():
    """This class hosts the main equation to be used for the steering vector.

    """
    # ======== Constructor ====================================================
    def __init__(self, formulation='true location'):
        """Initializes the SteeringVector using the passed formulation.

        Parameters
        ----------
        formulation : str or callable, optional
            Steering vector formulation to use. Choose from `'classic'`,
            `'inverse'`, `'true power'` or `'true location'`. These
            correspond to 1, 2, 3 and 4 of the reference paper, respectively.
            For an own formulation, pass a callable with signature::

                formulation(wave_number: numpy.ndarray, grid: Grid,
                            microphone_array: MicArray) -> numpy.ndarray:

            The output array should have shape (frequency, mic, grid) and be
            complex-valued. Default: `'true location'`.

        Methods
        -------
        - `get_vector()`: computes and returns steering vector for the passed
          frequencies, grid points and mic coordinates.

        References
        ----------
        - Sarradj, Ennes. (2012). Three-Dimensional Acoustic Source Mapping
          with Different Beamforming Steering Vector Formulations. Advances in
          Acoustics and Vibration. 2012. 10.1155/2012/292695.

        """
        if type(formulation) == str:
            formulation = formulation.lower()
            if formulation == 'classic':
                self.get_vector = classic_steering
            elif formulation == 'inverse':
                self.get_vector = inverse_steering
            elif formulation == 'true power':
                self.get_vector = true_power_steering
            elif formulation == 'true location':
                self.get_vector = true_location_steering
            else:
                raise ValueError(
                    'Incorrect formulation. Use either classic, inverse, ' +
                    'true power or true location')
        else:
            assert type(formulation) == callable, \
                'Formulation should be a callable or a string'
            self.get_vector = formulation


class BaseBeamformer():
    """Base class for a beamformer.

    """
    def __init__(self, multi_channel_signal: Signal,
                 mic_array: MicArray, c: float = 343):
        """Base constructor for Beamformer.

        Parameters
        ----------
        multi_channel_signal : `Signal`
            Signal with multiple channels. It is assumed that the channel order
            matches the order in the MicArray object.
        mic_array : `MicArray`
            Microphone array object containing microphone positions.
        c : float, optional
            Speed of sound in m/s. Default: 343.

        Methods
        -------
        - `set_csm_parameters()`: passes all necessary parameters to configure
          the cross-spectral matrix acquired via the multi-channel signal
          object.

        """
        assert type(multi_channel_signal) == Signal, \
            'Multi-channel signal must be of type Signal'
        assert type(mic_array) == MicArray, \
            'mic_array should be of type MicArray'
        assert c > 0, \
            'Speed of sound should be bigger than 0'
        assert multi_channel_signal.number_of_channels == \
            mic_array.number_of_points, \
            'Number of channels in signal and microphone array do not match'
        self.signal = multi_channel_signal
        self.mics = mic_array
        self.c = c
        self.beamformer_type = 'Base'
        self.set_csm_parameters = self.signal.set_csm_parameters

    # ======== Prints and plots ===============================================
    def plot_setting(self) -> tuple[Figure, Axes]:
        """Plots spatial setting of microphones and grid.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure.
        ax : `matplotlib.axes.Axes`
            Axes.

        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5),
                               subplot_kw={'projection': '3d'})
        ax.scatter(
            self.mics.coordinates[:, 0], self.mics.coordinates[:, 1],
            self.mics.coordinates[:, 2])
        if hasattr(self, 'grid'):
            if self.grid is not None:
                ax.scatter(
                    self.grid.coordinates[:, 0], self.grid.coordinates[:, 1],
                    self.grid.coordinates[:, 2])
        ax.scatter(
            self.mics.array_center_coordinates[0],
            self.mics.array_center_coordinates[1],
            self.mics.array_center_coordinates[2], c='xkcd:dark green')
        ax.set_xlabel('$x$ / m')
        ax.set_ylabel('$y$ / m')
        ax.set_zlabel('$z$ / m')
        ax.legend(['Mic Array', 'Grid', 'Center Mic'])
        return fig, ax

    # ======== Helpers ========================================================
    def get_frequency_range_from_he(self, range_he=[4, 10]) -> list:
        """Takes in frequency range in He (Helmholtz number) and returns it
        in Hz. This is done by taking the array's aperture and speed of
        sound as inputs.

        Parameters
        ----------
        range_he : array-like, optional
            Range in He (Helmholtz number [1]). Default: [4, 10].

        Returns
        -------
        frequency_range_hz : list
            Frequency range in Hz.

        """
        assert len(range_he) == 2, \
            'Range in He should have length two'
        return [self.mics.he_to_hz(i, self.c) for i in range_he]

    def show_info(self):
        """Helper for creating a string containing metadata.

        """
        txt = f"""Beamformer: {self.beamformer_type}"""
        txt = '\n'+txt+'\n'+'-'*len(txt)+'\n'
        txt += f'''Aperture: {self.mics.aperture}\n'''
        txt += f'''Min mic distance: {self.mics.min_distance}\n'''
        txt += f'''Recommended f range: {self.mics.
                                         get_maximum_frequency_range()}\n'''
        txt += f'''Number of mics: {self.mics.number_of_points}\n'''
        if hasattr(self, 'grid'):
            if self.grid is not None:
                txt += f'''Number of grid points: {self.grid.
                                                   number_of_points}\n'''
        print(txt)


class BeamformerGridded(BaseBeamformer):
    """Base class for beamformers that use a grid.

    """
    def __init__(self, multi_channel_signal: Signal,
                 mic_array: MicArray, grid: Grid,
                 steering_vector: SteeringVector,
                 c: float = 343):
        """Constructor for beamformer with grid and steering vector.

        Parameters
        ----------
        multi_channel_signal : `Signal`
            Signal with multiple channels. It is assumed that the channel order
            matches the order in the MicArray object.
        mic_array : `MicArray`
            Microphone array object containing microphone positions.
        grid : `Grid`
            Grid object with the information about the points to be sampled.
        steering_vector : `SteeringVector`
            Steering vector to be used for the beamforming.
        c : float, optional
            Speed of sound in m/s. Default: 343.

        Methods
        -------
        - `set_csm_parameters()`: passes all necessary parameters to configure
          the cross-spectral matrix acquired via the multi-channel signal
          object.
        - `get_beamformer_map()`: computes a map using all passed parameters.

        """
        super().__init__(multi_channel_signal, mic_array, c)
        assert type(steering_vector) == SteeringVector, \
            'steering_vector should be of type SteeringVector'
        assert issubclass(type(grid), Grid), \
            'grid should be a Grid object'
        self.grid = grid
        self.st_vec = steering_vector


class BeamformerDASFrequency(BeamformerGridded):
    """This is the base class for beamforming in frequency-domain.

    """
    beamformer_type = 'Delay-and-sum (Frequency)'

    # ======== Get beamforming map ============================================
    def get_beamformer_map(self, center_frequency_hz: float,
                           octave_fraction: int = 3,
                           remove_csm_diagonal: bool = True) -> np.ndarray:
        """Run delay-and-sum beamforming in the given frequency range.

        Parameters
        ----------
        center_frequenc_hz : float
            Center frequency for which to compute map.
        octave_fraction : int, optional
            Fractional octave bandwidth for computing the map. For instance,
            8 means 1/8-octave bandwidth. Default: 3.
        remove_csm_diagonal : bool, optional
            When `True`, the diagonal of the cross-spectral matrix is removed.
            Default: `True`.

        Returns
        -------
        map : `np.ndarray`
            Beamforming map

        """
        self.center_frequency_hz = center_frequency_hz
        self.octave_fraction = octave_fraction
        self.f_range_hz = _get_fractional_octave_bandwidth(
            self.center_frequency_hz, self.octave_fraction)

        txt = 'Beamformer computation has started successfully:'
        print('\n'+txt)
        print('-'*len(txt))
        print('...csm...')
        f, csm = self.signal.get_csm()
        if remove_csm_diagonal:
            # Account for energy loss
            csm *= self.signal.number_of_channels / \
                (self.signal.number_of_channels - 1)
            for i in range(len(f)):
                np.fill_diagonal(csm[i, :, :], 0)

        print('...Steering vector...')
        # Frequency selection, wave numbers and steering vector
        ids = _find_nearest(self.f_range_hz, f)
        id1, id2 = ids[0], ids[1]
        # In case of only one frequency bin
        if id1 == id2:
            id2 += 1
        f = f[id1:id2]
        csm = csm[id1:id2]
        number_frequency_bins = id2 - id1
        wave_numbers = f * np.pi * 2 / self.c
        h = self.st_vec.get_vector(
            wave_numbers, grid=self.grid, mic=self.mics)
        h_H = np.swapaxes(h, 1, 2).conjugate()
        self.f_range_hz = np.array([f[0], f[-1]])

        print('...Apply...')
        map = np.zeros((self.grid.number_of_points,
                        number_frequency_bins))
        for gind in range(self.grid.number_of_points):
            for find in range(len(f)):
                map[gind, find] = np.linalg.multi_dot(
                    [h_H[find, gind, :], csm[find, :, :],
                     h[find, :, gind]]).real

        # Unphysical values for removed diagonal of CSM
        if remove_csm_diagonal:
            map[map < 0] = 0

        # Integrate over all frequencies
        if number_frequency_bins > 1:
            map = simpson(map, dx=f[1]-f[0], axis=1)
        else:
            map = map.squeeze()
        self.map = self.grid.reconstruct_map_shape(map)
        return self.map.copy()


class BeamformerCleanSC(BeamformerGridded):
    """This class performs beamforming using the CLEAN-SC method presented in
    [1].

    References
    ----------
    - [1]: Sijtsma P. CLEAN Based on Spatial Source Coherence. International
      Journal of Aeroacoustics. 2007;6(4):357-374.
      doi: 10.1260/147547207783359459.

    """
    beamformer_type = 'CleanSC'

    def get_beamformer_map(self, center_frequency_hz: float,
                           octave_fraction: int = 3,
                           maximum_iterations: int = None,
                           safety_factor: float = 0.5,
                           remove_diagonal_csm: bool = False) -> np.ndarray:
        """Returns a deconvolved beaforming map.

        Parameters
        ----------
        center_frequenc_hz : float
            Center frequency for which to compute map.
        octave_fraction : int, optional
            Fractional octave bandwidth for computing the map. For instance,
            8 means 1/8-octave bandwidth. Default: 3.
        maximum_iterations : int, optional
            Set a maximum number of iterations for acquiring the degraded CSM.
            If `None` is passed, the double of the number of microphones is
            taken as the maximum iteration number. The stopping criterion
            given in [1] is always checked. Default: `None`.
        safety_factor : float, optional
            Also called loop gain, the safety factor dampens the result from
            each iteration during deconvolution. Should be between 0 and 1.
            See [1] for more details. Default: 0.5.
        remove_diagonal_csm : bool, optional
            When `True`, the main diagonal of the CSM is removed for a cleaner
            map (source powers might be wrongly estimated). Default: `False`.

        Returns
        -------
        map : `np.ndarray`
            Beamformer map.

        References
        ----------
        - [1]: Sijtsma P. CLEAN Based on Spatial Source Coherence.
          International Journal of Aeroacoustics. 2007;6(4):357-374.
          doi: 10.1260/147547207783359459.

        """
        if maximum_iterations is None:
            # Set maximum iterations to twice the number of channels
            maximum_iterations = self.signal.number_of_channels*2
        else:
            assert maximum_iterations > 0, \
                'Number of iterations must be positive'
        assert safety_factor > 0 and safety_factor <= 1, \
            f'{safety_factor} is not valid. The safety factor (loop gain) ' +\
            'should be in ]0, 1]'

        self.center_frequency_hz = center_frequency_hz
        self.octave_fraction = octave_fraction
        self.f_range_hz = _get_fractional_octave_bandwidth(
            self.center_frequency_hz, self.octave_fraction)

        txt = 'Beamformer computation has started successfully:'
        print('\n'+txt)
        print('-'*len(txt))
        print('...csm...')
        f, csm = self.signal.get_csm()

        print('...Steering vector...')
        # Frequency selection, wave numbers and steering vector
        ids = _find_nearest(self.f_range_hz, f)
        id1, id2 = ids[0], ids[1]
        # In case of only one frequency bin
        if id1 == id2:
            id2 += 1
        f = f[id1:id2]
        csm = csm[id1:id2]
        number_frequency_bins = id2 - id1

        # Steering vector
        wave_numbers = f * np.pi * 2 / self.c
        h = self.st_vec.get_vector(
            wave_numbers, grid=self.grid, mic=self.mics)
        h_H = np.swapaxes(h, 1, 2).conjugate()
        self.f_range_hz = np.array([f[0], f[-1]])

        # Remove diagonal CSM
        if remove_diagonal_csm:
            for find in range(len(f)):
                np.fill_diagonal(csm[find, :, :], 0)

        print('...Create and deconvolve map...')
        map = np.zeros((self.grid.number_of_points,
                        number_frequency_bins))
        for find in range(len(f)):
            for gind in range(self.grid.number_of_points):
                # Create initial map
                map[gind, find] = np.linalg.multi_dot(
                    [h_H[find, gind, :], csm[find, :, :],
                     h[find, :, gind]]).real
            map = _clean_sc_deconvolve(
                map[:, find], csm[find, :, :], h[find, :, :], h_H[find, :, :],
                maximum_iterations, remove_diagonal_csm, safety_factor).real

        # Integrate over all frequencies
        if number_frequency_bins > 1:
            map = simpson(map, dx=f[1]-f[0], axis=1)
        else:
            map = map.squeeze()
        self.map = self.grid.reconstruct_map_shape(map)
        return self.map.copy()


class BeamformerOrthogonal(BeamformerGridded):
    """This class performs orthogonal beamforming using the method presented in
    [1].

    References
    ----------
    - [1]: Ennes Sarradj, A fast signal subspace approach for the determination
      of absolute levels from phased microphone array measurements, Journal of
      Sound and Vibration, Volume 329, Issue 9, 2010, Pages 1553-1569,
      ISSN 0022-460X, https://doi.org/10.1016/j.jsv.2009.11.009.

    """
    beamformer_type = 'Orthogonal (Grid)'

    def get_beamformer_map(self, center_frequency_hz: float,
                           octave_fraction: int = 3,
                           number_eigenvalues: int = None) -> np.ndarray:
        """Returns a beaforming map created with orthogonal beamforming.

        Parameters
        ----------
        center_frequenc_hz : float
            Center frequency for which to compute map.
        octave_fraction : int, optional
            Fractional octave bandwidth for computing the map. For instance,
            8 means 1/8-octave bandwidth. Default: 3.
        number_eigenvalues : int, optional
            Set a number of eigenvalues to be regarded. Pass `None` to use at
            least half of what is possible (number of microphones).
            Default: `None`.

        Returns
        -------
        map : np.ndarray
            Beamformer map.

        References
        ----------
        - [1]: Ennes Sarradj, A fast signal subspace approach for the
          determination of absolute levels from phased microphone array
          measurements, Journal of Sound and Vibration, Volume 329, Issue 9,
          2010, Pages 1553-1569, ISSN 0022-460X,
          https://doi.org/10.1016/j.jsv.2009.11.009.

        """
        self.center_frequency_hz = center_frequency_hz
        self.octave_fraction = octave_fraction
        self.f_range_hz = _get_fractional_octave_bandwidth(
            self.center_frequency_hz, self.octave_fraction)

        if number_eigenvalues is None:
            number_eigenvalues = self.signal.number_of_channels // 2
        else:
            assert number_eigenvalues <= self.signal.number_of_channels, \
                'Number of eigenvalues cannot be more than number of ' +\
                'microphones'
            assert number_eigenvalues > 0, \
                'At least one eigenvalue of the CSM must be regarded'

        txt = 'Beamformer computation has started successfully:'
        print('\n'+txt)
        print('-'*len(txt))
        print('...csm...')
        f, csm = self.signal.get_csm()

        print('...Steering vector...')
        # Frequency selection, wave numbers and steering vector
        ids = _find_nearest(self.f_range_hz, f)
        id1, id2 = ids[0], ids[1]
        # In case of only one frequency bin
        if id1 == id2:
            id2 += 1
        f = f[id1:id2]
        csm = csm[id1:id2]
        number_frequency_bins = id2 - id1
        wave_numbers = f * np.pi * 2 / self.c
        h = self.st_vec.get_vector(
            wave_numbers, grid=self.grid, mic=self.mics)
        self.f_range_hz = np.array([f[0], f[-1]])

        print('...Apply...')
        eig_map = np.zeros((number_eigenvalues, self.grid.number_of_points,
                            number_frequency_bins))
        map = np.zeros((self.grid.number_of_points,
                        number_frequency_bins))

        for find in range(len(f)):
            # Spectral decomposition – eigenvalues are given in ascending order
            w, v = np.linalg.eigh(csm[find, :, :])
            for eig in range(number_eigenvalues):
                for gind in range(self.grid.number_of_points):
                    # Generate whole map
                    product = h[find, :, gind].conjugate() @ v[:, -eig-1]
                    eig_map[eig, gind, find] = \
                        (product * product.conjugate()).real
                # Find largest value
                source_ind = np.argmax(eig_map[eig, :, find])
                # Scale by eigenvalue and pass to final map
                map[source_ind, find] = \
                    eig_map[eig, source_ind, find] * w[-eig-1]

        # Integrate over all frequencies
        if number_frequency_bins > 1:
            map = simpson(map, dx=f[1]-f[0], axis=1)
        else:
            map = map.squeeze()
        self.map = self.grid.reconstruct_map_shape(map)
        return self.map.copy()


class BeamformerFunctional(BeamformerGridded):
    """This class performs functional beamforming using the method presented in
    [1].

    References
    ----------
    - [1]: Dougherty, Robert. (2014). Functional Beamforming.

    """
    beamformer_type = 'Functional'

    def get_beamformer_map(self, center_frequency_hz: float,
                           octave_fraction: int = 3,
                           gamma: float = 10) -> np.ndarray:
        """Returns a beaforming map created with functional beamforming.

        Parameters
        ----------
        center_frequenc_hz : float
            Center frequency for which to compute map.
        octave_fraction : int, optional
            Fractional octave bandwidth for computing the map. For instance,
            8 means 1/8-octave bandwidth. Default: 3.
        gamma : float, optional
            Set a gamma value as the power of the CSM. Default: 10.

        Returns
        -------
        map : np.ndarray
            Beamformer map.

        References
        ----------
        - [1]: Dougherty, Robert. (2014). Functional Beamforming.

        """
        self.center_frequency_hz = center_frequency_hz
        self.octave_fraction = octave_fraction
        self.f_range_hz = _get_fractional_octave_bandwidth(
            self.center_frequency_hz, self.octave_fraction)

        txt = 'Beamformer computation has started successfully:'
        print('\n'+txt)
        print('-'*len(txt))
        print('...csm...')
        f, csm = self.signal.get_csm()

        print('...Steering vector...')
        # Frequency selection, wave numbers and steering vector
        ids = _find_nearest(self.f_range_hz, f)
        id1, id2 = ids[0], ids[1]
        # In case of only one frequency bin
        if id1 == id2:
            id2 += 1
        f = f[id1:id2]
        csm = csm[id1:id2]
        number_frequency_bins = id2 - id1
        wave_numbers = f * np.pi * 2 / self.c

        # Generate steering vectors
        h = self.st_vec.get_vector(
            wave_numbers, grid=self.grid, mic=self.mics)
        h_H = np.swapaxes(h, 1, 2).conjugate()
        self.f_range_hz = np.array([f[0], f[-1]])

        print('...Apply...')
        map = np.zeros((self.grid.number_of_points,
                        number_frequency_bins))

        for find in range(len(f)):
            # SVD
            u, s, vh = np.linalg.svd(csm[find, :, :])
            s = np.diag(s**(1/gamma))
            # New CSM
            csm_ = np.linalg.multi_dot([u, s, vh])
            for gind in range(self.grid.number_of_points):
                map[gind, find] = np.linalg.multi_dot(
                    [h_H[find, gind, :], csm_, h[find, :, gind]]).real
                steering_normalization = (
                    h_H[find, gind, :] @ h[find, :, gind]).real
                map[gind, find] = (
                    map[gind, find]/steering_normalization)**gamma * \
                    steering_normalization

        # Integrate over all frequencies
        if number_frequency_bins > 1:
            map = simpson(map, dx=f[1]-f[0], axis=1)
        else:
            map = map.squeeze()
        self.map = self.grid.reconstruct_map_shape(map)
        return self.map.copy()


class BeamformerMVDR(BeamformerGridded):
    """This class performs minimum variance distortionless response (MVDR)
    beamforming using the method presented in [1]. This formulation is
    also referred to as Capon beamformer.

    References
    ----------
    - [1]: J. Capon, "High-resolution frequency-wavenumber spectrum analysis,"
      in Proceedings of the IEEE, vol. 57, no. 8, pp. 1408-1418, Aug. 1969,
      doi: 10.1109/PROC.1969.7278.

    """
    beamformer_type = 'MVDR'

    def get_beamformer_map(self, center_frequency_hz: float,
                           octave_fraction: int = 3,
                           gamma: float = 10) -> np.ndarray:
        """Returns a beaforming map created with MVDR beamforming.

        Parameters
        ----------
        center_frequenc_hz : float
            Center frequency for which to compute map.
        octave_fraction : int, optional
            Fractional octave bandwidth for computing the map. For instance,
            8 means 1/8-octave bandwidth. Default: 3.

        Returns
        -------
        map : np.ndarray
            Beamformer map.

        References
        ----------
        - [1]: J. Capon, "High-resolution frequency-wavenumber spectrum
          analysis," in Proceedings of the IEEE, vol. 57, no. 8,
          pp. 1408-1418, Aug. 1969, doi: 10.1109/PROC.1969.7278.

        """
        self.center_frequency_hz = center_frequency_hz
        self.octave_fraction = octave_fraction
        self.f_range_hz = _get_fractional_octave_bandwidth(
            self.center_frequency_hz, self.octave_fraction)

        txt = 'Beamformer computation has started successfully:'
        print('\n'+txt)
        print('-'*len(txt))
        print('...csm...')
        f, csm = self.signal.get_csm()

        print('...Steering vector...')
        # Frequency selection, wave numbers and steering vector
        ids = _find_nearest(self.f_range_hz, f)
        id1, id2 = ids[0], ids[1]
        # In case of only one frequency bin
        if id1 == id2:
            id2 += 1
        f = f[id1:id2]
        csm = csm[id1:id2]
        number_frequency_bins = id2 - id1
        wave_numbers = f * np.pi * 2 / self.c

        # Generate steering vectors
        h = self.st_vec.get_vector(
            wave_numbers, grid=self.grid, mic=self.mics)
        h_H = np.swapaxes(h, 1, 2).conjugate()
        self.f_range_hz = np.array([f[0], f[-1]])

        print('...Apply...')
        map = np.zeros((self.grid.number_of_points,
                        number_frequency_bins))

        for find in range(len(f)):
            csm_1 = np.linalg.inv(csm[find, :, :])
            for gind in range(self.grid.number_of_points):
                map[gind, find] = 1/np.linalg.multi_dot(
                    [h_H[find, gind, :], csm_1, h[find, :, gind]]).real

        # Integrate over all frequencies
        if number_frequency_bins > 1:
            map = simpson(map, dx=f[1]-f[0], axis=1)
        else:
            map = map.squeeze()
        self.map = self.grid.reconstruct_map_shape(map)
        return self.map.copy()


class BeamformerDASTime(BaseBeamformer):
    """Conventional delay-and-sum beamformer in time-domain.

    """
    def __init__(self, multi_channel_signal: Signal,
                 mic_array: MicArray, grid: Grid, c: float = 343):
        """Constructor for the traditional Delay-and-sum beamforming approach
        in time domain.

        Parameters
        ----------
        multi_channel_signal : `Signal`
            Signal with multiple channels. It is assumed that the channel order
            matches the order in the MicArray object.
        mic_array : `MicArray`
            Microphone array object containing microphone positions.
        grid : `Grid`
            Grid object with the information about the points to be sampled.
        c : float, optional
            Speed of sound in m/s. Default: 343.

        """
        super().__init__(multi_channel_signal, mic_array, c)
        assert issubclass(type(grid), Grid), \
            'grid should be a Grid object'
        self.grid = grid
        self.beamformer_type = 'Delay-and-sum (Time)'

    def get_beamformer_output(self) -> Signal:
        """Triggers the computation for beamforming in time domain.

        Returns
        -------
        out_sig : `Signal`
            Output signal focused to the points of the grid.

        """
        txt = 'Beamformer computation has started successfully:'
        print('\n'+txt)
        print('-'*len(txt))
        print('...get delays...')
        # Start Signal from one channel
        out_sig = self.signal.get_channels(0)

        # Get maximal distance in order to delay all signals to that
        r0 = -1
        min_distance = 1e20
        for gp in self.grid.coordinates:
            ds = self.mics.get_distances_to_point(gp)
            r0 = max(r0, np.max(ds))
            min_distance = min(min_distance, np.min(ds))

        # Get longest delay in order to pad all signals accordingly
        longest_delay_samples = \
            (r0 - min_distance)/self.c * self.signal.sampling_rate_hz
        longest_delay_samples = int(longest_delay_samples + 2)
        total_length_samples = \
            out_sig.time_data.shape[0] + longest_delay_samples
        out_sig = pad_trim(out_sig, total_length_samples)

        # Start computation for each grid point
        print('...grid focusing...')
        for ig in range(self.grid.number_of_points):
            if ig == self.grid.number_of_points//2:
                print(r'...50% grid done...')
            ds = self.mics.get_distances_to_point(self.grid.coordinates[ig, :])
            delays = (r0 - ds)/self.c
            # Accumulator
            new_time_data = np.zeros((total_length_samples, 1))
            for im in range(self.mics.number_of_points):
                ntd = fractional_delay(
                    self.signal.get_channels(im), delays[im]).time_data *\
                        ds[im]
                new_time_data += _pad_trim(ntd, total_length_samples)
            new_time_data *= (4*np.pi/self.mics.number_of_points)
            out_sig.add_channel(None, new_time_data, out_sig.sampling_rate_hz)
        out_sig.remove_channel(0)
        return out_sig


class MonopoleSource():
    """Base class for all sources. It has a monopole characteristic by
    default.

    """
    def __init__(self, signal: Signal, coordinates):
        """Constructor for a monopole source. It is defined by an emitted
        signal and spatial coordinates. Its emission characteristic is
        omnidirectional.

        Parameters
        ----------
        signal : `Signal`
            Emitted signal by the source. It is restricted to one channel.
        coordinates : array-like
            Array with room coordinates (in meters) defining source's position
            with shape (x, y, z).

        """
        assert signal.number_of_channels == 1, \
            'Only signals with a single channel are supported'
        coordinates = np.squeeze(coordinates)
        assert len(coordinates) == 3 and coordinates.ndim == 1, \
            'Coordinates should have exactly three values'
        self.emitted_signal = signal
        self.coordinates = coordinates

    def get_signals_on_array(self, mics: MicArray, c: float = 343) -> Signal:
        """Simulate transmission of emitted signal onto microphone array.

        Parameters
        ----------
        mics : `MicArray`
            Microphone array.
        c : float, optional
            Speed of sound in m/s. Default: 343.

        Returns
        -------
        multi_channel_signal : `Signal`
            Multi-channel signal corresponding array's microphones.

        """
        distances = mics.get_distances_to_point(self.coordinates)
        delays = distances/c

        multi_channel_signal = self.emitted_signal.copy()
        for i in range(len(distances)):
            # Delay
            ns = fractional_delay(
                self.emitted_signal, delays[i], keep_length=True)
            # Amplitude scaling – 1 on point and decays with distance
            ns.time_data = ns.time_data/(1+distances[i])
            # Append to final signal
            multi_channel_signal = merge_signals(
                multi_channel_signal, ns, padding_trimming=True)
        # Remove original signal
        multi_channel_signal.remove_channel(0)
        return multi_channel_signal


def mix_sources_on_array(sources: list | MonopoleSource, mics: MicArray,
                         c: float = 343) ->\
        Signal:
    """This function takes in a list containing multiple sources and gives back
    the multi-channel signal on the array of the combined sources.

    Parameters
    ----------
    sources : list or `MonopoleSource`
        List containing sources of object type `MonopoleSource`. It is also
        possible to pass a Source directly.
    mics : `MicArray`
        Microphone array on which to mixed the sources.
    c : float, optional
        Speed of sound in m/s. Default: 343.

    Returns
    -------
    multi_channel_sig : `Signal`
        Multi-channel signal containing combined source's signals.

    """
    # Convert to list if only Monopole source is passed
    if type(sources) == MonopoleSource:
        sources = [sources]
    assert len(sources) > 0, \
        'There must be at least one source to project on array'
    assert all([type(i) == MonopoleSource for i in sources]), \
        'All sources in list should be of type Source'
    # Take first source
    multi_channel_sig = sources[0].get_signals_on_array(mics, c)
    total_length_samples = multi_channel_sig.time_data.shape[0]
    sources.pop(0)

    # Add all other sources progressively checking for shortest duration
    for s in sources:
        # Warning if lengths do not match
        if total_length_samples != s.emitted_signal.time_data.shape[0]:
            warn('Emitted signals from sources differ in length. Trimming to '
                 'shortest will be done')
            total_length_samples = min(
                total_length_samples, s.emitted_signal.time_data.shape[0])
            multi_channel_sig = pad_trim(
                multi_channel_sig, total_length_samples)
            s.emitted_signal = pad_trim(s.emitted_signal, total_length_samples)
        # Add to multi-channel data
        ns = s.get_signals_on_array(mics, c)
        multi_channel_sig.time_data += ns.time_data
    return multi_channel_sig


# ========== Steering vector formulations =====================================
def classic_steering(wave_number: np.ndarray, grid: Grid,
                     mic: MicArray) -> np.ndarray:
    """Classic formulation for steering vector (formulation 1 in reference
    paper).

    Parameters
    ----------
    wave_number : float, array-like
        Wave number `k=omega/c` that represents frequency or frequency vector.
    grid : `Grid`
        Grid to be used for steering vector.
    mic : `MicArray`
        Microphone Array object.

    Returns
    -------
    steering_vector : `np.ndarray`
        Complex steering vector with shape (frequency, nmics, ngrid).

    References
    ----------
    - Sarradj, Ennes. (2012). Three-Dimensional Acoustic Source Mapping
      with Different Beamforming Steering Vector Formulations. Advances in
      Acoustics and Vibration. 2012. 10.1155/2012/292695.

    """
    wave_number = np.atleast_1d(wave_number)
    assert wave_number.ndim == 1, \
        'Wave number should be a 1D-array'
    # Number of mics and grid points
    N = mic.number_of_points
    NGrid = grid.number_of_points

    # rt0 with shape (ngrid)
    rt0 = grid.get_distances_to_point(mic.array_center_coordinates)

    # rti matrix with shape (nmic, ngrid)
    rti = np.zeros((N, NGrid))
    for i in range(N):
        rti[i, :] = grid.get_distances_to_point(mic.coordinates[i, :])

    return 1/N * np.exp(
        -1j*wave_number[:, nxs, nxs] * (rti[nxs, :, :] - rt0[nxs, nxs, :]))


def inverse_steering(wave_number: np.ndarray, grid: Grid,
                     mic: MicArray) -> np.ndarray:
    """Inverse formulation for steering vector (formulation 2 in reference
    paper).

    Parameters
    ----------
    wave_number : float, array-like
        Wave number `k=omega/c` that represents frequency or frequency vector.
    grid : `Grid`
        Grid to be used for steering vector.
    mic : `MicArray`
        Microphone Array object.

    Returns
    -------
    steering_vector : `np.ndarray`
        Complex steering vector with shape (frequency, nmics, ngrid).

    References
    ----------
    - Sarradj, Ennes. (2012). Three-Dimensional Acoustic Source Mapping
      with Different Beamforming Steering Vector Formulations. Advances in
      Acoustics and Vibration. 2012. 10.1155/2012/292695.

    """
    wave_number = np.atleast_1d(wave_number)
    assert wave_number.ndim == 1, \
        'Wave number should be a 1D-array'
    # Number of mics and grid points
    N = mic.number_of_points
    NGrid = grid.number_of_points

    # rt0 with shape (ngrid)
    rt0 = grid.get_distances_to_point(mic.array_center_coordinates)

    # rti matrix with shape (nmic, ngrid)
    rti = np.zeros((N, NGrid))
    for i in range(N):
        rti[i, :] = grid.get_distances_to_point(mic.coordinates[i, :])

    # Formulate vector
    return rti[nxs, :, :] / N / rt0[nxs, nxs, :] * \
        np.exp(-1j * wave_number[:, nxs, nxs] *
               (rti[nxs, :, :] - rt0[nxs, nxs, :]))


def true_power_steering(wave_number: np.ndarray, grid: Grid,
                        mic: MicArray) -> np.ndarray:
    """Formulation for true power steering vector (formulation 3 in reference
    paper).

    Parameters
    ----------
    wave_number : float, array-like
        Wave number `k=omega/c` that represents frequency or frequency vector.
    grid : `Grid`
        Grid to be used for steering vector.
    mic : `MicArray`
        Microphone Array object.

    Returns
    -------
    steering_vector : `np.ndarray`
        Complex steering vector with shape (frequency, nmics, ngrid).

    References
    ----------
    - Sarradj, Ennes. (2012). Three-Dimensional Acoustic Source Mapping
      with Different Beamforming Steering Vector Formulations. Advances in
      Acoustics and Vibration. 2012. 10.1155/2012/292695.

    """
    wave_number = np.atleast_1d(wave_number)
    assert wave_number.ndim == 1, \
        'Wave number should be a 1D-array'
    # Number of mics and grid points
    N = mic.number_of_points
    NGrid = grid.number_of_points

    # rt0 with shape (ngrid)
    rt0 = grid.get_distances_to_point(mic.array_center_coordinates)

    # rti matrix with shape (nmic, ngrid)
    rti = np.zeros((N, NGrid))
    for i in range(N):
        rti[i, :] = grid.get_distances_to_point(mic.coordinates[i, :])

    # rtj vector with shape (ngrid)
    rtj = np.zeros((NGrid))
    for i in range(NGrid):
        all_distances = mic.get_distances_to_point(grid.coordinates[i, :])
        rtj[i] = np.sum(1/all_distances**2)

    # Formulate vector
    return 1 / rt0[nxs, nxs, :] / rti[nxs, :, :] / rtj[nxs, nxs, :] * \
        np.exp(-1j * wave_number[:, nxs, nxs] *
               (rti[nxs, :, :] - rt0[nxs, nxs, :]))


def true_location_steering(wave_number: np.ndarray, grid: Grid,
                           mic: MicArray) -> np.ndarray:
    """Formulation for true location steering vector (formulation 4 in
    reference paper).

    Parameters
    ----------
    wave_number : float, array-like
        Wave number `k=omega/c` that represents frequency or frequency vector.
    grid : `Grid`
        Grid to be used for steering vector.
    mic : `MicArray`
        Microphone Array object.

    Returns
    -------
    steering_vector : `np.ndarray`
        Complex steering vector with shape (frequency, ngrid, nmics).

    References
    ----------
    - Sarradj, Ennes. (2012). Three-Dimensional Acoustic Source Mapping
      with Different Beamforming Steering Vector Formulations. Advances in
      Acoustics and Vibration. 2012. 10.1155/2012/292695.

    """
    wave_number = np.atleast_1d(wave_number)
    assert wave_number.ndim == 1, \
        'Wave number should be a 1D-array'
    # Number of mics and grid points
    N = mic.number_of_points
    NGrid = grid.number_of_points

    # rt0 with shape (ngrid)
    rt0 = grid.get_distances_to_point(mic.array_center_coordinates)

    # rti matrix with shape (nmic, ngrid)
    rti = np.zeros((N, NGrid))
    for i in range(N):
        rti[i, :] = grid.get_distances_to_point(mic.coordinates[i, :])

    # rtj vector with shape (ngrid)
    rtj = np.zeros((NGrid))
    for i in range(NGrid):
        all_distances = mic.get_distances_to_point(grid.coordinates[i, :])
        rtj[i] = N * np.sum(1/all_distances**2)

    return 1 / rti[nxs, :, :] / np.sqrt(rtj[nxs, nxs, :]) * \
        np.exp(-1j * wave_number[:, nxs, nxs] *
               (rti[nxs, :, :] - rt0[nxs, nxs, :]))
