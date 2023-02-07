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
from ._beamforming import BasePoints
from dsptoolbox.plots import general_matrix_plot

set_style('whitegrid')
nxs = np.newaxis


class Grid(BasePoints):
    """This class contains a grid to use for beamforming and all its metadata.
    This class is implemented only for right-hand systems with cartesian
    coordinates.

    """
    def __init__(self, positions: dict):
        """Construct a grid for beamforming by passing positions for the point
        coordinates. Additionally, there is a class method
        (reconstruct_map_shape) that you can manually add to the object in
        order to obtain beamformer maps with an specific shape immediately
        instead of vector shape.

        Parameters
        ----------
        positions : dict, Pandas.DataFrame
            Dictionary or pandas dataframe containing point positions.
            Use `'x'`, `'y'` and `'z'` as keys to pass array-like objects with
            the positions.

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
        same given map. Use inheritance from the `Grid` class to overwrite
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
        map = 20*np.log10(np.abs(map))
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
    def __init__(self, linex, liney, linez):
        """Constructor for a regular 3D grid.

        Parameters
        ----------
        linex : array-like
            Line to define points along the x coordinate.
        liney : array-like
            Line to define points along the y coordinate.
        linez : array-like
            Line to define points along the z coordinate.

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
        linex = np.asarray(linex).squeeze()
        liney = np.asarray(liney).squeeze()
        linez = np.asarray(linez).squeeze()
        self.lines = (linex, liney, linez)
        assert all([n.ndim == 1 for n in self.lines]), \
            'Shape of lines is invalid'

        # For reconstructing the matrix later
        self.original_lengths = (len(linex), len(liney), len(linez))
        xx, yy, zz = np.meshgrid(linex, liney, linez, indexing='ij')
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
        map = 20*np.log10(np.abs(map))
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
            Position for the grid points along the extended dimension.
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
        - `coordinates`: Coordinates of the grid points as numpy.ndarray with
          shape (point, coordinate xyz).
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
        - `coordinates`: Coordinates of the grid points as numpy.ndarray with
          shape (point, coordinate xyz).
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
        """Computes maximum recommended frequency range for microphone array
        based on lowest Helmholtz number and the criterion
        `min_distance = wavelength/2`.

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
                            mic_coordinates: MicArray) -> numpy.ndarray:

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
                 mic_array: MicArray, grid: Grid,
                 c: float = 343):
        """Base constructor for Beamformer.

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
        assert issubclass(type(grid), Grid), \
            'grid should be a Grid object'
        assert c > 0, \
            'Speed of sound should be bigger than 0'
        assert multi_channel_signal.number_of_channels == \
            mic_array.number_of_points, \
            'Number of channels in signal and microphone array do not match'
        self.signal = multi_channel_signal
        self.grid = grid
        self.mics = mic_array
        self.c = c
        self.beamformer_type = 'Base'

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
        txt += f'''Number of grid points: {self.grid.number_of_points}\n'''
        print(txt)


class BeamformerFrequencyDAS(BaseBeamformer):
    """This is the base class for beamforming in frequency-domain.

    """
    # ======== Constructor ====================================================
    def __init__(self, multi_channel_signal: Signal,
                 mic_array: MicArray, grid: Grid,
                 steering_vector: SteeringVector,
                 c: float = 343):
        """Classic delay-and-sum (DAS) beamforming in frequency domain.

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
        super().__init__(multi_channel_signal, mic_array, grid, c)
        assert type(steering_vector) == SteeringVector, \
            'steering_vector should be of type SteeringVector'
        self.st_vec = steering_vector
        self.set_csm_parameters = self.signal.set_csm_parameters
        self.beamformer_type = 'Delay-and-sum (Frequency)'

    # ======== Get beamforming map ============================================
    def get_beamformer_map(self, center_frequency_hz: float,
                           octave_fraction: int = 3,
                           remove_csm_diagonal: bool = True) -> np.ndarray:
        """Run delay-and-sum beamforming in the given frequency range.

        Parameters
        ----------
        center_frequency_hz : float
            Center frequency for band in which to compute the beamforming.
        octave_fraction : int, optional
            Bandwidth (1/octave_fraction)-octave for the map. Pass 0 to
            get a single-frequency-bin map. Default: 3.
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

        print('...Apply...')
        self.map = np.zeros((self.grid.number_of_points,
                             number_frequency_bins), dtype='cfloat')
        for gind in range(self.grid.number_of_points):
            for find in range(len(f)):
                self.map[gind, find] = np.linalg.multi_dot(
                    [h_H[find, gind, :], csm[find, :, :], h[find, :, gind]])

        # Integrate over all frequencies
        if number_frequency_bins > 1:
            self.map = simpson(self.map, dx=f[1]-f[0], axis=1)
        else:
            self.map = self.map.squeeze()
        self.map = self.grid.reconstruct_map_shape(self.map)
        return self.map


class BeamformerTime(BaseBeamformer):
    """Base class for a beamformer in time-domain.

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
        super().__init__(multi_channel_signal, mic_array, grid, c)
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
        """Constructor for Source object. It is defined by an emitted signal
        and spatial coordinates. Its emission characteristic is omnidirectional
        by default.

        Parameters
        ----------
        signal : `Signal`
            Emitted signal by the source. It is restricted to one channel.
        coordinates : array-like
            Array with room coordinates defining source's position with shape
            (x, y, z).

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
            # Amplitude
            ns.time_data /= distances[i]
            # Append to final signal
            multi_channel_signal = merge_signals(
                multi_channel_signal, ns, padding_trimming=True)
        multi_channel_signal.remove_channel(0)
        return multi_channel_signal


def mix_sources_on_array(source_list: list, mics: MicArray, c: float = 343) ->\
        Signal:
    """This function takes in a list containing multiple sources and gives back
    the multi-channel signal on the array of the combined sources.

    Parameters
    ----------
    source_list : list
        List containing sources of object type `Source`.
    mics : `MicArray`
        Microphone array on which to mixed the sources.
    c : float, optional
        Speed of sound in m/s. Default: 343.

    Returns
    -------
    multi_channel_sig : `Signal`
        Multi-channel signal containing combined source's signals.

    """
    assert len(source_list) > 1, \
        'There must be at least two sources to combine'
    assert all([type(i) == MonopoleSource for i in source_list]), \
        'All sources in list should be of type Source'
    # Take first source
    multi_channel_sig = source_list[0].get_signals_on_array(mics, c)
    total_length_samples = multi_channel_sig.time_data.shape[0]
    source_list.pop(0)

    # Add all other sources progressively checking for shortest duration
    for s in source_list:
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
