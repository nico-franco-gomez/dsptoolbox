import dsptoolbox as dsp
import numpy as np
from os.path import join

x = np.arange(0, 1.1, 0.25)
y = x.copy()
z = x.copy()
xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")


class TestBeamformingModule:
    points_uniform = dict(x=xx.flatten(), y=yy.flatten(), z=zz.flatten())

    def test_grid(self):
        # Mostly functionality
        g = dsp.beamforming.Grid(positions=self.points_uniform)

        # Check extent
        assert np.all([0, 1] == g.extent["x"])

        # Check number of points
        assert g.number_of_points == len(x) * len(y) * len(z)

        # Check other
        g.get_distances_to_point([0, 0, 0])
        g.find_nearest_point([-0.2, 0.1, -1])
        g.plot_points(projection=None)
        g.plot_points(projection="2d")
        g.plot_points(projection="3d")

        # g.reconstruct_map_shape()

    def test_regular_grids(self):
        # Only functionality
        # 2D
        g = dsp.beamforming.Regular2DGrid(
            line1=x, line2=y, dimensions=("x", "y"), value3=2
        )
        g.plot_points()

        # 3D
        g = dsp.beamforming.Regular3DGrid(x, y, z)
        g.plot_points()

        # Line
        g = dsp.beamforming.LineGrid(line=x, dimension="x", value2=0, value3=1)
        g.plot_points()

    def test_mic_array(self):
        # Only functionality
        m = dsp.beamforming.MicArray(self.points_uniform)
        m.array_center_channel_number
        m.array_center_coordinates
        m.aperture
        m.get_maximum_frequency_range()

    def test_steering_vector(self):
        dsp.beamforming.SteeringVector(formulation="true location")
        dsp.beamforming.SteeringVector(formulation="inverse")
        dsp.beamforming.SteeringVector(formulation="true power")

        # Check for steering vector classic
        ma = dsp.beamforming.MicArray(self.points_uniform)
        xval = np.arange(-0.5, 0.5, 0.1)
        yval = np.arange(-0.5, 0.5, 0.1)
        zval = 1
        g = dsp.beamforming.Regular2DGrid(xval, yval, ["x", "y"], value3=zval)
        r0 = ma.array_center_coordinates

        def dist(r1, r0):
            """Euclidean distance between two points"""
            return np.sqrt(np.sum((r1 - r0) ** 2))

        k = np.array([1000, 1200]) * np.pi * 2 / 343
        rt0 = g.get_distances_to_point(r0)
        h = np.zeros(
            (len(k), ma.number_of_points, g.number_of_points), dtype="cfloat"
        )
        N = ma.number_of_points
        for i0, kn in enumerate(k):
            for i1 in range(ma.number_of_points):
                for i2 in range(g.number_of_points):
                    rti = dist(g.coordinates[i2, :], ma.coordinates[i1, :])
                    rt0 = dist(g.coordinates[i2, :], r0)
                    h[i0, i1, i2] = 1 / N * np.exp(-1j * kn * (rti - rt0))

        st = dsp.beamforming.SteeringVector(formulation="classic")
        h_intern = st.get_vector(k, g, ma)

        # Test for difference
        assert np.all(np.isclose(h_intern, h))

    def test_monopole_source_transmission(self):
        # Only functionality
        ma = self.points_uniform.copy()
        ma["z"] = np.zeros(len(ma["x"]))
        ma = dsp.beamforming.MicArray(ma)

        # Single source
        ns = dsp.beamforming.MonopoleSource(
            dsp.generators.noise(length_seconds=0.5, sampling_rate_hz=20_000),
            [0, 0, 0.5],
        )
        # Simulate getting signals on the array
        ns.get_signals_on_array(ma)

        # Multiple sources
        sp = dsp.Signal(join("examples", "data", "speech.flac"))
        sp = dsp.pad_trim(sp, 20_000)
        ns = dsp.generators.noise(
            length_seconds=0.5, sampling_rate_hz=sp.sampling_rate_hz
        )
        sp = dsp.beamforming.MonopoleSource(sp, [0, -0.5, 0.4])
        ns = dsp.beamforming.MonopoleSource(ns, [0, 0, 0.5])
        # Simulate combining signals on array
        dsp.beamforming.mix_sources_on_array([sp, ns], ma)

    def test_beamformer_frequency(self):
        # Only functionality
        # Mic Array
        ma = self.points_uniform.copy()
        ma["z"] = np.zeros(len(ma["x"]))
        ma = dsp.beamforming.MicArray(ma)

        # Signal (simulated)
        ns = dsp.beamforming.MonopoleSource(
            dsp.generators.noise(length_seconds=2, sampling_rate_hz=10_000),
            [0, 0.4, 0.5],
        )
        s = ns.get_signals_on_array(ma)

        # Grid
        xval = np.arange(-0.2, 0.2, 0.1)
        yval = np.arange(-0.5, 0.5, 0.1)
        zval = 0.5
        g = dsp.beamforming.Regular2DGrid(xval, yval, ["x", "y"], value3=zval)

        # Steering vector
        st = dsp.beamforming.SteeringVector(formulation="true location")

        # Create beamformer and plot setting
        bf = dsp.beamforming.BeamformerDASFrequency(s, ma, g, st)
        # Get and show map
        bf.get_beamformer_map(2000, 0, remove_csm_diagonal=True)

        # Create beamformer and plot setting
        bf = dsp.beamforming.BeamformerOrthogonal(s, ma, g, st)
        # Get and show map
        bf.get_beamformer_map(2000, 0, number_eigenvalues=None)

        # Create beamformer and plot setting
        bf = dsp.beamforming.BeamformerFunctional(s, ma, g, st)
        # Get and show map
        bf.get_beamformer_map(2000, 0, gamma=10)

        try:
            # Create beamformer and plot setting
            bf = dsp.beamforming.BeamformerMVDR(s, ma, g, st)
            # Get and show map
            bf.get_beamformer_map(2000, 0, gamma=10)
        except np.linalg.LinAlgError as e:
            print(e)
            pass
        except Exception as e:
            print(e)
            assert False

        # Create beamformer and plot setting
        bf = dsp.beamforming.BeamformerCleanSC(s, ma, g, st)
        # Get and show map
        bf.get_beamformer_map(
            2000,
            0,
            maximum_iterations=10,
            safety_factor=0.5,
            remove_csm_diagonal=True,
        )

    def test_beamformer_time(self):
        # Only functionality
        # Mic Array
        ma = self.points_uniform.copy()
        ma["z"] = np.zeros(len(ma["x"]))
        ma = dsp.beamforming.MicArray(ma)
        # Signal (simulated)
        sp = dsp.Signal(join("examples", "data", "speech.flac"))
        sp = dsp.pad_trim(sp, 20_000)
        ns = dsp.generators.noise(
            length_seconds=0.3, sampling_rate_hz=sp.sampling_rate_hz
        )
        sp = dsp.beamforming.MonopoleSource(sp, [0, -0.5, 0.5])
        ns = dsp.beamforming.MonopoleSource(ns, [0, 0, 0.5])
        s = dsp.beamforming.mix_sources_on_array([sp, ns], ma)
        # Grid
        xval = np.arange(-0.5, 0.5, 0.1)
        g = dsp.beamforming.LineGrid(xval, "y", 0.5, 0)
        bf = dsp.beamforming.BeamformerDASTime(s, ma, g)
        bf.get_beamformer_output()
