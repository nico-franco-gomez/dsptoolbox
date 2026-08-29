import os
from os.path import join

import numpy as np

import dsptoolbox as dsp

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
            line1=x,
            line2=y,
            dimensions=(
                dsp.beamforming.SpatialDimension.X,
                dsp.beamforming.SpatialDimension.Y,
            ),
            value3=2,
        )
        g.plot_points()

        # 3D
        g = dsp.beamforming.Regular3DGrid(x, y, z)
        g.plot_points()

        # Line
        g = dsp.beamforming.LineGrid(
            line=x,
            dimension=dsp.beamforming.SpatialDimension.X,
            value2=0,
            value3=1,
        )
        g.plot_points()

    def test_mic_array(self):
        # Only functionality
        m = dsp.beamforming.MicArray(self.points_uniform)
        _ = m.array_center_channel_number
        _ = m.array_center_coordinates
        _ = m.aperture
        m.get_maximum_frequency_range()

    def test_steering_vector(self):
        dsp.beamforming.SteeringVector(
            formulation=dsp.beamforming.SteeringVectorType.TrueLocation
        )
        dsp.beamforming.SteeringVector(
            formulation=dsp.beamforming.SteeringVectorType.Inverse
        )
        dsp.beamforming.SteeringVector(
            formulation=dsp.beamforming.SteeringVectorType.TruePower
        )

        # Check for steering vector classic
        ma = dsp.beamforming.MicArray(self.points_uniform)
        xval = np.arange(-0.5, 0.5, 0.1)
        yval = np.arange(-0.5, 0.5, 0.1)
        zval = 1
        g = dsp.beamforming.Regular2DGrid(
            xval,
            yval,
            [dsp.beamforming.SpatialDimension.X, dsp.beamforming.SpatialDimension.Y],
            value3=zval,
        )
        r0 = ma.array_center_coordinates

        def dist(r1, r0):
            """Euclidean distance between two points"""
            return np.sqrt(np.sum((r1 - r0) ** 2))

        k = np.array([1000, 1200]) * np.pi * 2 / 343
        rt0 = g.get_distances_to_point(r0)
        h = np.zeros(
            (len(k), ma.number_of_points, g.number_of_points),
            dtype=np.complex128,
        )
        N = ma.number_of_points
        for i0, kn in enumerate(k):
            for i1 in range(ma.number_of_points):
                for i2 in range(g.number_of_points):
                    rti = dist(g.coordinates[i2, :], ma.coordinates[i1, :])
                    rt0 = dist(g.coordinates[i2, :], r0)
                    h[i0, i1, i2] = 1 / N * np.exp(-1j * kn * (rti - rt0))

        st = dsp.beamforming.SteeringVector(
            formulation=dsp.beamforming.SteeringVectorType.Classic
        )
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
        sp = dsp.Signal(
            join(os.path.dirname(__file__), "..", "example_data", "speech.flac")
        )
        sp = sp.pad_trim(20_000)
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
        g = dsp.beamforming.Regular2DGrid(
            xval,
            yval,
            [dsp.beamforming.SpatialDimension.X, dsp.beamforming.SpatialDimension.Y],
            value3=zval,
        )

        # Steering vector
        st = dsp.beamforming.SteeringVector(
            formulation=dsp.beamforming.SteeringVectorType.TrueLocation
        )

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

        # MVDR inverts the cross-spectral matrix per frequency bin. The
        # mic array shared by this smoke test is `points_uniform` with `z`
        # collapsed to 0: it was built from a 5x5x5 3D grid, so every one
        # of the 25 unique (x, y) locations is physically duplicated 5
        # times once z is flattened away. Duplicate mic positions record
        # numerically identical signals, which makes the CSM exactly
        # rank-deficient regardless of snapshot count -- an expected
        # degenerate-input failure of this specific shared smoke-test
        # fixture, not a defect in `BeamformerMVDR` itself. See
        # `test_beamformer_mvdr_localizes_source` below for MVDR exercised
        # on a non-degenerate array, where it is asserted on directly.
        try:
            # Create beamformer and plot setting
            bf = dsp.beamforming.BeamformerMVDR(s, ma, g, st)
            # Get and show map
            bf.get_beamformer_map(2000, 0, gamma=10)
        except np.linalg.LinAlgError as e:
            print(e)

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
        sp = dsp.Signal(
            join(os.path.dirname(__file__), "..", "example_data", "speech.flac")
        )
        sp = sp.pad_trim(20_000)
        ns = dsp.generators.noise(
            length_seconds=0.3, sampling_rate_hz=sp.sampling_rate_hz
        )
        sp = dsp.beamforming.MonopoleSource(sp, [0, -0.5, 0.5])
        ns = dsp.beamforming.MonopoleSource(ns, [0, 0, 0.5])
        s = dsp.beamforming.mix_sources_on_array([sp, ns], ma)
        # Grid
        xval = np.arange(-0.5, 0.5, 0.1)
        g = dsp.beamforming.LineGrid(xval, dsp.beamforming.SpatialDimension.Y, 0.5, 0)
        bf = dsp.beamforming.BeamformerDASTime(s, ma, g)
        bf.get_beamformer_output()

    def _make_planar_array(self, spacing=0.25, extent=1.0, z=0.0):
        """A non-degenerate (no duplicate positions) planar mic array,
        unlike this module's `points_uniform` fixture which, once its `z`
        coordinate is collapsed to a constant, physically duplicates every
        (x, y) location several times over (it was built as a 3D grid).

        """
        line = np.arange(0, extent + 1e-9, spacing)
        xx, yy = np.meshgrid(line, line, indexing="ij")
        return dsp.beamforming.MicArray(
            dict(x=xx.flatten(), y=yy.flatten(), z=np.full(xx.size, z))
        )

    def test_beamformer_das_frequency_localizes_source(self):
        """A single monopole source, propagated onto an idealized free-field
        planar array via `MonopoleSource.get_signals_on_array` (per its
        source: spherical-wave delay `distance/c` plus `1/(1+distance)`
        amplitude falloff), should have the DAS frequency-domain map peak
        exactly at the grid point coinciding with the true source location
        (verified empirically to land exactly on the source's grid node,
        not just nearby, for this noise-source/free-field setup).

        """
        fs = 10_000
        ma = self._make_planar_array(spacing=0.25, extent=1.0, z=0.0)

        true_xy = (0.4, 0.6)
        source = dsp.beamforming.MonopoleSource(
            dsp.generators.noise(length_seconds=2, sampling_rate_hz=fs),
            [true_xy[0], true_xy[1], 0.5],
        )
        s = source.get_signals_on_array(ma)

        gx = np.arange(0.0, 1.01, 0.2)
        gy = np.arange(0.0, 1.01, 0.2)
        grid = dsp.beamforming.Regular2DGrid(
            gx,
            gy,
            [dsp.beamforming.SpatialDimension.X, dsp.beamforming.SpatialDimension.Y],
            value3=0.5,
        )
        st = dsp.beamforming.SteeringVector(
            formulation=dsp.beamforming.SteeringVectorType.TrueLocation
        )

        bf = dsp.beamforming.BeamformerDASFrequency(s, ma, grid, st)
        beamformer_map = bf.get_beamformer_map(2000, 3, remove_csm_diagonal=True)

        peak_idx = np.unravel_index(np.argmax(beamformer_map), beamformer_map.shape)
        peak_xy = (gx[peak_idx[0]], gy[peak_idx[1]])
        np.testing.assert_allclose(peak_xy, true_xy, atol=1e-9)

    def test_beamformer_das_time_localizes_source(self):
        """Same idealized free-field peak-location check as the frequency-
        domain DAS test, but for `BeamformerDASTime`: the grid-focused
        output channel with the highest energy should coincide exactly
        with the true source location (verified empirically to be exact
        for this noise-source setup).

        """
        fs = 10_000
        ma = self._make_planar_array(spacing=0.25, extent=1.0, z=0.0)

        true_xy = (0.4, 0.6)
        source = dsp.beamforming.MonopoleSource(
            dsp.generators.noise(length_seconds=1, sampling_rate_hz=fs),
            [true_xy[0], true_xy[1], 0.5],
        )
        s = source.get_signals_on_array(ma)

        gx = np.arange(0.0, 1.01, 0.2)
        gy = np.arange(0.0, 1.01, 0.2)
        grid = dsp.beamforming.Regular2DGrid(
            gx,
            gy,
            [dsp.beamforming.SpatialDimension.X, dsp.beamforming.SpatialDimension.Y],
            value3=0.5,
        )

        bf = dsp.beamforming.BeamformerDASTime(s, ma, grid)
        out = bf.get_beamformer_output()

        energies = np.sum(out.time_data**2, axis=0)
        energy_map = grid.reconstruct_map_shape(energies)
        peak_idx = np.unravel_index(np.argmax(energy_map), energy_map.shape)
        peak_xy = (gx[peak_idx[0]], gy[peak_idx[1]])
        np.testing.assert_allclose(peak_xy, true_xy, atol=1e-9)

    def test_beamformer_mvdr_localizes_source(self):
        """MVDR needs an invertible cross-spectral matrix per frequency
        bin, which requires enough independent Welch snapshots relative to
        the channel count and (critically) no duplicate mic positions --
        neither of which the shared `points_uniform`-derived array in
        `test_beamformer_frequency` satisfies (see the comment there). With
        a proper non-degenerate array and a long-enough signal, MVDR works
        without needing any exception handling.

        Unlike the DAS variants above (a fixed, deterministic delay-sum
        that localizes exactly regardless of the specific noise
        realization), MVDR's CSM inverse makes it a statistical estimator
        sensitive to finite-data covariance noise -- an adaptive
        beamformer per the plan's own carve-out for cases "genuinely too
        complex for an exact reference". A fixed seed and a one-grid-step
        tolerance (rather than requiring the exact grid node) account for
        this; without the seed, the peak was empirically observed to
        occasionally land one grid cell away from the true location on an
        unlucky noise draw.

        """
        fs = 10_000
        ma = self._make_planar_array(spacing=0.25, extent=1.0, z=0.0)

        true_xy = (0.0, 0.4)
        rng_state = np.random.get_state()
        np.random.seed(0)
        try:
            noise_signal = dsp.generators.noise(length_seconds=5, sampling_rate_hz=fs)
        finally:
            np.random.set_state(rng_state)
        source = dsp.beamforming.MonopoleSource(
            noise_signal, [true_xy[0], true_xy[1], 0.5]
        )
        s = source.get_signals_on_array(ma)

        gx = np.arange(-0.2, 0.21, 0.1)
        gy = np.arange(-0.5, 0.51, 0.1)
        grid = dsp.beamforming.Regular2DGrid(
            gx,
            gy,
            [dsp.beamforming.SpatialDimension.X, dsp.beamforming.SpatialDimension.Y],
            value3=0.5,
        )
        st = dsp.beamforming.SteeringVector(
            formulation=dsp.beamforming.SteeringVectorType.TrueLocation
        )

        bf = dsp.beamforming.BeamformerMVDR(s, ma, grid, st)
        beamformer_map = bf.get_beamformer_map(2000, 3, gamma=10)

        peak_idx = np.unravel_index(np.argmax(beamformer_map), beamformer_map.shape)
        peak_xy = (gx[peak_idx[0]], gy[peak_idx[1]])
        np.testing.assert_allclose(peak_xy, true_xy, atol=0.1)
