import os
from os.path import join

import numpy as np
import pytest
from matplotlib.pyplot import close, subplots

import dsptoolbox as dsp

x = np.arange(0, 1.1, 0.25)
y = x.copy()
z = x.copy()
xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")


class TestBeamformingModule:
    points_uniform = dict(x=xx.flatten(), y=yy.flatten(), z=zz.flatten())

    def test_grid(self):
        g = dsp.beamforming.Grid(positions=self.points_uniform)

        assert np.all([0, 1] == g.extent["x"])
        assert g.number_of_points == len(x) * len(y) * len(z)

        g.get_distances_to_point([0, 0, 0])
        g.find_nearest_point([-0.2, 0.1, -1])
        g.plot_points()
        g.plot_points(dsp.beamforming.PointsProjection.TwoDimensional)
        g.plot_points(dsp.beamforming.PointsProjection.ThreeDimensional)

    def test_regular_grids(self):
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

        assert np.all(np.isclose(h_intern, h))

    def test_monopole_source_transmission(self):
        ma = self.points_uniform.copy()
        ma["z"] = np.zeros(len(ma["x"]))
        ma = dsp.beamforming.MicArray(ma)

        ns = dsp.beamforming.MonopoleSource(
            dsp.generators.noise(length_seconds=0.5, sampling_rate_hz=20_000, rng=100),
            [0, 0, 0.5],
        )
        ns.get_signals_on_array(ma)

        # Multiple sources mixed onto the array
        sp = dsp.Signal(
            join(os.path.dirname(__file__), "..", "example_data", "speech.flac")
        )
        sp = sp.pad_trim(20_000)
        ns = dsp.generators.noise(
            length_seconds=0.5,
            sampling_rate_hz=sp.sampling_rate_hz,
            rng=101,
        )
        sp = dsp.beamforming.MonopoleSource(sp, [0, -0.5, 0.4])
        ns = dsp.beamforming.MonopoleSource(ns, [0, 0, 0.5])
        dsp.beamforming.mix_sources_on_array([sp, ns], ma)

    def test_monopole_source_amplitude_follows_distance_law(self):
        """`MonopoleSource.get_signals_on_array` scales the emitted signal's
        amplitude by `1 / (1 + distance)` (see
        `dsptoolbox/beamforming/beamforming.py`, `MonopoleSource.
        get_signals_on_array`) -- a regularized version of the free-field
        monopole's `1/r` law that stays finite at `r=0`. This is an exact,
        known reference (not merely a plausibility property): for two
        receivers at distances `d0`, `d1` from the source, the ratio of
        captured RMS levels should equal `(1 + d1) / (1 + d0)`.

        """
        d0, d1 = 1.0, 3.0
        ma = dsp.beamforming.MicArray(
            dict(x=np.array([d0, d1]), y=np.zeros(2), z=np.zeros(2))
        )
        source_signal = dsp.generators.noise(
            length_seconds=2.0, sampling_rate_hz=20_000, rng=104
        )
        ns = dsp.beamforming.MonopoleSource(source_signal, [0, 0, 0])
        received = ns.get_signals_on_array(ma)

        rms0, rms1 = dsp.rms(received, in_dbfs=False)
        expected_ratio = (1.0 + d1) / (1.0 + d0)
        np.testing.assert_allclose(rms0 / rms1, expected_ratio, rtol=0.02)

    def test_beamformer_frequency(self):
        ma = self.points_uniform.copy()
        ma["z"] = np.zeros(len(ma["x"]))
        ma = dsp.beamforming.MicArray(ma)

        ns = dsp.beamforming.MonopoleSource(
            dsp.generators.noise(length_seconds=2, sampling_rate_hz=10_000, rng=102),
            [0, 0.4, 0.5],
        )
        s = ns.get_signals_on_array(ma)

        xval = np.arange(-0.2, 0.2, 0.1)
        yval = np.arange(-0.5, 0.5, 0.1)
        zval = 0.5
        g = dsp.beamforming.Regular2DGrid(
            xval,
            yval,
            [dsp.beamforming.SpatialDimension.X, dsp.beamforming.SpatialDimension.Y],
            value3=zval,
        )

        st = dsp.beamforming.SteeringVector(
            formulation=dsp.beamforming.SteeringVectorType.TrueLocation
        )

        bf = dsp.beamforming.BeamformerDASFrequency(s, ma, g, st)
        bf.get_beamformer_map(2000, 0, remove_csm_diagonal=True)

        bf = dsp.beamforming.BeamformerOrthogonal(s, ma, g, st)
        bf.get_beamformer_map(2000, 0, number_eigenvalues=None)

        bf = dsp.beamforming.BeamformerFunctional(s, ma, g, st)
        bf.get_beamformer_map(2000, 0, gamma=10)

        # MVDR inverts the cross-spectral matrix per frequency bin. The
        # mic array shared by this smoke test is `points_uniform` with `z`
        # collapsed to 0: it was built from a 5x5x5 3D grid, so every one
        # of the 25 unique (x, y) locations is physically duplicated 5
        # times once z is flattened away. Duplicate mic positions record
        # numerically identical signals, which makes the CSM exactly
        # rank-deficient regardless of snapshot count. This is exactly what
        # the diagonal loading regularizes, so the default run succeeds
        # while the unregularized one does not.
        bf = dsp.beamforming.BeamformerMVDR(s, ma, g, st)
        bf.get_beamformer_map(2000, 0)
        with pytest.raises(np.linalg.LinAlgError):
            bf.get_beamformer_map(2000, 0, diagonal_loading_db=None)

        bf = dsp.beamforming.BeamformerCleanSC(s, ma, g, st)
        bf.get_beamformer_map(
            2000,
            0,
            maximum_iterations=10,
            safety_factor=0.5,
            remove_csm_diagonal=True,
        )

    def test_beamformer_steering_vector_cache(self):
        ma = self._make_planar_array(spacing=0.5, extent=0.5)
        signal = dsp.generators.noise(
            length_seconds=0.1,
            sampling_rate_hz=10_000,
            number_of_channels=ma.number_of_points,
            rng=106,
        )
        grid = dsp.beamforming.LineGrid(
            np.array([0.0, 0.5]), dsp.beamforming.SpatialDimension.X, 0.0, 0.5
        )
        steering = dsp.beamforming.SteeringVector(
            formulation=dsp.beamforming.SteeringVectorType.TrueLocation
        )
        beamformer = dsp.beamforming.BeamformerDASFrequency(signal, ma, grid, steering)
        original_get_vector = steering.get_vector
        calls = 0

        def counted_get_vector(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_get_vector(*args, **kwargs)

        steering.get_vector = counted_get_vector
        wave_numbers = np.array([1.0, 2.0])
        first = beamformer._get_steering_vector(wave_numbers)
        second = beamformer._get_steering_vector(wave_numbers.copy())
        assert calls == 1
        assert first is second

        changed_frequency = beamformer._get_steering_vector(np.array([1.0, 3.0]))
        assert calls == 2
        assert changed_frequency is not second

        changed_coordinates = grid.coordinates.copy()
        changed_coordinates[0, 0] += 0.1
        grid.coordinates = changed_coordinates
        changed_grid = beamformer._get_steering_vector(np.array([1.0, 3.0]))
        assert calls == 3
        assert changed_grid is not changed_frequency

        beamformer.delete_cache()
        deleted = beamformer._get_steering_vector(np.array([1.0, 3.0]))
        assert calls == 4
        assert deleted is not changed_grid

    def test_beamformer_time(self):
        ma = self.points_uniform.copy()
        ma["z"] = np.zeros(len(ma["x"]))
        ma = dsp.beamforming.MicArray(ma)
        sp = dsp.Signal(
            join(os.path.dirname(__file__), "..", "example_data", "speech.flac")
        )
        sp = sp.pad_trim(20_000)
        ns = dsp.generators.noise(
            length_seconds=0.3,
            sampling_rate_hz=sp.sampling_rate_hz,
            rng=103,
        )
        sp = dsp.beamforming.MonopoleSource(sp, [0, -0.5, 0.5])
        ns = dsp.beamforming.MonopoleSource(ns, [0, 0, 0.5])
        s = dsp.beamforming.mix_sources_on_array([sp, ns], ma)
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
            dsp.generators.noise(length_seconds=2, sampling_rate_hz=fs, rng=104),
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
            dsp.generators.noise(length_seconds=1, sampling_rate_hz=fs, rng=105),
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

    def test_beamformer_das_time_integer_delays(self):
        """The integer-delay DAS path should match rounded-delay accumulation."""
        fs = 10_000
        ma = self._make_planar_array(spacing=0.25, extent=1.0, z=0.0)

        source = dsp.beamforming.MonopoleSource(
            dsp.generators.noise(length_seconds=1, sampling_rate_hz=fs, rng=105),
            [0.4, 0.6, 0.5],
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
        out = bf.get_beamformer_output(fractional_delay=False)

        distances = ma.get_distances_to_point(grid.coordinates)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        longest_delay_samples = int((max_distance - min_distance) / 343 * fs + 2)
        total_length_samples = s.time_data.shape[0] + longest_delay_samples
        expected = np.zeros((total_length_samples, grid.number_of_points))
        for grid_index in range(grid.number_of_points):
            delays = (max_distance - distances[:, grid_index]) / 343
            for mic_index in range(ma.number_of_points):
                delay_samples = int(delays[mic_index] * fs + 0.5)
                delayed = s.get_channels(mic_index).delay(delay_samples).time_data
                expected[: delayed.shape[0], grid_index] += (
                    delayed[:, 0] * distances[mic_index, grid_index]
                )
            expected[:, grid_index] /= ma.number_of_points

        np.testing.assert_array_equal(out.time_data, expected)

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
        noise_signal = dsp.generators.noise(
            length_seconds=5, sampling_rate_hz=fs, rng=0
        )
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
        beamformer_map = bf.get_beamformer_map(2000, 3)

        peak_idx = np.unravel_index(np.argmax(beamformer_map), beamformer_map.shape)
        peak_xy = (gx[peak_idx[0]], gy[peak_idx[1]])
        np.testing.assert_allclose(peak_xy, true_xy, atol=0.1)

    def test_regular_grid_plot_map(self):
        """Both regular grids used to pass a `returns` keyword that
        `general_matrix_plot` does not accept.

        """
        rng = np.random.default_rng(0)

        g2 = dsp.beamforming.Regular2DGrid(
            line1=x,
            line2=y,
            dimensions=(
                dsp.beamforming.SpatialDimension.X,
                dsp.beamforming.SpatialDimension.Y,
            ),
            value3=2,
        )
        fig, ax = g2.plot_map(rng.uniform(size=g2.number_of_points))
        close(fig)

        g3 = dsp.beamforming.Regular3DGrid(x, y, z)
        fig, _ = g3.plot_map(
            rng.uniform(size=g3.number_of_points),
            dsp.beamforming.SpatialDimension.Z,
            0.5,
        )
        close(fig)

        # Both accept an existing axis (A10)
        _, shared = subplots(1, 1)
        g2.plot_map(rng.uniform(size=g2.number_of_points), ax=shared)
        g3.plot_map(
            rng.uniform(size=g3.number_of_points),
            dsp.beamforming.SpatialDimension.Z,
            0.5,
            ax=shared,
        )
