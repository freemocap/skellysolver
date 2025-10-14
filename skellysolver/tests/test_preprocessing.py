"""Tests for data preprocessing (Phase 2).

Tests preprocessing functions.
"""

import numpy as np

from skellysolver.data.data_models import Trajectory3D, TrajectoryDataset
from skellysolver.data.preprocessing import (
    interpolate_missing,
    filter_by_confidence,
    smooth_trajectories,
    remove_outliers,
    center_data,
    scale_data,
    subsample_frames,
)


class TestInterpolateMissing:
    """Test missing data interpolation."""
    
    def test_interpolate_linear(self) -> None:
        """Should interpolate missing values linearly."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
            [2.0, 2.0, 2.0],
        ])
        
        traj = Trajectory3D(marker_name="m1", positions=positions)
        dataset = TrajectoryDataset(data={"m1": traj}, frame_indices=np.arange(3))
        
        # Interpolate
        interpolated = interpolate_missing(dataset=dataset, method="linear")
        
        # Check interpolated value
        interp_pos = interpolated.data["m1"].positions
        expected_middle = np.array([1.0, 1.0, 1.0])
        
        assert np.allclose(interp_pos[1], expected_middle)
    
    def test_interpolate_preserves_valid_data(self) -> None:
        """Should not change valid data."""
        positions = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
            [4.0, 5.0, 6.0],
        ])
        
        traj = Trajectory3D(marker_name="m1", positions=positions)
        dataset = TrajectoryDataset(data={"m1": traj}, frame_indices=np.arange(3))
        
        interpolated = interpolate_missing(dataset=dataset)
        
        interp_pos = interpolated.data["m1"].positions
        
        # First and last should be unchanged
        assert np.allclose(interp_pos[0], [1.0, 2.0, 3.0])
        assert np.allclose(interp_pos[2], [4.0, 5.0, 6.0])


class TestFilterByConfidence:
    """Test confidence filtering."""
    
    def test_filter_removes_low_confidence(self) -> None:
        """Should remove low-confidence frames."""
        positions = np.ones((10, 3))
        confidence = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.9, 0.3, 0.8, 0.2, 0.9])
        
        traj = Trajectory3D(
            marker_name="m1",
            positions=positions,
            confidence=confidence
        )
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(10)
        )
        
        # Filter
        filtered = filter_by_confidence(
            dataset=dataset,
            min_confidence=0.5,
            min_valid_markers=1
        )
        
        # Should keep 6 frames (confidence >= 0.5)
        assert filtered.n_frames == 6


class TestSmoothTrajectories:
    """Test trajectory smoothing."""
    
    def test_smooth_reduces_noise(self) -> None:
        """Should reduce noise in trajectories."""
        # Create noisy trajectory
        t = np.linspace(0, 2 * np.pi, 100)
        positions_clean = np.column_stack([np.cos(t), np.sin(t), np.zeros(100)])
        noise = np.random.normal(loc=0.0, scale=0.1, size=positions_clean.shape)
        positions_noisy = positions_clean + noise
        
        traj = Trajectory3D(marker_name="m1", positions=positions_noisy)
        dataset = TrajectoryDataset(data={"m1": traj}, frame_indices=np.arange(100))
        
        # Smooth
        smoothed = smooth_trajectories(
            dataset=dataset,
            window_size=5,
            poly_order=2
        )
        
        # Compute noise before and after
        noise_before = np.linalg.norm(positions_noisy - positions_clean, axis=1)
        noise_after = np.linalg.norm(
            smoothed.data["m1"].positions - positions_clean,
            axis=1
        )
        
        # Smoothing should reduce average noise
        assert np.mean(noise_after) < np.mean(noise_before)


class TestRemoveOutliers:
    """Test outlier removal."""
    
    def test_remove_velocity_outliers(self) -> None:
        """Should remove velocity-based outliers."""
        # Smooth trajectory with one outlier
        positions = np.ones((10, 3))
        positions[5] = np.array([100.0, 100.0, 100.0])  # Outlier
        
        traj = Trajectory3D(marker_name="m1", positions=positions)
        dataset = TrajectoryDataset(data={"m1": traj}, frame_indices=np.arange(10))
        
        # Remove outliers
        cleaned = remove_outliers(
            dataset=dataset,
            threshold=3.0,
            method="velocity"
        )
        
        # Outlier should be set to NaN
        cleaned_pos = cleaned.data["m1"].positions
        assert np.isnan(cleaned_pos[5, 0])


class TestCenterData:
    """Test data centering."""
    
    def test_center_at_origin(self) -> None:
        """Should center data at origin."""
        positions = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ])
        
        traj = Trajectory3D(marker_name="m1", positions=positions)
        dataset = TrajectoryDataset(data={"m1": traj}, frame_indices=np.arange(3))
        
        # Center
        centered, centroid = center_data(dataset=dataset, method="mean")
        
        # Check centroid
        expected_centroid = np.mean(positions, axis=0)
        assert np.allclose(centroid, expected_centroid)
        
        # Check centered data
        centered_pos = centered.data["m1"].positions
        assert np.allclose(np.mean(centered_pos, axis=0), 0.0, atol=1e-10)


class TestScaleData:
    """Test data scaling."""
    
    def test_scale_positions(self) -> None:
        """Should scale all positions."""
        positions = np.ones((10, 3))
        
        traj = Trajectory3D(marker_name="m1", positions=positions)
        dataset = TrajectoryDataset(data={"m1": traj}, frame_indices=np.arange(10))
        
        # Scale
        scaled = scale_data(dataset=dataset, scale_factor=0.001)
        
        # Check scaling
        scaled_pos = scaled.data["m1"].positions
        assert np.allclose(scaled_pos, 0.001)


class TestSubsampleFrames:
    """Test frame subsampling."""
    
    def test_subsample_by_step(self) -> None:
        """Should subsample frames."""
        positions = np.arange(100).reshape(100, 1).repeat(3, axis=1).astype(float)
        
        traj = Trajectory3D(marker_name="m1", positions=positions)
        dataset = TrajectoryDataset(data={"m1": traj}, frame_indices=np.arange(100))
        
        # Subsample every 2nd frame
        subsampled = subsample_frames(dataset=dataset, step=2)
        
        assert subsampled.n_frames == 50
        
        # Check frames are correct
        subsampled_pos = subsampled.data["m1"].positions
        assert np.allclose(subsampled_pos[0, 0], 0.0)
        assert np.allclose(subsampled_pos[1, 0], 2.0)
        assert np.allclose(subsampled_pos[2, 0], 4.0)
