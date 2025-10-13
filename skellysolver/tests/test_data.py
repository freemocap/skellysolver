"""Tests for data structures (Phase 2).

Tests Trajectory3D, Observation2D, and TrajectoryDataset.
"""

import numpy as np
import pytest

from skellysolver.data.base_data import Trajectory3D, Observation2D, TrajectoryDataset


class TestTrajectory3D:
    """Test Trajectory3D data structure."""
    
    def test_create_trajectory(self, sample_3d_trajectory: np.ndarray) -> None:
        """Should create trajectory."""
        positions = sample_3d_trajectory[:, 0, :]  # First marker only
        
        traj = Trajectory3D(
            marker_name="test_marker",
            positions=positions,
            confidence=None
        )
        
        assert traj.marker_name == "test_marker"
        assert traj.n_frames == len(positions)
        assert traj.positions.shape == positions.shape
    
    def test_is_valid_no_confidence(self) -> None:
        """Should check validity based on NaN when no confidence."""
        positions = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
            [4.0, 5.0, 6.0],
        ])
        
        traj = Trajectory3D(marker_name="test", positions=positions)
        valid_mask = traj.is_valid()
        
        assert valid_mask[0] is True
        assert valid_mask[1] is False
        assert valid_mask[2] is True
    
    def test_is_valid_with_confidence(self) -> None:
        """Should check validity based on confidence threshold."""
        positions = np.ones((3, 3))
        confidence = np.array([0.9, 0.2, 0.5])
        
        traj = Trajectory3D(
            marker_name="test",
            positions=positions,
            confidence=confidence
        )
        
        valid_mask = traj.is_valid(min_confidence=0.3)
        
        assert valid_mask[0] is True
        assert valid_mask[1] is False
        assert valid_mask[2] is True
    
    def test_get_valid_positions(self) -> None:
        """Should return only valid positions."""
        positions = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
            [4.0, 5.0, 6.0],
        ])
        
        traj = Trajectory3D(marker_name="test", positions=positions)
        valid_pos = traj.get_valid_positions()
        
        assert len(valid_pos) == 2
        assert np.allclose(valid_pos[0], [1.0, 2.0, 3.0])
        assert np.allclose(valid_pos[1], [4.0, 5.0, 6.0])
    
    def test_get_centroid(self) -> None:
        """Should compute centroid of valid positions."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [np.nan, np.nan, np.nan],
        ])
        
        traj = Trajectory3D(marker_name="test", positions=positions)
        centroid = traj.get_centroid()
        
        expected = np.array([1.0, 1.0, 1.0])
        assert np.allclose(centroid, expected)


class TestObservation2D:
    """Test Observation2D data structure."""
    
    def test_create_observation(self, sample_2d_observations: np.ndarray) -> None:
        """Should create 2D observation."""
        positions = sample_2d_observations[:, 0, :]  # First point only
        
        obs = Observation2D(
            point_name="pupil_p1",
            positions=positions,
            confidence=None
        )
        
        assert obs.point_name == "pupil_p1"
        assert obs.n_frames == len(positions)
        assert obs.positions.shape == positions.shape
    
    def test_is_valid_2d(self) -> None:
        """Should check 2D validity."""
        positions = np.array([
            [100.0, 200.0],
            [np.nan, np.nan],
            [150.0, 250.0],
        ])
        
        obs = Observation2D(point_name="test", positions=positions)
        valid_mask = obs.is_valid()
        
        assert valid_mask[0] is True
        assert valid_mask[1] is False
        assert valid_mask[2] is True


class TestTrajectoryDataset:
    """Test TrajectoryDataset."""
    
    def test_create_dataset(self) -> None:
        """Should create dataset from trajectories."""
        traj_1 = Trajectory3D(
            marker_name="m1",
            positions=np.random.randn(10, 3)
        )
        traj_2 = Trajectory3D(
            marker_name="m2",
            positions=np.random.randn(10, 3)
        )
        
        dataset = TrajectoryDataset(
            data={"m1": traj_1, "m2": traj_2},
            frame_indices=np.arange(10)
        )
        
        assert dataset.n_frames == 10
        assert dataset.n_markers == 2
        assert dataset.is_3d
    
    def test_to_array(self) -> None:
        """Should convert to numpy array."""
        traj_1 = Trajectory3D(
            marker_name="m1",
            positions=np.ones((10, 3))
        )
        traj_2 = Trajectory3D(
            marker_name="m2",
            positions=np.ones((10, 3)) * 2
        )
        
        dataset = TrajectoryDataset(
            data={"m1": traj_1, "m2": traj_2},
            frame_indices=np.arange(10)
        )
        
        array = dataset.to_array()
        
        assert array.shape == (10, 2, 3)
        assert np.allclose(array[:, 0, :], 1.0)
        assert np.allclose(array[:, 1, :], 2.0)
    
    def test_to_array_specific_markers(self) -> None:
        """Should convert specific markers to array."""
        traj_1 = Trajectory3D(marker_name="m1", positions=np.ones((10, 3)))
        traj_2 = Trajectory3D(marker_name="m2", positions=np.ones((10, 3)) * 2)
        traj_3 = Trajectory3D(marker_name="m3", positions=np.ones((10, 3)) * 3)
        
        dataset = TrajectoryDataset(
            data={"m1": traj_1, "m2": traj_2, "m3": traj_3},
            frame_indices=np.arange(10)
        )
        
        # Get only m1 and m3
        array = dataset.to_array(marker_names=["m1", "m3"])
        
        assert array.shape == (10, 2, 3)
        assert np.allclose(array[:, 0, :], 1.0)
        assert np.allclose(array[:, 1, :], 3.0)
    
    def test_filter_by_confidence(self) -> None:
        """Should filter frames by confidence."""
        # Create trajectory with varying confidence
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
        filtered = dataset.filter_by_confidence(
            min_confidence=0.5,
            min_valid_markers=1
        )
        
        # Should keep frames with confidence >= 0.5
        # That's frames: 0, 2, 4, 5, 7, 9 = 6 frames
        assert filtered.n_frames == 6
    
    def test_get_summary(self) -> None:
        """Should generate summary."""
        traj = Trajectory3D(
            marker_name="m1",
            positions=np.random.randn(10, 3)
        )
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(10)
        )
        
        summary = dataset.get_summary()
        
        assert summary["n_frames"] == 10
        assert summary["n_markers"] == 1
        assert "m1" in summary["marker_names"]


class TestDatasetValidation:
    """Test dataset validation."""
    
    def test_mismatched_lengths_raises(self) -> None:
        """Should raise if trajectories have different lengths."""
        traj_1 = Trajectory3D(marker_name="m1", positions=np.ones((10, 3)))
        traj_2 = Trajectory3D(marker_name="m2", positions=np.ones((20, 3)))
        
        with pytest.raises(ValueError, match="same length"):
            dataset = TrajectoryDataset(
                data={"m1": traj_1, "m2": traj_2},
                frame_indices=np.arange(10)
            )
    
    def test_empty_dataset_raises(self) -> None:
        """Should raise if dataset is empty."""
        with pytest.raises(ValueError, match="at least one"):
            dataset = TrajectoryDataset(
                data={},
                frame_indices=np.arange(10)
            )
