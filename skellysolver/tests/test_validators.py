"""Tests for data validators (Phase 2).

Tests validation functions.
"""

import numpy as np
import pytest

from skellysolver.data.base_data import Trajectory3D, TrajectoryDataset
from skellysolver.data.validators import (
    validate_dataset,
    check_required_markers,
    check_temporal_gaps,
    check_spatial_outliers,
    check_data_quality,
)


class TestValidateDataset:
    """Test dataset validation."""
    
    def test_valid_dataset(self) -> None:
        """Should validate good dataset."""
        traj = Trajectory3D(
            marker_name="m1",
            positions=np.random.randn(100, 3),
            confidence=np.ones(100) * 0.9
        )
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(100)
        )
        
        report = validate_dataset(
            dataset=dataset,
            min_valid_frames=10,
            min_confidence=0.3
        )
        
        assert report["valid"] is True
        assert len(report["errors"]) == 0
    
    def test_insufficient_frames(self) -> None:
        """Should fail if too few frames."""
        traj = Trajectory3D(
            marker_name="m1",
            positions=np.random.randn(5, 3)
        )
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(5)
        )
        
        report = validate_dataset(
            dataset=dataset,
            min_valid_frames=10
        )
        
        assert report["valid"] is False
        assert any("Insufficient frames" in e for e in report["errors"])
    
    def test_missing_required_markers(self) -> None:
        """Should fail if required markers missing."""
        traj = Trajectory3D(
            marker_name="m1",
            positions=np.random.randn(100, 3)
        )
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(100)
        )
        
        report = validate_dataset(
            dataset=dataset,
            required_markers=["m1", "m2", "m3"]
        )
        
        assert report["valid"] is False
        assert any("Missing required markers" in e for e in report["errors"])


class TestCheckRequiredMarkers:
    """Test required markers check."""
    
    def test_all_markers_present(self) -> None:
        """Should pass when all markers present."""
        traj_1 = Trajectory3D(marker_name="m1", positions=np.ones((10, 3)))
        traj_2 = Trajectory3D(marker_name="m2", positions=np.ones((10, 3)))
        
        dataset = TrajectoryDataset(
            data={"m1": traj_1, "m2": traj_2},
            frame_indices=np.arange(10)
        )
        
        result = check_required_markers(
            dataset=dataset,
            required_markers=["m1", "m2"]
        )
        
        assert result["has_all_required"] is True
        assert len(result["missing_markers"]) == 0
    
    def test_markers_missing(self) -> None:
        """Should detect missing markers."""
        traj = Trajectory3D(marker_name="m1", positions=np.ones((10, 3)))
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(10)
        )
        
        result = check_required_markers(
            dataset=dataset,
            required_markers=["m1", "m2", "m3"]
        )
        
        assert result["has_all_required"] is False
        assert "m2" in result["missing_markers"]
        assert "m3" in result["missing_markers"]
        assert result["n_missing"] == 2


class TestCheckTemporalGaps:
    """Test temporal gap detection."""
    
    def test_no_gaps(self) -> None:
        """Should detect no gaps when data is complete."""
        positions = np.ones((10, 3))
        traj = Trajectory3D(marker_name="m1", positions=positions)
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(10)
        )
        
        result = check_temporal_gaps(dataset=dataset, min_confidence=0.3)
        
        assert result["has_gaps"] is False
        assert result["n_gaps"] == 0
    
    def test_detect_gaps(self) -> None:
        """Should detect temporal gaps."""
        positions = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [4.0, 5.0, 6.0],
        ])
        
        traj = Trajectory3D(marker_name="m1", positions=positions)
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(4)
        )
        
        result = check_temporal_gaps(
            dataset=dataset,
            min_confidence=0.3,
            max_gap_size=1
        )
        
        assert result["has_gaps"] is True
        assert result["n_gaps"] >= 1


class TestCheckSpatialOutliers:
    """Test spatial outlier detection."""
    
    def test_no_outliers(self) -> None:
        """Should detect no outliers in clean data."""
        # Smooth trajectory
        t = np.linspace(0, 2 * np.pi, 100)
        positions = np.column_stack([np.cos(t), np.sin(t), np.zeros(100)])
        
        traj = Trajectory3D(marker_name="m1", positions=positions)
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(100)
        )
        
        result = check_spatial_outliers(dataset=dataset, threshold=5.0)
        
        assert result["has_outliers"] is False
    
    def test_detect_outliers(self) -> None:
        """Should detect outliers in data with jumps."""
        positions = np.ones((10, 3))
        
        # Add outlier at frame 5
        positions[5] = np.array([100.0, 100.0, 100.0])
        
        traj = Trajectory3D(marker_name="m1", positions=positions)
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(10)
        )
        
        result = check_spatial_outliers(dataset=dataset, threshold=3.0)
        
        assert result["has_outliers"] is True
        assert result["n_outliers"] >= 1


class TestCheckDataQuality:
    """Test data quality check."""
    
    def test_quality_metrics(self) -> None:
        """Should compute quality metrics."""
        positions = np.ones((100, 3))
        confidence = np.random.uniform(low=0.0, high=1.0, size=100)
        
        traj = Trajectory3D(
            marker_name="m1",
            positions=positions,
            confidence=confidence
        )
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(100)
        )
        
        quality = check_data_quality(dataset=dataset, min_confidence=0.5)
        
        assert "n_frames" in quality
        assert "n_markers" in quality
        assert "marker_validity" in quality
        assert "m1" in quality["marker_validity"]
    
    def test_confidence_statistics(self) -> None:
        """Should compute confidence statistics."""
        positions = np.ones((100, 3))
        confidence = np.ones(100) * 0.8
        
        traj = Trajectory3D(
            marker_name="m1",
            positions=positions,
            confidence=confidence
        )
        
        dataset = TrajectoryDataset(
            data={"m1": traj},
            frame_indices=np.arange(100)
        )
        
        quality = check_data_quality(dataset=dataset)
        
        assert "confidence_stats" in quality
        assert quality["confidence_stats"]["mean"] == 0.8
