"""Tests for data loaders (Phase 2).

Tests CSV loading for all formats.
"""

import numpy as np
import pytest
from pathlib import Path

from skellysolver.data.loaders import (
    load_trajectories,
    load_tidy_format,
    load_wide_format,
    load_dlc_format,
    load_from_dict,
    concatenate_datasets,
)
from skellysolver.io.formats import detect_csv_format
from skellysolver.data.base_data import TrajectoryDataset, Trajectory3D


class TestTidyFormatLoader:
    """Test tidy format CSV loading."""
    
    def test_load_tidy_csv(self, create_tidy_csv: Path) -> None:
        """Should load tidy format CSV."""
        dataset = load_tidy_format(filepath=create_tidy_csv, scale_factor=1.0)
        
        assert isinstance(dataset, TrajectoryDataset)
        assert dataset.n_markers == 3
        assert dataset.n_frames == 2
        assert "marker1" in dataset.marker_names
    
    def test_tidy_format_detection(self, create_tidy_csv: Path) -> None:
        """Should detect tidy format."""
        format_type = detect_csv_format(filepath=create_tidy_csv)
        assert format_type == "tidy"
    
    def test_load_tidy_auto(self, create_tidy_csv: Path) -> None:
        """Should auto-load tidy format."""
        dataset = load_trajectories(filepath=create_tidy_csv)
        
        assert dataset.n_markers == 3
        assert dataset.is_3d


class TestWideFormatLoader:
    """Test wide format CSV loading."""
    
    def test_load_wide_csv(self, create_wide_csv: Path) -> None:
        """Should load wide format CSV."""
        dataset = load_wide_format(
            filepath=create_wide_csv,
            scale_factor=1.0,
            z_value=0.0
        )
        
        assert isinstance(dataset, TrajectoryDataset)
        assert dataset.n_markers == 2
        assert dataset.n_frames == 3
        assert "marker1" in dataset.marker_names
    
    def test_wide_format_detection(self, create_wide_csv: Path) -> None:
        """Should detect wide format."""
        format_type = detect_csv_format(filepath=create_wide_csv)
        assert format_type == "wide"
    
    def test_load_wide_auto(self, create_wide_csv: Path) -> None:
        """Should auto-load wide format."""
        dataset = load_trajectories(filepath=create_wide_csv)
        
        assert dataset.n_markers == 2
        assert dataset.is_3d


class TestDLCFormatLoader:
    """Test DeepLabCut format CSV loading."""
    
    def test_load_dlc_csv(self, create_dlc_csv: Path) -> None:
        """Should load DLC format CSV."""
        dataset = load_dlc_format(
            filepath=create_dlc_csv,
            scale_factor=1.0,
            z_value=0.0,
            likelihood_threshold=None
        )
        
        assert isinstance(dataset, TrajectoryDataset)
        assert dataset.n_markers == 2
        assert dataset.n_frames == 3
        assert "bodypart1" in dataset.marker_names
    
    def test_dlc_format_detection(self, create_dlc_csv: Path) -> None:
        """Should detect DLC format."""
        format_type = detect_csv_format(filepath=create_dlc_csv)
        assert format_type == "dlc"
    
    def test_load_dlc_auto(self, create_dlc_csv: Path) -> None:
        """Should auto-load DLC format."""
        dataset = load_trajectories(filepath=create_dlc_csv)
        
        assert dataset.n_markers == 2
        assert dataset.is_2d  # DLC is typically 2D
    
    def test_dlc_likelihood_filtering(self, create_dlc_csv: Path) -> None:
        """Should filter by likelihood threshold."""
        # Load with high threshold
        dataset = load_dlc_format(
            filepath=create_dlc_csv,
            scale_factor=1.0,
            z_value=0.0,
            likelihood_threshold=0.94
        )
        
        # Some points should be filtered (set to NaN)
        # bodypart1 has likelihoods: 0.95, 0.93, 0.96
        # With threshold=0.94, frame 1 should be NaN
        
        bodypart1 = dataset.data["bodypart1"]
        assert not np.isnan(bodypart1.positions[0, 0])  # 0.95 >= 0.94
        assert np.isnan(bodypart1.positions[1, 0])      # 0.93 < 0.94
        assert not np.isnan(bodypart1.positions[2, 0])  # 0.96 >= 0.94


class TestLoadFromDict:
    """Test loading from dictionary."""
    
    def test_load_from_dict_3d(self) -> None:
        """Should create dataset from dict of 3D arrays."""
        data_dict = {
            "m1": np.random.randn(10, 3),
            "m2": np.random.randn(10, 3),
        }
        
        dataset = load_from_dict(trajectory_dict=data_dict)
        
        assert dataset.n_frames == 10
        assert dataset.n_markers == 2
        assert dataset.is_3d
    
    def test_load_from_dict_2d(self) -> None:
        """Should create dataset from dict of 2D arrays."""
        data_dict = {
            "p1": np.random.randn(10, 2),
            "p2": np.random.randn(10, 2),
        }
        
        dataset = load_from_dict(trajectory_dict=data_dict)
        
        assert dataset.n_frames == 10
        assert dataset.n_markers == 2
        assert dataset.is_2d


class TestConcatenateDatasets:
    """Test dataset concatenation."""
    
    def test_concatenate_two_datasets(self) -> None:
        """Should concatenate two datasets."""
        # Create two datasets
        data_1 = {
            "m1": Trajectory3D(marker_name="m1", positions=np.ones((10, 3))),
            "m2": Trajectory3D(marker_name="m2", positions=np.ones((10, 3)) * 2),
        }
        
        data_2 = {
            "m1": Trajectory3D(marker_name="m1", positions=np.ones((5, 3)) * 3),
            "m2": Trajectory3D(marker_name="m2", positions=np.ones((5, 3)) * 4),
        }
        
        dataset_1 = TrajectoryDataset(data=data_1, frame_indices=np.arange(10))
        dataset_2 = TrajectoryDataset(data=data_2, frame_indices=np.arange(5))
        
        # Concatenate
        combined = concatenate_datasets(datasets=[dataset_1, dataset_2])
        
        assert combined.n_frames == 15
        assert combined.n_markers == 2
        
        # Check values
        array = combined.to_array()
        assert np.allclose(array[:10, 0, :], 1.0)
        assert np.allclose(array[10:, 0, :], 3.0)
    
    def test_concatenate_incompatible_raises(self) -> None:
        """Should raise if datasets have different markers."""
        data_1 = {
            "m1": Trajectory3D(marker_name="m1", positions=np.ones((10, 3))),
        }
        
        data_2 = {
            "m2": Trajectory3D(marker_name="m2", positions=np.ones((10, 3))),
        }
        
        dataset_1 = TrajectoryDataset(data=data_1, frame_indices=np.arange(10))
        dataset_2 = TrajectoryDataset(data=data_2, frame_indices=np.arange(10))
        
        with pytest.raises(ValueError, match="different markers"):
            combined = concatenate_datasets(datasets=[dataset_1, dataset_2])


class TestLoadTrajectories:
    """Test main load_trajectories function."""
    
    def test_load_auto_detect_tidy(self, create_tidy_csv: Path) -> None:
        """Should auto-detect and load tidy format."""
        dataset = load_trajectories(filepath=create_tidy_csv)
        
        assert dataset.n_markers == 3
        assert dataset.is_3d
    
    def test_load_auto_detect_wide(self, create_wide_csv: Path) -> None:
        """Should auto-detect and load wide format."""
        dataset = load_trajectories(filepath=create_wide_csv)
        
        assert dataset.n_markers == 2
        assert dataset.is_3d
    
    def test_load_auto_detect_dlc(self, create_dlc_csv: Path) -> None:
        """Should auto-detect and load DLC format."""
        dataset = load_trajectories(filepath=create_dlc_csv)
        
        assert dataset.n_markers == 2
        assert dataset.is_2d
    
    def test_load_with_scale_factor(self, create_tidy_csv: Path) -> None:
        """Should apply scale factor to positions."""
        # Load without scaling
        dataset_1 = load_trajectories(filepath=create_tidy_csv, scale_factor=1.0)
        
        # Load with scaling
        dataset_2 = load_trajectories(filepath=create_tidy_csv, scale_factor=0.001)
        
        # Positions should be scaled
        array_1 = dataset_1.to_array()
        array_2 = dataset_2.to_array()
        
        assert np.allclose(array_2, array_1 * 0.001)
    
    def test_load_nonexistent_file_raises(self, temp_dir: Path) -> None:
        """Should raise if file doesn't exist."""
        nonexistent = temp_dir / "doesnt_exist.csv"
        
        with pytest.raises(FileNotFoundError):
            dataset = load_trajectories(filepath=nonexistent)
