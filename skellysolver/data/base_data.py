"""Base data structures for all SkellySolver pipelines.

This module provides fundamental data structures used across all pipelines:
- Trajectory3D: 3D marker trajectories (rigid body tracking)
- Observation2D: 2D image observations (eye tracking, camera calibration)
- TrajectoryDataset: Collection of trajectories with metadata

These replace scattered data handling across multiple files.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from pydantic import model_validator
from typing_extensions import Self

from skellysolver.data.arbitrary_types_model import ArbitraryTypesModel


class Trajectory3D(ArbitraryTypesModel):
    """3D trajectory data for a single marker.
    
    Used by rigid body tracking to store marker positions over time.
    
    Attributes:
        marker_name: Identifier for this marker
        positions: (n_frames, 3) XYZ positions in meters
        confidence: Optional (n_frames,) confidence scores [0-1]
        metadata: Optional additional data
    """
    
    marker_name: str
    positions: np.ndarray
    confidence: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @model_validator(mode='after')
    def validate(self) -> Self:
        """Validate trajectory data."""
        if self.positions.ndim != 2:
            raise ValueError(f"Positions must be 2D array, got shape {self.positions.shape}")
        
        if self.positions.shape[1] != 3:
            raise ValueError(f"Positions must have 3 columns (XYZ), got {self.positions.shape[1]}")
        
        if self.confidence is not None:
            if len(self.confidence) != len(self.positions):
                raise ValueError(
                    f"Confidence length {len(self.confidence)} != positions length {len(self.positions)}"
                )
        return self

    @property
    def n_frames(self) -> int:
        """Number of frames in trajectory."""
        return len(self.positions)
    
    def is_valid(self, *, min_confidence: float = 0.3) -> np.ndarray:
        """Get mask of valid frames.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            (n_frames,) boolean mask
        """
        if self.confidence is None:
            # If no confidence, check for NaN
            return ~np.isnan(self.positions[:, 0])
        
        # Valid if confidence above threshold AND not NaN
        above_threshold = self.confidence >= min_confidence
        not_nan = ~np.isnan(self.positions[:, 0])
        return above_threshold & not_nan
    
    def get_valid_positions(self, *, min_confidence: float = 0.3) -> np.ndarray:
        """Get positions for valid frames only.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            (n_valid, 3) valid positions
        """
        mask = self.is_valid(min_confidence=min_confidence)
        return self.positions[mask]
    
    def get_centroid(self, *, min_confidence: float = 0.3) -> np.ndarray:
        """Compute centroid of valid positions.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            (3,) centroid position
        """
        valid_pos = self.get_valid_positions(min_confidence=min_confidence)
        if len(valid_pos) == 0:
            return np.zeros(3)
        return np.mean(valid_pos, axis=0)
    
    def interpolate_missing(self, *, method: str = "linear") -> "Trajectory3D":
        """Interpolate missing (NaN) positions.
        
        Args:
            method: Interpolation method ("linear", "cubic")
            
        Returns:
            New Trajectory3D with interpolated positions
        """
        from scipy.interpolate import interp1d
        
        positions_interp = self.positions.copy()
        
        for axis in range(3):
            data = self.positions[:, axis]
            valid_mask = ~np.isnan(data)
            
            if np.sum(valid_mask) < 2:
                # Not enough points to interpolate
                continue
            
            valid_indices = np.where(valid_mask)[0]
            valid_values = data[valid_mask]
            
            # Interpolate
            interp_func = interp1d(
                valid_indices,
                valid_values,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            # Fill missing values
            missing_mask = np.isnan(data)
            if np.any(missing_mask):
                missing_indices = np.where(missing_mask)[0]
                positions_interp[missing_indices, axis] = interp_func(missing_indices)
        
        return Trajectory3D(
            marker_name=self.marker_name,
            positions=positions_interp,
            confidence=self.confidence,
            metadata=self.metadata
        )



class Observation2D(ArbitraryTypesModel):
    """2D observation data for a single point.
    
    Used by eye tracking and camera calibration to store image points.
    
    Attributes:
        point_name: Identifier for this point
        positions: (n_frames, 2) UV pixel coordinates
        confidence: Optional (n_frames,) confidence scores [0-1]
        metadata: Optional additional data
    """
    
    point_name: str
    positions: np.ndarray
    confidence: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @model_validator(mode='after')
    def validate(self) -> Self:
        """Validate observation data."""
        if self.positions.ndim != 2:
            raise ValueError(f"Positions must be 2D array, got shape {self.positions.shape}")
        
        if self.positions.shape[1] != 2:
            raise ValueError(f"Positions must have 2 columns (UV), got {self.positions.shape[1]}")
        
        if self.confidence is not None:
            if len(self.confidence) != len(self.positions):
                raise ValueError(
                    f"Confidence length {len(self.confidence)} != positions length {len(self.positions)}"
                )
        return self
    
    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.positions)
    
    def is_valid(self, *, min_confidence: float = 0.3) -> np.ndarray:
        """Get mask of valid frames.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            (n_frames,) boolean mask
        """
        if self.confidence is None:
            return ~np.isnan(self.positions[:, 0])
        
        above_threshold = self.confidence >= min_confidence
        not_nan = ~np.isnan(self.positions[:, 0])
        return above_threshold & not_nan
    
    def get_valid_positions(self, *, min_confidence: float = 0.3) -> np.ndarray:
        """Get positions for valid frames only.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            (n_valid, 2) valid positions
        """
        mask = self.is_valid(min_confidence=min_confidence)
        return self.positions[mask]
    
    def interpolate_missing(self, *, method: str = "linear") -> "Observation2D":
        """Interpolate missing (NaN) positions.
        
        Args:
            method: Interpolation method ("linear", "cubic")
            
        Returns:
            New Observation2D with interpolated positions
        """
        from scipy.interpolate import interp1d
        
        positions_interp = self.positions.copy()
        
        for axis in range(2):
            data = self.positions[:, axis]
            valid_mask = ~np.isnan(data)
            
            if np.sum(valid_mask) < 2:
                continue
            
            valid_indices = np.where(valid_mask)[0]
            valid_values = data[valid_mask]
            
            interp_func = interp1d(
                valid_indices,
                valid_values,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            missing_mask = np.isnan(data)
            if np.any(missing_mask):
                missing_indices = np.where(missing_mask)[0]
                positions_interp[missing_indices, axis] = interp_func(missing_indices)
        
        return Observation2D(
            point_name=self.point_name,
            positions=positions_interp,
            confidence=self.confidence,
            metadata=self.metadata
        )



class TrajectoryDataset(ArbitraryTypesModel):
    """Collection of trajectories or observations with metadata.
    
    Can contain either 3D trajectories or 2D observations.
    Used by all pipelines to manage multiple markers/points.
    
    Attributes:
        data: Dictionary mapping names to Trajectory3D or Observation2D
        frame_indices: Frame numbers (may not start at 0)
        metadata: Optional dataset-level metadata
    """
    
    data: dict[str, Trajectory3D | Observation2D]
    frame_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate(self) -> Self:
        """Validate dataset."""
        if len(self.data) == 0:
            raise ValueError("Dataset must contain at least one trajectory")
        
        # Check all trajectories have same length
        n_frames_list = [traj.n_frames for traj in self.data.values()]
        if len(set(n_frames_list)) > 1:
            raise ValueError(f"All trajectories must have same length, got {n_frames_list}")
        
        if len(self.frame_indices) != n_frames_list[0]:
            raise ValueError(
                f"Frame indices length {len(self.frame_indices)} != trajectory length {n_frames_list[0]}"
            )

        return  self
    
    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.frame_indices)
    
    @property
    def marker_names(self) -> list[str]:
        """Get list of marker/point names."""
        return list(self.data.keys())
    
    @property
    def n_markers(self) -> int:
        """Number of markers/points."""
        return len(self.data)
    
    @property
    def is_3d(self) -> bool:
        """Check if dataset contains 3D trajectories."""
        first_item = next(iter(self.data.values()))
        return isinstance(first_item, Trajectory3D)
    
    @property
    def is_2d(self) -> bool:
        """Check if dataset contains 2D observations."""
        first_item = next(iter(self.data.values()))
        return isinstance(first_item, Observation2D)
    
    def to_array(self, *, marker_names: list[str] | None = None) -> np.ndarray:
        """Convert to numpy array.
        
        Args:
            marker_names: Optional list of markers to include (default: all)
            
        Returns:
            (n_frames, n_markers, n_dims) array where n_dims is 3 for 3D, 2 for 2D
        """
        if marker_names is None:
            marker_names = self.marker_names
        
        # Check all requested markers exist
        missing = set(marker_names) - set(self.marker_names)
        if missing:
            raise ValueError(f"Markers not in dataset: {missing}")
        
        # Stack positions
        arrays = [self.data[name].positions for name in marker_names]
        return np.stack(arrays, axis=1)
    
    def get_valid_frames(
        self,
        *,
        min_confidence: float = 0.3,
        min_valid_markers: int | None = None
    ) -> np.ndarray:
        """Get mask of frames with sufficient valid markers.
        
        Args:
            min_confidence: Minimum confidence threshold
            min_valid_markers: Minimum number of valid markers required (default: all)
            
        Returns:
            (n_frames,) boolean mask
        """
        if min_valid_markers is None:
            min_valid_markers = self.n_markers
        
        # Count valid markers per frame
        valid_counts = np.zeros(self.n_frames, dtype=int)
        for traj in self.data.values():
            valid_counts += traj.is_valid(min_confidence=min_confidence).astype(int)
        
        return valid_counts >= min_valid_markers
    
    def filter_by_confidence(
        self,
        *,
        min_confidence: float = 0.3,
        min_valid_markers: int | None = None
    ) -> "TrajectoryDataset":
        """Filter dataset to keep only high-confidence frames.
        
        Args:
            min_confidence: Minimum confidence threshold
            min_valid_markers: Minimum valid markers per frame
            
        Returns:
            New filtered dataset
        """
        valid_mask = self.get_valid_frames(
            min_confidence=min_confidence,
            min_valid_markers=min_valid_markers
        )
        
        if not np.any(valid_mask):
            raise ValueError("No valid frames after filtering")
        
        # Filter each trajectory
        filtered_data = {}
        for name, traj in self.data.items():
            filtered_positions = traj.positions[valid_mask]
            filtered_confidence = traj.confidence[valid_mask] if traj.confidence is not None else None
            
            if isinstance(traj, Trajectory3D):
                filtered_data[name] = Trajectory3D(
                    marker_name=name,
                    positions=filtered_positions,
                    confidence=filtered_confidence,
                    metadata=traj.metadata
                )
            else:
                filtered_data[name] = Observation2D(
                    point_name=name,
                    positions=filtered_positions,
                    confidence=filtered_confidence,
                    metadata=traj.metadata
                )
        
        return TrajectoryDataset(
            data=filtered_data,
            frame_indices=self.frame_indices[valid_mask],
            metadata=self.metadata
        )
    
    def interpolate_missing(self, *, method: str = "linear") -> "TrajectoryDataset":
        """Interpolate missing data in all trajectories.
        
        Args:
            method: Interpolation method
            
        Returns:
            New dataset with interpolated trajectories
        """
        interpolated_data = {
            name: traj.interpolate_missing(method=method)
            for name, traj in self.data.items()
        }
        
        return TrajectoryDataset(
            data=interpolated_data,
            frame_indices=self.frame_indices,
            metadata=self.metadata
        )
    
    def get_summary(self) -> dict[str, Any]:
        """Get dataset summary statistics.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            "n_frames": self.n_frames,
            "n_markers": self.n_markers,
            "marker_names": self.marker_names,
            "data_type": "3D" if self.is_3d else "2D",
        }
        
        # Compute validity statistics
        if self.data:
            first_traj = next(iter(self.data.values()))
            if first_traj.confidence is not None:
                all_valid = self.get_valid_frames(min_confidence=0.3, min_valid_markers=self.n_markers)
                summary["n_fully_valid_frames"] = int(np.sum(all_valid))
                summary["percent_fully_valid"] = float(np.mean(all_valid) * 100)
        
        return summary
