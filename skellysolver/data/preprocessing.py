"""Data preprocessing for SkellySolver.

Common preprocessing operations:
- Interpolation of missing data
- Filtering by confidence
- Smoothing
- Outlier removal
- Centering/scaling
"""

from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from .data_models import TrajectoryDataset, Trajectory3D, Observation2D


def interpolate_missing(
    *,
    dataset: TrajectoryDataset,
    method: str = "linear",
    max_gap: int | None = None
) -> TrajectoryDataset:
    """Interpolate missing (NaN) values in dataset.
    
    Args:
        dataset: Dataset with missing values
        method: Interpolation method ("linear", "cubic", "nearest")
        max_gap: Maximum gap size to interpolate (None = no limit)
        
    Returns:
        Dataset with interpolated values
    """
    print(f"Interpolating missing data (method={method}, max_gap={max_gap})...")
    
    interpolated_data = {}
    for marker_name, traj in dataset.data.items():
        # Interpolate this trajectory
        if max_gap is not None:
            # Only interpolate small gaps
            interp_traj = _interpolate_with_max_gap(
                trajectory=traj,
                method=method,
                max_gap=max_gap
            )
        else:
            # Interpolate all gaps
            interp_traj = traj.interpolate_missing(method=method)
        
        interpolated_data[marker_name] = interp_traj
    
    print("  ✓ Interpolation complete")
    
    return TrajectoryDataset(
        data=interpolated_data,
        frame_indices=dataset.frame_indices,
        metadata=dataset.metadata
    )


def _interpolate_with_max_gap(
    *,
    trajectory: Trajectory3D | Observation2D,
    method: str,
    max_gap: int
) -> Trajectory3D | Observation2D:
    """Interpolate trajectory with maximum gap size limit.
    
    Args:
        trajectory: Trajectory to interpolate
        method: Interpolation method
        max_gap: Maximum gap size to fill
        
    Returns:
        Interpolated trajectory
    """
    positions = trajectory.positions.copy()
    n_dims = positions.shape[1]
    
    for axis in range(n_dims):
        data = positions[:, axis]
        
        # Find gaps
        is_valid = ~np.isnan(data)
        
        # Identify gap regions
        gap_start = None
        for i in range(len(data)):
            if np.isnan(data[i]) and gap_start is None:
                gap_start = i
            elif not np.isnan(data[i]) and gap_start is not None:
                gap_size = i - gap_start
                
                # Only interpolate if gap is small enough
                if gap_size <= max_gap:
                    # Interpolate this gap
                    valid_indices = np.where(is_valid)[0]
                    valid_values = data[is_valid]
                    
                    if len(valid_indices) >= 2:
                        interp_func = interp1d(
                            valid_indices,
                            valid_values,
                            kind=method,
                            bounds_error=False,
                            fill_value="extrapolate"
                        )
                        
                        gap_indices = np.arange(gap_start, i)
                        positions[gap_indices, axis] = interp_func(gap_indices)
                
                gap_start = None
    
    if isinstance(trajectory, Trajectory3D):
        return Trajectory3D(
            marker_name=trajectory.marker_name,
            positions=positions,
            confidence=trajectory.confidence,
            metadata=trajectory.metadata
        )
    else:
        return Observation2D(
            point_name=trajectory.point_name,
            positions=positions,
            confidence=trajectory.confidence,
            metadata=trajectory.metadata
        )


def filter_by_confidence(
    *,
    dataset: TrajectoryDataset,
    min_confidence: float = 0.3,
    min_valid_markers: int | None = None
) -> TrajectoryDataset:
    """Filter dataset to keep only high-confidence frames.
    
    Args:
        dataset: Dataset to filter
        min_confidence: Minimum confidence threshold
        min_valid_markers: Minimum valid markers per frame (default: all)
        
    Returns:
        Filtered dataset
    """
    print(f"Filtering by confidence (min={min_confidence}, min_markers={min_valid_markers})...")
    
    filtered = dataset.filter_by_confidence(
        min_confidence=min_confidence,
        min_valid_markers=min_valid_markers
    )
    
    n_before = dataset.n_frames
    n_after = filtered.n_frames
    n_removed = n_before - n_after
    
    print(f"  ✓ Filtered: {n_before} → {n_after} frames ({n_removed} removed)")
    
    return filtered


def smooth_trajectories(
    *,
    dataset: TrajectoryDataset,
    window_size: int = 5,
    poly_order: int = 2
) -> TrajectoryDataset:
    """Smooth trajectories using Savitzky-Golay filter.
    
    Args:
        dataset: Dataset to smooth
        window_size: Window size for filter (must be odd)
        poly_order: Polynomial order for filter
        
    Returns:
        Smoothed dataset
    """
    print(f"Smoothing trajectories (window={window_size}, order={poly_order})...")
    
    if window_size % 2 == 0:
        window_size += 1
    
    if window_size > dataset.n_frames:
        window_size = dataset.n_frames if dataset.n_frames % 2 == 1 else dataset.n_frames - 1
    
    smoothed_data = {}
    for marker_name, traj in dataset.data.items():
        positions = traj.positions.copy()
        n_dims = positions.shape[1]
        
        # Apply filter to each dimension
        for axis in range(n_dims):
            data = positions[:, axis]
            
            # Only smooth non-NaN regions
            valid_mask = ~np.isnan(data)
            if np.sum(valid_mask) >= window_size:
                # Smooth valid data
                valid_data = data[valid_mask]
                smoothed_valid = savgol_filter(
                    valid_data,
                    window_length=window_size,
                    polyorder=poly_order
                )
                
                # Put back
                data[valid_mask] = smoothed_valid
                positions[:, axis] = data
        
        # Create smoothed trajectory
        if isinstance(traj, Trajectory3D):
            smoothed_data[marker_name] = Trajectory3D(
                marker_name=marker_name,
                positions=positions,
                confidence=traj.confidence,
                metadata=traj.metadata
            )
        else:
            smoothed_data[marker_name] = Observation2D(
                point_name=marker_name,
                positions=positions,
                confidence=traj.confidence,
                metadata=traj.metadata
            )
    
    print("  ✓ Smoothing complete")
    
    return TrajectoryDataset(
        data=smoothed_data,
        frame_indices=dataset.frame_indices,
        metadata=dataset.metadata
    )


def remove_outliers(
        *,
        dataset: TrajectoryDataset,
        threshold: float = 5.0,
        method: str = "velocity"
) -> TrajectoryDataset:
    """Remove spatial outliers from dataset.

    Args:
        dataset: Dataset to clean
        threshold: Number of median absolute deviations for outlier detection
        method: Detection method ("velocity" or "position")

    Returns:
        Dataset with outliers removed (set to NaN)
    """
    if not dataset.is_3d:
        raise ValueError("Outlier removal only supported for 3D data")

    print(f"Removing outliers (method={method}, threshold={threshold})...")

    cleaned_data = {}
    n_total_outliers = 0

    for marker_name, traj in dataset.data.items():
        if not isinstance(traj, Trajectory3D):
            cleaned_data[marker_name] = traj
            continue

        positions = traj.positions.copy()
        outlier_mask = np.zeros(len(positions), dtype=bool)

        if method == "velocity":
            # Velocity-based outlier detection using MAD (Median Absolute Deviation)
            velocities = np.diff(positions, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)

            valid_speeds = speeds[~np.isnan(speeds)]
            if len(valid_speeds) < 2:
                cleaned_data[marker_name] = traj
                continue

            # Use median and MAD for robust outlier detection
            median_speed = np.median(valid_speeds)
            mad = np.median(np.abs(valid_speeds - median_speed))

            # Avoid division by zero and handle tight distributions
            if mad < 1e-10:
                # If MAD is zero, distribution is very tight - use std as fallback
                # with a more sensitive threshold
                std_speed = np.std(valid_speeds)
                if std_speed > 0:
                    modified_z_scores = np.abs((speeds - median_speed) / std_speed)
                    # Use lower threshold for tight distributions
                    effective_threshold = min(threshold / 2.0, 2.0)
                else:
                    cleaned_data[marker_name] = traj
                    continue
            else:
                # Modified z-score using MAD
                # Factor 1.4826 makes MAD consistent with std for normal distribution
                modified_z_scores = np.abs((speeds - median_speed) / (1.4826 * mad))
                effective_threshold = threshold

            # Find outlier velocities
            outlier_velocities = modified_z_scores > effective_threshold

            # Mark frames involved in outlier velocities
            # If velocity i (from frame i to i+1) is an outlier,
            # mark both frames i and i+1 as potential outliers
            for i in np.where(outlier_velocities)[0]:
                outlier_mask[i] = True
                outlier_mask[i + 1] = True

        elif method == "position":
            # Position-based outlier detection using MAD
            centroid = traj.get_centroid()
            distances = np.linalg.norm(positions - centroid, axis=1)

            valid_distances = distances[~np.isnan(distances)]
            if len(valid_distances) < 2:
                cleaned_data[marker_name] = traj
                continue

            median_dist = np.median(valid_distances)
            mad = np.median(np.abs(valid_distances - median_dist))

            if mad < 1e-10:
                # Tight distribution - use std with adjusted threshold
                std_dist = np.std(valid_distances)
                if std_dist > 0:
                    modified_z_scores = np.abs((distances - median_dist) / std_dist)
                    effective_threshold = min(threshold / 2.0, 2.0)
                else:
                    cleaned_data[marker_name] = traj
                    continue
            else:
                modified_z_scores = np.abs((distances - median_dist) / (1.4826 * mad))
                effective_threshold = threshold

            outlier_mask = modified_z_scores > effective_threshold

        # Set outlier positions to NaN
        positions[outlier_mask] = np.nan
        n_total_outliers += np.sum(outlier_mask)

        cleaned_data[marker_name] = Trajectory3D(
            marker_name=marker_name,
            positions=positions,
            confidence=traj.confidence,
            metadata=traj.metadata
        )

    print(f"  ✓ Removed {n_total_outliers} outlier points")

    return TrajectoryDataset(
        data=cleaned_data,
        frame_indices=dataset.frame_indices,
        metadata=dataset.metadata
    )
def center_data(
    *,
    dataset: TrajectoryDataset,
    method: str = "mean"
) -> tuple[TrajectoryDataset, np.ndarray]:
    """Center data by subtracting centroid.
    
    Args:
        dataset: Dataset to center
        method: Centering method ("mean" or "median")
        
    Returns:
        Tuple of (centered_dataset, centroid)
    """
    if not dataset.is_3d:
        raise ValueError("Centering only supported for 3D data")
    
    print(f"Centering data (method={method})...")
    
    # Compute centroid across all markers and frames
    all_positions = []
    for traj in dataset.data.values():
        if isinstance(traj, Trajectory3D):
            valid_pos = traj.get_valid_positions()
            if len(valid_pos) > 0:
                all_positions.append(valid_pos)
    
    if not all_positions:
        raise ValueError("No valid positions to compute centroid")
    
    all_positions_array = np.concatenate(all_positions, axis=0)
    
    if method == "mean":
        centroid = np.mean(all_positions_array, axis=0)
    elif method == "median":
        centroid = np.median(all_positions_array, axis=0)
    else:
        raise ValueError(f"Unknown centering method: {method}")
    
    # Center each trajectory
    centered_data = {}
    for marker_name, traj in dataset.data.items():
        if not isinstance(traj, Trajectory3D):
            centered_data[marker_name] = traj
            continue
        
        centered_positions = traj.positions - centroid
        
        centered_data[marker_name] = Trajectory3D(
            marker_name=marker_name,
            positions=centered_positions,
            confidence=traj.confidence,
            metadata=traj.metadata
        )
    
    print(f"  ✓ Centered at {centroid}")
    
    return TrajectoryDataset(
        data=centered_data,
        frame_indices=dataset.frame_indices,
        metadata=dataset.metadata
    ), centroid


def scale_data(
    *,
    dataset: TrajectoryDataset,
    scale_factor: float
) -> TrajectoryDataset:
    """Scale all positions by constant factor.
    
    Args:
        dataset: Dataset to scale
        scale_factor: Scaling factor (e.g., 0.001 for mm to m)
        
    Returns:
        Scaled dataset
    """
    print(f"Scaling data by factor {scale_factor}...")
    
    scaled_data = {}
    for marker_name, traj in dataset.data.items():
        scaled_positions = traj.positions * scale_factor
        
        if isinstance(traj, Trajectory3D):
            scaled_data[marker_name] = Trajectory3D(
                marker_name=marker_name,
                positions=scaled_positions,
                confidence=traj.confidence,
                metadata=traj.metadata
            )
        else:
            scaled_data[marker_name] = Observation2D(
                point_name=marker_name,
                positions=scaled_positions,
                confidence=traj.confidence,
                metadata=traj.metadata
            )
    
    print("  ✓ Scaling complete")
    
    return TrajectoryDataset(
        data=scaled_data,
        frame_indices=dataset.frame_indices,
        metadata=dataset.metadata
    )


def subsample_frames(
    *,
    dataset: TrajectoryDataset,
    step: int = 2
) -> TrajectoryDataset:
    """Subsample frames to reduce data size.
    
    Args:
        dataset: Dataset to subsample
        step: Take every Nth frame
        
    Returns:
        Subsampled dataset
    """
    print(f"Subsampling frames (step={step})...")
    
    # Subsample indices
    subsampled_indices = dataset.frame_indices[::step]
    
    # Subsample each trajectory
    subsampled_data = {}
    for marker_name, traj in dataset.data.items():
        subsampled_positions = traj.positions[::step]
        subsampled_confidence = traj.confidence[::step] if traj.confidence is not None else None
        
        if isinstance(traj, Trajectory3D):
            subsampled_data[marker_name] = Trajectory3D(
                marker_name=marker_name,
                positions=subsampled_positions,
                confidence=subsampled_confidence,
                metadata=traj.metadata
            )
        else:
            subsampled_data[marker_name] = Observation2D(
                point_name=marker_name,
                positions=subsampled_positions,
                confidence=subsampled_confidence,
                metadata=traj.metadata
            )
    
    n_before = dataset.n_frames
    n_after = len(subsampled_indices)
    
    print(f"  ✓ Subsampled: {n_before} → {n_after} frames")
    
    return TrajectoryDataset(
        data=subsampled_data,
        frame_indices=subsampled_indices,
        metadata=dataset.metadata
    )


def apply_preprocessing_pipeline(
    *,
    dataset: TrajectoryDataset,
    steps: list[dict[str, Any]]
) -> TrajectoryDataset:
    """Apply sequence of preprocessing steps.
    
    Args:
        dataset: Input dataset
        steps: List of preprocessing steps, each a dict with:
            - "operation": function name
            - other keys: kwargs for that function
            
    Returns:
        Preprocessed dataset
        
    Example:
        steps = [
            {"operation": "filter_by_confidence", "min_confidence": 0.5},
            {"operation": "interpolate_missing", "method": "cubic"},
            {"operation": "smooth_trajectories", "window_size": 5},
        ]
    """
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE")
    print("="*80)
    
    current_dataset = dataset
    
    for i, step in enumerate(steps, 1):
        operation = step.pop("operation")
        print(f"\nStep {i}: {operation}")
        
        # Get function
        if operation == "filter_by_confidence":
            current_dataset = filter_by_confidence(dataset=current_dataset, **step)
        elif operation == "interpolate_missing":
            current_dataset = interpolate_missing(dataset=current_dataset, **step)
        elif operation == "smooth_trajectories":
            current_dataset = smooth_trajectories(dataset=current_dataset, **step)
        elif operation == "remove_outliers":
            current_dataset = remove_outliers(dataset=current_dataset, **step)
        elif operation == "center_data":
            current_dataset, _ = center_data(dataset=current_dataset, **step)
        elif operation == "scale_data":
            current_dataset = scale_data(dataset=current_dataset, **step)
        elif operation == "subsample_frames":
            current_dataset = subsample_frames(dataset=current_dataset, **step)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    return current_dataset
