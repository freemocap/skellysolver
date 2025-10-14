"""Unified CSV loaders for all SkellySolver pipelines.

This module consolidates CSV loading from:
- io/loaders.py (multiple format loaders)
- io/load_trajectories.py (trajectory loading)
- io/eye_data_loader.py (eye tracking data)

Provides single interface: load_trajectories()
"""

import csv
import numpy as np
from pathlib import Path
from typing import Any

from skellysolver.data.data_models import Trajectory3D, Observation2D, TrajectoryDataset
from skellysolver.io.formats import detect_csv_format, CSVFormat


def load_trajectories(
    *,
    filepath: Path,
    csv_format: CSVFormat | None = None,
    scale_factor: float = 1.0,
    z_value: float = 0.0,
    likelihood_threshold: float | None = None
) -> TrajectoryDataset:
    """Load trajectory data from CSV file (auto-detect format).
    
    This is the main entry point for loading data in SkellySolver.
    Automatically detects and handles all supported formats.
    
    Args:
        filepath: Path to CSV file
        csv_format: Force specific format (None = auto-detect)
        scale_factor: Multiplier for coordinates (e.g., 0.001 for mm to m)
        z_value: Default z-coordinate for 2D data
        likelihood_threshold: Filter low-confidence points (for DLC format)
        
    Returns:
        TrajectoryDataset with loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect format if not specified
    if csv_format is None:
        csv_format = detect_csv_format(filepath=filepath)
    
    # Load based on format
    if csv_format == "tidy":
        return load_tidy_format(
            filepath=filepath,
            scale_factor=scale_factor
        )
    elif csv_format == "wide":
        return load_wide_format(
            filepath=filepath,
            scale_factor=scale_factor,
            z_value=z_value
        )
    elif csv_format == "dlc":
        return load_dlc_format(
            filepath=filepath,
            scale_factor=scale_factor,
            z_value=z_value,
            likelihood_threshold=likelihood_threshold
        )
    else:
        raise ValueError(f"Unknown format: {csv_format}")


def load_tidy_format(
    *,
    filepath: Path,
    scale_factor: float = 1.0
) -> TrajectoryDataset:
    """Load tidy/long-format CSV.
    
    Expected columns: frame, keypoint, x, y, z
    
    Args:
        filepath: Path to CSV file
        scale_factor: Multiplier for coordinates
        
    Returns:
        TrajectoryDataset with 3D trajectories
    """
    print(f"Loading tidy format: {filepath.name}")
    
    # Read all data into dictionary
    data_dict = {}
    frame_set = set()
    
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            frame = int(row['frame'])
            keypoint = row['keypoint']
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z']) if 'z' in row and row['z'] else 0.0
            
            frame_set.add(frame)
            
            if keypoint not in data_dict:
                data_dict[keypoint] = {}
            
            data_dict[keypoint][frame] = np.array([x, y, z])
    
    # Convert to arrays
    frame_indices = np.array(sorted(frame_set))
    n_frames = len(frame_indices)
    
    trajectories = {}
    for keypoint, frame_data in data_dict.items():
        positions = np.zeros((n_frames, 3))
        positions[:] = np.nan
        
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx in frame_data:
                positions[i] = frame_data[frame_idx] * scale_factor
        
        trajectories[keypoint] = Trajectory3D(
            marker_name=keypoint,
            positions=positions,
            confidence=None
        )
    
    print(f"  ✓ Loaded {len(trajectories)} markers × {n_frames} frames")
    
    return TrajectoryDataset(
        data=trajectories,
        frame_indices=frame_indices,
        metadata={"format": "tidy", "source": str(filepath)}
    )


def load_wide_format(
    *,
    filepath: Path,
    scale_factor: float = 1.0,
    z_value: float = 0.0
) -> TrajectoryDataset:
    """Load wide-format CSV.
    
    Expected columns: frame, {marker}_x, {marker}_y, {marker}_z (optional)
    
    Args:
        filepath: Path to CSV file
        scale_factor: Multiplier for coordinates
        z_value: Default z-coordinate if no _z column
        
    Returns:
        TrajectoryDataset with 3D trajectories
    """
    print(f"Loading wide format: {filepath.name}")
    
    # Read CSV
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        if headers is None:
            raise ValueError("CSV has no headers")
        
        # Find marker names
        marker_names = set()
        for header in headers:
            if header.endswith('_x'):
                marker_names.add(header[:-2])
        
        if not marker_names:
            raise ValueError("No markers found")
        
        # Read all rows
        rows = list(reader)
    
    n_frames = len(rows)
    frame_indices = np.arange(n_frames)
    
    # Extract data for each marker
    trajectories = {}
    for marker_name in marker_names:
        positions = np.zeros((n_frames, 3))
        
        x_col = f"{marker_name}_x"
        y_col = f"{marker_name}_y"
        z_col = f"{marker_name}_z"
        
        has_z = z_col in headers
        
        for i, row in enumerate(rows):
            try:
                x_str = row[x_col].strip()
                y_str = row[y_col].strip()
                
                x = float(x_str) if x_str else np.nan
                y = float(y_str) if y_str else np.nan
                
                if has_z:
                    z_str = row[z_col].strip()
                    z = float(z_str) if z_str else z_value
                else:
                    z = z_value
                
                positions[i] = np.array([x, y, z]) * scale_factor
                
            except (ValueError, KeyError):
                positions[i] = np.array([np.nan, np.nan, z_value])
        
        trajectories[marker_name] = Trajectory3D(
            marker_name=marker_name,
            positions=positions,
            confidence=None
        )
    
    print(f"  ✓ Loaded {len(trajectories)} markers × {n_frames} frames")
    
    return TrajectoryDataset(
        data=trajectories,
        frame_indices=frame_indices,
        metadata={"format": "wide", "source": str(filepath)}
    )


def load_dlc_format(
    *,
    filepath: Path,
    scale_factor: float = 1.0,
    z_value: float = 0.0,
    likelihood_threshold: float | None = None
) -> TrajectoryDataset:
    """Load DeepLabCut CSV with 3-row header.
    
    Expected format:
        Row 0: scorer names
        Row 1: bodypart names
        Row 2: coordinate types (x, y, likelihood)
        Row 3+: data
    
    Args:
        filepath: Path to CSV file
        scale_factor: Multiplier for coordinates
        z_value: Default z-coordinate (DLC is typically 2D)
        likelihood_threshold: Filter points below this confidence
        
    Returns:
        TrajectoryDataset with 2D or 3D data
    """
    print(f"Loading DLC format: {filepath.name}")
    
    # Read file
    with open(filepath, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 4:
        raise ValueError("DLC CSV must have at least 4 rows")
    
    # Parse 3-row header
    scorer_row = lines[0].strip().split(',')
    bodypart_row = lines[1].strip().split(',')
    coords_row = lines[2].strip().split(',')
    
    # Build column mapping
    column_map = {}
    for col_idx, (bodypart, coord_type) in enumerate(zip(bodypart_row, coords_row)):
        bodypart = bodypart.strip()
        coord_type = coord_type.strip()
        
        if not bodypart or bodypart.lower() == 'scorer':
            continue
        
        if bodypart not in column_map:
            column_map[bodypart] = {}
        
        column_map[bodypart][coord_type] = col_idx
    
    # Filter to valid bodyparts
    valid_bodyparts = [
        bp for bp, coords in column_map.items()
        if 'x' in coords and 'y' in coords
    ]
    
    if not valid_bodyparts:
        raise ValueError("No valid bodyparts with x and y coordinates")
    
    # Parse data rows
    n_frames = len(lines) - 3
    frame_indices = np.arange(n_frames)
    
    # Check if 3D data
    has_z = any('z' in column_map[bp] for bp in valid_bodyparts)
    
    # Extract data for each bodypart
    trajectories = {}
    for bodypart in valid_bodyparts:
        coords = column_map[bodypart]
        
        positions = np.zeros((n_frames, 3 if has_z else 2))
        confidence = np.zeros(n_frames) if 'likelihood' in coords else None
        
        for i, line in enumerate(lines[3:]):
            values = line.strip().split(',')
            
            try:
                x_str = values[coords['x']].strip()
                y_str = values[coords['y']].strip()
                
                x = float(x_str) if x_str else np.nan
                y = float(y_str) if y_str else np.nan
                
                # Check likelihood
                if 'likelihood' in coords:
                    likelihood_str = values[coords['likelihood']].strip()
                    likelihood = float(likelihood_str) if likelihood_str else 0.0
                    confidence[i] = likelihood
                    
                    # Filter by threshold
                    if likelihood_threshold is not None and likelihood < likelihood_threshold:
                        x = np.nan
                        y = np.nan
                
                # Get z if available
                if has_z and 'z' in coords:
                    z_str = values[coords['z']].strip()
                    z = float(z_str) if z_str else z_value
                else:
                    z = z_value
                
                if has_z:
                    positions[i] = np.array([x, y, z]) * scale_factor
                else:
                    positions[i] = np.array([x, y]) * scale_factor
                
            except (ValueError, IndexError):
                if has_z:
                    positions[i] = np.array([np.nan, np.nan, z_value])
                else:
                    positions[i] = np.array([np.nan, np.nan])
        
        # Create appropriate data structure
        if has_z:
            trajectories[bodypart] = Trajectory3D(
                marker_name=bodypart,
                positions=positions,
                confidence=confidence
            )
        else:
            trajectories[bodypart] = Observation2D(
                point_name=bodypart,
                positions=positions,
                confidence=confidence
            )
    
    print(f"  ✓ Loaded {len(trajectories)} bodyparts × {n_frames} frames")
    
    return TrajectoryDataset(
        data=trajectories,
        frame_indices=frame_indices,
        metadata={"format": "dlc", "source": str(filepath)}
    )


def load_from_dict(
    *,
    trajectory_dict: dict[str, np.ndarray],
    frame_indices: np.ndarray | None = None
) -> TrajectoryDataset:
    """Create TrajectoryDataset from dictionary of arrays.
    
    Utility function for when you already have data in memory.
    
    Args:
        trajectory_dict: Maps marker names to (n_frames, 3) or (n_frames, 2) arrays
        frame_indices: Optional frame indices (default: 0..n_frames-1)
        
    Returns:
        TrajectoryDataset
    """
    if not trajectory_dict:
        raise ValueError("Empty trajectory dictionary")
    
    # Determine n_frames
    first_array = next(iter(trajectory_dict.values()))
    n_frames = len(first_array)
    
    if frame_indices is None:
        frame_indices = np.arange(n_frames)
    
    # Determine if 3D or 2D
    n_dims = first_array.shape[1]
    is_3d = n_dims == 3
    
    # Create trajectory objects
    trajectories = {}
    for name, positions in trajectory_dict.items():
        if is_3d:
            trajectories[name] = Trajectory3D(
                marker_name=name,
                positions=positions,
                confidence=None
            )
        else:
            trajectories[name] = Observation2D(
                point_name=name,
                positions=positions,
                confidence=None
            )
    
    return TrajectoryDataset(
        data=trajectories,
        frame_indices=frame_indices,
        metadata={"source": "dictionary"}
    )


def load_multiple_files(
    *,
    filepaths: list[Path],
    **kwargs: Any
) -> list[TrajectoryDataset]:
    """Load multiple CSV files.
    
    Args:
        filepaths: List of CSV file paths
        **kwargs: Additional arguments passed to load_trajectories
        
    Returns:
        List of TrajectoryDatasets
    """
    datasets = []
    for filepath in filepaths:
        print(f"\nLoading {filepath.name}...")
        dataset = load_trajectories(filepath=filepath, **kwargs)
        datasets.append(dataset)
    
    return datasets


def concatenate_datasets(
    *,
    datasets: list[TrajectoryDataset]
) -> TrajectoryDataset:
    """Concatenate multiple datasets along time axis.
    
    All datasets must have same markers and data type.
    
    Args:
        datasets: List of datasets to concatenate
        
    Returns:
        Combined dataset
    """
    if not datasets:
        raise ValueError("Empty dataset list")
    
    if len(datasets) == 1:
        return datasets[0]
    
    # Check compatibility
    first_markers = set(datasets[0].marker_names)
    is_3d = datasets[0].is_3d
    
    for i, dataset in enumerate(datasets[1:], 1):
        if set(dataset.marker_names) != first_markers:
            raise ValueError(f"Dataset {i} has different markers")
        if dataset.is_3d != is_3d:
            raise ValueError(f"Dataset {i} has different dimensionality")
    
    # Concatenate frame indices
    frame_offset = 0
    all_frame_indices = []
    for dataset in datasets:
        all_frame_indices.append(dataset.frame_indices + frame_offset)
        frame_offset += dataset.n_frames
    
    combined_frame_indices = np.concatenate(all_frame_indices)
    
    # Concatenate data for each marker
    combined_data = {}
    for marker_name in datasets[0].marker_names:
        # Concatenate positions
        positions_list = [d.data[marker_name].positions for d in datasets]
        combined_positions = np.concatenate(positions_list, axis=0)
        
        # Concatenate confidence if available
        confidence_list = [d.data[marker_name].confidence for d in datasets]
        if all(c is not None for c in confidence_list):
            combined_confidence = np.concatenate(confidence_list, axis=0)
        else:
            combined_confidence = None
        
        # Create trajectory
        if is_3d:
            combined_data[marker_name] = Trajectory3D(
                marker_name=marker_name,
                positions=combined_positions,
                confidence=combined_confidence
            )
        else:
            combined_data[marker_name] = Observation2D(
                point_name=marker_name,
                positions=combined_positions,
                confidence=combined_confidence
            )
    
    return TrajectoryDataset(
        data=combined_data,
        frame_indices=combined_frame_indices,
        metadata={"source": "concatenated", "n_datasets": len(datasets)}
    )
