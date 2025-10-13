"""Data handling module for SkellySolver.

This module provides unified data structures, loading, validation, and preprocessing
for all pipelines.

Consolidates:
- io/loaders.py
- io/load_trajectories.py
- io/eye_data_loader.py

Main Components:

Base Data Structures:
- Trajectory3D: 3D marker trajectories
- Observation2D: 2D image observations
- TrajectoryDataset: Collection of trajectories

Format Detection:
- detect_csv_format(): Auto-detect CSV format
- CSVFormat: Type for format specification

Loading:
- load_trajectories(): Main loading function (auto-detects format)
- load_tidy_format(): Load tidy/long format
- load_wide_format(): Load wide format
- load_dlc_format(): Load DeepLabCut format

Validation:
- validate_dataset(): Comprehensive validation
- check_required_markers(): Check marker presence
- check_temporal_gaps(): Find gaps in data
- check_spatial_outliers(): Detect outliers
- check_data_quality(): Quality metrics

Preprocessing:
- interpolate_missing(): Fill missing data
- filter_by_confidence(): Remove low-confidence frames
- smooth_trajectories(): Smooth using Savitzky-Golay
- remove_outliers(): Remove spatial outliers
- center_data(): Center around origin
- scale_data(): Scale coordinates
- subsample_frames(): Reduce frame count

Usage:
    from skellysolver.data import load_trajectories, validate_dataset
    
    # Load data (auto-detects format)
    dataset = load_trajectories(filepath=Path("data.csv"))
    
    # Validate
    report = validate_dataset(dataset=dataset)
    
    # Preprocess
    from skellysolver.data import interpolate_missing, smooth_trajectories
    dataset = interpolate_missing(dataset=dataset)
    dataset = smooth_trajectories(dataset=dataset)
"""

# Base data structures
from .base_data import (
    Trajectory3D,
    Observation2D,
    TrajectoryDataset,
)

# Format detection
from .formats import (
    CSVFormat,
    detect_csv_format,
    validate_tidy_format,
    validate_wide_format,
    validate_dlc_format,
    get_format_info,
    is_3d_data,
)

# Loading
from .loaders import (
    load_trajectories,
    load_tidy_format,
    load_wide_format,
    load_dlc_format,
    load_from_dict,
    load_multiple_files,
    concatenate_datasets,
)

# Validation
from .validators import (
    validate_dataset,
    check_required_markers,
    check_temporal_gaps,
    check_spatial_outliers,
    check_data_quality,
    validate_topology_compatibility,
    suggest_preprocessing,
    print_validation_report,
)

# Preprocessing
from .preprocessing import (
    interpolate_missing,
    filter_by_confidence,
    smooth_trajectories,
    remove_outliers,
    center_data,
    scale_data,
    subsample_frames,
    apply_preprocessing_pipeline,
)

__all__ = [
    # Base
    "Trajectory3D",
    "Observation2D",
    "TrajectoryDataset",
    # Formats
    "CSVFormat",
    "detect_csv_format",
    "validate_tidy_format",
    "validate_wide_format",
    "validate_dlc_format",
    "get_format_info",
    "is_3d_data",
    # Loaders
    "load_trajectories",
    "load_tidy_format",
    "load_wide_format",
    "load_dlc_format",
    "load_from_dict",
    "load_multiple_files",
    "concatenate_datasets",
    # Validators
    "validate_dataset",
    "check_required_markers",
    "check_temporal_gaps",
    "check_spatial_outliers",
    "check_data_quality",
    "validate_topology_compatibility",
    "suggest_preprocessing",
    "print_validation_report",
    # Preprocessing
    "interpolate_missing",
    "filter_by_confidence",
    "smooth_trajectories",
    "remove_outliers",
    "center_data",
    "scale_data",
    "subsample_frames",
    "apply_preprocessing_pipeline",
]
