"""Data validation for SkellySolver.

Validates trajectory data and checks for common issues:
- Missing markers
- Insufficient valid frames
- Outliers
- Temporal gaps
"""

import numpy as np
from typing import Any

from .base_data import TrajectoryDataset, Trajectory3D, Observation2D


def validate_dataset(
    *,
    dataset: TrajectoryDataset,
    required_markers: list[str] | None = None,
    min_valid_frames: int = 10,
    min_confidence: float = 0.3,
    check_outliers: bool = True,
    outlier_threshold: float = 5.0
) -> dict[str, Any]:
    """Validate trajectory dataset and return validation report.
    
    Args:
        dataset: Dataset to validate
        required_markers: List of required marker names (None = no requirement)
        min_valid_frames: Minimum number of valid frames required
        min_confidence: Confidence threshold for validity
        check_outliers: Whether to check for spatial outliers
        outlier_threshold: Standard deviations for outlier detection
        
    Returns:
        Validation report dictionary
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    # Check required markers
    if required_markers is not None:
        missing_markers = set(required_markers) - set(dataset.marker_names)
        if missing_markers:
            report["valid"] = False
            report["errors"].append(f"Missing required markers: {missing_markers}")
    
    # Check number of frames
    if dataset.n_frames < min_valid_frames:
        report["valid"] = False
        report["errors"].append(
            f"Insufficient frames: {dataset.n_frames} < {min_valid_frames}"
        )
    
    # Check validity per marker
    for marker_name, traj in dataset.data.items():
        valid_mask = traj.is_valid(min_confidence=min_confidence)
        n_valid = np.sum(valid_mask)
        percent_valid = (n_valid / traj.n_frames) * 100
        
        report["info"][f"{marker_name}_valid_frames"] = n_valid
        report["info"][f"{marker_name}_valid_percent"] = percent_valid
        
        if n_valid < min_valid_frames:
            report["warnings"].append(
                f"Marker '{marker_name}' has only {n_valid} valid frames ({percent_valid:.1f}%)"
            )
    
    # Check for temporal gaps
    gap_report = check_temporal_gaps(dataset=dataset, min_confidence=min_confidence)
    if gap_report["has_gaps"]:
        report["warnings"].append(f"Temporal gaps detected: {gap_report['max_gap_size']} frames")
        report["info"]["temporal_gaps"] = gap_report
    
    # Check for outliers
    if check_outliers and dataset.is_3d:
        outlier_report = check_spatial_outliers(
            dataset=dataset,
            threshold=outlier_threshold
        )
        if outlier_report["has_outliers"]:
            report["warnings"].append(
                f"Spatial outliers detected: {outlier_report['n_outliers']} points"
            )
            report["info"]["spatial_outliers"] = outlier_report
    
    return report


def check_required_markers(
    *,
    dataset: TrajectoryDataset,
    required_markers: list[str]
) -> dict[str, Any]:
    """Check if dataset contains all required markers.
    
    Args:
        dataset: Dataset to check
        required_markers: List of required marker names
        
    Returns:
        Report dictionary with missing markers
    """
    existing_markers = set(dataset.marker_names)
    required_set = set(required_markers)
    
    missing = required_set - existing_markers
    extra = existing_markers - required_set
    
    return {
        "has_all_required": len(missing) == 0,
        "missing_markers": list(missing),
        "extra_markers": list(extra),
        "n_missing": len(missing),
        "n_extra": len(extra)
    }


def check_temporal_gaps(
    *,
    dataset: TrajectoryDataset,
    min_confidence: float = 0.3,
    max_gap_size: int = 10
) -> dict[str, Any]:
    """Check for temporal gaps in valid data.
    
    Args:
        dataset: Dataset to check
        min_confidence: Confidence threshold
        max_gap_size: Maximum acceptable gap size
        
    Returns:
        Report dictionary with gap information
    """
    gaps = []
    
    for marker_name, traj in dataset.data.items():
        valid_mask = traj.is_valid(min_confidence=min_confidence)
        
        # Find gaps (consecutive False values)
        gap_start = None
        for i, is_valid in enumerate(valid_mask):
            if not is_valid and gap_start is None:
                gap_start = i
            elif is_valid and gap_start is not None:
                gap_size = i - gap_start
                if gap_size > max_gap_size:
                    gaps.append({
                        "marker": marker_name,
                        "start_frame": gap_start,
                        "end_frame": i,
                        "size": gap_size
                    })
                gap_start = None
    
    return {
        "has_gaps": len(gaps) > 0,
        "n_gaps": len(gaps),
        "gaps": gaps,
        "max_gap_size": max(g["size"] for g in gaps) if gaps else 0
    }


def check_spatial_outliers(
    *,
    dataset: TrajectoryDataset,
    threshold: float = 5.0
) -> dict[str, Any]:
    """Check for spatial outliers using velocity-based detection.
    
    Args:
        dataset: Dataset to check (must be 3D)
        threshold: Number of standard deviations for outlier detection
        
    Returns:
        Report dictionary with outlier information
    """
    if not dataset.is_3d:
        raise ValueError("Outlier detection only supported for 3D data")
    
    outliers = []
    
    for marker_name, traj in dataset.data.items():
        if not isinstance(traj, Trajectory3D):
            continue
        
        # Compute velocities
        positions = traj.positions
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Detect outliers using z-score
        valid_speeds = speeds[~np.isnan(speeds)]
        if len(valid_speeds) < 2:
            continue
        
        mean_speed = np.mean(valid_speeds)
        std_speed = np.std(valid_speeds)
        
        if std_speed == 0:
            continue
        
        z_scores = np.abs((speeds - mean_speed) / std_speed)
        outlier_mask = z_scores > threshold
        
        outlier_indices = np.where(outlier_mask)[0]
        if len(outlier_indices) > 0:
            outliers.append({
                "marker": marker_name,
                "n_outliers": len(outlier_indices),
                "outlier_frames": outlier_indices.tolist(),
                "max_z_score": float(np.max(z_scores[outlier_mask]))
            })
    
    return {
        "has_outliers": len(outliers) > 0,
        "n_outliers": sum(o["n_outliers"] for o in outliers),
        "outliers_by_marker": outliers
    }


def check_data_quality(
    *,
    dataset: TrajectoryDataset,
    min_confidence: float = 0.3
) -> dict[str, Any]:
    """Compute data quality metrics.
    
    Args:
        dataset: Dataset to analyze
        min_confidence: Confidence threshold
        
    Returns:
        Quality metrics dictionary
    """
    metrics = {
        "n_frames": dataset.n_frames,
        "n_markers": dataset.n_markers,
        "marker_names": dataset.marker_names
    }
    
    # Compute per-marker validity
    marker_validity = {}
    for marker_name, traj in dataset.data.items():
        valid_mask = traj.is_valid(min_confidence=min_confidence)
        n_valid = np.sum(valid_mask)
        
        marker_validity[marker_name] = {
            "n_valid": int(n_valid),
            "percent_valid": float(n_valid / traj.n_frames * 100),
            "n_missing": int(traj.n_frames - n_valid)
        }
    
    metrics["marker_validity"] = marker_validity
    
    # Compute overall validity
    all_valid = dataset.get_valid_frames(
        min_confidence=min_confidence,
        min_valid_markers=dataset.n_markers
    )
    
    metrics["n_fully_valid_frames"] = int(np.sum(all_valid))
    metrics["percent_fully_valid"] = float(np.mean(all_valid) * 100)
    
    # Compute confidence statistics if available
    first_traj = next(iter(dataset.data.values()))
    if first_traj.confidence is not None:
        all_confidences = []
        for traj in dataset.data.values():
            if traj.confidence is not None:
                all_confidences.append(traj.confidence)
        
        if all_confidences:
            all_conf = np.concatenate(all_confidences)
            valid_conf = all_conf[~np.isnan(all_conf)]
            
            metrics["confidence_stats"] = {
                "mean": float(np.mean(valid_conf)),
                "std": float(np.std(valid_conf)),
                "min": float(np.min(valid_conf)),
                "max": float(np.max(valid_conf)),
                "median": float(np.median(valid_conf))
            }
    
    return metrics


def validate_topology_compatibility(
    *,
    dataset: TrajectoryDataset,
    topology_marker_names: list[str]
) -> dict[str, Any]:
    """Check if dataset is compatible with a topology.
    
    Used by rigid body tracking to ensure dataset has all required markers.
    
    Args:
        dataset: Dataset to check
        topology_marker_names: Marker names required by topology
        
    Returns:
        Compatibility report
    """
    dataset_markers = set(dataset.marker_names)
    required_markers = set(topology_marker_names)
    
    missing = required_markers - dataset_markers
    extra = dataset_markers - required_markers
    
    compatible = len(missing) == 0
    
    report = {
        "compatible": compatible,
        "missing_markers": list(missing),
        "extra_markers": list(extra),
        "n_missing": len(missing),
        "n_extra": len(extra)
    }
    
    if not compatible:
        report["error"] = f"Dataset missing required markers: {missing}"
    
    return report


def suggest_preprocessing(
    *,
    dataset: TrajectoryDataset,
    min_confidence: float = 0.3
) -> list[str]:
    """Suggest preprocessing steps based on data quality.
    
    Args:
        dataset: Dataset to analyze
        min_confidence: Confidence threshold
        
    Returns:
        List of suggested preprocessing steps
    """
    suggestions = []
    
    # Check validity
    quality = check_data_quality(dataset=dataset, min_confidence=min_confidence)
    
    if quality["percent_fully_valid"] < 80.0:
        suggestions.append("Filter low-confidence frames (< 80% fully valid)")
    
    # Check for missing data
    for marker_name, validity in quality["marker_validity"].items():
        if validity["n_missing"] > 0:
            if validity["percent_valid"] > 50.0:
                suggestions.append(f"Interpolate missing data for '{marker_name}'")
            else:
                suggestions.append(f"Consider removing marker '{marker_name}' (< 50% valid)")
    
    # Check for gaps
    gap_report = check_temporal_gaps(dataset=dataset, min_confidence=min_confidence)
    if gap_report["has_gaps"]:
        suggestions.append("Interpolate temporal gaps in data")
    
    # Check for outliers
    if dataset.is_3d:
        outlier_report = check_spatial_outliers(dataset=dataset, threshold=5.0)
        if outlier_report["has_outliers"]:
            suggestions.append("Filter or smooth spatial outliers")
    
    if not suggestions:
        suggestions.append("No preprocessing needed - data quality is good!")
    
    return suggestions


def print_validation_report(*, report: dict[str, Any]) -> None:
    """Print validation report in human-readable format.
    
    Args:
        report: Validation report from validate_dataset()
    """
    print("\n" + "="*80)
    print("DATA VALIDATION REPORT")
    print("="*80)
    
    if report["valid"]:
        print("✓ Validation passed")
    else:
        print("✗ Validation failed")
    
    if report["errors"]:
        print("\nERRORS:")
        for error in report["errors"]:
            print(f"  ✗ {error}")
    
    if report["warnings"]:
        print("\nWARNINGS:")
        for warning in report["warnings"]:
            print(f"  ⚠ {warning}")
    
    if report["info"]:
        print("\nINFO:")
        for key, value in report["info"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    print("="*80)
