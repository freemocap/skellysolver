"""CSV writers for trajectory and observation data.

Provides specialized writers for:
- Trajectory data (noisy + optimized)
- Simple trajectory format (for input files)
- Eye tracking results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from skellysolver.io.writers.writer_base import CSVWriter


class TrajectoryCSVWriter(CSVWriter):
    """Writer for trajectory data with noisy and optimized columns.
    
    Writes CSV with columns:
        frame, noisy_{marker}_x/y/z, optimized_{marker}_x/y/z, [gt_{marker}_x/y/z]
    """
    
    def write(
        self,
        *,
        filepath: Path,
        data: dict[str, Any]
    ) -> None:
        """Write trajectory data to CSV.
        
        Args:
            filepath: Path to output CSV
            data: Dictionary with:
                - raw_data: (n_frames, n_markers, 3) noisy positions
                - optimized_data: (n_frames, n_markers, 3) optimized positions
                - marker_names: list of marker names
                - ground_truth_data: optional (n_frames, n_markers, 3) ground truth
        """
        self.validate_data(
            data=data,
            required_keys=["raw_data", "optimized_data", "marker_names"]
        )
        
        raw_data = data["raw_data"]
        optimized_data = data["optimized_data"]
        marker_names = data["marker_names"]
        ground_truth_data = data.get("ground_truth_data")
        
        n_frames, n_markers, _ = raw_data.shape
        
        # Build DataFrame
        df_data = {"frame": np.arange(n_frames)}
        
        # Add noisy data
        for idx, marker_name in enumerate(marker_names):
            for coord_idx, coord_name in enumerate(["x", "y", "z"]):
                df_data[f"noisy_{marker_name}.{coord_name}"] = raw_data[:, idx, coord_idx]
        
        # Add optimized data
        for idx, marker_name in enumerate(marker_names):
            for coord_idx, coord_name in enumerate(["x", "y", "z"]):
                df_data[f"optimized_{marker_name}.{coord_name}"] = optimized_data[:, idx, coord_idx]
        
        # Add ground truth if provided
        if ground_truth_data is not None:
            for idx, marker_name in enumerate(marker_names):
                for coord_idx, coord_name in enumerate(["x", "y", "z"]):
                    df_data[f"gt_{marker_name}.{coord_name}"] = ground_truth_data[:, idx, coord_idx]
        
        # Add centroids
        noisy_center = np.mean(raw_data, axis=1)
        optimized_center = np.mean(optimized_data, axis=1)
        
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            df_data[f"noisy_center.{coord_name}"] = noisy_center[:, coord_idx]
            df_data[f"optimized_center.{coord_name}"] = optimized_center[:, coord_idx]
        
        if ground_truth_data is not None:
            gt_center = np.mean(ground_truth_data, axis=1)
            for coord_idx, coord_name in enumerate(["x", "y", "z"]):
                df_data[f"gt_center.{coord_name}"] = gt_center[:, coord_idx]
        
        # Write
        self.ensure_directory(filepath=filepath)
        
        df = pd.DataFrame(data=df_data)
        df.to_csv(path_or_buf=filepath, index=False)
        


class SimpleTrajectoryCSVWriter(CSVWriter):
    """Writer for simple trajectory format (for input files).
    
    Writes CSV with columns:
        frame, {marker}_x, {marker}_y, {marker}_z
    
    No prefixes like "noisy_" or "optimized_".
    """
    
    def write(
        self,
        *,
        filepath: Path,
        data: dict[str, Any]
    ) -> None:
        """Write simple trajectory data to CSV.
        
        Args:
            filepath: Path to output CSV
            data: Dictionary with:
                - positions: (n_frames, n_markers, 3) trajectory positions
                - marker_names: list of marker names
        """
        self.validate_data(
            data=data,
            required_keys=["positions", "marker_names"]
        )
        
        positions = data["positions"]
        marker_names = data["marker_names"]
        
        n_frames, n_markers, _ = positions.shape
        
        # Build DataFrame
        df_data = {"frame": np.arange(n_frames)}
        
        for idx, marker_name in enumerate(marker_names):
            for coord_idx, coord_name in enumerate(["x", "y", "z"]):
                df_data[f"{marker_name}.{coord_name}"] = positions[:, idx, coord_idx]
        
        # Write
        self.ensure_directory(filepath=filepath)
        
        df = pd.DataFrame(data=df_data)
        df.to_csv(path_or_buf=filepath, index=False)
        



class EyeTrackingCSVWriter(CSVWriter):
    """Writer for eye tracking results.
    
    Writes CSV with gaze directions, angles, and errors.
    """
    
    def write(
        self,
        *,
        filepath: Path,
        data: dict[str, Any]
    ) -> None:
        """Write eye tracking results to CSV.
        
        Args:
            filepath: Path to output CSV
            data: Dictionary with:
                - frame_indices: (n_frames,) frame numbers
                - gaze_directions: (n_frames, 3) gaze vectors
                - pupil_scales: (n_frames,) pupil dilation scales
                - Optional: reprojection_errors, pupil_centers, etc.
        """
        self.validate_data(
            data=data,
            required_keys=["frame_indices", "gaze_directions", "pupil_scales"]
        )
        
        frame_indices = data["frame_indices"]
        gaze_directions = data["gaze_directions"]
        pupil_scales = data["pupil_scales"]
        
        # Compute gaze angles
        azimuth = np.arctan2(gaze_directions[:, 0], gaze_directions[:, 2])
        elevation = np.arcsin(gaze_directions[:, 1])
        
        # Build DataFrame
        df_data = {
            "frame": frame_indices,
            "gaze.x": gaze_directions[:, 0],
            "gaze.y": gaze_directions[:, 1],
            "gaze.z": gaze_directions[:, 2],
            "gaze_azimuth_rad": azimuth,
            "gaze_elevation_rad": elevation,
            "gaze_azimuth_deg": np.rad2deg(azimuth),
            "gaze_elevation_deg": np.rad2deg(elevation),
            "pupil_scale": pupil_scales,
        }
        
        # Add optional fields
        if "reprojection_errors" in data:
            df_data["reprojection_error_px"] = data["reprojection_errors"]
        
        if "pupil_centers_3d" in data:
            centers = data["pupil_centers_3d"]
            df_data["pupil_3d_x_mm"] = centers[:, 0]
            df_data["pupil_3d_y_mm"] = centers[:, 1]
            df_data["pupil_3d_z_mm"] = centers[:, 2]
        
        if "eyeball_centers" in data:
            centers = data["eyeball_centers"]
            df_data["eyeball_x_mm"] = centers[:, 0]
            df_data["eyeball_y_mm"] = centers[:, 1]
            df_data["eyeball_z_mm"] = centers[:, 2]
        
        # Write
        self.ensure_directory(filepath=filepath)
        
        df = pd.DataFrame(data=df_data)
        df.to_csv(path_or_buf=filepath, index=False)
        



class TidyCSVWriter(CSVWriter):
    """Writer for tidy/long-format CSV.
    
    Writes CSV in long format:
        frame, keypoint, x, y, z
        0, marker1, 1.0, 2.0, 3.0
        0, marker2, 4.0, 5.0, 6.0
        1, marker1, 1.1, 2.1, 3.1
        ...
    """
    
    def write(
        self,
        *,
        filepath: Path,
        data: dict[str, Any]
    ) -> None:
        """Write trajectory data in tidy format.
        
        Args:
            filepath: Path to output CSV
            data: Dictionary with:
                - positions: (n_frames, n_markers, 3) positions
                - marker_names: list of marker names
        """
        self.validate_data(
            data=data,
            required_keys=["positions", "marker_names"]
        )
        
        positions = data["positions"]
        marker_names = data["marker_names"]
        
        n_frames, n_markers, _ = positions.shape
        
        # Build rows
        rows = []
        for frame_idx in range(n_frames):
            for marker_idx, marker_name in enumerate(marker_names):
                x, y, z = positions[frame_idx, marker_idx]
                rows.append({
                    "frame": frame_idx,
                    "keypoint": marker_name,
                    "x": x,
                    "y": y,
                    "z": z,
                })
        
        # Write
        self.write_rows(
            filepath=filepath,
            rows=rows,
            fieldnames=["frame", "keypoint", "x", "y", "z"]
        )
        

