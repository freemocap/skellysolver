"""Unified results writer for all pipelines.

Consolidates result saving from:
- io/savers.py (rigid body results)
- io/eye_savers.py (eye tracking results)

Provides single interface for saving optimization results.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Any
import shutil

from .base import BaseWriter
from .csv_writer import TrajectoryCSVWriter, EyeTrackingCSVWriter
from ...core.result import OptimizationResult, RigidBodyResult, EyeTrackingResult


class ResultsWriter:
    """Unified writer for optimization results.
    
    Handles saving for all pipeline types:
    - Rigid body tracking
    - Eye tracking
    - Custom pipelines
    
    Saves:
    - CSV files with trajectory/observation data
    - JSON files with metrics and configuration
    - NPY files with arrays (reference geometry, quaternions, etc.)
    - HTML viewer (optional)
    """
    
    def __init__(self, *, output_dir: Path) -> None:
        """Initialize results writer.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_rigid_body_results(
        self,
        *,
        result: RigidBodyResult,
        noisy_data: np.ndarray,
        marker_names: list[str],
        topology_dict: dict[str, Any],
        metrics: dict[str, Any],
        ground_truth_data: np.ndarray | None = None,
        copy_viewer: bool = True,
        viewer_template_path: Path | None = None
    ) -> None:
        """Save complete rigid body tracking results.
        
        Saves:
        - trajectory_data.csv: Noisy and optimized trajectories
        - topology.json: Topology metadata
        - metrics.json: Evaluation metrics
        - reference_geometry.npy: Optimized reference shape
        - rigid_body_viewer.html: Interactive viewer (optional)
        
        Args:
            result: Optimization result
            noisy_data: (n_frames, n_markers, 3) noisy measurements
            marker_names: List of marker names
            topology_dict: Topology dictionary
            metrics: Evaluation metrics
            ground_truth_data: Optional ground truth
            copy_viewer: Whether to copy HTML viewer
            viewer_template_path: Path to viewer template
        """
        print(f"\nSaving rigid body results to {self.output_dir}...")
        
        # Save trajectory CSV
        print("  Saving trajectory_data.csv...")
        traj_writer = TrajectoryCSVWriter()
        traj_writer.write(
            filepath=self.output_dir / "trajectory_data.csv",
            data={
                "noisy_data": noisy_data,
                "optimized_data": result.reconstructed,
                "marker_names": marker_names,
                "ground_truth_data": ground_truth_data,
            }
        )
        
        # Save topology JSON
        print("  Saving topology.json...")
        topology_data = {
            "topology": topology_dict,
            "marker_names": marker_names,
            "n_frames": result.n_frames,
            "n_markers": result.n_markers,
            "has_ground_truth": ground_truth_data is not None,
        }
        
        with open(self.output_dir / "topology.json", mode='w') as f:
            json.dump(obj=topology_data, fp=f, indent=2)
        
        # Save metrics JSON
        print("  Saving metrics.json...")
        metrics_data = {
            "metrics": metrics,
            "optimization": {
                "success": result.success,
                "iterations": result.num_iterations,
                "initial_cost": result.initial_cost,
                "final_cost": result.final_cost,
                "solve_time_seconds": result.solve_time_seconds,
            }
        }
        
        with open(self.output_dir / "metrics.json", mode='w') as f:
            json.dump(obj=metrics_data, fp=f, indent=2)
        
        # Save reference geometry
        print("  Saving reference_geometry.npy...")
        np.save(
            file=self.output_dir / "reference_geometry.npy",
            arr=result.reference_geometry
        )
        
        # Save rotations
        print("  Saving rotations.npy...")
        np.save(
            file=self.output_dir / "rotations.npy",
            arr=result.rotations
        )
        
        # Save translations
        print("  Saving translations.npy...")
        np.save(
            file=self.output_dir / "translations.npy",
            arr=result.translations
        )
        
        # Copy viewer
        if copy_viewer and viewer_template_path is not None:
            if viewer_template_path.exists():
                print("  Copying rigid_body_viewer.html...")
                shutil.copy(
                    src=viewer_template_path,
                    dst=self.output_dir / "rigid_body_viewer.html"
                )
            else:
                print(f"  ⚠ Viewer template not found: {viewer_template_path}")
        
        print(f"✓ Results saved to {self.output_dir}")
    
    def save_eye_tracking_results(
        self,
        *,
        result: EyeTrackingResult,
        metrics: dict[str, Any],
        copy_viewer: bool = True,
        viewer_template_path: Path | None = None
    ) -> None:
        """Save complete eye tracking results.
        
        Saves:
        - eye_tracking_results.csv: Gaze directions, angles, scales
        - metrics.json: Evaluation metrics
        - quaternions.npy: Eye orientations
        - pupil_scales.npy: Pupil dilation
        - gaze_directions.npy: Gaze vectors
        - eye_tracking_viewer.html: Interactive viewer (optional)
        
        Args:
            result: Optimization result
            metrics: Evaluation metrics
            copy_viewer: Whether to copy HTML viewer
            viewer_template_path: Path to viewer template
        """
        print(f"\nSaving eye tracking results to {self.output_dir}...")
        
        # Save eye tracking CSV
        print("  Saving eye_tracking_results.csv...")
        eye_writer = EyeTrackingCSVWriter()
        
        eye_data = {
            "frame_indices": np.arange(result.n_frames),
            "gaze_directions": result.gaze_directions,
            "pupil_scales": result.pupil_scales,
        }
        
        # Add optional fields
        if result.pupil_centers_3d is not None:
            eye_data["pupil_centers_3d"] = result.pupil_centers_3d
        
        if result.pupil_errors is not None:
            eye_data["reprojection_errors"] = result.pupil_errors
        
        eye_writer.write(
            filepath=self.output_dir / "eye_tracking_results.csv",
            data=eye_data
        )
        
        # Save metrics JSON
        print("  Saving metrics.json...")
        metrics_data = {
            "metrics": metrics,
            "optimization": {
                "success": result.success,
                "iterations": result.num_iterations,
                "initial_cost": result.initial_cost,
                "final_cost": result.final_cost,
                "solve_time_seconds": result.solve_time_seconds,
            }
        }
        
        with open(self.output_dir / "metrics.json", mode='w') as f:
            json.dump(obj=metrics_data, fp=f, indent=2)
        
        # Save quaternions
        print("  Saving quaternions.npy...")
        np.save(
            file=self.output_dir / "quaternions.npy",
            arr=result.rotations
        )
        
        # Save pupil scales
        print("  Saving pupil_scales.npy...")
        np.save(
            file=self.output_dir / "pupil_scales.npy",
            arr=result.pupil_scales
        )
        
        # Save gaze directions
        print("  Saving gaze_directions.npy...")
        np.save(
            file=self.output_dir / "gaze_directions.npy",
            arr=result.gaze_directions
        )
        
        # Copy viewer
        if copy_viewer and viewer_template_path is not None:
            if viewer_template_path.exists():
                print("  Copying eye_tracking_viewer.html...")
                shutil.copy(
                    src=viewer_template_path,
                    dst=self.output_dir / "eye_tracking_viewer.html"
                )
            else:
                print(f"  ⚠ Viewer template not found: {viewer_template_path}")
        
        print(f"✓ Results saved to {self.output_dir}")
    
    def save_generic_results(
        self,
        *,
        result: OptimizationResult,
        metrics: dict[str, Any],
        additional_arrays: dict[str, np.ndarray] | None = None
    ) -> None:
        """Save generic optimization results.
        
        For custom pipelines that don't fit rigid body or eye tracking.
        
        Args:
            result: Optimization result
            metrics: Evaluation metrics
            additional_arrays: Optional additional numpy arrays to save
        """
        print(f"\nSaving generic results to {self.output_dir}...")
        
        # Save metrics JSON
        print("  Saving metrics.json...")
        metrics_data = {
            "metrics": metrics,
            "optimization": {
                "success": result.success,
                "iterations": result.num_iterations,
                "initial_cost": result.initial_cost,
                "final_cost": result.final_cost,
                "solve_time_seconds": result.solve_time_seconds,
            }
        }
        
        with open(self.output_dir / "metrics.json", mode='w') as f:
            json.dump(obj=metrics_data, fp=f, indent=2)
        
        # Save any arrays in result
        if result.reconstructed is not None:
            print("  Saving reconstructed.npy...")
            np.save(
                file=self.output_dir / "reconstructed.npy",
                arr=result.reconstructed
            )
        
        if result.rotations is not None:
            print("  Saving rotations.npy...")
            np.save(
                file=self.output_dir / "rotations.npy",
                arr=result.rotations
            )
        
        if result.translations is not None:
            print("  Saving translations.npy...")
            np.save(
                file=self.output_dir / "translations.npy",
                arr=result.translations
            )
        
        # Save additional arrays
        if additional_arrays is not None:
            for name, array in additional_arrays.items():
                print(f"  Saving {name}.npy...")
                np.save(
                    file=self.output_dir / f"{name}.npy",
                    arr=array
                )
        
        print(f"✓ Results saved to {self.output_dir}")
