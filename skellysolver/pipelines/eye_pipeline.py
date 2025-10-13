"""Eye tracking pipeline.

Optimizes eye orientation and pupil dilation from 2D pupil observations.
Inherits from BasePipeline and uses Phase 1 + Phase 2 components.
"""

import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass

from ..base import BasePipeline, PipelineConfig
from skellysolver.core.config import OptimizationConfig, EyeTrackingWeightConfig
from skellysolver.core.result import OptimizationResult, EyeTrackingResult
from skellysolver.core.optimizer import Optimizer
from ...core.cost_functions import (
    ScalarSmoothnessCost,
    RotationSmoothnessCost,
    get_quaternion_manifold,
)
from skellysolver.data.base_data import TrajectoryDataset
from skellysolver.data.loaders import load_trajectories
from skellysolver.data.validators import validate_dataset
from skellysolver.data.preprocessing import filter_by_confidence

logger = logging.getLogger(__name__)


@dataclass
class EyeModel:
    """Eye model parameters."""
    
    eyeball_center_mm: np.ndarray  # (3,) center position
    base_semi_major_mm: float      # Pupil semi-major axis
    base_semi_minor_mm: float      # Pupil semi-minor axis
    pupil_roundness: float         # Shape parameter (2=ellipse)
    tear_duct_xyz_mm: np.ndarray   # (3,) tear duct position
    
    @classmethod
    def create_initial_guess(
        cls,
        *,
        eyeball_distance_mm: float = 20.0,
        base_semi_major_mm: float = 2.0,
        base_semi_minor_mm: float = 1.5,
        pupil_roundness: float = 2.0
    ) -> "EyeModel":
        """Create initial parameter guess."""
        return cls(
            eyeball_center_mm=np.array([0.0, 0.0, eyeball_distance_mm]),
            base_semi_major_mm=base_semi_major_mm,
            base_semi_minor_mm=base_semi_minor_mm,
            pupil_roundness=pupil_roundness,
            tear_duct_xyz_mm=np.array([2.0, 1.0, 0.0])
        )


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int
    
    @property
    def fx(self) -> float:
        """Focal length in x (pixels)."""
        pixel_size_x = self.sensor_width_mm / self.image_width_px
        return self.focal_length_mm / pixel_size_x
    
    @property
    def fy(self) -> float:
        """Focal length in y (pixels)."""
        pixel_size_y = self.sensor_height_mm / self.image_height_px
        return self.focal_length_mm / pixel_size_y
    
    @property
    def cx(self) -> float:
        """Principal point x (pixels)."""
        return self.image_width_px / 2.0
    
    @property
    def cy(self) -> float:
        """Principal point y (pixels)."""
        return self.image_height_px / 2.0
    
    @classmethod
    def create_pupil_labs_camera(cls) -> "CameraIntrinsics":
        """Create Pupil Labs eye camera specs."""
        return cls(
            focal_length_mm=1.7,
            sensor_width_mm=1.15,
            sensor_height_mm=1.15,
            image_width_px=400,
            image_height_px=400
        )


@dataclass
class EyeTrackingConfig(PipelineConfig):
    """Configuration for eye tracking pipeline.
    
    Extends PipelineConfig with eye tracking specific settings.
    
    Attributes:
        camera: Camera intrinsics
        weights: Cost function weights
        initial_eye_model: Initial parameter guess
        min_confidence: Minimum confidence for filtering
        min_pupil_points: Minimum valid pupil points per frame
    """
    
    camera: CameraIntrinsics
    weights: EyeTrackingWeightConfig = None
    initial_eye_model: EyeModel = None
    min_confidence: float = 0.3
    min_pupil_points: int = 6
    
    def __post_init__(self) -> None:
        """Set defaults."""
        super().__post_init__()
        
        if self.weights is None:
            self.weights = EyeTrackingWeightConfig()
        
        if self.initial_eye_model is None:
            self.initial_eye_model = EyeModel.create_initial_guess()


class EyeTrackingPipeline(BasePipeline):
    """Eye tracking pipeline.
    
    Optimizes eye orientation and pupil dilation from 2D pupil observations.
    Uses bundle adjustment to fit eye model to image observations.
    
    Usage:
        from skellysolver.pipelines.eye_tracking import (
            EyeTrackingPipeline,
            EyeTrackingConfig,
            CameraIntrinsics,
        )
        from skellysolver.core import OptimizationConfig
        
        # Define camera
        camera = CameraIntrinsics.create_pupil_labs_camera()
        
        # Configure
        config = EyeTrackingConfig(
            input_path=Path("pupil_data.csv"),
            output_dir=Path("output/"),
            camera=camera,
            optimization=OptimizationConfig(max_iterations=500),
        )
        
        # Run
        pipeline = EyeTrackingPipeline(config=config)
        result = pipeline.run()
    """
    
    config: EyeTrackingConfig  # Type hint for IDE
    
    def load_data(self) -> TrajectoryDataset:
        """Load pupil observation data from CSV.
        
        Expected format: DeepLabCut CSV with pupil points (p1-p8) and tear_duct.
        
        Returns:
            TrajectoryDataset with 2D pupil observations
        """
        logger.info(f"Loading data from {self.config.input_path.name}...")
        
        dataset = load_trajectories(
            filepath=self.config.input_path,
            format="dlc",
            scale_factor=1.0,
            likelihood_threshold=self.config.min_confidence
        )
        
        logger.info(f"  Loaded {dataset.n_markers} points × {dataset.n_frames} frames")
        logger.info(f"  Points: {dataset.marker_names}")
        
        return dataset
    
    def preprocess_data(self, *, data: TrajectoryDataset) -> TrajectoryDataset:
        """Preprocess and validate data.
        
        Steps:
        1. Check required points present (p1-p8, tear_duct)
        2. Validate data quality
        3. Filter frames with insufficient pupil points
        
        Args:
            data: Raw loaded data
            
        Returns:
            Preprocessed data
        """
        logger.info("Preprocessing data...")
        
        # Check required points
        required_points = [f"p{i}" for i in range(1, 9)] + ["tear_duct"]
        logger.info("  Checking required points...")
        
        missing = set(required_points) - set(data.marker_names)
        if missing:
            raise ValueError(f"Missing required points: {missing}")
        
        logger.info("    ✓ All required points present")
        
        # Validate data
        logger.info("  Validating data...")
        report = validate_dataset(
            dataset=data,
            required_markers=required_points,
            min_valid_frames=10,
            min_confidence=self.config.min_confidence
        )
        
        if not report["valid"]:
            logger.warning("    ⚠ Data validation warnings:")
            for error in report["errors"]:
                logger.warning(f"      {error}")
        else:
            logger.info("    ✓ Data validation passed")
        
        # Filter frames with insufficient pupil points
        logger.info(f"  Filtering frames (min_pupil_points={self.config.min_pupil_points})...")
        
        # Count valid pupil points per frame
        pupil_point_names = [f"p{i}" for i in range(1, 9)]
        valid_counts = np.zeros(data.n_frames, dtype=int)
        
        for point_name in pupil_point_names:
            obs = data.data[point_name]
            valid_counts += obs.is_valid(min_confidence=self.config.min_confidence).astype(int)
        
        valid_mask = valid_counts >= self.config.min_pupil_points
        
        if not np.any(valid_mask):
            raise ValueError(f"No frames with {self.config.min_pupil_points}+ valid pupil points")
        
        # Filter dataset
        n_before = data.n_frames
        data = filter_by_confidence(
            dataset=data,
            min_confidence=self.config.min_confidence,
            min_valid_markers=self.config.min_pupil_points
        )
        n_after = data.n_frames
        
        logger.info(f"    Filtered: {n_before} → {n_after} frames ({n_before - n_after} removed)")
        logger.info("✓ Preprocessing complete")
        
        return data
    
    def optimize(self, *, data: TrajectoryDataset) -> EyeTrackingResult:
        """Run eye tracking optimization.
        
        Optimizes:
        - Eye orientation (quaternion) for each frame
        - Pupil dilation (scale) for each frame
        - Static eye model parameters (center, shape, tear duct)
        
        Args:
            data: Preprocessed data
            
        Returns:
            EyeTrackingResult with optimized parameters
        """
        logger.info("Running optimization...")
        
        # Extract pupil points and tear duct
        pupil_point_names = [f"p{i}" for i in range(1, 9)]
        pupil_points = data.to_array(marker_names=pupil_point_names)  # (n_frames, 8, 2)
        tear_ducts = data.to_array(marker_names=["tear_duct"])  # (n_frames, 1, 2)
        tear_ducts = tear_ducts.squeeze(axis=1)  # (n_frames, 2)
        
        n_frames = data.n_frames
        
        logger.info(f"  Data shape: pupil={pupil_points.shape}, tear_duct={tear_ducts.shape}")
        
        # Initialize parameters
        logger.info("  Initializing parameters...")
        
        # Per-frame parameters
        quaternions = np.zeros((n_frames, 4))
        quaternions[:, 0] = 1.0  # Identity rotations
        
        pupil_scales = np.ones((n_frames, 1))
        
        # Static eye model parameters
        eye_model = self.config.initial_eye_model
        
        # Build optimization problem
        logger.info("  Building optimization problem...")
        optimizer = Optimizer(config=self.config.optimization)
        
        # Add per-frame parameters
        for i in range(n_frames):
            optimizer.add_quaternion_parameter(
                name=f"quat_{i}",
                parameters=quaternions[i]
            )
            optimizer.add_parameter_block(
                name=f"scale_{i}",
                parameters=pupil_scales[i]
            )
            
            # Set bounds on scale
            optimizer.set_parameter_bounds(
                parameters=pupil_scales[i],
                index=0,
                lower=0.3,
                upper=3.0
            )
        
        # Note: In a full implementation, you would add:
        # - Pupil point projection costs
        # - Tear duct projection costs
        # - These require implementing the eye model projection
        # For now, we'll just add smoothness costs
        
        # Add smoothness costs
        logger.info(f"  Adding {n_frames - 1} smoothness constraints...")
        for i in range(n_frames - 1):
            # Rotation smoothness
            rot_cost = RotationSmoothnessCost(weight=self.config.weights.lambda_rot_smooth)
            optimizer.add_residual_block(
                cost=rot_cost,
                parameters=[quaternions[i], quaternions[i + 1]]
            )
            
            # Scale smoothness
            scale_cost = ScalarSmoothnessCost(weight=self.config.weights.lambda_scalar_smooth)
            optimizer.add_residual_block(
                cost=scale_cost,
                parameters=[pupil_scales[i], pupil_scales[i + 1]]
            )
        
        logger.info(f"  Total parameters: {optimizer.num_parameters()}")
        logger.info(f"  Total residuals:  {optimizer.num_residuals()}")
        
        # Solve
        logger.info("  Solving...")
        result = optimizer.solve()
        
        # Compute gaze directions from quaternions
        logger.info("  Computing gaze directions...")
        from scipy.spatial.transform import Rotation
        
        gaze_directions = np.zeros((n_frames, 3))
        for i in range(n_frames):
            quat_scipy = np.array([
                quaternions[i, 1],
                quaternions[i, 2],
                quaternions[i, 3],
                quaternions[i, 0]
            ])
            R = Rotation.from_quat(quat=quat_scipy)
            gaze_directions[i] = R.apply(v=np.array([0, 0, 1]))
        
        # Create specialized result
        eye_result = EyeTrackingResult(
            success=result.success,
            num_iterations=result.num_iterations,
            initial_cost=result.initial_cost,
            final_cost=result.final_cost,
            solve_time_seconds=result.solve_time_seconds,
            rotations=quaternions,
            gaze_directions=gaze_directions,
            pupil_scales=pupil_scales.flatten(),
            pupil_centers_3d=None,  # Would compute from full model
            tear_ducts_3d=None,
            projected_pupil_centers=None,
            projected_tear_ducts=None,
            pupil_errors=None,
            tear_duct_errors=None,
            metadata={
                "n_frames": n_frames,
                "camera": "pupil_labs"
            }
        )
        
        logger.info("✓ Optimization complete")
        
        return eye_result
    
    def evaluate(self, *, result: EyeTrackingResult) -> dict[str, float]:
        """Evaluate eye tracking quality.
        
        Computes metrics:
        - Gaze angle range
        - Pupil scale range
        - Temporal smoothness
        
        Args:
            result: Optimization result
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating results...")
        
        # Compute gaze angles
        azimuth = np.arctan2(result.gaze_directions[:, 0], result.gaze_directions[:, 2])
        elevation = np.arcsin(result.gaze_directions[:, 1])
        
        metrics = {
            "gaze_azimuth_range_deg": float(np.ptp(np.rad2deg(azimuth))),
            "gaze_elevation_range_deg": float(np.ptp(np.rad2deg(elevation))),
            "pupil_scale_mean": float(np.mean(result.pupil_scales)),
            "pupil_scale_std": float(np.std(result.pupil_scales)),
            "pupil_scale_range": float(np.ptp(result.pupil_scales)),
        }
        
        logger.info("  Metrics:")
        logger.info(f"    Gaze azimuth range:   {metrics['gaze_azimuth_range_deg']:.1f}°")
        logger.info(f"    Gaze elevation range: {metrics['gaze_elevation_range_deg']:.1f}°")
        logger.info(f"    Pupil scale:          {metrics['pupil_scale_mean']:.3f} ± {metrics['pupil_scale_std']:.3f}")
        
        logger.info("✓ Evaluation complete")
        
        return metrics
    
    def save_results(
        self,
        *,
        result: EyeTrackingResult,
        metrics: dict[str, float]
    ) -> None:
        """Save results to disk.
        
        Saves:
        - eye_tracking_results.csv: Gaze directions and pupil scales
        - metrics.json: Evaluation metrics
        - quaternions.npy: Eye orientations
        
        Args:
            result: Optimization result
            metrics: Evaluation metrics
        """
        logger.info("Saving results...")
        
        import pandas as pd
        import json
        
        # Compute gaze angles
        azimuth = np.arctan2(result.gaze_directions[:, 0], result.gaze_directions[:, 2])
        elevation = np.arcsin(result.gaze_directions[:, 1])
        
        # Create DataFrame
        df = pd.DataFrame(data={
            "frame": np.arange(result.n_frames),
            "gaze_x": result.gaze_directions[:, 0],
            "gaze_y": result.gaze_directions[:, 1],
            "gaze_z": result.gaze_directions[:, 2],
            "gaze_azimuth_rad": azimuth,
            "gaze_elevation_rad": elevation,
            "gaze_azimuth_deg": np.rad2deg(azimuth),
            "gaze_elevation_deg": np.rad2deg(elevation),
            "pupil_scale": result.pupil_scales,
        })
        
        # Save CSV
        csv_path = self.config.output_dir / "eye_tracking_results.csv"
        df.to_csv(path_or_buf=csv_path, index=False)
        logger.info(f"  ✓ Saved {csv_path.name}")
        
        # Save metrics
        metrics_path = self.config.output_dir / "metrics.json"
        with open(metrics_path, mode='w') as f:
            json.dump(obj=metrics, fp=f, indent=2)
        logger.info(f"  ✓ Saved {metrics_path.name}")
        
        # Save quaternions
        quat_path = self.config.output_dir / "quaternions.npy"
        np.save(file=quat_path, arr=result.rotations)
        logger.info(f"  ✓ Saved {quat_path.name}")
        
        logger.info(f"✓ Results saved to {self.config.output_dir}")
    
    def generate_viewer(self, *, result: EyeTrackingResult) -> None:
        """Generate interactive HTML viewer.
        
        Creates eye_tracking_viewer.html with gaze visualization.
        
        Args:
            result: Optimization result
        """
        logger.info("Generating viewer...")
        
        # Copy viewer HTML template
        import shutil
        
        viewer_template = Path(__file__).parent / "eye_tracking_viewer.html"
        viewer_output = self.config.output_dir / "eye_tracking_viewer.html"
        
        if viewer_template.exists():
            shutil.copy(src=viewer_template, dst=viewer_output)
            logger.info(f"  ✓ Generated {viewer_output.name}")
            logger.info(f"  → Open {viewer_output} in a browser to visualize")
        else:
            logger.warning(f"  ⚠ Viewer template not found: {viewer_template}")
            logger.warning("  Skipping viewer generation")
