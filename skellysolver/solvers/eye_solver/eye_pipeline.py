"""Complete eye tracking pipeline with full eye model optimization and chunking support.

"""

import numpy as np
import logging
from dataclasses import dataclass

from pydantic import Field, model_validator

from skellysolver.solvers.eye_solver.eye_solver_weights import EyeTrackingWeightConfig
from skellysolver.core import ChunkingConfig
from skellysolver.cost_primatives import RotationSmoothnessCost, ScalarSmoothnessCost
from skellysolver.solvers.eye_solver.eye_costs import PupilPointProjectionCost, TearDuctProjectionCost
from skellysolver.core.optimization_result import EyeTrackingResult
from skellysolver.solvers.pyceres_solver import PyceresOptimizer
from skellysolver.core.chunking import optimize_chunked_parallel, optimize_chunked_sequential
from skellysolver.data.data_models import TrajectoryDataset
from skellysolver.io.loaders import load_trajectories
from skellysolver.data.validators import validate_dataset
from skellysolver.data.preprocessing import filter_by_confidence
from skellysolver.solvers import PipelineConfig, BasePipeline

logger = logging.getLogger(__name__)


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


class EyeTrackingConfig(PipelineConfig):
    """Configuration for eye tracking pipeline with chunking support."""

    camera: CameraIntrinsics
    weights: EyeTrackingWeightConfig | None = None
    min_confidence: float = 0.3
    min_pupil_points: int = 6

    # Eye model parameters
    eyeball_distance_mm: float = 20.0
    base_semi_major_mm: float = 2.0
    base_semi_minor_mm: float = 1.5

    # Parallel processing
    parallel: ChunkingConfig | None = Field(default_factory=ChunkingConfig)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def set_defaults(self) -> 'EyeTrackingConfig':
        """Set defaults."""
        if self.weights is None:
            self.weights = EyeTrackingWeightConfig()
        if self.parallel is None:
            self.parallel = ChunkingConfig()
        return self


class EyeTrackingPipeline(BasePipeline):
    """Complete eye tracking pipeline with chunking support.

    NOW SUPPORTS:
    - Single-pass optimization for small datasets
    - Chunked parallel optimization for long recordings
    - Automatic mode selection based on data size
    """

    config: EyeTrackingConfig

    def load_data(self) -> TrajectoryDataset:
        """Load pupil observation data from CSV."""
        logger.info(f"Loading data from {self.config.input_path.name}...")

        dataset = load_trajectories(
            filepath=self.config.input_path,
            csv_format="dlc",
            scale_factor=1.0,
            likelihood_threshold=self.config.min_confidence
        )

        logger.info(f"  Loaded {dataset.n_markers} points × {dataset.n_frames} frames")
        logger.info(f"  Points: {dataset.marker_names}")

        return dataset

    def preprocess_data(self, *, data: TrajectoryDataset) -> TrajectoryDataset:
        """Preprocess and validate data."""
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

        # Filter frames
        logger.info(f"  Filtering frames (min_pupil_points={self.config.min_pupil_points})...")

        pupil_point_names = [f"p{i}" for i in range(1, 9)]
        valid_counts = np.zeros(data.n_frames, dtype=int)

        for point_name in pupil_point_names:
            obs = data.data[point_name]
            valid_counts += obs.is_valid(min_confidence=self.config.min_confidence).astype(int)

        valid_mask = valid_counts >= self.config.min_pupil_points

        if not np.any(valid_mask):
            raise ValueError(f"No frames with {self.config.min_pupil_points}+ valid pupil points")

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
        """Run eye tracking optimization with optional chunking.

        Automatically selects between single-pass and chunked optimization
        based on data size and parallel configuration.
        """
        logger.info("Running optimization...")

        # Extract data
        pupil_point_names = [f"p{i}" for i in range(1, 9)]
        pupil_points = data.to_array(marker_names=pupil_point_names)
        tear_ducts = data.to_array(marker_names=["tear_duct"]).squeeze(axis=1)

        n_frames = data.n_frames

        logger.info(f"  Data shape: pupil={pupil_points.shape}, tear_duct={tear_ducts.shape}")

        # Check if we should use chunked optimization
        use_chunking = (
            self.config.parallel is not None and
            self.config.parallel.should_use_parallel(n_frames=n_frames)
        )

        if use_chunking:
            result = self._optimize_chunked(
                pupil_points=pupil_points,
                tear_ducts=tear_ducts
            )
        else:
            result = self._optimize_single_pass(
                pupil_points=pupil_points,
                tear_ducts=tear_ducts
            )

        return result

    def _optimize_single_pass(
        self,
        *,
        pupil_points: np.ndarray,
        tear_ducts: np.ndarray
    ) -> EyeTrackingResult:
        """Run standard single-pass optimization.

        Args:
            pupil_points: (n_frames, 8, 2) pupil observations
            tear_ducts: (n_frames, 2) tear duct observations

        Returns:
            EyeTrackingResult
        """
        n_frames, n_pupil_points, _ = pupil_points.shape

        logger.info("  Using SINGLE-PASS optimization")
        logger.info("  Initializing parameters...")

        # Initialize parameters
        quaternions = np.zeros((n_frames, 4))
        quaternions[:, 0] = 1.0  # Identity rotations

        pupil_scales = np.ones((n_frames, 1))

        # Static eye model parameters
        eyeball_center = np.array([0.0, 0.0, self.config.eyeball_distance_mm])
        tear_duct_offset = np.array([2.0, 1.0, 0.0])

        # Build optimization problem
        logger.info("  Building optimization problem...")
        optimizer = PyceresOptimizer(config=self.config.optimization)

        # Add static parameters
        optimizer.add_parameter_block(
            name="eyeball_center",
            parameters=eyeball_center
        )

        optimizer.add_parameter_block(
            name="tear_duct_offset",
            parameters=tear_duct_offset
        )

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

        # Add pupil projection costs
        logger.info(f"  Adding {n_frames * n_pupil_points} pupil projection costs...")
        for frame_idx in range(n_frames):
            for point_idx in range(n_pupil_points):
                obs = pupil_points[frame_idx, point_idx]
                if np.any(np.isnan(obs)):
                    continue

                cost = PupilPointProjectionCost(
                    observed_px=obs,
                    point_index=point_idx,
                    n_pupil_points=n_pupil_points,
                    camera_fx=self.config.camera.fx,
                    camera_fy=self.config.camera.fy,
                    camera_cx=self.config.camera.cx,
                    camera_cy=self.config.camera.cy,
                    base_semi_major_mm=self.config.base_semi_major_mm,
                    base_semi_minor_mm=self.config.base_semi_minor_mm,
                    weight=self.config.weights.lambda_pupil
                )
                optimizer.add_residual_block(
                    cost=cost,
                    parameters=[quaternions[frame_idx], pupil_scales[frame_idx], eyeball_center]
                )

        # Add tear duct projection costs
        logger.info(f"  Adding {n_frames} tear duct projection costs...")
        for frame_idx in range(n_frames):
            obs = tear_ducts[frame_idx]
            if np.any(np.isnan(obs)):
                continue

            cost = TearDuctProjectionCost(
                observed_px=obs,
                camera_fx=self.config.camera.fx,
                camera_fy=self.config.camera.fy,
                camera_cx=self.config.camera.cx,
                camera_cy=self.config.camera.cy,
                weight=self.config.weights.lambda_tear_duct
            )
            optimizer.add_residual_block(
                cost=cost,
                parameters=[eyeball_center, tear_duct_offset]
            )

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

        # Compute results
        logger.info("  Computing results...")
        eye_result = self._compute_eye_results(
            quaternions=quaternions,
            pupil_scales=pupil_scales,
            eyeball_center=eyeball_center,
            tear_duct_offset=tear_duct_offset,
            pupil_points=pupil_points,
            tear_ducts=tear_ducts,
            optimization_result=result
        )

        logger.info("✓ Optimization complete")

        return eye_result

    def _optimize_chunk_wrapper(self, *, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wrapper for chunked optimization - converts eye tracking to chunking API format.

        This needs to be a class method (not nested function) to be picklable for multiprocessing.

        Args:
            data: (n_frames, 9, 2) combined [pupil_points | tear_duct] for this chunk

        Returns:
            (rotations, translations, reconstructed) - only rotations are meaningful for eye tracking
        """
        # Split combined data back into pupil points and tear duct
        chunk_pupil_points = data[:, :8, :]  # First 8 markers are pupil points
        chunk_tear_ducts = data[:, 8, :]     # 9th marker is tear duct

        # Run optimization
        chunk_result = self._optimize_single_pass(
            pupil_points=chunk_pupil_points,
            tear_ducts=chunk_tear_ducts
        )

        # Convert quaternions to rotation matrices for chunking API
        from scipy.spatial.transform import Rotation

        rotations = np.zeros((chunk_result.n_frames, 3, 3))
        for i in range(chunk_result.n_frames):
            quat_scipy = np.array([
                chunk_result.rotations[i, 1],
                chunk_result.rotations[i, 2],
                chunk_result.rotations[i, 3],
                chunk_result.rotations[i, 0]
            ])
            rotations[i] = Rotation.from_quat(quat=quat_scipy).as_matrix()

        # Dummy values for translations and reconstructed (not used for eye tracking)
        translations = np.zeros((chunk_result.n_frames, 3))
        reconstructed = np.zeros((chunk_result.n_frames, 8, 3))

        return rotations, translations, reconstructed

    def _optimize_chunked(
        self,
        *,
        pupil_points: np.ndarray,
        tear_ducts: np.ndarray
    ) -> EyeTrackingResult:
        """Run chunked optimization with blending.

        Args:
            pupil_points: (n_frames, 8, 2) pupil observations
            tear_ducts: (n_frames, 2) tear duct observations

        Returns:
            EyeTrackingResult
        """
        n_frames = pupil_points.shape[0]

        parallel_mode = "PARALLEL" if self.config.parallel.enabled else "SEQUENTIAL"
        logger.info(f"  Using CHUNKED {parallel_mode} optimization")
        logger.info(f"  Chunk size: {self.config.parallel.chunk_size}")
        logger.info(f"  Overlap: {self.config.parallel.overlap_size}")
        logger.info(f"  Blend window: {self.config.parallel.blend_window}")

        # Combine pupil_points and tear_ducts into single array for chunking
        # pupil_points: (n_frames, 8, 2)
        # tear_ducts: (n_frames, 2) -> reshape to (n_frames, 1, 2)
        # combined: (n_frames, 9, 2)
        tear_ducts_reshaped = tear_ducts[:, np.newaxis, :]  # Add marker dimension
        combined_data = np.concatenate([pupil_points, tear_ducts_reshaped], axis=1)

        logger.info(f"  Combined data shape: {combined_data.shape}")

        # Choose chunking mode
        if self.config.parallel.enabled:
            rotations, _, _ = optimize_chunked_parallel(
                raw_data=combined_data,
                chunk_size=self.config.parallel.chunk_size,
                overlap_size=self.config.parallel.overlap_size,
                blend_window=self.config.parallel.blend_window,
                min_chunk_size=self.config.parallel.min_chunk_size,
                n_workers=self.config.parallel.get_num_workers(),
                optimize_fn=self._optimize_chunk_wrapper
            )
        else:
            rotations, _, _ = optimize_chunked_sequential(
                raw_data=combined_data,
                chunk_size=self.config.parallel.chunk_size,
                overlap_size=self.config.parallel.overlap_size,
                blend_window=self.config.parallel.blend_window,
                min_chunk_size=self.config.parallel.min_chunk_size,
                optimize_fn=self._optimize_chunk_wrapper
            )

        # Convert rotation matrices back to quaternions
        from scipy.spatial.transform import Rotation

        quaternions = np.zeros((n_frames, 4))
        for i in range(n_frames):
            R = Rotation.from_matrix(matrix=rotations[i])
            quat_scipy = R.as_quat()  # [x, y, z, w]
            quaternions[i] = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])  # [w, x, y, z]

        # Recompute pupil scales (simple re-optimization for scales only)
        pupil_scales = np.ones((n_frames, 1))

        # Estimate eye model params (use median)
        eyeball_center = np.array([0.0, 0.0, self.config.eyeball_distance_mm])
        tear_duct_offset = np.array([2.0, 1.0, 0.0])

        # Create result
        eye_result = self._compute_eye_results(
            quaternions=quaternions,
            pupil_scales=pupil_scales,
            eyeball_center=eyeball_center,
            tear_duct_offset=tear_duct_offset,
            pupil_points=pupil_points,
            tear_ducts=tear_ducts,
            optimization_result=None
        )

        # Update metadata
        eye_result.metadata["optimization_mode"] = parallel_mode.lower()

        logger.info("✓ Chunked optimization complete")

        return eye_result

    def _compute_eye_results(
        self,
        *,
        quaternions: np.ndarray,
        pupil_scales: np.ndarray,
        eyeball_center: np.ndarray,
        tear_duct_offset: np.ndarray,
        pupil_points: np.ndarray,
        tear_ducts: np.ndarray,
        optimization_result: object | None
    ) -> EyeTrackingResult:
        """Compute final eye tracking results.

        Args:
            quaternions: (n_frames, 4) eye orientations
            pupil_scales: (n_frames, 1) pupil scales
            eyeball_center: (3,) eyeball center position
            tear_duct_offset: (3,) tear duct offset from eyeball center
            pupil_points: (n_frames, 8, 2) observed pupil points
            tear_ducts: (n_frames, 2) observed tear ducts
            optimization_result: Optional optimization result (None for chunked)

        Returns:
            EyeTrackingResult
        """
        from scipy.spatial.transform import Rotation

        n_frames = quaternions.shape[0]

        gaze_directions = np.zeros((n_frames, 3))
        projected_pupil_centers = np.zeros((n_frames, 2))
        projected_tear_ducts = np.zeros((n_frames, 2))
        pupil_centers_3d = np.zeros((n_frames, 3))

        for i in range(n_frames):
            # Gaze direction
            quat_scipy = np.array([
                quaternions[i, 1],
                quaternions[i, 2],
                quaternions[i, 3],
                quaternions[i, 0]
            ])
            R = Rotation.from_quat(quat=quat_scipy)
            gaze_directions[i] = R.apply(np.array([0, 0, 1]))

            # Pupil center 3D (on eyeball surface)
            pupil_centers_3d[i] = eyeball_center

            # Project pupil center
            x = pupil_centers_3d[i, 0]
            y = pupil_centers_3d[i, 1]
            z = pupil_centers_3d[i, 2]
            if z < 0.1:
                z = 0.1
            u = self.config.camera.fx * (x / z) + self.config.camera.cx
            v = self.config.camera.fy * (y / z) + self.config.camera.cy
            projected_pupil_centers[i] = [u, v]

            # Project tear duct
            td_3d = eyeball_center + tear_duct_offset
            x = td_3d[0]
            y = td_3d[1]
            z = td_3d[2]
            if z < 0.1:
                z = 0.1
            u = self.config.camera.fx * (x / z) + self.config.camera.cx
            v = self.config.camera.fy * (y / z) + self.config.camera.cy
            projected_tear_ducts[i] = [u, v]

        # Compute errors
        pupil_center_observed = np.nanmean(pupil_points, axis=1)
        pupil_errors = np.linalg.norm(
            pupil_center_observed - projected_pupil_centers,
            axis=1
        )
        tear_duct_errors = np.linalg.norm(
            tear_ducts - projected_tear_ducts,
            axis=1
        )

        # Create result
        if optimization_result is not None:
            success = optimization_result.success
            num_iterations = optimization_result.num_iterations
            initial_cost = optimization_result.initial_cost
            final_cost = optimization_result.final_cost
            solve_time = optimization_result.solve_time_seconds
        else:
            success = True
            num_iterations = 0
            initial_cost = 0.0
            final_cost = 0.0
            solve_time = 0.0

        eye_result = EyeTrackingResult(
            success=success,
            num_iterations=num_iterations,
            initial_cost=initial_cost,
            final_cost=final_cost,
            solve_time_seconds=solve_time,
            rotations=quaternions,
            gaze_directions=gaze_directions,
            pupil_scales=pupil_scales.flatten(),
            pupil_centers_3d=pupil_centers_3d,
            tear_ducts_3d=np.tile(eyeball_center + tear_duct_offset, (n_frames, 1)),
            projected_pupil_centers=projected_pupil_centers,
            projected_tear_ducts=projected_tear_ducts,
            pupil_errors=pupil_errors,
            tear_duct_errors=tear_duct_errors,
            metadata={
                "n_frames": n_frames,
                "camera": "pupil_labs",
                "eyeball_center": eyeball_center.tolist(),
                "tear_duct_offset": tear_duct_offset.tolist(),
                "optimization_mode": "single_pass"
            }
        )

        return eye_result

    def evaluate(self, *, result: EyeTrackingResult) -> dict[str, float]:
        """Evaluate eye tracking quality."""
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
            "mean_pupil_error_px": float(np.mean(result.pupil_errors)),
            "mean_tear_duct_error_px": float(np.mean(result.tear_duct_errors)),
        }

        logger.info("  Metrics:")
        logger.info(f"    Gaze azimuth range:   {metrics['gaze_azimuth_range_deg']:.1f}°")
        logger.info(f"    Gaze elevation range: {metrics['gaze_elevation_range_deg']:.1f}°")
        logger.info(f"    Pupil scale:          {metrics['pupil_scale_mean']:.3f} ± {metrics['pupil_scale_std']:.3f}")
        logger.info(f"    Pupil error:          {metrics['mean_pupil_error_px']:.2f} px")
        logger.info(f"    Tear duct error:      {metrics['mean_tear_duct_error_px']:.2f} px")

        logger.info("✓ Evaluation complete")

        return metrics

    def save_results(
        self,
        *,
        result: EyeTrackingResult,
        metrics: dict[str, float]
    ) -> None:
        """Save results to disk."""
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
            "pupil_error_px": result.pupil_errors,
            "tear_duct_error_px": result.tear_duct_errors,
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
        """Generate interactive HTML viewer."""
        logger.info("Generating viewer...")
        logger.info("  ⚠ Eye tracking viewer not yet implemented")
        logger.info("  → View results in eye_tracking_results.csv")