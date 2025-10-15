"""Skeleton-based optimization pipeline for motion capture data.

This pipeline connects CSV mocap data to the Skeleton model system,
automatically building costs and running optimization.
"""

from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd
import pyceres
from pydantic import Field

from skellysolver.cost_primatives.cost_info_model import CostCollection
from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.data.primitive_objects.skeleton_cost_builder import SkeletonCostBuilder
from skellysolver.data.primitive_objects.skeleton_model import Skeleton
from skellysolver.data.primitive_objects.keypoint_model import Keypoint
from skellysolver.data.data_models import TrajectoryDataset, TrajectoryND
from skellysolver.cost_primatives.manifold_helpers import (
    get_quaternion_manifold,
    normalize_quaternion,
)
from skellysolver.io.readers.csv_reader import AutoCSVReader

logger = logging.getLogger(__name__)


class SkeletonPipelineConfig(ABaseModel):
    """Configuration for skeleton-based pipeline.

    Attributes:
        input_path: Path to input CSV file
        output_dir: Directory for outputs
        skeleton: Skeleton model defining structure
        scale_factor: Scale factor for data (e.g., 0.001 for mm->m)
        min_confidence: Minimum confidence threshold
        rigidity_threshold: Minimum rigidity to enforce (0-1)
        stiffness_threshold: Minimum stiffness to enforce (0-1)
        rotation_smoothness_weight: Weight for rotation smoothness
        translation_smoothness_weight: Weight for translation smoothness
        anchor_weight: Weight for reference anchor
        max_iterations: Maximum optimization iterations
        use_robust_loss: Whether to use robust loss
        robust_loss_type: Type of robust loss ("huber", "cauchy", "arctan")
        robust_loss_param: Robust loss parameter
    """

    input_path: Path
    output_dir: Path
    skeleton: Skeleton
    scale_factor: float = 1.0
    min_confidence: float = 0.3
    rigidity_threshold: float = 0.5
    stiffness_threshold: float = 0.1
    rotation_smoothness_weight: float = 10.0
    translation_smoothness_weight: float = 10.0
    anchor_weight: float = 1.0
    max_iterations: int = 500
    use_robust_loss: bool = True
    robust_loss_type: str = "huber"
    robust_loss_param: float = 2.0
    symmetry_pairs: list[tuple[str, str]] = Field(default_factory=list)


class SkeletonPipelineResult(ABaseModel):
    """Results from skeleton pipeline.

    Attributes:
        optimized_trajectory: Final optimized trajectory dataset
        reference_geometry: Final reference geometry (n_keypoints, 3)
        quaternions: Final rotations (n_frames, 4)
        translations: Final translations (n_frames, 3)
        costs: Cost collection used
        optimization_summary: Summary of optimization
    """

    optimized_trajectory: TrajectoryDataset
    reference_geometry: np.ndarray
    quaternions: np.ndarray
    translations: np.ndarray
    costs: CostCollection
    optimization_summary: dict[str, Any]


class SkeletonPipeline:
    """Pipeline for skeleton-based motion capture optimization.

    Workflow:
        1. Load CSV data into TrajectoryDataset
        2. Initialize reference geometry from skeleton
        3. Initialize poses (quaternions, translations)
        4. Build costs automatically from skeleton
        5. Run optimization
        6. Export results
    """

    def __init__(self, *, config: SkeletonPipelineConfig) -> None:
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Create reverse mapping: Keypoint -> tracked name
        self._keypoint_to_tracked: dict[Keypoint, str] = {
            kp: tracked_name
            for tracked_name, kp in self.config.skeleton.tracked_to_keypoint_mapping.items()
        }

        # Validate all keypoints have tracked names
        missing_mapping = [
            kp for kp in self.config.skeleton.keypoints
            if kp not in self._keypoint_to_tracked
        ]
        if missing_mapping:
            names = [kp.name for kp in missing_mapping]
            raise ValueError(
                f"Keypoints missing from skeleton.tracked_to_keypoint_mapping: {names}"
            )

    def load_data(self) -> TrajectoryDataset:
        """Load CSV data into TrajectoryDataset using automatic format detection.

        Returns:
            TrajectoryDataset with loaded data
        """
        logger.info(f"Loading data from {self.config.input_path}")

        # Use AutoCSVReader for automatic format detection
        reader = AutoCSVReader(
            min_likelihood=self.config.min_confidence,
            encoding='utf-8'
        )

        # Read and parse CSV
        reader_data = reader.read(filepath=self.config.input_path)

        # Log detected format
        format_name = reader_data.get('format', 'unknown')
        logger.info(f"Detected format: {format_name}")

        # Extract trajectories from reader output
        trajectories: dict[str, np.ndarray] = reader_data['trajectories']
        frame_indices: np.ndarray = reader_data['frame_indices']
        confidence_data: dict[str, np.ndarray] | None = reader_data.get('confidence')

        # Apply scale factor
        if self.config.scale_factor != 1.0:
            trajectories = {
                name: values * self.config.scale_factor
                for name, values in trajectories.items()
            }

        # Convert to TrajectoryDataset format
        data: dict[str, TrajectoryND] = {}
        for marker_name, values in trajectories.items():
            confidence = confidence_data.get(marker_name) if confidence_data else None

            data[marker_name] = TrajectoryND(
                name=marker_name,
                values=values,
                confidence=confidence
            )

        dataset = TrajectoryDataset(
            data=data,
            frame_indices=frame_indices
        )

        logger.info(
            f"Loaded {dataset.n_markers} markers, {dataset.n_frames} frames "
        )

        return dataset

    def initialize_reference_geometry(
        self,
        *,
        dataset: TrajectoryDataset
    ) -> np.ndarray:
        """Initialize reference geometry from first frame.

        Uses tracked_to_keypoint_mapping to map CSV marker names to skeleton keypoints.

        Args:
            dataset: Input trajectory dataset

        Returns:
            (n_keypoints * 3,) flattened reference geometry
        """
        logger.info("Initializing reference geometry")

        skeleton_keypoints = self.config.skeleton.keypoints

        # Get tracked names for all keypoints
        tracked_names = [self._keypoint_to_tracked[kp] for kp in skeleton_keypoints]

        # Check all tracked names exist in dataset
        missing_in_dataset = set(tracked_names) - set(dataset.marker_names)
        if missing_in_dataset:
            available = sorted(dataset.marker_names)
            raise ValueError(
                f"Tracked marker names not in dataset: {missing_in_dataset}\n"
                f"Available markers: {available}"
            )

        # Get first valid frame as reference
        ref_positions = []
        for kp in skeleton_keypoints:
            tracked_name = self._keypoint_to_tracked[kp]
            traj = dataset.data[tracked_name]
            # Use first valid position
            valid_mask = ~np.isnan(traj.values[:, 0])
            if not np.any(valid_mask):
                raise ValueError(
                    f"No valid data for tracked marker '{tracked_name}' "
                    f"(keypoint '{kp.name}')"
                )
            first_valid = traj.values[valid_mask][0]
            ref_positions.append(first_valid)

        ref_geometry = np.concatenate(ref_positions)

        logger.info(
            f"Reference geometry: {len(ref_geometry)} values "
            f"({len(skeleton_keypoints)} keypoints)"
        )

        return ref_geometry

    def initialize_poses(
        self,
        *,
        n_frames: int,
        reference_geometry: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize pose parameters (quaternions and translations).

        Args:
            n_frames: Number of frames
            reference_geometry: (n_keypoints * 3,) reference geometry

        Returns:
            Tuple of (quaternions, translations)
            - quaternions: (n_frames, 4) identity rotations
            - translations: (n_frames, 3) centroid of reference
        """
        logger.info("Initializing poses")

        # Identity quaternions [w, x, y, z]
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (n_frames, 1))

        # Translations at reference centroid
        ref_reshaped = reference_geometry.reshape(-1, 3)
        centroid = np.mean(ref_reshaped, axis=0)
        translations = np.tile(centroid, (n_frames, 1))

        logger.info(f"Initialized {n_frames} poses")

        return quaternions, translations

    def prepare_measurements(
        self,
        *,
        dataset: TrajectoryDataset
    ) -> list[dict[str, np.ndarray]]:
        """Prepare measurements for each frame.

        Uses tracked_to_keypoint_mapping to map CSV marker names to skeleton keypoints.
        The returned dictionaries use keypoint names as keys for consistency with
        the cost builder.

        Args:
            dataset: Input trajectory dataset

        Returns:
            List of dicts mapping keypoint names to (3,) positions per frame
        """
        skeleton_keypoints = self.config.skeleton.keypoints

        measurements_per_frame = []
        for frame_idx in range(dataset.n_frames):
            frame_measurements = {}
            for kp in skeleton_keypoints:
                tracked_name = self._keypoint_to_tracked[kp]
                traj = dataset.data[tracked_name]
                pos = traj.values[frame_idx]
                # Only add if valid (not NaN)
                if not np.isnan(pos[0]):
                    # Use keypoint name as key for consistency with cost builder
                    frame_measurements[kp.name] = pos
            measurements_per_frame.append(frame_measurements)

        return measurements_per_frame

    def build_costs(
        self,
        *,
        reference_geometry: np.ndarray,
        initial_reference: np.ndarray,
        quaternions: np.ndarray,
        translations: np.ndarray,
        measured_positions_per_frame: list[dict[str, np.ndarray]]
    ) -> CostCollection:
        """Build all costs using SkeletonCostBuilder.

        Args:
            reference_geometry: Current reference geometry
            initial_reference: Initial reference (for anchor)
            quaternions: Rotation quaternions per frame
            translations: Translation vectors per frame
            measured_positions_per_frame: Measurements per frame

        Returns:
            CostCollection with all costs
        """
        logger.info("Building costs from skeleton")

        builder = SkeletonCostBuilder(skeleton=self.config.skeleton)

        costs = builder.build_all_constraint_costs(
            reference_geometry=reference_geometry,
            initial_reference=initial_reference,
            quaternions=quaternions,
            translations=translations,
            measured_positions_per_frame=measured_positions_per_frame,
            symmetry_pairs=self.config.symmetry_pairs,
            rigidity_threshold=self.config.rigidity_threshold,
            stiffness_threshold=self.config.stiffness_threshold,
            include_measurements=True,
            include_temporal_smoothness=True,
            include_anchor=True,
            rotation_smoothness_weight=self.config.rotation_smoothness_weight,
            translation_smoothness_weight=self.config.translation_smoothness_weight,
            anchor_weight=self.config.anchor_weight
        )

        costs.print_summary()

        return costs

    def run_optimization(
            self,
            *,
            reference_geometry: np.ndarray,
            quaternions: np.ndarray,
            translations: np.ndarray,
            costs: CostCollection
    ) -> dict[str, Any]:
        """Run pyceres optimization.

        Args:
            reference_geometry: Initial reference (modified in place)
            quaternions: Initial quaternions (modified in place)
            translations: Initial translations (modified in place)
            costs: Cost collection

        Returns:
            Optimization summary
        """
        logger.info("Running optimization")

        # Create optimizer
        options = pyceres.SolverOptions()
        options.max_num_iterations = self.config.max_iterations
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        options.minimizer_progress_to_stdout = True

        problem = pyceres.Problem()

        # Add quaternion manifolds
        quat_manifold = get_quaternion_manifold()
        for quat in quaternions:
            # NOTE: pyceres is a C++ binding and requires positional arguments
            problem.add_parameter_block(quat, 4)
            problem.set_manifold(quat, quat_manifold)

        # Add costs to problem
        for cost_info in costs.costs:
            # NOTE: pyceres.add_residual_block signature: (cost, loss, parameters_list)
            # The loss function is the second argument (None for no robust loss)
            # Parameters must be passed as a list

            # Determine loss function
            loss_func = None
            if self.config.use_robust_loss and cost_info.cost_type == "measurement":
                # Create appropriate robust loss function
                if self.config.robust_loss_type == "huber":
                    loss_func = pyceres.HuberLoss(self.config.robust_loss_param)
                elif self.config.robust_loss_type == "cauchy":
                    loss_func = pyceres.CauchyLoss(self.config.robust_loss_param)
                else:  # arctan
                    loss_func = pyceres.ArctanLoss(self.config.robust_loss_param)

            # Add residual block with positional arguments
            problem.add_residual_block(
                cost_info.cost,
                loss_func,
                cost_info.parameters
            )

        # Solve
        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        logger.info("\n" + str(summary.BriefReport()))

        return {
            "success": summary.termination_type == pyceres.TerminationType.CONVERGENCE,
            "final_cost": summary.final_cost,
            "iterations": summary.iterations.size,
            "termination_type": str(summary.termination_type)
        }

    def create_optimized_trajectory(
        self,
        *,
        dataset: TrajectoryDataset,
        reference_geometry: np.ndarray,
        quaternions: np.ndarray,
        translations: np.ndarray
    ) -> TrajectoryDataset:
        """Create optimized trajectory from results.

        Output uses keypoint names (not tracked names) for clarity.

        Args:
            dataset: Original dataset (for frame indices)
            reference_geometry: Optimized reference geometry
            quaternions: Optimized quaternions
            translations: Optimized translations

        Returns:
            TrajectoryDataset with optimized positions (using keypoint names)
        """
        from scipy.spatial.transform import Rotation

        skeleton_keypoints = self.config.skeleton.keypoints
        ref_reshaped = reference_geometry.reshape(-1, 3)

        optimized_data: dict[str, TrajectoryND] = {}

        for kp_idx, kp in enumerate(skeleton_keypoints):
            ref_point = ref_reshaped[kp_idx]

            # Transform reference point for each frame
            positions = np.zeros((dataset.n_frames, 3))
            for frame_idx in range(dataset.n_frames):
                quat = quaternions[frame_idx]
                trans = translations[frame_idx]

                # Convert to scipy format [x, y, z, w]
                quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
                R = Rotation.from_quat(quat=quat_scipy).as_matrix()

                positions[frame_idx] = R @ ref_point + trans

            # Use keypoint name for output
            optimized_data[kp.name] = TrajectoryND(
                name=kp.name,
                values=positions,
                confidence=None
            )

        return TrajectoryDataset(
            data=optimized_data,
            frame_indices=dataset.frame_indices
        )

    def save_results(
        self,
        *,
        result: SkeletonPipelineResult
    ) -> None:
        """Save optimization results.

        Args:
            result: Pipeline result
        """
        logger.info(f"Saving results to {self.config.output_dir}")

        # Save trajectory as CSV
        output_csv = self.config.output_dir / "optimized_trajectory.csv"

        # Convert to DataFrame
        df_data = {"frame": result.optimized_trajectory.frame_indices}
        for name, traj in result.optimized_trajectory.data.items():
            df_data[f"{name}_x"] = traj.values[:, 0]
            df_data[f"{name}_y"] = traj.values[:, 1]
            df_data[f"{name}_z"] = traj.values[:, 2]

        df = pd.DataFrame(data=df_data)
        df.to_csv(path_or_buf=output_csv, index=False)

        logger.info(f"Saved trajectory: {output_csv}")

        # Save reference geometry
        ref_csv = self.config.output_dir / "reference_geometry.csv"
        ref_df = pd.DataFrame(
            data=result.reference_geometry.reshape(-1, 3),
            columns=["x", "y", "z"]
        )
        ref_df.insert(
            loc=0,
            column="keypoint",
            value=[kp.name for kp in self.config.skeleton.keypoints]
        )
        ref_df.to_csv(path_or_buf=ref_csv, index=False)

        logger.info(f"Saved reference: {ref_csv}")

        # Save summary
        summary_file = self.config.output_dir / "optimization_summary.txt"
        with open(file=summary_file, mode='w') as f:
            f.write("SKELETON-BASED OPTIMIZATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Skeleton: {self.config.skeleton.name}\n")
            f.write(f"Keypoints: {len(self.config.skeleton.keypoints)}\n")
            f.write(f"Segments: {len(self.config.skeleton.segments)}\n")
            f.write(f"Linkages: {len(self.config.skeleton.linkages)}\n")
            f.write(f"Frames: {len(result.quaternions)}\n\n")
            f.write("Optimization Results:\n")
            for key, value in result.optimization_summary.items():
                f.write(f"  {key}: {value}\n")

        logger.info(f"Saved summary: {summary_file}")

    def run(self) -> SkeletonPipelineResult:
        """Run complete pipeline.

        Returns:
            SkeletonPipelineResult with optimized data
        """
        logger.info("\n" + "="*80)
        logger.info("SKELETON-BASED OPTIMIZATION PIPELINE")
        logger.info("="*80)

        # 1. Load data
        dataset = self.load_data()

        # 2. Initialize reference geometry
        reference_geometry = self.initialize_reference_geometry(dataset=dataset)
        initial_reference = reference_geometry.copy()

        # 3. Initialize poses
        quaternions, translations = self.initialize_poses(
            n_frames=dataset.n_frames,
            reference_geometry=reference_geometry
        )

        # 4. Prepare measurements
        measurements_per_frame = self.prepare_measurements(dataset=dataset)

        # 5. Build costs
        costs = self.build_costs(
            reference_geometry=reference_geometry,
            initial_reference=initial_reference,
            quaternions=quaternions,
            translations=translations,
            measured_positions_per_frame=measurements_per_frame
        )

        # 6. Run optimization
        opt_summary = self.run_optimization(
            reference_geometry=reference_geometry,
            quaternions=quaternions,
            translations=translations,
            costs=costs
        )

        # 7. Create optimized trajectory
        optimized_trajectory = self.create_optimized_trajectory(
            dataset=dataset,
            reference_geometry=reference_geometry,
            quaternions=quaternions,
            translations=translations
        )

        # 8. Create result
        result = SkeletonPipelineResult(
            optimized_trajectory=optimized_trajectory,
            reference_geometry=reference_geometry,
            quaternions=quaternions,
            translations=translations,
            costs=costs,
            optimization_summary=opt_summary
        )

        # 9. Save results
        self.save_results(result=result)

        logger.info("\n" + "="*80)
        logger.info("✓ PIPELINE COMPLETE")
        logger.info("="*80)

        return result