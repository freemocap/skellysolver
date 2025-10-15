"""Skeleton-based optimization pipeline.

This pipeline connects CSV mocap data to the Skeleton model system,
automatically building costs and running optimization with automatic
parallel chunked processing for long recordings.
"""

import logging

import numpy as np
import pyceres
from pydantic import model_validator
from typing_extensions import Self

from skellysolver.data.trajectory_dataset import TrajectoryDataset, TrajectoryND
from skellysolver.pipelines.base_pipeline import BasePipeline, PipelineConfig
from skellysolver.solvers.base_solver import SolverResult, SolverConfig, PyceresSolver
from skellysolver.solvers.constraints.skeleton_constraint import SkeletonConstraint
from skellysolver.pipelines.skeleton_pipeline.skeleton_cost_builder import SkeletonCostBuilder

logger = logging.getLogger(__name__)


class SkeletonSolverConfig(SolverConfig):
    """Configuration specific to skeleton solving."""
    # Add skeleton-specific solver config here if needed
    pass


class SkeletonPipelineConfig(PipelineConfig):
    """Configuration for skeleton pipeline."""
    skeleton: SkeletonConstraint
    solver_config: SkeletonSolverConfig

    # Cost weights
    rigidity_threshold: float = 0.5
    stiffness_threshold: float = 0.1
    rotation_smoothness_weight: float = 10.0
    translation_smoothness_weight: float = 10.0
    anchor_weight: float = 1.0
    measurement_weight: float = 1.0


class SkeletonSolverResult(SolverResult):
    """Result from skeleton optimization."""
    reference_geometry: np.ndarray
    quaternions: TrajectoryND
    translations: TrajectoryND

    @classmethod
    def from_pyceres_summary(
        cls,
        *,
        summary: pyceres.SolverSummary,
        solve_time_seconds: float,
        raw_data: TrajectoryDataset,
        reference_geometry: np.ndarray,
        quaternions: np.ndarray,
        translations: np.ndarray
    ) -> Self:
        """Create result from pyceres summary.

        Args:
            summary: pyceres solver summary
            solve_time_seconds: Measured solve time
            raw_data: Original input data
            reference_geometry: Final reference geometry
            quaternions: Final quaternion trajectories
            translations: Final translation trajectories

        Returns:
            SkeletonSolverResult instance
        """
        success = summary.termination_type in [
            pyceres.TerminationType.CONVERGENCE,
            pyceres.TerminationType.USER_SUCCESS
        ]

        return cls(
            success=success,
            num_iterations=summary.num_successful_steps,
            initial_cost=summary.initial_cost,
            final_cost=summary.final_cost,
            solve_time_seconds=solve_time_seconds,
            raw_data=raw_data,
        )


class SkeletonPipeline(BasePipeline):
    """Pipeline for skeleton-based optimization."""
    config: SkeletonPipelineConfig
    _initial_reference: np.ndarray | None = None


    @model_validator(mode="after")
    def validate(self) -> Self:
        """Validate configuration."""
        # Validate all keypoints have tracked names
        missing_mapping = [
            kp for kp in self.config.skeleton.keypoints
            if kp not in self._keypoint_to_tracked
        ]
        if missing_mapping:
            names = [kp.name for kp in missing_mapping]
            raise ValueError(
                f"Keypoints missing from skeleton.keypoint_to_tracked_mapping: {names}"
            )

        # Check all required markers exist
        skeleton_keypoints = self.config.skeleton.keypoints
        tracked_names = [self._keypoint_to_tracked[kp] for kp in skeleton_keypoints]

        missing_in_dataset = set(tracked_names) - set(self.input_data.marker_names)
        if missing_in_dataset:
            available = sorted(self.input_data.marker_names)
            raise ValueError(
                f"Tracked marker names not in dataset: {missing_in_dataset}\n"
                f"Available markers: {available}"
            )

        # Initialize reference geometry once
        self._initialize_reference_geometry()

        return self

    def _initialize_reference_geometry(self) -> None:
        """Initialize reference geometry from first valid frame."""
        logger.info("Initializing reference geometry from first valid frame")
        ref_positions = []

        for kp in self.config.skeleton.keypoints:
            tracked_name = self._keypoint_to_tracked[kp]
            traj = self.input_data.data[tracked_name]

            # Use first valid position
            valid_mask = ~np.isnan(traj.values[:, 0])
            if not np.any(valid_mask):
                raise ValueError(
                    f"No valid data for tracked marker '{tracked_name}' "
                    f"(keypoint '{kp.name}')"
                )
            first_valid = traj.values[valid_mask][0]
            ref_positions.append(first_valid)

        self._initial_reference = np.concatenate(ref_positions)
        logger.info(
            f"Reference geometry: {len(self._initial_reference)} values "
            f"({len(self.config.skeleton.keypoints)} keypoints)"
        )

    def setup_and_solve(self, *, chunk_data: TrajectoryDataset) -> SolverResult:
        """Setup solver and solve for a chunk of data.

        This method is called once per chunk (or once for non-chunked data).

        Args:
            chunk_data: Trajectory data for this chunk

        Returns:
            SkeletonSolverResult from optimization
        """
        n_frames = chunk_data.n_frames
        n_keypoints = len(self.config.skeleton.keypoints)

        logger.info(f"Setting up solver for chunk with {n_frames} frames, {n_keypoints} keypoints")

        # 1. Initialize parameters for this chunk
        reference_geometry = self._initial_reference.copy()
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (n_frames, 1))
        translations = np.zeros((n_frames, 3))

        # 2. Extract measurements for this chunk
        measured_positions_per_frame = []
        for frame_idx in range(n_frames):
            frame_measurements = {}
            for kp in self.config.skeleton.keypoints:
                tracked_name = self._keypoint_to_tracked[kp]
                position = chunk_data.data[tracked_name].values[frame_idx]
                if not np.any(np.isnan(position)):
                    frame_measurements[kp.name] = position
            measured_positions_per_frame.append(frame_measurements)

        # 3. Create solver
        solver = PyceresSolver(config=self.config.solver_config)

        # 4. Add parameter blocks
        solver.add_parameter_block(
            name="reference_geometry",
            parameters=reference_geometry
        )

        for frame_idx in range(n_frames):
            solver.add_quaternion_parameter(
                name=f"quat_{frame_idx}",
                parameters=quaternions[frame_idx]
            )
            solver.add_parameter_block(
                name=f"trans_{frame_idx}",
                parameters=translations[frame_idx]
            )

        # 5. Build costs using cost builder
        cost_builder = SkeletonCostBuilder(constraint=self.config.skeleton)

        costs = cost_builder.build_all_costs(
            reference_geometry=reference_geometry,
            initial_reference=self._initial_reference,
            quaternions=quaternions,
            translations=translations,
            measured_positions_per_frame=measured_positions_per_frame,
            rigidity_threshold=self.config.rigidity_threshold,
            stiffness_threshold=self.config.stiffness_threshold,
            rotation_smoothness_weight=self.config.rotation_smoothness_weight,
            translation_smoothness_weight=self.config.translation_smoothness_weight,
            anchor_weight=self.config.anchor_weight,
            include_measurements=True,
            include_temporal_smoothness=True,
            include_anchor=True
        )

        # 6. Add costs to solver
        for cost_info in costs.all_costs():
            solver.add_residual_block(
                cost=cost_info.cost,
                parameters=cost_info.parameters
            )

        logger.info(f"Added {len(costs.all_costs())} cost functions to solver")

        # 7. Solve
        summary, solve_time = solver.solve()

        # 8. Extract optimized data
        optimized_data = self._extract_optimized_data(
            chunk_data=chunk_data,
            reference_geometry=reference_geometry,
            quaternions=quaternions,
            translations=translations
        )

        # 9. Create result
        result = SkeletonSolverResult.from_pyceres_summary(
            summary=summary,
            solve_time_seconds=solve_time,
            raw_data=chunk_data,
            optimized_data=optimized_data,
            reference_geometry=reference_geometry,
            quaternions=quaternions,
            translations=translations
        )

        return result

    def generate_viewer(self) -> None:
        """Generate interactive visualization.

        TODO: Implement viewer generation (HTML, Blender, etc.)
        """
        logger.info("Viewer generation not yet implemented")
        pass