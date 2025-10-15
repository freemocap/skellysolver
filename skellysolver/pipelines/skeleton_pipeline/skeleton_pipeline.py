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

from skellysolver.data.trajectory_dataset import TrajectoryDataset, TrajectoryND, TrajectoryType
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
    pass

class SkeletonPipeline(BasePipeline):
    """Pipeline for skeleton-based optimization."""
    config: SkeletonPipelineConfig


    def setup_and_solve(self, input_data: TrajectoryDataset) -> SolverResult:
        """Setup and solve optimization for a chunk of data.

        Args:
            input_data: TrajectoryDataset for this chunk

        Returns:
            SolverResult from optimization
        """
        n_frames = input_data.n_frames
        n_keypoints = len(self.config.skeleton.keypoints)

        logger.info(
            f"Setting up solver for chunk with {n_frames} frames, {n_keypoints} keypoints"
        )

        # 1. Initialize pose parameters
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (n_frames, 1))
        translations = np.zeros((n_frames, 3))

        # 2. Create solver
        solver = PyceresSolver(config=self.config.solver_config)

        # 3. Add pose parameters to solver
        for frame_idx in range(n_frames):
            solver.add_quaternion_parameter(
                name=f"quat_{frame_idx}",
                parameters=quaternions[frame_idx]
            )
            solver.add_parameter_block(
                name=f"trans_{frame_idx}",
                parameters=translations[frame_idx]
            )

        # 4. Build costs using refactored cost builder
        cost_builder = SkeletonCostBuilder(constraint=self.config.skeleton)

        # This now does all the data wrangling internally!
        result = cost_builder.build_all_costs(
            quaternions=quaternions,
            translations=translations,
            input_data=input_data,
            config=self.config
        )

        # 5. Add reference geometry as a parameter
        solver.add_parameter_block(
            name="reference_geometry",
            parameters=result.reference_geometry
        )

        # 6. Add all costs to solver
        for cost_info in result.costs.costs:
            solver.add_residual_block(
                cost=cost_info.cost,
                parameters=cost_info.parameters
            )

        logger.info(f"Added {len(result.costs.costs)} cost functions to solver")

        # 7. Solve
        summary, solve_time = solver.solve()

        # 8. Create result
        solver_result = SkeletonSolverResult.from_pyceres_summary(
            summary=summary,
            solve_time_seconds=solve_time,
            raw_data=input_data,
        )

        return solver_result
    def generate_viewer(self) -> None:
        """Generate interactive visualization.

        TODO: Implement viewer generation (HTML, Blender, etc.)
        """
        logger.info("Viewer generation not yet implemented")
        pass