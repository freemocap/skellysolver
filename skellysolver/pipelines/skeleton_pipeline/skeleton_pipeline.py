"""Skeleton pipeline with direct keypoint trajectory optimization.

This pipeline optimizes keypoint positions directly (not rigid body poses).
Each keypoint position at each frame is an optimization parameter.

Key differences from rigid body approach:
- Parameters: positions (n_frames, n_keypoints, 3) not (quaternion, translation, reference)
- Constraints apply per-frame to actual keypoint positions
- Better suited for articulated structures with flexible joints
"""

import logging

import numpy as np

from skellysolver.data.dataset_manager import save_trajectory_csv
from skellysolver.data.trajectory_dataset import TrajectoryDataset, TrajectoryND, TrajectoryType
from skellysolver.pipelines.base_pipeline import BasePipeline, PipelineConfig
from skellysolver.pipelines.skeleton_pipeline.skeleton_cost_builder import SkeletonCostBuilder
from skellysolver.solvers.base_solver import SolverResult, SolverConfig, PyceresSolver
from skellysolver.solvers.constraints.skeleton_constraint import SkeletonConstraint
from skellysolver.utilities.chunk_processor import ChunkProcessor

logger = logging.getLogger(__name__)


class SkeletonSolverConfig(SolverConfig):
    """Configuration for skeleton keypoint solving."""
    pass


class SkeletonPipelineConfig(PipelineConfig):
    """Configuration for skeleton keypoint pipeline."""
    skeleton: SkeletonConstraint
    solver_config: SkeletonSolverConfig

    # Cost weights
    rigidity_threshold: float = 0.5
    stiffness_threshold: float = 0.1
    smoothness_weight: float = 10.0
    measurement_weight: float = 1.0


class SkeletonSolverResult(SolverResult):
    """Result from skeleton keypoint optimization."""
    pass


class SkeletonPipeline(BasePipeline):
    """Pipeline for direct keypoint trajectory optimization."""
    config: SkeletonPipelineConfig

    def _initialize_positions(
        self,
        *,
        input_data: TrajectoryDataset
    ) -> np.ndarray:
        """Initialize position parameters from input data.

        Args:
            input_data: Measured trajectories

        Returns:
            (n_frames, n_keypoints, 3) initial positions
        """
        n_frames = input_data.n_frames
        n_keypoints = len(self.config.skeleton.keypoints)

        # Extract keypoint names in skeleton order
        keypoint_names = [kp.name for kp in self.config.skeleton.keypoints]

        # Initialize with measured data (NaNs will be handled)
        positions = np.zeros((n_frames, n_keypoints, 3))

        for kp_idx, kp_name in enumerate(keypoint_names):
            if kp_name not in input_data.data:
                logger.warning(f"Keypoint '{kp_name}' not found in input data")
                continue

            traj = input_data.data[kp_name]
            positions[:, kp_idx, :] = traj.values

        # Interpolate any NaNs for initialization
        for kp_idx in range(n_keypoints):
            for axis in range(3):
                data = positions[:, kp_idx, axis]
                valid_mask = ~np.isnan(data)

                if not np.any(valid_mask):
                    # No valid data - use zeros
                    positions[:, kp_idx, axis] = 0.0
                    continue

                if np.all(valid_mask):
                    # All valid - nothing to do
                    continue

                # Interpolate missing values
                valid_indices = np.where(valid_mask)[0]
                valid_values = data[valid_mask]

                missing_indices = np.where(~valid_mask)[0]
                positions[missing_indices, kp_idx, axis] = np.interp(
                    x=missing_indices,
                    xp=valid_indices,
                    fp=valid_values
                )

        logger.info(f"Initialized positions: {positions.shape}")
        return positions

    def setup_and_solve(
        self,
        input_data: TrajectoryDataset
    ) -> SolverResult:
        """Setup and solve optimization for keypoint positions.

        Args:
            input_data: Measured trajectories

        Returns:
            SolverResult with optimized positions
        """
        n_frames = input_data.n_frames
        n_keypoints = len(self.config.skeleton.keypoints)

        logger.info(
            f"Setting up solver for {n_frames} frames, {n_keypoints} keypoints"
        )

        # Initialize positions from measurements
        positions = self._initialize_positions(input_data=input_data)

        # Create solver
        solver = PyceresSolver(config=self.config.solver_config)

        # Add position parameters to solver
        logger.info("Adding position parameters...")
        for frame_idx in range(n_frames):
            for kp_idx in range(n_keypoints):
                solver.add_parameter_block(
                    name=f"pos_{frame_idx}_{kp_idx}",
                    parameters=positions[frame_idx, kp_idx]
                )


        cost_builder = SkeletonCostBuilder(constraint=self.config.skeleton)

        logger.info("Building costs...")
        result = cost_builder.build_all_costs(
            positions=positions,
            input_data=input_data,
            config=self.config
        )

        # Add all costs to solver
        logger.info("Adding costs to solver...")
        for cost_info in result.costs.costs:
            solver.add_residual_block(
                cost=cost_info.cost,
                parameters=cost_info.parameters
            )

        logger.info(f"Added {len(result.costs.costs)} cost functions")

        # Solve
        logger.info("Solving...")
        summary, solve_time = solver.solve()

        # Create optimized trajectory dataset
        optimized_data = self._create_optimized_dataset(
            positions=positions,
            input_data=input_data
        )

        # Create result
        solver_result = SkeletonSolverResult.from_pyceres_summary(
            summary=summary,
            solve_time_seconds=solve_time,
            raw_data=input_data,
        )
        solver_result.optimized_data = optimized_data

        return solver_result

    def _create_optimized_dataset(
        self,
        *,
        positions: np.ndarray,
        input_data: TrajectoryDataset
    ) -> TrajectoryDataset:
        """Create TrajectoryDataset from optimized positions.

        Args:
            positions: (n_frames, n_keypoints, 3) optimized positions
            input_data: Original input data (for metadata)

        Returns:
            TrajectoryDataset with optimized trajectories
        """
        n_frames = positions.shape[0]
        keypoint_names = [kp.name for kp in self.config.skeleton.keypoints]

        optimized_trajectories: dict[str, TrajectoryND] = {}

        for kp_idx, kp_name in enumerate(keypoint_names):
            optimized_trajectories[kp_name] = TrajectoryND(
                name=kp_name,
                values=positions[:, kp_idx, :].copy(),
                trajectory_type=TrajectoryType.POSITION,
                confidence=None,  # Confidence not preserved
                metadata={}
            )

        return TrajectoryDataset(
            data=optimized_trajectories,
            frame_indices=np.arange(n_frames),
            metadata=input_data.metadata
        )

    def run(self) -> SolverResult:
        """Run the pipeline with optional chunking."""
        logger.info(f"Data shape: {self.input_data.get_summary()}")
        logger.info(f"Pipeline config: {self.config}")

        # chunk_processor = ChunkProcessor(config=self.config.parallel)
        # self.solver_result = chunk_processor.chunk_run_pipeline(
        #     input_data=self.input_data,
        #     setup_and_solve_fn=self.setup_and_solve,
        # )
        self.solver_result = self.setup_and_solve(input_data=self.input_data)
        raw_csv_path = self.config.output_dir / "raw_trajectories.csv"
        save_trajectory_csv(
            dataset=self.input_data,
            filepath=raw_csv_path
        )
        logger.info(f"Saved raw trajectories to {raw_csv_path}")
        optimized_csv_path = self.config.output_dir / "trajectories.csv"
        save_trajectory_csv(
            dataset=self.solver_result.optimized_data,
            filepath=optimized_csv_path
        )

        logger.info(f"\n{self.solver_result.summary}")

        return self.solver_result

    def generate_viewer(self) -> None:
        """Generate interactive visualization."""
        logger.info("Viewer generation not yet implemented")
        pass