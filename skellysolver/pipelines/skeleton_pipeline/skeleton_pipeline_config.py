from skellysolver.pipelines.base_pipeline import PipelineConfig
from skellysolver.solvers.base_solver import SolverConfig
from skellysolver.solvers.constraints.skeleton_constraint import SkeletonConstraint


class SkeletonSolverConfig(SolverConfig):
    """Configuration for skeleton keypoint solving."""
    pass


class SkeletonPipelineConfig(PipelineConfig):
    """Configuration for skeleton keypoint pipeline."""
    skeleton: SkeletonConstraint
    solver_config: SkeletonSolverConfig

    # Cost weights
    rigidity_threshold: float = 1.0 # Ignore rigidity constraints below this value, for now.
    segment_length_change_threshold: float = 0.01 # Don't let segments change length by more than this fraction from initial estimate

    measurement_weight: float = 1.0 # Weight for fitting to measured data
    rigidity_weight: float = 100.0 # Weight for rigidity constraints
    smoothness_weight: float = 2.0 # Weight for temporal smoothness
    distance_anchor_weight: float = .1 # gauge freedom - low weight anchor to initial position
