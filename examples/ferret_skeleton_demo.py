"""Ferret skeleton tracking demo.

This demo shows how to use the skeleton pipeline with automatic chunking
for long recordings.
"""

from pathlib import Path
import logging

from skellysolver.utilities.chunk_processor import ChunkingConfig
from skellysolver.pipelines.skeleton_pipeline.skeleton_definitions.ferret_skeleton_v1 import FERRET_SKELETON_V1
from skellysolver.pipelines.skeleton_pipeline.skeleton_pipeline import SkeletonPipeline
from skellysolver.pipelines.skeleton_pipeline.skeleton_pipeline_config import SkeletonSolverConfig, \
    SkeletonPipelineConfig
from skellysolver.pipelines.base_pipeline import PipelineConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def run_ferret_skeleton_tracking(*, config: SkeletonPipelineConfig) -> None:
    """Run ferret tracking with skeleton system.

    Args:
        config: Pipeline configuration
    """
    logger.info(str(config))

    # Create pipeline
    pipeline = SkeletonPipeline.from_config(config=config)

    # Set skeleton and other required attributes
    # TODO: These should be in SkeletonPipelineConfig
    pipeline.skeleton = FERRET_SKELETON_V1
    pipeline._keypoint_to_tracked = {
        v: k for k, v in FERRET_SKELETON_V1.keypoint_to_trajectory_mapping.items()
    }

    # Run pipeline
    result = pipeline.run()

    # Print results
    if result is not None:
        logger.info(f"\n{result.summary()}")


if __name__ == "__main__":
    input_csv = Path(
        r"D:\bs\ferret_recordings\session_2025-07-01_ferret_757_EyeCameras_P33_EO5"
        r"\clips\1m_20s-2m_20s\mocap_data\output_data\processed_data"
        r"\head_spine_body_rigid_3d_xyz.csv"
    )
    output_dir = Path("output/ferret_skeleton_EO5")

    chunking_config = ChunkingConfig(
        enabled=True,
        chunk_size=500,
        overlap_size=50,
        blend_window=25,
        min_chunk_size=100,
        num_workers=None
    )

    solver_config = SkeletonSolverConfig(
        max_num_iterations=500,
        function_tolerance=1e-6,
        gradient_tolerance=1e-8,
        parameter_tolerance=1e-7,
        minimizer_progress_to_stdout=True,
    )

    config = SkeletonPipelineConfig(
        input_path=input_csv,
        output_dir=output_dir,
        solver_config=solver_config,
        parallel=chunking_config,
        input_data_confidence_threshold=0.3,
        skeleton_constrints= FERRET_SKELETON_V1
    )

    run_ferret_skeleton_tracking(config=config)