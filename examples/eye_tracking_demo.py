"""Eye tracking demo using the new skellysolver framework with PARALLEL CHUNKING! 🚀

Simple workflow:
1. Load DeepLabCut pupil data
2. Configure camera and eye model
3. Configure parallel processing (auto-chunks for long recordings)
4. Run pipeline (automatically optimizes gaze + pupil dilation)
5. View results
"""

from pathlib import Path
import logging

from skellysolver.core import OptimizationConfig, EyeTrackingWeightConfig, ParallelConfig
from skellysolver.pipelines import CameraIntrinsics, EyeTrackingConfig, EyeTrackingPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def run_eye_tracking(*, input_csv: Path, output_dir: Path) -> None:
    """Run complete eye tracking workflow with automatic parallel chunking.

    Args:
        input_csv: Path to DeepLabCut CSV with pupil points (p1-p8) and tear_duct
        output_dir: Directory to save results
    """

    logger.info("="*80)
    logger.info("EYE TRACKING DEMO (WITH PARALLEL CHUNKING)")
    logger.info("="*80)

    # Configure pipeline
    config = EyeTrackingConfig(
        input_path=input_csv,
        output_dir=output_dir,
        camera=CameraIntrinsics.create_pupil_labs_camera(),
        optimization=OptimizationConfig(
            max_iterations=500,
            use_robust_loss=True,
            robust_loss_type="huber",
            robust_loss_param=2.0,
        ),
        weights=EyeTrackingWeightConfig(
            lambda_pupil=1.0,
            lambda_tear_duct=1.0,
            lambda_rot_smooth=10.0,
            lambda_scalar_smooth=5.0,
        ),
        min_confidence=0.3,
        min_pupil_points=6,
        # Eye model parameters
        eyeball_distance_mm=20.0,
        base_semi_major_mm=2.0,
        base_semi_minor_mm=1.5,
        # Parallel processing - auto-decides based on data size
        parallel=ParallelConfig(
            enabled=True,  # Enable parallel processing
            chunk_size=1000,  # Frames per chunk
            overlap_size=100,  # Overlap between chunks
            blend_window=50,  # Smooth blending window
        )
    )

    # Run pipeline - handles everything automatically
    logger.info("\nRunning pipeline...")
    logger.info("(Will automatically use parallel chunking for long recordings)")

    pipeline = EyeTrackingPipeline(config=config)
    result = pipeline.run()

    # Done! Results contain:
    # - Gaze directions for each frame
    # - Pupil dilation scales
    # - Reprojection errors
    # - Eye model parameters

    logger.info("\n" + "="*80)
    logger.info("✓ COMPLETE")
    logger.info("="*80)
    logger.info(f"Results: {output_dir}")
    logger.info(f"  - eye_tracking_results.csv: Gaze directions and pupil scales")
    logger.info(f"  - metrics.json: Evaluation metrics")
    logger.info(f"  - quaternions.npy: Eye orientations")
    logger.info(f"\nOptimization mode: {result.metadata.get('optimization_mode', 'single_pass')}")


def run_eye_tracking_sequential(*, input_csv: Path, output_dir: Path) -> None:
    """Run eye tracking with SEQUENTIAL chunking (no parallelism).

    Useful for debugging or systems with limited memory.

    Args:
        input_csv: Path to DeepLabCut CSV
        output_dir: Directory to save results
    """

    logger.info("="*80)
    logger.info("EYE TRACKING DEMO (SEQUENTIAL CHUNKING)")
    logger.info("="*80)

    config = EyeTrackingConfig(
        input_path=input_csv,
        output_dir=output_dir,
        camera=CameraIntrinsics.create_pupil_labs_camera(),
        optimization=OptimizationConfig(
            max_iterations=500,
            use_robust_loss=True,
            robust_loss_type="huber",
            robust_loss_param=2.0,
        ),
        weights=EyeTrackingWeightConfig(
            lambda_pupil=1.0,
            lambda_tear_duct=1.0,
            lambda_rot_smooth=10.0,
            lambda_scalar_smooth=5.0,
        ),
        min_confidence=0.3,
        min_pupil_points=6,
        eyeball_distance_mm=20.0,
        base_semi_major_mm=2.0,
        base_semi_minor_mm=1.5,
        # Sequential chunking
        parallel=ParallelConfig(
            enabled=False,  # Disable parallelism
            chunk_size=1000,
            overlap_size=100,
            blend_window=50
        )
    )

    pipeline = EyeTrackingPipeline(config=config)
    result = pipeline.run()

    logger.info("\n✓ Complete (sequential mode)")


def run_eye_tracking_single_pass(*, input_csv: Path, output_dir: Path) -> None:
    """Run eye tracking with NO chunking (single optimization pass).

    Best for short recordings (< 2000 frames).

    Args:
        input_csv: Path to DeepLabCut CSV
        output_dir: Directory to save results
    """

    logger.info("="*80)
    logger.info("EYE TRACKING DEMO (SINGLE-PASS, NO CHUNKING)")
    logger.info("="*80)

    config = EyeTrackingConfig(
        input_path=input_csv,
        output_dir=output_dir,
        camera=CameraIntrinsics.create_pupil_labs_camera(),
        optimization=OptimizationConfig(
            max_iterations=500,
            use_robust_loss=True,
            robust_loss_type="huber",
            robust_loss_param=2.0,
        ),
        weights=EyeTrackingWeightConfig(
            lambda_pupil=1.0,
            lambda_tear_duct=1.0,
            lambda_rot_smooth=10.0,
            lambda_scalar_smooth=5.0,
        ),
        min_confidence=0.3,
        min_pupil_points=6,
        eyeball_distance_mm=20.0,
        base_semi_major_mm=2.0,
        base_semi_minor_mm=1.5,
        # No chunking - will force single-pass even for long recordings
        parallel=ParallelConfig(
            enabled=False,
            min_chunk_threshold=1_000_000  # Very high threshold
        )
    )

    pipeline = EyeTrackingPipeline(config=config)
    result = pipeline.run()

    logger.info("\n✓ Complete (single-pass mode)")


if __name__ == "__main__":
    # Example usage - replace with your actual file path
    input_csv = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1"
        r"\clips\0m_37s-1m_37s\eye_data\dlc_output\model_outputs_iteration_11"
        r"\eye0_clipped_4354_11523DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )
    output_dir = Path("output/eye_tracking_demo")

    # Default: Auto-decides between single-pass and parallel chunking
    run_eye_tracking(input_csv=input_csv, output_dir=output_dir)

    # Or explicitly use sequential chunking:
    # run_eye_tracking_sequential(input_csv=input_csv, output_dir=output_dir / "sequential")

    # Or force single-pass mode:
    # run_eye_tracking_single_pass(input_csv=input_csv, output_dir=output_dir / "single_pass")