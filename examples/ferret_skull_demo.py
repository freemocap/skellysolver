"""Ferret skull tracking with passthrough spine markers.

Simple, single workflow:
1. Load data (all markers)
2. Define topology (skull only)
3. Configure passthrough markers (spine)
4. Run pipeline (automatically combines optimized skull + raw spine)
5. Visualize

NO manual CSV creation!
NO manual marker concatenation!
NO manual edge list management!
"""

from pathlib import Path
import logging

from skellysolver.core import OptimizationConfig, RigidBodyWeightConfig, ParallelConfig
from skellysolver.pipelines.rigid_body_pipeline import RigidBodyConfig, RigidBodyPipeline
from skellysolver.pipelines.topology import RigidBodyTopology

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def create_skull_topology() -> RigidBodyTopology:
    """Create skull topology with all rigid edges."""
    marker_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "base", "left_cam_tip", "right_cam_tip"
    ]

    rigid_edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 5), (4, 6), (4, 7),
        (5, 6), (5, 7),
        (6, 7)
    ]

    return RigidBodyTopology(
        marker_names=marker_names,
        rigid_edges=rigid_edges,
        name="ferret_skull"
    )


def run_ferret_tracking(*, input_csv: Path, output_dir: Path) -> None:
    """Run complete ferret tracking workflow with passthrough markers.

    Args:
        input_csv: Path to input CSV with all markers (skull + spine)
        output_dir: Directory to save results
    """

    logger.info("="*80)
    logger.info("FERRET SKULL TRACKING (with Passthrough Spine)")
    logger.info("="*80)

    # Define marker groups
    skull_markers = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "base", "left_cam_tip", "right_cam_tip"
    ]
    spine_markers = ["spine_t1", "sacrum", "tail_tip"]

    logger.info(f"\nMarker configuration:")
    logger.info(f"  Skull (optimized): {len(skull_markers)} markers")
    logger.info(f"  Spine (passthrough): {len(spine_markers)} markers")

    # Configure pipeline with passthrough markers
    config = RigidBodyConfig(
        input_path=input_csv,
        output_dir=output_dir,
        topology=create_skull_topology(),
        passthrough_markers=spine_markers,
        # scale_factor=0.001,  # mm to m
        optimization=OptimizationConfig(
            max_iterations=500,
            use_robust_loss=True,
            robust_loss_type="huber",
            robust_loss_param=2.0,
        ),
        weights=RigidBodyWeightConfig(
            lambda_data=100.0,
            lambda_rigid=1000.0,
            lambda_rot_smooth=500.0,
            lambda_trans_smooth=500.0,
        ),
        parallel=ParallelConfig()  # Auto-decides based on data size
    )

    # Run pipeline - handles everything automatically
    logger.info("\nRunning pipeline...")
    pipeline = RigidBodyPipeline(config=config)
    result = pipeline.run()

    # Done! Result contains:
    # - Optimized skull markers (8 markers)
    # - Raw passthrough spine markers (3 markers)
    # - Combined in single trajectory_data.csv
    # - Proper topology with display edges
    # - Interactive viewer

    logger.info("\n" + "="*80)
    logger.info("✓ COMPLETE")
    logger.info("="*80)
    logger.info(f"Results: {output_dir}")
    logger.info(f"  - trajectory_data.csv: {result.n_markers} markers (skull + spine)")
    logger.info(f"  - topology.json: Display edges for visualization")
    logger.info(f"  - rigid_body_viewer.html: Interactive 3D viewer")
    logger.info(f"\nOpen {output_dir / 'rigid_body_viewer.html'} to visualize!")


if __name__ == "__main__":
    input_csv = Path(
        r"D:\bs\ferret_recordings\session_2025-07-01_ferret_757_EyeCameras_P33_EO5"
        r"\clips\1m_20s-2m_20s\mocap_data\output_data\processed_data"
        r"\head_spine_body_rigid_3d_xyz.csv"
    )
    output_dir = Path("output/ferret_skull_tracking_EO5")

    run_ferret_tracking(input_csv=input_csv, output_dir=output_dir)

    # input_csv = Path(
    #     r"D:\bs\ferret_recordings\session_2025-07-11_ferret_757_EyeCameras_P43_E15__1"
    #     r"\clips\0m_37s-10m_37s\mocap_data\output_data"
    #     r"\head_spine_body_rigid_3d_xyz.csv"
    # )
    # output_dir = Path("output/ferret_skull_tracking_E15")
    #
    # run_ferret_tracking(input_csv=input_csv, output_dir=output_dir)