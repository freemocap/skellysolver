"""Ferret skeleton tracking demo using new Skeleton system.

This replaces the old RigidBodyPipeline approach with the new
skeleton-based automatic cost building system.

Key improvements:
- Automatic cost generation from Skeleton definition
- Type-safe cost functions with CostInfo models
- Flexible segment rigidity and linkage stiffness
- Automatic symmetry constraints
- Cleaner, more maintainable code
"""

from pathlib import Path
import logging

from skellysolver.solvers.mocap_solver.mocap_models.ferret_skeleton_v1 import FERRET_SKELETON_V1
from skellysolver.solvers.skeleton_pipeline import SkeletonPipelineConfig, SkeletonPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def run_ferret_skeleton_tracking(
    *,
    input_csv: Path,
    output_dir: Path
) -> None:
    """Run ferret tracking with new skeleton system.
    
    Args:
        input_csv: Path to input CSV with marker data
        output_dir: Directory to save results
    """
    
    logger.info("="*80)
    logger.info("FERRET SKELETON TRACKING (New System)")
    logger.info("="*80)
    
    # Define bilateral symmetry pairs for ferret
    symmetry_pairs = [
        ("left_eye_camera", "right_eye_camera"),
        ("left_eye_center", "right_eye_center"),
        ("left_acoustic_meatus", "right_acoustic_meatus"),
    ]
    
    # Configure pipeline
    config = SkeletonPipelineConfig(
        input_path=input_csv,
        output_dir=output_dir,
        skeleton=FERRET_SKELETON_V1,
        scale_factor=0.001,  # mm to m
        min_confidence=0.3,
        # Segment rigidity threshold - only enforce edges for segments above this
        rigidity_threshold=0.5,  # Skull is 1.0, spine segments are 0.2
        # Linkage stiffness threshold - only enforce linkages above this
        stiffness_threshold=0.1,  # Eye cameras are 0.99, spine linkages are 0.1
        # Smoothness weights
        rotation_smoothness_weight=10.0,
        translation_smoothness_weight=10.0,
        # Anchor weight (prevent reference drift)
        anchor_weight=1.0,
        # Optimization settings
        max_iterations=500,
        use_robust_loss=True,
        robust_loss_type="huber",
        robust_loss_param=2.0,
        # Symmetry
        symmetry_pairs=symmetry_pairs,
    )
    
    # Print configuration
    logger.info("\nSkeleton Configuration:")
    logger.info(f"  Name: {FERRET_SKELETON_V1.name}")
    logger.info(f"  Keypoints: {len(FERRET_SKELETON_V1.keypoints)}")
    logger.info(f"  Segments: {len(FERRET_SKELETON_V1.segments)}")
    for seg in FERRET_SKELETON_V1.segments:
        logger.info(f"    - {seg.name}: rigidity={seg.rigidity:.2f}")
    logger.info(f"  Linkages: {len(FERRET_SKELETON_V1.linkages)}")
    for link in FERRET_SKELETON_V1.linkages:
        logger.info(f"    - {link.name}: stiffness={link.stiffness:.2f}")
    logger.info(f"  Chains: {len(FERRET_SKELETON_V1.chains)}")
    
    logger.info("\nCost Configuration:")
    logger.info(f"  Rigidity threshold: {config.rigidity_threshold}")
    logger.info(f"  Stiffness threshold: {config.stiffness_threshold}")
    logger.info(f"  Rotation smoothness: {config.rotation_smoothness_weight}")
    logger.info(f"  Translation smoothness: {config.translation_smoothness_weight}")
    logger.info(f"  Symmetry pairs: {len(symmetry_pairs)}")
    
    # Run pipeline - automatically:
    # 1. Loads CSV data
    # 2. Initializes reference geometry
    # 3. Builds all costs from skeleton
    # 4. Runs optimization
    # 5. Saves results
    pipeline = SkeletonPipeline(config=config)
    result = pipeline.run()
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Optimized trajectory: {output_dir / 'optimized_trajectory.csv'}")
    logger.info(f"Reference geometry: {output_dir / 'reference_geometry.csv'}")
    logger.info(f"Optimization summary: {output_dir / 'optimization_summary.txt'}")
    
    logger.info("\nOptimization Summary:")
    for key, value in result.optimization_summary.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nCost Summary:")
    summary = result.costs.get_summary()
    logger.info(f"  Total costs: {summary['total_costs']}")
    logger.info(f"  Total weight: {summary['total_weight']:.1f}")
    logger.info("\n  By type:")
    for cost_type, info in summary["by_type"].items():
        logger.info(
            f"    {cost_type:25s}: {info['count']:4d} costs, "
            f"weight={info['total_weight']:8.1f} "
            f"(avg={info['avg_weight']:.2f})"
        )
    
    logger.info("\n" + "="*80)
    logger.info("✓ COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    # Example 1
    input_csv = Path(
        r"D:\bs\ferret_recordings\session_2025-07-01_ferret_757_EyeCameras_P33_EO5"
        r"\clips\1m_20s-2m_20s\mocap_data\output_data\processed_data"
        r"\head_spine_body_rigid_3d_xyz.csv"
    )
    output_dir = Path("output/ferret_skeleton_EO5")
    
    run_ferret_skeleton_tracking(
        input_csv=input_csv,
        output_dir=output_dir
    )
    
    # Example 2
    # input_csv = Path(
    #     r"D:\bs\ferret_recordings\session_2025-07-11_ferret_757_EyeCameras_P43_E15__1"
    #     r"\clips\0m_37s-10m_37s\mocap_data\output_data"
    #     r"\head_spine_body_rigid_3d_xyz.csv"
    # )
    # output_dir = Path("output/ferret_skeleton_E15")
    #
    # run_ferret_skeleton_tracking(
    #     input_csv=input_csv,
    #     output_dir=output_dir
    # )
