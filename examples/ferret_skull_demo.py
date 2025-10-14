"""Ferret skull tracking with chunked parallel optimization.

Simple, single workflow:
1. Load data
2. Create skull subset
3. Optimize skull (with automatic chunking if needed)
4. Attach raw spine
5. Save and visualize
"""

from pathlib import Path
import logging
import numpy as np

from skellysolver.core import OptimizationConfig, RigidBodyWeightConfig, ParallelConfig
from skellysolver.pipelines.rigid_body_pipeline import RigidBodyConfig, RigidBodyPipeline
from skellysolver.pipelines.topology import RigidBodyTopology
from skellysolver.data.loaders import load_trajectories
from skellysolver.pipelines.savers import save_trajectory_csv, save_topology_json

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


def create_skull_csv(*, input_csv: Path, output_csv: Path, skull_markers: list[str]) -> None:
    """Extract skull markers to separate CSV."""
    dataset = load_trajectories(filepath=input_csv, scale_factor=1.0, z_value=0.0)
    skull_data = dataset.to_array(marker_names=skull_markers)
    n_frames, n_markers, _ = skull_data.shape

    header = ["frame"] + [f"{name}_{axis}" for name in skull_markers for axis in ["x", "y", "z"]]
    output_data = np.zeros((n_frames, 1 + n_markers * 3))
    output_data[:, 0] = np.arange(n_frames)

    for i in range(n_markers):
        output_data[:, 1 + i * 3:1 + (i + 1) * 3] = skull_data[:, i, :]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(fname=output_csv, X=output_data, delimiter=",",
               header=",".join(header), comments="", fmt="%.6f")

    logger.info(f"Created skull CSV: {n_markers} markers × {n_frames} frames")


def run_ferret_tracking(input_csv: Path, output_dir: Path) -> None:
    """Run complete ferret tracking workflow."""

    logger.info("="*80)
    logger.info("FERRET SKULL TRACKING")
    logger.info("="*80)

    skull_markers = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                     "base", "left_cam_tip", "right_cam_tip"]
    spine_markers = ["spine_t1", "sacrum", "tail_tip"]

    # Step 1: Create skull-only CSV
    skull_csv = output_dir / "skull_only.csv"
    create_skull_csv(input_csv=input_csv, output_csv=skull_csv, skull_markers=skull_markers)

    # Step 2: Optimize skull (automatic chunking for long recordings)
    logger.info("\nOptimizing skull...")
    config = RigidBodyConfig(
        input_path=skull_csv,
        output_dir=output_dir / "skull_opt",
        topology=create_skull_topology(),
        optimization=OptimizationConfig(
            max_iterations=500,
            use_robust_loss=True,
            robust_loss_type="huber",
            robust_loss_param=2.0,
        ),
        weights=RigidBodyWeightConfig(
            lambda_data=100.0,
            lambda_rigid=200.0,
            lambda_rot_smooth=100.0,
            lambda_trans_smooth=100.0,
        ),
        parallel=ParallelConfig()  # Auto-decides based on data size
    )

    pipeline = RigidBodyPipeline(config=config)
    result = pipeline.run()

    # Step 3: Attach raw spine
    logger.info("\nAttaching raw spine markers...")
    full_dataset = load_trajectories(filepath=input_csv, scale_factor=1.0, z_value=0.0)
    raw_spine = full_dataset.to_array(marker_names=spine_markers)

    combined_data = np.concatenate([result.reconstructed, raw_spine], axis=1)
    combined_names = skull_markers + spine_markers
    noisy_all = full_dataset.to_array(marker_names=combined_names)

    logger.info(f"  Skull (optimized): {len(skull_markers)} markers")
    logger.info(f"  Spine (raw): {len(spine_markers)} markers")

    # Step 4: Save results
    logger.info("\nSaving results...")

    save_trajectory_csv(
        filepath=output_dir / "trajectory_data.csv",
        noisy_data=noisy_all,
        optimized_data=combined_data,
        marker_names=combined_names,
        ground_truth_data=None
    )

    n_skull = len(skull_markers)
    display_edges = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5),
        (5, 6), (5, 7), (6, 7),  # Skull edges
        (5, n_skull), (3, n_skull), (4, n_skull),  # Skull to spine
        (n_skull, n_skull+1), (n_skull+1, n_skull+2)  # Spine chain
    ]

    save_topology_json(
        filepath=output_dir / "topology.json",
        topology_dict={
            "name": "ferret_skull_plus_spine",
            "marker_names": combined_names,
            "rigid_edges": create_skull_topology().rigid_edges,
            "display_edges": display_edges,
        },
        marker_names=combined_names,
        n_frames=result.n_frames,
        has_ground_truth=False,
        soft_edges=None
    )

    # Step 5: Generate viewer
    from skellysolver.io.viewers.rigid_viewer import RigidBodyViewerGenerator
    viewer_gen = RigidBodyViewerGenerator()
    viewer_path = viewer_gen.generate(
        output_dir=output_dir,
        data_csv_path=output_dir / "trajectory_data.csv",
        topology_json_path=output_dir / "topology.json",
        video_path=None
    )

    logger.info("\n" + "="*80)
    logger.info("✓ COMPLETE")
    logger.info("="*80)
    logger.info(f"Results: {output_dir}")
    logger.info(f"Viewer: {viewer_path}")


if __name__ == "__main__":
    input_csv = Path(
        "D:/bs/ferret_recordings/2025-07-11_ferret_757_EyeCameras_P43_E15__1"
        "/clips/0m_37s-1m_37s/mocap_data/output_data"
        "/output_data_head_body_eyecam_retrain_test_v2_model_outputs_iteration_1"
        "/dlc/dlc_body_rigid_3d_xyz.csv"
    )

    output_dir = Path("output/ferret_skull_tracking")

    run_ferret_tracking(input_csv=input_csv, output_dir=output_dir)