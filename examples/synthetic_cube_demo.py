"""Synthetic data demo with cube markers - updated for refactored SkellySolver."""

from pathlib import Path
import numpy as np
import logging
from scipy.spatial.transform import Rotation

from skellysolver.core import OptimizationConfig, RigidBodyWeightConfig
from skellysolver.pipelines import RigidBodyConfig, RigidBodyPipeline
from skellysolver.pipelines.topology import RigidBodyTopology

logger = logging.getLogger(__name__)


def rotation_matrix_from_axis_angle(*, axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix from axis-angle."""
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def generate_cube_markers(*, size: float = 1.0, n_extra: int = 3) -> np.ndarray:
    """
    Generate cube vertices plus asymmetric markers.

    Args:
        size: Cube half-width
        n_extra: Number of extra asymmetric markers

    Returns:
        (8 + n_extra, 3) marker positions
    """
    s = size

    # Base cube
    cube = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ])

    # Add asymmetric markers to break symmetry
    extra_markers = []
    if n_extra >= 1:
        extra_markers.append([0.0, -s * 1.5, 0.0])
    if n_extra >= 2:
        extra_markers.append([s * 1.3, -s, -s * 0.7])
    if n_extra >= 3:
        extra_markers.append([-s * 0.8, -s * 0.8, s * 1.4])

    if extra_markers:
        return np.vstack([cube, np.array(extra_markers)])
    return cube


def generate_synthetic_trajectory(
        *,
        reference_markers: np.ndarray,
        n_frames: int ,
        noise_std: float,
        random_seed: int ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic trajectory with circular motion.

    Args:
        reference_markers: (n_markers, 3) marker configuration
        n_frames: Number of frames to generate
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        ground_truth: (n_frames, n_markers, 3)
        noisy: (n_frames, n_markers, 3)
    """
    n_markers = len(reference_markers)
    ground_truth = np.zeros((n_frames, n_markers, 3))

    for i in range(n_frames):
        t = i / n_frames

        # Circular trajectory with vertical oscillation
        radius = 3.0
        translation = np.array([
            radius * np.cos(t * 2 * np.pi),
            radius * np.sin(t * 2 * np.pi),
            1.5 * np.sin(t * 4 * np.pi)
        ])

        # Rotation around diagonal axis
        rot_axis = np.array([0.3, 1.0, 0.2])
        rot_angle = t * 4 * np.pi
        R = rotation_matrix_from_axis_angle(axis=rot_axis, angle=rot_angle)

        ground_truth[i] = (R @ reference_markers.T).T + translation

    # Add noise
    np.random.seed(seed=random_seed)
    noise = np.random.normal(loc=0, scale=noise_std, size=ground_truth.shape)
    noisy = ground_truth + noise

    return ground_truth, noisy


def create_cube_topology() -> RigidBodyTopology:
    """Create topology for cube with asymmetric markers."""

    marker_names = [
        "v0", "v1", "v2", "v3",  # Bottom face
        "v4", "v5", "v6", "v7",  # Top face
        "m0", "m1", "m2"  # Asymmetric markers
    ]

    # Cube edges (12 edges)
    rigid_edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Diagonal connections to asymmetric markers
        (0, 8), (1, 8), (4, 8),
        (1, 9), (2, 9), (5, 9),
        (4, 10), (7, 10), (0, 10),
    ]

    return RigidBodyTopology(
        marker_names=marker_names,
        rigid_edges=rigid_edges,
        name="cube_asymmetric"
    )


def save_simple_csv(
        *,
        filepath: Path,
        data: np.ndarray,
        marker_names: list[str]
) -> None:
    """
    Save trajectory data as simple wide-format CSV.

    Args:
        filepath: Output CSV path
        data: (n_frames, n_markers, 3) trajectory data
        marker_names: List of marker names
    """
    n_frames, n_markers, _ = data.shape

    # Create header
    header_parts = ["frame"]
    for name in marker_names:
        header_parts.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    # Create data array
    output_data = np.zeros((n_frames, 1 + n_markers * 3))
    output_data[:, 0] = np.arange(n_frames)

    for i, name in enumerate(marker_names):
        output_data[:, 1 + i * 3:1 + (i + 1) * 3] = data[:, i, :]

    # Save
    np.savetxt(
        filepath,
        output_data,
        delimiter=",",
        header=",".join(header_parts),
        comments="",
        fmt="%.6f"
    )


def compute_reconstruction_error(
        *,
        ground_truth: np.ndarray,
        reconstructed: np.ndarray
) -> dict[str, float]:
    """
    Compute reconstruction metrics.

    Args:
        ground_truth: (n_frames, n_markers, 3) ground truth
        reconstructed: (n_frames, n_markers, 3) reconstructed

    Returns:
        Dictionary of metrics
    """
    errors = np.linalg.norm(reconstructed - ground_truth, axis=2)

    return {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "max_error": float(np.max(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def run_synthetic_demo() -> None:
    """Run complete synthetic data demonstration."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    logger.info("=" * 80)
    logger.info("SYNTHETIC CUBE DEMO")
    logger.info("=" * 80)

    # Generate synthetic data
    logger.info("\nGenerating synthetic data...")
    reference_markers = generate_cube_markers(size=1.0, n_extra=3)
    ground_truth, noisy = generate_synthetic_trajectory(
        reference_markers=reference_markers,
        n_frames=200,
        noise_std=0.1,
        random_seed=42
    )

    logger.info(f"  Generated {len(noisy)} frames")
    logger.info(f"  Number of markers: {len(reference_markers)}")


    # Setup output directory
    output_dir = Path("output/synthetic_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create topology
    topology = create_cube_topology()

    # Save noisy input data
    input_csv = output_dir / "input_data.csv"
    save_simple_csv(
        filepath=input_csv,
        data=noisy,
        marker_names=topology.marker_names
    )
    logger.info(f"  Saved input data to {input_csv}")

    # Configure pipeline
    logger.info("\nConfiguring pipeline...")
    config = RigidBodyConfig(
        input_path=input_csv,
        output_dir=output_dir,
        topology=topology,
        optimization=OptimizationConfig(
            max_iterations=300,
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
    )

    # Run pipeline
    logger.info("\nRunning optimization pipeline...")
    pipeline = RigidBodyPipeline(config=config)
    result = pipeline.run()

    # Compute reconstruction error against ground truth
    logger.info("\nEvaluating reconstruction quality...")
    if result.reconstructed is not None:
        metrics = compute_reconstruction_error(
            ground_truth=ground_truth,
            reconstructed=result.reconstructed
        )

        logger.info("\nReconstruction Metrics:")
        logger.info(f"  Mean error:   {metrics['mean_error']:.4f}")
        logger.info(f"  Median error: {metrics['median_error']:.4f}")
        logger.info(f"  Std error:    {metrics['std_error']:.4f}")
        logger.info(f"  Max error:    {metrics['max_error']:.4f}")
        logger.info(f"  RMSE:         {metrics['rmse']:.4f}")

        # Save metrics
        import json
        with open(output_dir / "reconstruction_vs_ground_truth_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Open {output_dir / 'rigid_body_viewer.html'} to visualize")


if __name__ == "__main__":
    run_synthetic_demo()