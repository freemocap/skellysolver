"""Demo script for rigid body tracking pipeline.

Generates synthetic data with a cube + asymmetric markers,
adds noise, and optimizes using bundle adjustment.
"""

import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from skellysolver.pipelines.rigid_body_pipeline.rigid_body_pipeline import (
    RigidBodyPipeline,
    RigidBodyPipelineConfig
)
from skellysolver.pipelines.rigid_body_pipeline.rigid_body_topology import RigidBodyTopology
from skellysolver.solvers.base_solver import SolverConfig
from skellysolver.utilities.chunk_processor import ChunkingConfig
from skellysolver.data.trajectory_dataset import TrajectoryDataset, TrajectoryND, TrajectoryType
from skellysolver.data.dataset_manager import save_trajectory_csv

logger = logging.getLogger(__name__)


def generate_cube_markers(
    *,
    size: float = 1.0,
    n_extra: int = 3
) -> np.ndarray:
    """Generate cube vertices plus asymmetric markers.
    
    Args:
        size: Cube half-width
        n_extra: Number of extra asymmetric markers to break symmetry
        
    Returns:
        (8 + n_extra, 3) marker positions
    """
    s = size
    
    # Base cube vertices
    cube = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Bottom face
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],      # Top face
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


def rotation_matrix_from_axis_angle(
    *,
    axis: np.ndarray,
    angle: float
) -> np.ndarray:
    """Create rotation matrix from axis-angle representation.
    
    Args:
        axis: Rotation axis (will be normalized)
        angle: Rotation angle in radians
        
    Returns:
        (3, 3) rotation matrix
    """
    axis_normalized = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(rotvec=axis_normalized * angle).as_matrix()


def generate_synthetic_trajectory(
    *,
    reference_markers: np.ndarray,
    n_frames: int = 200,
    noise_std: float = 0.1,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic trajectory with circular motion.
    
    Args:
        reference_markers: (n_markers, 3) marker configuration
        n_frames: Number of frames to generate
        noise_std: Standard deviation of Gaussian noise (in meters)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - ground_truth: (n_frames, n_markers, 3) clean trajectory
        - noisy: (n_frames, n_markers, 3) noisy measurements
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
        
        # Transform markers
        ground_truth[i] = (R @ reference_markers.T).T + translation
    
    # Add Gaussian noise
    np.random.seed(seed=random_seed)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=ground_truth.shape)
    noisy = ground_truth + noise
    
    return ground_truth, noisy


def create_cube_topology() -> RigidBodyTopology:
    """Create topology for cube with asymmetric markers.
    
    Returns:
        RigidBodyTopology with cube edges + connections to asymmetric markers
    """
    marker_names = [
        "v0", "v1", "v2", "v3",  # Bottom face
        "v4", "v5", "v6", "v7",  # Top face
        "m0", "m1", "m2"          # Asymmetric markers
    ]
    
    # Cube edges (12 edges)
    rigid_edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Connections to asymmetric markers
        (0, 8), (1, 8), (4, 8),
        (1, 9), (2, 9), (5, 9),
        (4, 10), (7, 10), (0, 10),
    ]
    
    return RigidBodyTopology(
        marker_names=marker_names,
        rigid_edges=rigid_edges,
        name="cube_asymmetric"
    )


def create_trajectory_dataset(
    *,
    data_array: np.ndarray,
    marker_names: list[str]
) -> TrajectoryDataset:
    """Convert numpy array to TrajectoryDataset.
    
    Args:
        data_array: (n_frames, n_markers, 3) trajectory data
        marker_names: List of marker names
        
    Returns:
        TrajectoryDataset
    """
    n_frames, n_markers, _ = data_array.shape
    
    trajectories = {}
    for marker_idx, marker_name in enumerate(marker_names):
        trajectories[marker_name] = TrajectoryND(
            name=marker_name,
            data=data_array[:, marker_idx, :],
            trajectory_type=TrajectoryType.POSITION,
            confidence=None,
            metadata={}
        )
    
    return TrajectoryDataset(
        data=trajectories,
        frame_indices=np.arange(n_frames),
        metadata={"source": "synthetic_generation"}
    )


def run_synthetic_demo() -> None:
    """Run complete synthetic demonstration."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-8s | %(name)-30s | %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("SYNTHETIC RIGID BODY DEMO")
    logger.info("=" * 80)
    
    # 1. Generate synthetic data
    logger.info("\nGenerating synthetic data...")
    reference_markers = generate_cube_markers(size=1.0, n_extra=3)
    ground_truth, noisy = generate_synthetic_trajectory(
        reference_markers=reference_markers,
        n_frames=200,
        noise_std=0.1,  # 100mm noise
        random_seed=42
    )
    
    logger.info(f"  Generated {len(noisy)} frames")
    logger.info(f"  Markers: {len(reference_markers)}")
    logger.info(f"  Noise level: σ=100mm")
    
    # 2. Create topology
    logger.info("\nCreating topology...")
    topology = create_cube_topology()
    logger.info(f"  {topology}")
    
    # 3. Convert to TrajectoryDataset
    logger.info("\nConverting to TrajectoryDataset...")
    noisy_dataset = create_trajectory_dataset(
        data_array=noisy,
        marker_names=topology.marker_names
    )
    ground_truth_dataset = create_trajectory_dataset(
        data_array=ground_truth,
        marker_names=topology.marker_names
    )
    
    # 4. Save input data
    output_dir = Path("output/synthetic_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_csv = output_dir / "input_data.csv"
    save_trajectory_csv(dataset=noisy_dataset, filepath=input_csv)
    logger.info(f"  Saved input data: {input_csv}")
    
    # 5. Create pipeline configuration
    logger.info("\nCreating pipeline configuration...")
    config = RigidBodyPipelineConfig(
        input_path=input_csv,
        output_dir=output_dir,
        topology=topology,
        solver_config=SolverConfig(
            max_num_iterations=300,
            function_tolerance=1e-9,
            gradient_tolerance=1e-11,
            parameter_tolerance=1e-10,
            minimizer_progress_to_stdout=True
        ),
        parallel=ChunkingConfig(
            enabled=False  # Small dataset, no chunking needed
        ),
        measurement_weight=100.0,
        rigidity_weight=500.0,
        smoothness_weight=200.0,
        soft_weight=10.0,
        anchor_weight=10.0
    )
    
    # 6. Create and run pipeline
    logger.info("\nCreating pipeline...")
    pipeline = RigidBodyPipeline.from_config(config=config)
    
    logger.info("\nRunning pipeline...")
    # FIX: Capture the returned result
    solver_result = pipeline.setup_problem_and_solve()

    # 7. Evaluate results
    if solver_result is not None and solver_result.success:
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION")
        logger.info("=" * 80)

        # Compute accuracy vs ground truth
        accuracy_metrics = solver_result.compute_reconstruction_accuracy(
            ground_truth_data=ground_truth_dataset
        )

        if accuracy_metrics:
            logger.info("\nAccuracy vs Ground Truth:")
            logger.info(f"  Raw error:  {accuracy_metrics['raw_position_error_mean_mm']:.2f}mm "
                       f"(max: {accuracy_metrics['raw_position_error_max_mm']:.2f}mm)")
            logger.info(f"  Opt error:  {accuracy_metrics['opt_position_error_mean_mm']:.2f}mm "
                       f"(max: {accuracy_metrics['opt_position_error_max_mm']:.2f}mm)")
            logger.info(f"  Improvement: {accuracy_metrics['improvement_percent']:.1f}%")

        # Save results
        output_csv = output_dir / "optimized_data.csv"
        save_trajectory_csv(dataset=solver_result.optimized_data, filepath=output_csv)
        logger.info(f"\nSaved optimized data: {output_csv}")

        logger.info("\n" + "=" * 80)
        logger.info("DEMO COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"  - input_data.csv (noisy measurements)")
        logger.info(f"  - optimized_data.csv (optimized results)")
    else:
        logger.error("Pipeline failed to produce results!")
        if solver_result is not None:
            logger.error(f"Solver status: {solver_result.summary.termination_type}")


if __name__ == "__main__":
    run_synthetic_demo()