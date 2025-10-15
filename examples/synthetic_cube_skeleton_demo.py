"""Synthetic cube skeleton demo with wiggling tail.

Generates synthetic data of a cube with a flexible tail chain,
adds noise, and reconstructs using the skeleton pipeline.
"""

from pathlib import Path
import numpy as np
import logging
from scipy.spatial.transform import Rotation

from skellysolver.data.dataset_manager import save_trajectory_csv
from skellysolver.data.trajectory_dataset import TrajectoryDataset, TrajectoryType, TrajectoryND
from skellysolver.pipelines.skeleton_pipeline.skeleton_definitions.synthetic_cube_skeleton_v1 import \
    SYNTHETIC_CUBE_SKELETON
from skellysolver.pipelines.skeleton_pipeline.skeleton_pipeline import (
    SkeletonPipelineConfig, 
    SkeletonPipeline, 
    SkeletonSolverConfig
)
from skellysolver.utilities.chunk_processor import ChunkingConfig

logger = logging.getLogger(__name__)


def generate_cube_with_tail_reference(
    *, 
    cube_size: float = 1.0, 
    tail_segment_length: float = 1.5
) -> dict[str, np.ndarray]:
    """Generate reference positions for cube vertices and tail segments.
    
    Args:
        cube_size: Half-width of cube
        tail_segment_length: Length of each tail segment
        
    Returns:
        Dictionary mapping keypoint names to (3,) positions
    """
    s = cube_size
    
    # Cube vertices (8 corners)
    reference: dict[str, np.ndarray] = {
        "cube_v0": np.array([-s, -s, -s]),  # front-bottom-left
        "cube_v1": np.array([s, -s, -s]),   # front-bottom-right
        "cube_v2": np.array([s, s, -s]),    # front-top-right
        "cube_v3": np.array([-s, s, -s]),   # front-top-left
        "cube_v4": np.array([-s, -s, s]),   # back-bottom-left
        "cube_v5": np.array([s, -s, s]),    # back-bottom-right
        "cube_v6": np.array([s, s, s]),     # back-top-right
        "cube_v7": np.array([-s, s, s]),    # back-top-left
        "cube_asymmetric_v8": np.array([0.0, 0.0, -s]),  # front-center (asymmetric point)
    }
    
    # Tail extends from back face (positive z direction)
    # Start from center of back face
    back_center = np.array([0.0, 0.0, s])
    
    reference["tail_base"] = back_center + np.array([0.0, 0.0, 0.2])
    reference["tail_mid1"] = reference["tail_base"] + np.array([0.0, 0.0, tail_segment_length])
    reference["tail_mid2"] = reference["tail_mid1"] + np.array([0.0, 0.0, tail_segment_length])
    reference["tail_tip"] = reference["tail_mid2"] + np.array([0.0, 0.0, tail_segment_length])
    
    return reference


def generate_wiggle_rotation(
    *, 
    t: float, 
    frequency: float, 
    amplitude: float
) -> np.ndarray:
    """Generate a wiggling rotation matrix.
    
    Args:
        t: Time parameter [0, 1]
        frequency: Wiggle frequency (cycles per time unit)
        amplitude: Wiggle amplitude in radians
        
    Returns:
        (3, 3) rotation matrix
    """
    # Wiggle around x and y axes
    angle_x = amplitude * np.sin(2 * np.pi * frequency * t)
    angle_y = amplitude * np.cos(2 * np.pi * frequency * t * 1.3)
    
    R_x = Rotation.from_euler(seq='x', angles=angle_x).as_matrix()
    R_y = Rotation.from_euler(seq='y', angles=angle_y).as_matrix()
    
    return R_y @ R_x


def generate_synthetic_trajectory(
    *,
    reference_positions: dict[str, np.ndarray],
    n_frames: int,
    noise_std: float,
    random_seed: int,
    wiggle_amplitude: float = 0.3,
    wiggle_frequency: float = 2.0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate synthetic trajectory with global motion and tail wiggling.
    
    Args:
        reference_positions: Dictionary of reference keypoint positions
        n_frames: Number of frames to generate
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility
        wiggle_amplitude: Amplitude of tail wiggling (radians)
        wiggle_frequency: Frequency of tail wiggling
        
    Returns:
        ground_truth: Dictionary mapping keypoint names to (n_frames, 3) positions
        noisy: Dictionary mapping keypoint names to (n_frames, 3) noisy positions
    """
    keypoint_names = list(reference_positions.keys())
    
    # Initialize storage
    ground_truth: dict[str, np.ndarray] = {
        name: np.zeros((n_frames, 3)) 
        for name in keypoint_names
    }
    
    # Cube keypoints (rigid body motion only)
    cube_keypoints = [f"cube_v{i}" for i in range(8)]
    
    # Tail keypoints (rigid + wiggle)
    tail_keypoints = ["tail_base", "tail_mid1", "tail_mid2", "tail_tip"]
    
    for frame_idx in range(n_frames):
        t = frame_idx / n_frames
        
        # Global circular trajectory
        radius = 3.0
        translation = np.array([
            radius * np.cos(t * 2 * np.pi),
            radius * np.sin(t * 2 * np.pi),
            1.5 * np.sin(t * 4 * np.pi)
        ])
        
        # Global rotation
        rot_axis = np.array([0.3, 1.0, 0.2])
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        rot_angle = t * 4 * np.pi
        R_global = Rotation.from_rotvec(rotvec=rot_axis * rot_angle).as_matrix()
        
        # Transform cube (rigid motion only)
        for kp_name in cube_keypoints:
            ref_pos = reference_positions[kp_name]
            ground_truth[kp_name][frame_idx] = R_global @ ref_pos + translation
        
        # Transform tail base (rigid motion only)
        ref_base = reference_positions["tail_base"]
        base_global = R_global @ ref_base + translation
        ground_truth["tail_base"][frame_idx] = base_global
        
        # Transform tail segments with cumulative wiggling
        # Each segment wiggles relative to its parent
        
        # Tail segment 0: base -> mid1
        R_wiggle_0 = generate_wiggle_rotation(
            t=t, 
            frequency=wiggle_frequency, 
            amplitude=wiggle_amplitude * 0.5
        )
        ref_mid1_local = reference_positions["tail_mid1"] - reference_positions["tail_base"]
        mid1_wiggled = base_global + R_global @ (R_wiggle_0 @ ref_mid1_local)
        ground_truth["tail_mid1"][frame_idx] = mid1_wiggled
        
        # Tail segment 1: mid1 -> mid2
        R_wiggle_1 = generate_wiggle_rotation(
            t=t, 
            frequency=wiggle_frequency * 1.2, 
            amplitude=wiggle_amplitude * 0.8
        )
        ref_mid2_local = reference_positions["tail_mid2"] - reference_positions["tail_mid1"]
        mid2_wiggled = mid1_wiggled + R_global @ R_wiggle_0 @ (R_wiggle_1 @ ref_mid2_local)
        ground_truth["tail_mid2"][frame_idx] = mid2_wiggled
        
        # Tail segment 2: mid2 -> tip
        R_wiggle_2 = generate_wiggle_rotation(
            t=t, 
            frequency=wiggle_frequency * 1.5, 
            amplitude=wiggle_amplitude
        )
        ref_tip_local = reference_positions["tail_tip"] - reference_positions["tail_mid2"]
        tip_wiggled = mid2_wiggled + R_global @ R_wiggle_0 @ R_wiggle_1 @ (R_wiggle_2 @ ref_tip_local)
        ground_truth["tail_tip"][frame_idx] = tip_wiggled
    
    # Add noise
    np.random.seed(seed=random_seed)
    noisy: dict[str, np.ndarray] = {}
    for name, positions in ground_truth.items():
        noise = np.random.normal(loc=0, scale=noise_std, size=positions.shape)
        noisy[name] = positions + noise
    
    return ground_truth, noisy





def compute_reconstruction_metrics(
    *,
    ground_truth: dict[str, np.ndarray],
    reconstructed: dict[str, np.ndarray]
) -> dict[str, float]:
    """Compute reconstruction error metrics.
    
    Args:
        ground_truth: Ground truth positions per keypoint
        reconstructed: Reconstructed positions per keypoint
        
    Returns:
        Dictionary of metrics
    """
    all_errors = []
    
    for kp_name in ground_truth.keys():
        gt_pos = ground_truth[kp_name]
        rec_pos = reconstructed[kp_name]
        
        errors = np.linalg.norm(rec_pos - gt_pos, axis=1)
        all_errors.extend(errors)
    
    all_errors = np.array(all_errors)
    
    return {
        "mean_error": float(np.mean(all_errors)),
        "median_error": float(np.median(all_errors)),
        "std_error": float(np.std(all_errors)),
        "max_error": float(np.max(all_errors)),
        "rmse": float(np.sqrt(np.mean(all_errors ** 2))),
    }


def run_synthetic_cube_skeleton_demo() -> None:
    """Run complete synthetic cube skeleton demonstration."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("SYNTHETIC CUBE SKELETON DEMO")
    logger.info("=" * 80)
    
    # Generate synthetic data
    logger.info("\nGenerating synthetic data...")
    reference = generate_cube_with_tail_reference(
        cube_size=1.0,
        tail_segment_length=1.5
    )
    
    ground_truth, noisy = generate_synthetic_trajectory(
        reference_positions=reference,
        n_frames=200,
        noise_std=0.05,
        random_seed=42,
        wiggle_amplitude=0.3,
        wiggle_frequency=2.0
    )
    
    logger.info(f"  Generated {len(noisy[list(noisy.keys())[0]])} frames")
    logger.info(f"  Number of keypoints: {len(reference)}")
    
    # Setup output directory
    output_dir = Path("output/synthetic_cube_skeleton")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  Converting to TrajectoryDataset...")
    trajectory_data: dict[str, TrajectoryND] = {}
    n_frames = len(next(iter(noisy.values())))

    for kp_name, positions in noisy.items():
        trajectory_data[kp_name] = TrajectoryND(
            name=kp_name,
            values=positions,
            trajectory_type=TrajectoryType.POSITION,
            confidence=None,
            metadata={}
        )

    noisy_dataset = TrajectoryDataset(
        data=trajectory_data,
        frame_indices=np.arange(n_frames),
        metadata={}
    )

    # Save noisy input data
    input_csv = output_dir / "input_data.csv"
    save_trajectory_csv(
        dataset=noisy_dataset,
        filepath=input_csv
    )
    logger.info(f"  Saved input data to: {input_csv}")
    
    # Configure pipeline
    logger.info("\nConfiguring skeleton pipeline...")
    
    solver_config = SkeletonSolverConfig(
        max_num_iterations=500,
        function_tolerance=1e-6,
        gradient_tolerance=1e-8,
        parameter_tolerance=1e-7,
    )
    
    chunking_config = ChunkingConfig(
        chunk_size=50,
        overlap_size=10,
        blend_window=5,
    )
    
    config = SkeletonPipelineConfig(
        input_path=input_csv,
        output_dir=output_dir,
        solver_config=solver_config,
        parallel=chunking_config,
        skeleton=SYNTHETIC_CUBE_SKELETON,
        input_data_confidence_threshold=None,
        rigidity_threshold=0.5,
        stiffness_threshold=0.1,
        rotation_smoothness_weight=10.0,
        translation_smoothness_weight=10.0,
        anchor_weight=1.0,
        measurement_weight=1.0,
    )
    
    # Run pipeline
    logger.info("\nRunning skeleton optimization pipeline...")
    pipeline = SkeletonPipeline.from_config(config=config)
    

    result = pipeline.run()
    
    # Compute reconstruction error
    logger.info("\nEvaluating reconstruction quality...")
    if result is not None and result.optimized_data is not None:
        # Extract reconstructed positions
        reconstructed: dict[str, np.ndarray] = {}
        for kp_name in reference.keys():
            traj = result.optimized_data.data.get(kp_name)
            if traj is not None:
                reconstructed[kp_name] = traj.values
        
        metrics = compute_reconstruction_metrics(
            ground_truth=ground_truth,
            reconstructed=reconstructed
        )
        
        logger.info("\nReconstruction Metrics:")
        logger.info(f"  Mean error:   {metrics['mean_error']:.4f}")
        logger.info(f"  Median error: {metrics['median_error']:.4f}")
        logger.info(f"  Std error:    {metrics['std_error']:.4f}")
        logger.info(f"  Max error:    {metrics['max_error']:.4f}")
        logger.info(f"  RMSE:         {metrics['rmse']:.4f}")
        
        # Save metrics
        import json
        with open(output_dir / "reconstruction_metrics.json", mode='w', encoding='utf-8') as f:
            json.dump(obj=metrics, fp=f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    run_synthetic_cube_skeleton_demo()
