"""Solver result for rigid body bundle adjustment."""

import logging

import numpy as np
import pyceres
from scipy.spatial.transform import Rotation

from skellysolver.solvers.base_solver import SolverResult
from skellysolver.data.trajectory_dataset import TrajectoryDataset, TrajectoryND, TrajectoryType

logger = logging.getLogger(__name__)


class RigidBodySolverResult(SolverResult):
    """Results from rigid body bundle adjustment optimization.
    
    Stores:
    - Optimized poses (rotations + translations) for each frame
    - Optimized reference geometry
    - Reconstructed marker positions
    - Original raw data for comparison
    
    Attributes:
        rotations: (n_frames, 3, 3) rotation matrices
        translations: (n_frames, 3) translation vectors
        reference_geometry: (n_markers, 3) optimized reference positions
    """
    
    rotations: np.ndarray | None = None
    translations: np.ndarray | None = None
    reference_geometry: np.ndarray | None = None
    
    @classmethod
    def from_solver_and_params(
        cls,
        *,
        summary: pyceres.SolverSummary,
        solve_time_seconds: float,
        raw_data: TrajectoryDataset,
        pose_params: list[tuple[np.ndarray, np.ndarray]],
        reference_params: np.ndarray,
        marker_names: list[str]
    ) -> "RigidBodySolverResult":
        """Create result from solved parameters.
        
        Args:
            summary: pyceres solver summary
            solve_time_seconds: Time taken to solve
            raw_data: Original input data
            pose_params: List of (quat, trans) for each frame (modified in-place by solver)
            reference_params: Flattened reference geometry (modified in-place by solver)
            marker_names: List of marker names
            
        Returns:
            RigidBodySolverResult with extracted data
        """
        n_frames = len(pose_params)
        n_markers = len(marker_names)
        
        # Extract rotations and translations
        rotations = np.zeros((n_frames, 3, 3))
        translations = np.zeros((n_frames, 3))
        
        for frame_idx in range(n_frames):
            quat_ceres, trans = pose_params[frame_idx]
            
            # Convert quaternion from pyceres [w,x,y,z] to scipy [x,y,z,w]
            quat_scipy = np.array([
                quat_ceres[1],  # x
                quat_ceres[2],  # y
                quat_ceres[3],  # z
                quat_ceres[0],  # w
            ])
            
            R = Rotation.from_quat(quat=quat_scipy).as_matrix()
            
            rotations[frame_idx] = R
            translations[frame_idx] = trans
        
        # Extract reference geometry
        reference_geometry = reference_params.reshape((n_markers, 3))
        
        # Reconstruct marker positions
        reconstructed = np.zeros((n_frames, n_markers, 3))
        for frame_idx in range(n_frames):
            R = rotations[frame_idx]
            t = translations[frame_idx]
            reconstructed[frame_idx] = (R @ reference_geometry.T).T + t
        
        # Create optimized TrajectoryDataset
        optimized_trajectories = {}
        for marker_idx, marker_name in enumerate(marker_names):
            optimized_trajectories[marker_name] = TrajectoryND(
                name=marker_name,
                data=reconstructed[:, marker_idx, :],
                trajectory_type=TrajectoryType.POSITION,
                confidence=None,
                metadata={"source": "rigid_body_optimization"}
            )
        
        optimized_data = TrajectoryDataset(
            data=optimized_trajectories,
            frame_indices=raw_data.frame_indices.copy(),
            metadata={
                **raw_data.metadata,
                "optimization": "rigid_body_bundle_adjustment"
            }
        )
        
        # Check success
        success = summary.termination_type in [
            pyceres.TerminationType.CONVERGENCE,
            pyceres.TerminationType.USER_SUCCESS
        ]
        
        result = cls(
            success=success,
            summary=summary,
            solve_time_seconds=solve_time_seconds,
            raw_data=raw_data,
            optimized_data=optimized_data,
            rotations=rotations,
            translations=translations,
            reference_geometry=reference_geometry
        )
        
        logger.info(f"\n{result.optimization_report()}")
        
        return result
    
    def optimization_report(self) -> str:
        """Generate detailed optimization report."""
        status_symbol = "✓" if self.success else "✗"
        status_text = "Converged" if self.success else "Did not converge"
        
        lines = [
            "=" * 80,
            "RIGID BODY OPTIMIZATION RESULT",
            "=" * 80,
            f"Status:       {status_symbol} {status_text}",
            f"Iterations:   {self.summary.num_successful_steps}",
            f"Solve time:   {self.solve_time_seconds:.2f}s",
            f"Initial cost: {self.summary.initial_cost:.6f}",
            f"Final cost:   {self.summary.final_cost:.6f}",
            f"Reduction:    {self.cost_reduction_percent:.1f}%",
        ]
        
        if self.reference_geometry is not None:
            lines.append("")
            lines.append(f"Reference geometry: {self.reference_geometry.shape}")
            lines.append(f"Rotations: {self.rotations.shape}")
            lines.append(f"Translations: {self.translations.shape}")
        
        if self.optimized_data is not None:
            lines.append("")
            lines.append("Reconstructed data:")
            lines.append(f"  Frames:  {self.optimized_data.n_frames}")
            lines.append(f"  Markers: {self.optimized_data.n_markers}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def compute_rigidity_metrics(
        self,
        *,
        rigid_edges: list[tuple[int, int]],
        target_distances: np.ndarray
    ) -> dict[str, float]:
        """Compute rigidity preservation metrics.
        
        Args:
            rigid_edges: List of (i, j) rigid edge pairs
            target_distances: (n_markers, n_markers) target distance matrix
            
        Returns:
            Dictionary with rigidity metrics
        """
        if self.optimized_data is None:
            raise ValueError("No optimized data available")
        
        # Get optimized positions
        opt_array = self.optimized_data.to_array()  # (n_frames, n_markers, 3)
        raw_array = self.raw_data.to_array()
        
        # Compute edge errors
        raw_edge_errors = []
        opt_edge_errors = []
        
        for i, j in rigid_edges:
            target_dist = target_distances[i, j]
            
            # Compute distances over all frames
            raw_dists = np.linalg.norm(
                raw_array[:, i, :] - raw_array[:, j, :],
                axis=1
            )
            opt_dists = np.linalg.norm(
                opt_array[:, i, :] - opt_array[:, j, :],
                axis=1
            )
            
            # Errors from target
            raw_edge_errors.extend(np.abs(raw_dists - target_dist))
            opt_edge_errors.extend(np.abs(opt_dists - target_dist))
        
        metrics = {
            "raw_edge_error_mean_mm": float(np.mean(raw_edge_errors) * 1000),
            "raw_edge_error_max_mm": float(np.max(raw_edge_errors) * 1000),
            "opt_edge_error_mean_mm": float(np.mean(opt_edge_errors) * 1000),
            "opt_edge_error_max_mm": float(np.max(opt_edge_errors) * 1000),
        }
        
        logger.info("\nRigidity metrics:")
        logger.info(f"  Raw edge errors:  {metrics['raw_edge_error_mean_mm']:.2f}mm "
                   f"(max: {metrics['raw_edge_error_max_mm']:.2f}mm)")
        logger.info(f"  Opt edge errors:  {metrics['opt_edge_error_mean_mm']:.2f}mm "
                   f"(max: {metrics['opt_edge_error_max_mm']:.2f}mm)")
        
        return metrics
    
    def compute_reconstruction_accuracy(
        self,
        *,
        ground_truth_data: TrajectoryDataset | None = None
    ) -> dict[str, float] | None:
        """Compute reconstruction accuracy vs ground truth.
        
        Args:
            ground_truth_data: Optional ground truth trajectory data
            
        Returns:
            Dictionary with accuracy metrics, or None if no ground truth
        """
        if ground_truth_data is None:
            return None
        
        if self.optimized_data is None:
            raise ValueError("No optimized data available")
        
        # Get arrays
        raw_array = self.raw_data.to_array()
        opt_array = self.optimized_data.to_array()
        gt_array = ground_truth_data.to_array()
        
        # Compute errors
        raw_errors = np.linalg.norm(raw_array - gt_array, axis=2)
        opt_errors = np.linalg.norm(opt_array - gt_array, axis=2)
        
        metrics = {
            "raw_position_error_mean_mm": float(np.mean(raw_errors) * 1000),
            "raw_position_error_max_mm": float(np.max(raw_errors) * 1000),
            "opt_position_error_mean_mm": float(np.mean(opt_errors) * 1000),
            "opt_position_error_max_mm": float(np.max(opt_errors) * 1000),
        }
        
        # Improvement
        improvement = (
            (metrics["raw_position_error_mean_mm"] - metrics["opt_position_error_mean_mm"]) 
            / metrics["raw_position_error_mean_mm"] * 100
        )
        metrics["improvement_percent"] = float(improvement)
        
        logger.info("\nAccuracy metrics (vs ground truth):")
        logger.info(f"  Raw position error:  {metrics['raw_position_error_mean_mm']:.2f}mm "
                   f"(max: {metrics['raw_position_error_max_mm']:.2f}mm)")
        logger.info(f"  Opt position error:  {metrics['opt_position_error_mean_mm']:.2f}mm "
                   f"(max: {metrics['opt_position_error_max_mm']:.2f}mm)")
        logger.info(f"  Improvement: {improvement:.1f}%")
        
        return metrics
    
    def get_pose_trajectory(self) -> TrajectoryDataset:
        """Get pose parameters as TrajectoryDataset.
        
        Returns:
            TrajectoryDataset with rotation and translation trajectories
        """
        if self.rotations is None or self.translations is None:
            raise ValueError("No pose data available")
        
        n_frames = len(self.rotations)
        
        # Flatten rotations to (n_frames, 9)
        rotations_flat = self.rotations.reshape((n_frames, 9))
        
        trajectories = {
            "rotation": TrajectoryND(
                name="rotation",
                data=rotations_flat,
                trajectory_type=TrajectoryType.ROTATION_MATRIX,
                confidence=None,
                metadata={"description": "Rigid body rotation matrices"}
            ),
            "translation": TrajectoryND(
                name="translation",
                data=self.translations,
                trajectory_type=TrajectoryType.POSITION,
                confidence=None,
                metadata={"description": "Rigid body translations"}
            )
        }
        
        return TrajectoryDataset(
            data=trajectories,
            frame_indices=self.raw_data.frame_indices.copy(),
            metadata={"source": "rigid_body_poses"}
        )
