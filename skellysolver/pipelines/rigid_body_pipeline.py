"""Rigid body tracking pipeline.

Optimizes rigid body pose from noisy marker measurements.
Inherits from BasePipeline and uses Phase 1 + Phase 2 components.
"""

import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass

from skellysolver.pipelines.topology import RigidBodyTopology
from skellysolver.pipelines.metrics import evaluate_reconstruction
from skellysolver.pipelines.savers import save_topology_json, save_trajectory_csv, save_evaluation_report
from skellysolver.core.config import RigidBodyWeightConfig
from skellysolver.core.cost_functions import RigidPoint3DMeasurementBundleAdjustment, RigidEdgeCost, SoftEdgeCost, \
    RotationSmoothnessCost, TranslationSmoothnessCost, ReferenceAnchorCost
from skellysolver.core.result import RigidBodyResult
from skellysolver.core.optimizer import Optimizer
from skellysolver.data.base_data import TrajectoryDataset
from skellysolver.data.loaders import load_trajectories
from skellysolver.data.validators import (
    validate_dataset,
    validate_topology_compatibility,
)
from skellysolver.data.preprocessing import (
    filter_by_confidence,
    interpolate_missing,
)
from skellysolver.pipelines import PipelineConfig, BasePipeline

logger = logging.getLogger(__name__)


class RigidBodyConfig(PipelineConfig):
    """Configuration for rigid body tracking pipeline.
    
    Extends PipelineConfig with rigid body specific settings.
    
    Attributes:
        topology: Rigid body topology (markers and edges)
        weights: Cost function weights
        soft_edges: Optional soft (flexible) edges
        soft_distances: Distances for soft edges
        min_confidence: Minimum confidence for filtering
        interpolate_missing: Whether to interpolate missing data
        use_bundle_adjustment: Joint optimization of pose + geometry
    """
    
    topology: RigidBodyTopology
    weights: RigidBodyWeightConfig = None
    soft_edges: list[tuple[int, int]] | None = None
    soft_distances: np.ndarray | None = None
    min_confidence: float = 0.3
    interpolate_missing_data: bool = True
    use_bundle_adjustment: bool = True
    
    def __post_init__(self) -> None:
        """Set defaults and validate."""
        super().__post_init__()
        
        if self.weights is None:
            self.weights = RigidBodyWeightConfig()
        
        # Validate soft edges match soft distances
        if self.soft_edges is not None:
            if self.soft_distances is None:
                raise ValueError("soft_distances required when soft_edges provided")


class RigidBodyPipeline(BasePipeline):
    """Rigid body tracking pipeline.
    
    Optimizes rigid body pose from noisy marker measurements.
    Uses bundle adjustment to jointly optimize reference geometry and poses.
    
    Usage:
        from skellysolver.pipelines.rigid_body import (
            RigidBodyPipeline,
            RigidBodyConfig,
        )
        from skellysolver.core import OptimizationConfig, RigidBodyWeightConfig
        from skellysolver.core.topology import RigidBodyTopology
        
        # Define topology
        topology = RigidBodyTopology(
            marker_names=["nose", "left_eye", "right_eye"],
            rigid_edges=[(0, 1), (1, 2), (2, 0)],
            name="simple_triangle"
        )
        
        # Configure
        config = RigidBodyConfig(
            input_path=Path("data.csv"),
            output_dir=Path("output/"),
            topology=topology,
            optimization=OptimizationConfig(max_iterations=300),
        )
        
        # Run
        pipeline = RigidBodyPipeline(config=config)
        result = pipeline.run()
    """
    
    config: RigidBodyConfig  # Type hint for IDE
    
    def load_data(self) -> TrajectoryDataset:
        """Load trajectory data from CSV.
        
        Returns:
            TrajectoryDataset with marker trajectories
        """
        logger.info(f"Loading data from {self.config.input_path.name}...")
        
        dataset = load_trajectories(
            filepath=self.config.input_path,
            scale_factor=1.0,
            z_value=0.0
        )
        
        logger.info(f"  Loaded {dataset.n_markers} markers × {dataset.n_frames} frames")
        logger.info(f"  Markers: {dataset.marker_names}")
        
        return dataset
    
    def preprocess_data(self, *, data: TrajectoryDataset) -> TrajectoryDataset:
        """Preprocess and validate data.
        
        Steps:
        1. Check topology compatibility
        2. Validate data quality
        3. Filter low-confidence frames (optional)
        4. Interpolate missing data (optional)
        
        Args:
            data: Raw loaded data
            
        Returns:
            Preprocessed data
        """
        logger.info("Preprocessing data...")
        
        # Check topology compatibility
        logger.info("  Checking topology compatibility...")
        compat = validate_topology_compatibility(
            dataset=data,
            topology_marker_names=self.config.topology.marker_names
        )
        
        if not compat["compatible"]:
            raise ValueError(
                f"Dataset incompatible with topology: {compat['error']}"
            )
        
        logger.info("    ✓ All required markers present")
        
        # Validate data
        logger.info("  Validating data...")
        report = validate_dataset(
            dataset=data,
            required_markers=self.config.topology.marker_names,
            min_valid_frames=10,
            min_confidence=self.config.min_confidence
        )
        
        if not report["valid"]:
            logger.warning("    ⚠ Data validation warnings:")
            for error in report["errors"]:
                logger.warning(f"      {error}")
        else:
            logger.info("    ✓ Data validation passed")
        
        # Filter by confidence
        if self.config.min_confidence > 0:
            logger.info(f"  Filtering by confidence (min={self.config.min_confidence})...")
            n_before = data.n_frames
            data = filter_by_confidence(
                dataset=data,
                min_confidence=self.config.min_confidence,
                min_valid_markers=len(self.config.topology.marker_names)
            )
            n_after = data.n_frames
            logger.info(f"    Filtered: {n_before} → {n_after} frames ({n_before - n_after} removed)")
        
        # Interpolate missing
        if self.config.interpolate_missing_data:
            logger.info("  Interpolating missing data...")
            data = interpolate_missing(
                dataset=data,
                method="linear",
                max_gap=10
            )
            logger.info("    ✓ Interpolation complete")
        
        logger.info("✓ Preprocessing complete")
        
        return data
    
    def optimize(self, *, data: TrajectoryDataset) -> RigidBodyResult:
        """Run rigid body optimization.
        
        Uses bundle adjustment to jointly optimize:
        - Reference geometry (marker positions in body frame)
        - Poses for each frame (rotation + translation)
        
        Args:
            data: Preprocessed data
            
        Returns:
            RigidBodyResult with optimized parameters
        """
        logger.info("Running optimization...")
        
        # Extract data as array
        noisy_data = data.to_array(marker_names=self.config.topology.marker_names)
        n_frames, n_markers, _ = noisy_data.shape
        
        logger.info(f"  Data shape: {noisy_data.shape}")
        
        # Estimate initial reference geometry (median frame, centered)
        logger.info("  Estimating initial reference geometry...")
        median_frame = np.median(noisy_data, axis=0)
        reference_geometry = median_frame - np.mean(median_frame, axis=0)
        reference_params = reference_geometry.flatten().copy()
        
        # Estimate rigid edge distances
        logger.info("  Estimating rigid edge distances...")
        reference_distances = self._estimate_edge_distances(
            data=noisy_data,
            edges=self.config.topology.rigid_edges
        )
        
        # Estimate soft edge distances if needed
        soft_distances_dict = None
        if self.config.soft_edges is not None:
            logger.info("  Estimating soft edge distances...")
            soft_distances_dict = self._estimate_edge_distances(
                data=noisy_data,
                edges=self.config.soft_edges
            )
        
        # Initialize poses
        logger.info("  Initializing poses...")
        quaternions = np.zeros((n_frames, 4))
        quaternions[:, 0] = 1.0  # Identity rotations (w=1)
        
        translations = np.zeros((n_frames, 3))
        for i in range(n_frames):
            translations[i] = np.mean(noisy_data[i], axis=0)
        
        # Build optimization problem
        logger.info("  Building optimization problem...")
        optimizer = Optimizer(config=self.config.optimization)
        
        # Add reference geometry
        optimizer.add_parameter_block(
            name="reference",
            parameters=reference_params
        )
        
        # Add poses
        for i in range(n_frames):
            optimizer.add_quaternion_parameter(
                name=f"quat_{i}",
                parameters=quaternions[i]
            )
            optimizer.add_parameter_block(
                name=f"trans_{i}",
                parameters=translations[i]
            )
        
        # Add measurement costs
        logger.info(f"  Adding {n_frames * n_markers} measurement costs...")
        for i in range(n_frames):
            for j in range(n_markers):
                cost = RigidPoint3DMeasurementBundleAdjustment(
                    measured_point=noisy_data[i, j],
                    marker_idx=j,
                    n_markers=n_markers,
                    weight=self.config.weights.lambda_data
                )
                optimizer.add_residual_block(
                    cost=cost,
                    parameters=[quaternions[i], translations[i], reference_params]
                )
        
        # Add rigid edge constraints
        logger.info(f"  Adding {len(self.config.topology.rigid_edges)} rigid edge constraints...")
        for i, j in self.config.topology.rigid_edges:
            cost = RigidEdgeCost(
                marker_i=i,
                marker_j=j,
                n_markers=n_markers,
                target_distance=reference_distances[i, j],
                weight=self.config.weights.lambda_rigid
            )
            optimizer.add_residual_block(
                cost=cost,
                parameters=[reference_params]
            )
        
        # Add soft edge constraints
        if self.config.soft_edges is not None:
            logger.info(f"  Adding {len(self.config.soft_edges)} soft edge constraints...")
            for frame_idx in range(n_frames):
                for i, j in self.config.soft_edges:
                    cost = SoftEdgeCost(
                        measured_point=noisy_data[frame_idx, j],
                        marker_idx=i,
                        n_markers=n_markers,
                        target_distance=soft_distances_dict[i, j],
                        weight=self.config.weights.lambda_soft
                    )
                    optimizer.add_residual_block(
                        cost=cost,
                        parameters=[quaternions[frame_idx], translations[frame_idx], reference_params]
                    )
        
        # Add temporal smoothness
        logger.info(f"  Adding {n_frames - 1} smoothness constraints...")
        for i in range(n_frames - 1):
            # Rotation smoothness
            rot_cost = RotationSmoothnessCost(weight=self.config.weights.lambda_rot_smooth)
            optimizer.add_residual_block(
                cost=rot_cost,
                parameters=[quaternions[i], quaternions[i + 1]]
            )
            
            # Translation smoothness
            trans_cost = TranslationSmoothnessCost(weight=self.config.weights.lambda_trans_smooth)
            optimizer.add_residual_block(
                cost=trans_cost,
                parameters=[translations[i], translations[i + 1]]
            )
        
        # Add reference anchor
        logger.info("  Adding reference anchor...")
        anchor_cost = ReferenceAnchorCost(
            initial_reference=reference_params.copy(),
            weight=self.config.weights.lambda_anchor
        )
        optimizer.add_residual_block(
            cost=anchor_cost,
            parameters=[reference_params]
        )
        
        logger.info(f"  Total parameters: {optimizer.num_parameters()}")
        logger.info(f"  Total residuals:  {optimizer.num_residuals()}")
        
        # Solve
        logger.info("  Solving...")
        result = optimizer.solve()
        
        # Extract results
        logger.info("  Extracting results...")
        optimized_reference = reference_params.reshape(n_markers, 3)
        
        # Compute rotations and reconstructed positions
        from scipy.spatial.transform import Rotation
        
        rotations = np.zeros((n_frames, 3, 3))
        reconstructed = np.zeros((n_frames, n_markers, 3))
        
        for i in range(n_frames):
            # Convert quaternion to rotation matrix
            quat_scipy = np.array([
                quaternions[i, 1],
                quaternions[i, 2],
                quaternions[i, 3],
                quaternions[i, 0]
            ])
            R = Rotation.from_quat(quat=quat_scipy).as_matrix()
            
            rotations[i] = R
            reconstructed[i] = (R @ optimized_reference.T).T + translations[i]
        
        # Create specialized result
        rigid_result = RigidBodyResult(
            success=result.success,
            num_iterations=result.num_iterations,
            initial_cost=result.initial_cost,
            final_cost=result.final_cost,
            solve_time_seconds=result.solve_time_seconds,
            reconstructed=reconstructed,
            rotations=rotations,
            translations=translations,
            reference_geometry=optimized_reference,
            metadata={
                "n_frames": n_frames,
                "n_markers": n_markers,
                "topology": self.config.topology.name
            }
        )
        
        logger.info("✓ Optimization complete")
        
        return rigid_result
    
    def evaluate(self, *, result: RigidBodyResult) -> dict[str, float]:
        """Evaluate reconstruction quality.
        
        Computes metrics:
        - Reconstruction error (RMSE)
        - Edge length consistency
        - Temporal smoothness
        
        Args:
            result: Optimization result
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating results...")
        
        # Get noisy data
        noisy_data = self.data.to_array(marker_names=self.config.topology.marker_names)
        
        # Compute reference distances
        reference_distances = self._estimate_edge_distances(
            data=noisy_data,
            edges=self.config.topology.rigid_edges
        )
        
        # Evaluate reconstruction
        metrics = evaluate_reconstruction(
            noisy_data=noisy_data,
            optimized_data=result.reconstructed,
            reference_distances=reference_distances,
            topology=self.config.topology,
            ground_truth_data=None
        )
        
        logger.info("  Metrics:")
        logger.info(f"    Reconstruction error: {metrics['reconstruction_error_mean']:.4f}m")
        logger.info(f"    Edge consistency:     {metrics.get('edge_consistency_std', 0):.4f}m")
        
        logger.info("✓ Evaluation complete")
        
        return metrics
    
    def save_results(
        self,
        *,
        result: RigidBodyResult,
        metrics: dict[str, float]
    ) -> None:
        """Save results to disk.
        
        Saves:
        - trajectory_data.csv: Noisy and optimized trajectories
        - topology.json: Topology metadata
        - metrics.json: Evaluation metrics
        - reference_geometry.npy: Optimized reference shape
        
        Args:
            result: Optimization result
            metrics: Evaluation metrics
        """
        logger.info("Saving results...")
        

        
        # Get noisy data
        noisy_data = self.data.to_array(marker_names=self.config.topology.marker_names)
        
        # Save trajectory CSV
        save_trajectory_csv(
            filepath=self.config.output_dir / "trajectory_data.csv",
            noisy_data=noisy_data,
            optimized_data=result.reconstructed,
            marker_names=self.config.topology.marker_names,
            ground_truth_data=None
        )
        logger.info("  ✓ Saved trajectory_data.csv")
        
        # Save topology JSON
        save_topology_json(
            filepath=self.config.output_dir / "topology.json",
            topology_dict=self.config.topology.to_dict(),
            marker_names=self.config.topology.marker_names,
            n_frames=result.n_frames,
            has_ground_truth=False,
            soft_edges=self.config.soft_edges
        )
        logger.info("  ✓ Saved topology.json")
        
        # Save metrics
        save_evaluation_report(
            filepath=self.config.output_dir / "metrics.json",
            metrics=metrics,
            config={
                "topology": self.config.topology.name,
                "n_frames": result.n_frames,
                "n_markers": result.n_markers,
                "optimization": {
                    "max_iterations": self.config.optimization.max_iterations,
                    "weights": {
                        "data": self.config.weights.lambda_data,
                        "rigid": self.config.weights.lambda_rigid,
                        "soft": self.config.weights.lambda_soft,
                        "rot_smooth": self.config.weights.lambda_rot_smooth,
                        "trans_smooth": self.config.weights.lambda_trans_smooth,
                    }
                }
            }
        )
        logger.info("  ✓ Saved metrics.json")
        
        # Save reference geometry
        np.save(
            file=self.config.output_dir / "reference_geometry.npy",
            arr=result.reference_geometry
        )
        logger.info("  ✓ Saved reference_geometry.npy")
        
        logger.info(f"✓ Results saved to {self.config.output_dir}")
    
    def generate_viewer(self, *, result: RigidBodyResult) -> None:
        """Generate interactive HTML viewer.
        
        Creates rigid_body_viewer.html with 3D visualization.
        
        Args:
            result: Optimization result
        """
        logger.info("Generating viewer...")
        
        # Copy viewer HTML template
        import shutil
        
        viewer_template = Path(__file__).parent / "rigid_body_viewer.html"
        viewer_output = self.config.output_dir / "rigid_body_viewer.html"
        
        if viewer_template.exists():
            shutil.copy(src=viewer_template, dst=viewer_output)
            logger.info(f"  ✓ Generated {viewer_output.name}")
            logger.info(f"  → Open {viewer_output} in a browser to visualize")
        else:
            logger.warning(f"  ⚠ Viewer template not found: {viewer_template}")
            logger.warning("  Skipping viewer generation")
    
    def _estimate_edge_distances(
        self,
        *,
        data: np.ndarray,
        edges: list[tuple[int, int]]
    ) -> np.ndarray:
        """Estimate edge distances from data.
        
        Args:
            data: (n_frames, n_markers, 3) positions
            edges: List of (i, j) edge pairs
            
        Returns:
            (n_markers, n_markers) distance matrix
        """
        n_markers = data.shape[1]
        distances = np.zeros((n_markers, n_markers))
        
        for i, j in edges:
            frame_distances = np.linalg.norm(
                data[:, i, :] - data[:, j, :],
                axis=1
            )
            median_dist = np.median(frame_distances)
            distances[i, j] = median_dist
            distances[j, i] = median_dist
        
        return distances
