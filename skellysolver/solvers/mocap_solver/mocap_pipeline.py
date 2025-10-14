"""Rigid body tracking pipeline with chunking support and passthrough markers.

Optimizes rigid body pose from noisy marker measurements.
Supports both single-pass and chunked parallel optimization for long recordings.
NEW: Supports passthrough markers that bypass optimization and are included raw.
"""
import logging

import numpy as np
from pydantic import model_validator, Field
from scipy.spatial.transform import Rotation

from skellysolver.core.chunking import optimize_chunked_parallel, optimize_chunked_sequential, ChunkingConfig
from skellysolver.cost_primatives.edge_consts import RigidEdgeCost, SoftEdgeCost
from skellysolver.cost_primatives.measurement_costs import RigidPoint3DMeasurementBundleAdjustment
from skellysolver.cost_primatives.smoothness_costs import RotationSmoothnessCost, TranslationSmoothnessCost
from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.io.viewers.html_viewers.rigid_viewer import MocapViewerGenerator
from skellysolver.solvers.pyceres_solver import PyceresOptimizer
from skellysolver.core.optimization_result import RigidBodyResult
from skellysolver.data.data_models import TrajectoryDataset
from skellysolver.io.loaders import load_trajectories
from skellysolver.data.preprocessing import (
    filter_by_confidence,
    interpolate_missing,
)
from skellysolver.data.validators import (
    validate_dataset,
    validate_topology_compatibility,
)
from skellysolver.solvers import PipelineConfig, BasePipeline
from skellysolver.solvers.mocap_solver.mocap_metrics import evaluate_mocap_reconstruction
from skellysolver.io.writers.csv_writer import TidyCSVWriter
from skellysolver.solvers.mocap_solver.mocap_topology import RigidBodyTopology

logger = logging.getLogger(__name__)


class MocapWeightConfig(ABaseModel):
    """Weights specifically for rigid body tracking.

    Provides sensible defaults for rigid body optimization.
    """

    lambda_data: float = 100.0
    lambda_rigid: float = 500.0
    lambda_soft: float = 10.0
    lambda_rot_smooth: float = 200.0
    lambda_trans_smooth: float = 200.0
    lambda_anchor: float = 10.0

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
        parallel: Optional parallel processing configuration
        passthrough_markers: Optional list of marker names to include in output
                           without optimization (passed through raw from input).
                           These markers must exist in the input data but not
                           be part of the topology.
    """


    topology: RigidBodyTopology
    scale_factor: float = 1.0
    z_value: float = 0.0
    weights: MocapWeightConfig | None = None
    soft_edges: list[tuple[int, int]] | None = None
    soft_distances: np.ndarray | None = None
    min_confidence: float = 0.3
    interpolate_missing_data: bool = True
    use_bundle_adjustment: bool = True
    parallel: ChunkingConfig | None = Field(default_factory=ChunkingConfig)
    passthrough_markers: list[str] | None = None

    @model_validator(mode='after')
    def set_defaults_and_validate(self) -> 'RigidBodyConfig':
        """Set defaults and validate."""
        if self.weights is None:
            self.weights = MocapWeightConfig()

        # Validate soft edges match soft distances
        if self.soft_edges is not None:
            if self.soft_distances is None:
                raise ValueError("soft_distances required when soft_edges provided")

        # Set default parallel config if not provided
        if self.parallel is None:
            self.parallel = ChunkingConfig()

        # Validate passthrough markers don't overlap with topology
        if self.passthrough_markers is not None:
            overlap = set(self.passthrough_markers) & set(self.topology.marker_names)
            if overlap:
                raise ValueError(
                    f"Passthrough markers cannot overlap with topology markers: {overlap}"
                )

        return self


class RigidBodyPipeline(BasePipeline):
    """Rigid body tracking pipeline with chunking support and passthrough markers.

    Optimizes rigid body pose from noisy marker measurements.
    Uses bundle adjustment to jointly optimize reference geometry and poses.
    Supports chunked parallel optimization for long recordings.

    NEW: Supports passthrough markers - markers that are included in the output
    without optimization. Useful for combining optimized rigid bodies (e.g., skull)
    with raw flexible markers (e.g., spine).

    Usage:
        from skellysolver.pipelines.rigid_body import (
            RigidBodyPipeline,
            RigidBodyConfig,
        )
        from skellysolver.core import OptimizationConfig, RigidBodyWeightConfig, ParallelConfig
        from skellysolver.core.topology import RigidBodyTopology

        # Define topology (only rigid markers)
        topology = RigidBodyTopology(
            marker_names=["nose", "left_eye", "right_eye"],
            rigid_edges=[(0, 1), (1, 2), (2, 0)],
            name="skull"
        )

        # Configure with passthrough markers (flexible markers)
        config = RigidBodyConfig(
            input_path=Path("data.csv"),
            output_dir=Path("output/"),
            topology=topology,
            passthrough_markers=["spine_t1", "sacrum", "tail_tip"],  # Raw markers
            optimization=OptimizationConfig(max_iterations=300),
            parallel=ParallelConfig(
                enabled=True,
                chunk_size=500,
                overlap_size=50,
                blend_window=25
            )
        )

        # Run - output includes optimized skull + raw spine
        pipeline = RigidBodyPipeline(config=config)
        result = pipeline.run()
    """

    config: RigidBodyConfig
    _passthrough_data: np.ndarray | None = None

    def load_data(self) -> TrajectoryDataset:
        """Load trajectory data from CSV.

        Returns:
            TrajectoryDataset with marker trajectories
        """
        logger.info(f"Loading data from {self.config.input_path.name}...")

        dataset = load_trajectories(
            filepath=self.config.input_path,
            scale_factor=self.config.scale_factor,
            z_value=self.config.z_value
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
        5. Store passthrough markers if specified

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

        # Check passthrough markers exist
        if self.config.passthrough_markers:
            logger.info("  Checking passthrough markers...")
            missing_passthrough = set(self.config.passthrough_markers) - set(data.marker_names)
            if missing_passthrough:
                raise ValueError(
                    f"Passthrough markers not in dataset: {missing_passthrough}"
                )
            logger.info(f"    ✓ All {len(self.config.passthrough_markers)} passthrough markers present")

        # Validate data
        logger.info("  Validating data...")
        all_required_markers = self.config.topology.marker_names
        if self.config.passthrough_markers:
            all_required_markers = all_required_markers + self.config.passthrough_markers

        report = validate_dataset(
            dataset=data,
            required_markers=all_required_markers,
            min_valid_frames=10,
            min_confidence=self.config.min_confidence
        )

        if not report["valid"]:
            logger.error("    ⚠ Data validation warnings:")
            for error in report["errors"]:
                logger.error(f"      {error}")
            raise ValueError("Data validation failed")
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

        # Store passthrough markers if specified
        if self.config.passthrough_markers:
            logger.info(f"  Storing {len(self.config.passthrough_markers)} passthrough markers (raw)...")
            self._passthrough_data = data.to_array(marker_names=self.config.passthrough_markers)
            logger.info(f"    Shape: {self._passthrough_data.shape}")
        else:
            self._passthrough_data = None

        logger.info("✓ Preprocessing complete")

        return data

    def optimize(self, *, data: TrajectoryDataset) -> RigidBodyResult:
        """Run rigid body optimization with optional chunking and passthrough.

        Automatically selects between single-pass and chunked optimization
        based on data size and parallel configuration.

        If passthrough markers are configured, combines optimized topology markers
        with raw passthrough markers in the final result.

        Args:
            data: Preprocessed data

        Returns:
            RigidBodyResult with optimized parameters (and passthrough if configured)
        """
        logger.info("Running optimization...")

        # Extract data as array
        raw_data = data.to_array(marker_names=self.config.topology.marker_names)
        n_frames, n_markers, _ = raw_data.shape

        logger.info(f"  Data shape: {raw_data.shape}")

        # Check if we should use chunked optimization
        use_chunking = (
            self.config.parallel is not None and
            self.config.parallel.should_use_parallel(n_frames=n_frames)
        )

        if use_chunking:
            result = self._optimize_chunked(data=raw_data)
        else:
            result = self._optimize_single_pass(data=raw_data)

        # Combine with passthrough if needed
        if self._passthrough_data is not None:
            logger.info("  Combining optimized + passthrough markers...")
            result = self._combine_with_passthrough(result=result)
            logger.info(f"    Final shape: {result.reconstructed.shape}")

        return result

    def _combine_with_passthrough(self, *, result: RigidBodyResult) -> RigidBodyResult:
        """Combine optimized topology markers with raw passthrough markers.

        Creates a new result with:
        - reconstructed: [optimized_topology | raw_passthrough]
        - Updated metadata with passthrough info

        Args:
            result: Optimization result with topology markers only

        Returns:
            New result with combined markers
        """
        # Validate shapes match
        if result.reconstructed.shape[0] != self._passthrough_data.shape[0]:
            raise ValueError(
                f"Frame count mismatch: optimized={result.reconstructed.shape[0]}, "
                f"passthrough={self._passthrough_data.shape[0]}"
            )

        # Concatenate along marker axis
        combined_reconstructed = np.concatenate(
            [result.reconstructed, self._passthrough_data],
            axis=1
        )

        # Update metadata
        new_metadata = result.metadata.copy()
        new_metadata["passthrough_markers"] = self.config.passthrough_markers
        new_metadata["n_topology_markers"] = len(self.config.topology.marker_names)
        new_metadata["n_passthrough_markers"] = len(self.config.passthrough_markers)
        new_metadata["n_markers"] = combined_reconstructed.shape[1]

        return RigidBodyResult(
            success=result.success,
            num_iterations=result.num_iterations,
            initial_cost=result.initial_cost,
            final_cost=result.final_cost,
            solve_time_seconds=result.solve_time_seconds,
            reconstructed=combined_reconstructed,
            rotations=result.rotations,
            translations=result.translations,
            reference_geometry=result.reference_geometry,
            metadata=new_metadata
        )

    def _optimize_single_pass(self, *, data: np.ndarray) -> RigidBodyResult:
        """Run standard single-pass optimization.

        Processes all frames in one optimization run. This is the core optimizer
        used both for small datasets and as the per-chunk optimizer in chunked mode.

        Args:
            data: (n_frames, n_markers, 3) positions

        Returns:
            RigidBodyResult
        """
        n_frames, n_markers, _ = data.shape

        logger.info("  Using SINGLE-PASS optimization")

        # Estimate initial reference geometry
        logger.info("  Estimating initial reference geometry...")
        median_frame = np.median(data, axis=0)
        reference_geometry = median_frame - np.mean(median_frame, axis=0)
        reference_params = reference_geometry.flatten().copy()

        # Estimate rigid edge distances
        logger.info("  Estimating rigid edge distances...")
        reference_distances = self._estimate_edge_distances(
            data=data,
            edges=self.config.topology.rigid_edges
        )

        # Estimate soft edge distances if needed
        soft_distances_dict = None
        if self.config.soft_edges is not None:
            logger.info("  Estimating soft edge distances...")
            soft_distances_dict = self._estimate_edge_distances(
                data=data,
                edges=self.config.soft_edges
            )

        # Initialize poses
        logger.info("  Initializing poses...")
        quaternions = np.zeros((n_frames, 4))
        quaternions[:, 0] = 1.0  # Identity rotations (w=1)

        translations = np.zeros((n_frames, 3))
        for i in range(n_frames):
            translations[i] = np.mean(data[i], axis=0)

        # Build optimization problem
        logger.info("  Building optimization problem...")
        optimizer = PyceresOptimizer(config=self.config.optimization)

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
                    measured_point=data[i, j],
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
                target_distance=float(reference_distances[i, j]),
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
                        measured_point=data[frame_idx, j],
                        marker_idx=i,
                        n_markers=n_markers,
                        target_distance=float(soft_distances_dict[i, j]),
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

        # # Add reference anchor
        # logger.info("  Adding reference anchor...")
        # anchor_cost = ReferenceAnchorCost(
        #     initial_reference=reference_params.copy(),
        #     weight=self.config.weights.lambda_anchor
        # )
        # optimizer.add_residual_block(
        #     cost=anchor_cost,
        #     parameters=[reference_params]
        # )

        logger.info(f"  Total parameters: {optimizer.num_parameters()}")
        logger.info(f"  Total residuals:  {optimizer.num_residuals()}")

        # Solve
        logger.info("  Solving...")
        result = optimizer.solve()

        # Extract results
        logger.info("  Extracting results...")
        optimized_reference = reference_params.reshape(n_markers, 3)

        # Compute rotations and reconstructed positions

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
                "topology": self.config.topology.name,
                "optimization_mode": "single_pass"
            }
        )

        logger.info("✓ Optimization complete")

        return rigid_result

    def _optimize_chunked(self, *, data: np.ndarray) -> RigidBodyResult:
        """Run chunked optimization with blending.

        Splits data into overlapping chunks, optimizes each chunk independently
        (using _optimize_single_pass), then smoothly blends the results.

        Args:
            data: (n_frames, n_markers, 3) positions

        Returns:
            RigidBodyResult
        """
        n_frames, n_markers, _ = data.shape

        parallel_mode = "PARALLEL" if self.config.parallel.enabled else "SEQUENTIAL"
        logger.info(f"  Using CHUNKED {parallel_mode} optimization")
        logger.info(f"  Chunk size: {self.config.parallel.chunk_size}")
        logger.info(f"  Overlap: {self.config.parallel.overlap_size}")
        logger.info(f"  Blend window: {self.config.parallel.blend_window}")

        # Choose chunking mode
        if self.config.parallel.enabled:
            rotations, translations, reconstructed = optimize_chunked_parallel(
                raw_data=data,
                chunk_size=self.config.parallel.chunk_size,
                overlap_size=self.config.parallel.overlap_size,
                blend_window=self.config.parallel.blend_window,
                min_chunk_size=self.config.parallel.min_chunk_size,
                n_workers=self.config.parallel.get_num_workers(),
                optimize_fn=self._optimize_single_pass
            )
        else:
            rotations, translations, reconstructed = optimize_chunked_sequential(
                raw_data=data,
                chunk_size=self.config.parallel.chunk_size,
                overlap_size=self.config.parallel.overlap_size,
                blend_window=self.config.parallel.blend_window,
                min_chunk_size=self.config.parallel.min_chunk_size,
                optimize_fn=self._optimize_single_pass
            )

        # Extract reference geometry (use median frame as approximation)
        median_frame = np.median(data, axis=0)
        reference_geometry = median_frame - np.mean(median_frame, axis=0)

        # Create result
        rigid_result = RigidBodyResult(
            success=True,
            num_iterations=0,  # Not applicable for chunked
            initial_cost=0.0,
            final_cost=0.0,
            solve_time_seconds=0.0,  # Tracked separately
            reconstructed=reconstructed,
            rotations=rotations,
            translations=translations,
            reference_geometry=reference_geometry,
            metadata={
                "n_frames": n_frames,
                "n_markers": n_markers,
                "topology": self.config.topology.name,
                "optimization_mode": parallel_mode.lower()
            }
        )

        logger.info("✓ Chunked optimization complete")

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

        # Get marker names (topology only, not passthrough)
        topology_marker_names = self.config.topology.marker_names

        # Get noisy data for topology markers
        raw_data = self.data.to_array(marker_names=topology_marker_names)

        # Extract only topology markers from result if passthrough is present
        if self._passthrough_data is not None:
            n_topology = len(topology_marker_names)
            optimized_data = result.reconstructed[:, :n_topology, :]
        else:
            optimized_data = result.reconstructed

        # Compute reference distances
        reference_distances = self._estimate_edge_distances(
            data=raw_data,
            edges=self.config.topology.rigid_edges
        )

        # Evaluate reconstruction
        metrics = evaluate_mocap_reconstruction(
            raw_data=raw_data,
            optimized_data=optimized_data,
            reference_distances=reference_distances,
            topology=self.config.topology,
            ground_truth_data=None
        )

        logger.info("  Metrics:")
        logger.info(f"    Reconstruction error: {metrics.get('reconstruction_error_mean', 0):.4f}m")
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
        - trajectory_data.csv: Optimized trajectories (tidy format)
        - raw_trajectory_data.csv: Raw/noisy trajectories (tidy format)
        - topology.json: Topology metadata
        - metrics.json: Evaluation metrics
        - reference_geometry.npy: Optimized reference shape

        Args:
            result: Optimization result
            metrics: Evaluation metrics
        """
        logger.info("Saving results...")

        # Determine all marker names (topology + passthrough)
        marker_names = self.config.topology.marker_names
        if self.config.passthrough_markers:
            marker_names = marker_names + self.config.passthrough_markers

        # Get noisy data for all markers
        raw_data = self.data.to_array(marker_names=marker_names)

        # Save optimized trajectory CSV (unadorned name)
        tidy_writer = TidyCSVWriter()
        tidy_writer.write(
            filepath=self.config.output_dir / "trajectory_data.csv",
            data={
                "positions": result.reconstructed,
                "marker_names": marker_names
            }
        )
        logger.info("  ✓ Saved trajectory_data.csv")

        # Save raw trajectory CSV (labeled as raw)
        tidy_writer.write(
            filepath=self.config.output_dir / "raw_trajectory_data.csv",
            data={
                "positions": raw_data,
                "marker_names": marker_names
            }
        )
        logger.info("  ✓ Saved raw_trajectory_data.csv")

        # Save topology JSON (with display edges if passthrough markers exist)
        topology_dict = self.config.topology.to_dict()

        # If we have passthrough markers, add display edges connecting them
        if self.config.passthrough_markers:
            # Get the indices where passthrough markers start
            n_topology = len(self.config.topology.marker_names)

            # You may want to customize this - for now, just chain passthrough markers
            display_edges = list(self.config.topology.rigid_edges)

            # Chain passthrough markers together
            for i in range(len(self.config.passthrough_markers) - 1):
                display_edges.append((n_topology + i, n_topology + i + 1))

            topology_dict["display_edges"] = display_edges

        topology_path=self.config.output_dir / "topology.json"
        topology_path.write_text(data=topology_dict, encoding='utf-8')
        logger.info(f"  ✓ Saved topology to {topology_path.name}")


        logger.info(f"✓ Results saved to {self.config.output_dir}")

    def generate_viewer(self, *, result: RigidBodyResult) -> None:
        """Generate interactive HTML viewer.

        Creates rigid_body_viewer.html with 3D visualization.

        Args:
            result: Optimization result
        """
        logger.info("Generating viewer...")


        # Create generator instance
        viewer_generator = MocapViewerGenerator()

        # Generate the viewer with embedded data
        viewer_path = viewer_generator.generate(
            output_dir=self.config.output_dir,
            data_csv_path=self.config.output_dir / "trajectory_data.csv",
            raw_csv_path=self.config.output_dir / "raw_trajectory_data.csv",
            topology_json_path=self.config.output_dir / "topology.json",
            video_path=None
        )

        logger.info(f"  ✓ Generated {viewer_path.name}")
        logger.info(f"  → Open {viewer_path} in a browser to visualize")

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