"""Rigid body tracking pipeline - pose optimization with fixed reference geometry.

This pipeline optimizes ONLY the poses (rotation and translation) while keeping
the reference geometry fixed based on median distances from raw data.

The optimization:
1. Estimates reference geometry from median frame positions
2. Fixes all inter-marker distances at their median values from raw data
3. Optimizes only rotations and translations to fit the fixed geometry to measurements
4. Enforces temporal smoothness on the poses
"""
import json
import logging
import time

import numpy as np
import pyceres
from sklearn.manifold import MDS

from skellysolver.data.trajectory_dataset import TrajectoryDataset
from skellysolver.data.dataset_manager import save_trajectory_csv
from skellysolver.pipelines.base_pipeline import PipelineConfig, BasePipeline
from skellysolver.pipelines.rigid_body_pipeline.rigid_body_cost_builder import (
    MeasurementFactorBA,
    RotationSmoothnessFactor,
    TranslationSmoothnessFactor,
)
from skellysolver.pipelines.rigid_body_pipeline.rigid_body_result import RigidBodySolverResult
from skellysolver.pipelines.rigid_body_pipeline.rigid_body_topology import RigidBodyTopology
from skellysolver.solvers.base_solver import PyceresProblemSolver
from skellysolver.viewers.html_viewers.rigid_viewer import MocapViewerGenerator

logger = logging.getLogger(__name__)


class RigidBodyPipelineConfig(PipelineConfig):
    """Configuration for rigid body tracking pipeline.

    Attributes:
        topology: Defines which markers form rigid body and connectivity
        measurement_weight: Weight for fitting to measured data
        smoothness_weight: Weight for temporal smoothness
    """

    topology: RigidBodyTopology

    # Cost weights
    measurement_weight: float = 100.0
    smoothness_weight: float = 200.0

    def __str__(self) -> str:
        """Human-readable config."""
        lines = [
            "=" * 80,
            "RIGID BODY PIPELINE CONFIGURATION (FIXED GEOMETRY)",
            "=" * 80,
            f"Input:  {self.input_path}",
            f"Output: {self.output_dir}",
            "",
            f"Topology: {self.topology}",
            "",
            "Cost Weights:",
            f"  Measurement (data fitting):  {self.measurement_weight}",
            f"  Smoothness (temporal):       {self.smoothness_weight}",
            "",
            "Note: Reference geometry is FIXED (not optimized)",
            "",
            str(self.solver_config),
            "",
            str(self.parallel),
            "=" * 80,
        ]
        return "\n".join(lines)


class RigidBodyPipeline(BasePipeline):
    """Pipeline for rigid body tracking with fixed reference geometry.

    Flow:
    1. Load trajectory data (markers over time)
    2. Chunk data if needed (handled by base class)
    3. For each chunk:
       a. Estimate reference geometry from median frame
       b. Fix inter-marker distances at median values
       c. Initialize poses (identity rotations, centroids)
       d. Build cost functions (measurement, smoothness ONLY)
       e. Solve for poses only (reference geometry is constant)
       f. Extract optimized poses
    4. Stitch chunks with domain-aware blending (handled by base class)
    5. Save results
    """

    config: RigidBodyPipelineConfig

    def setup_problem_and_solve(
        self,
        *,
        chunk_data: TrajectoryDataset | None = None
    ) -> RigidBodySolverResult:
        """Setup solver and solve for a chunk of data.

        This is called for each chunk (or once for non-chunked data).

        Args:
            chunk_data: Trajectory data for this chunk

        Returns:
            RigidBodySolverResult with optimized poses and fixed geometry
        """
        if chunk_data is None:
            chunk_data = self.input_data

        logger.info("=" * 80)
        logger.info("RIGID BODY POSE OPTIMIZATION - FIXED REFERENCE GEOMETRY")
        logger.info("=" * 80)
        logger.info(f"Chunk: {chunk_data.n_frames} frames, {chunk_data.n_markers} markers")

        # Validate that chunk has all required markers
        self.config.topology.validate_trajectory_data(
            trajectory_dict={name: traj.data for name, traj in chunk_data.data.items()}
        )

        # =========================================================================
        # ESTIMATE FIXED REFERENCE GEOMETRY
        # =========================================================================
        logger.info("\nEstimating fixed reference geometry from raw data...")
        reference_geometry = self._estimate_reference_geometry_from_median(
            chunk_data=chunk_data
        )
        reference_params = reference_geometry.flatten()

        logger.info(f"  Reference shape: {reference_geometry.shape}")
        logger.info(f"  Reference centroid: {np.mean(reference_geometry, axis=0)}")
        logger.info(f"  Reference span: {np.ptp(reference_geometry):.3f}m")

        # Compute and log edge distances
        n_markers = len(self.config.topology.marker_names)
        logger.info("\nFixed edge distances:")
        for i, j in self.config.topology.rigid_edges[:5]:  # Show first 5
            dist = np.linalg.norm(reference_geometry[i] - reference_geometry[j])
            logger.info(f"  Edge ({i},{j}): {dist:.4f}m")
        if len(self.config.topology.rigid_edges) > 5:
            logger.info(f"  ... and {len(self.config.topology.rigid_edges) - 5} more edges")

        # =========================================================================
        # INITIALIZE POSES
        # =========================================================================
        logger.info("\nInitializing poses...")
        poses: list[tuple[np.ndarray, np.ndarray]] = []
        input_data_array = chunk_data.to_array()  # (n_frames, n_markers, 3)
        n_frames, n_markers, _ = input_data_array.shape

        for frame_idx in range(n_frames):
            quat_ceres = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation [w,x,y,z]
            translation = np.nanmean(input_data_array[frame_idx], axis=0)
            if np.any(np.isnan(translation)):
                translation = np.zeros(3)
            poses.append((quat_ceres, translation))

        # =========================================================================
        # BUILD OPTIMIZATION PROBLEM
        # =========================================================================
        logger.info("\nBuilding optimization problem...")
        problem = pyceres.Problem()

        # NOTE: We do NOT add reference_params as a parameter block
        # This keeps it fixed during optimization

        # Add pose parameters (these WILL be optimized)
        pose_params: list[tuple[np.ndarray, np.ndarray]] = []
        for quat, trans in poses:
            problem.add_parameter_block(quat, 4)
            problem.add_parameter_block(trans, 3)
            problem.set_manifold(quat, pyceres.QuaternionManifold())
            pose_params.append((quat, trans))

        # =========================================================================
        # ADD MEASUREMENT COSTS (FIT POSES TO DATA)
        # =========================================================================
        logger.info("  Adding measurement factors...")
        loss = pyceres.HuberLoss(0.05)  # ~50mm threshold

        measurement_count = 0
        for frame_idx in range(n_frames):
            quat, trans = pose_params[frame_idx]
            for point_idx in range(n_markers):
                measured_point = input_data_array[frame_idx, point_idx]

                # Skip NaN measurements
                if np.any(np.isnan(measured_point)):
                    continue

                cost = MeasurementFactorBA(
                    measured_point=measured_point,
                    marker_idx=point_idx,
                    n_markers=n_markers,
                    weight=self.config.measurement_weight
                )
                # reference_params is passed but not optimized (not added as parameter block)
                problem.add_residual_block(cost, loss, [quat, trans, reference_params])
                measurement_count += 1

        logger.info(f"    Added {measurement_count} measurement costs")

        # =========================================================================
        # ADD SMOOTHNESS COSTS (TEMPORAL CONSISTENCY)
        # =========================================================================
        logger.info("  Adding smoothness factors...")
        smoothness_count = 0
        for frame_idx in range(n_frames - 1):
            quat_t, trans_t = pose_params[frame_idx]
            quat_t1, trans_t1 = pose_params[frame_idx + 1]

            rot_cost = RotationSmoothnessFactor(weight=self.config.smoothness_weight)
            problem.add_residual_block(rot_cost, None, [quat_t, quat_t1])

            trans_cost = TranslationSmoothnessFactor(weight=self.config.smoothness_weight)
            problem.add_residual_block(trans_cost, None, [trans_t, trans_t1])
            smoothness_count += 2

        logger.info(f"    Added {smoothness_count} smoothness costs")

        logger.info(f"\n  Total residual blocks: {problem.num_residual_blocks()}")
        logger.info(f"  Total parameters: {problem.num_parameters()}")
        logger.info(f"  Note: Reference geometry ({reference_params.size} params) is FIXED")

        # =========================================================================
        # SOLVE
        # =========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("SOLVING (POSES ONLY)")
        logger.info("=" * 80)

        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        options.minimizer_progress_to_stdout = self.config.solver_config.minimizer_progress_to_stdout
        options.max_num_iterations = self.config.solver_config.max_num_iterations
        options.function_tolerance = self.config.solver_config.function_tolerance
        options.gradient_tolerance = self.config.solver_config.gradient_tolerance
        options.parameter_tolerance = self.config.solver_config.parameter_tolerance

        summary = pyceres.SolverSummary()
        tik = time.perf_counter()
        pyceres.solve(options, problem, summary)
        solve_time = time.perf_counter() - tik

        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Status: {summary.termination_type}")
        logger.info(f"  Initial cost: {summary.initial_cost:.2f}")
        logger.info(f"  Final cost: {summary.final_cost:.2f}")
        logger.info(f"  Cost reduction: {((summary.initial_cost - summary.final_cost) / summary.initial_cost * 100):.1f}%")
        logger.info(f"  Iterations: {summary.num_successful_steps}")
        logger.info(f"  Time: {summary.total_time_in_seconds:.2f}s")

        # =========================================================================
        # CREATE RESULT
        # =========================================================================
        logger.info("\nExtracting results...")
        result = RigidBodySolverResult.from_solver_and_params(
            summary=summary,
            solve_time_seconds=solve_time,
            raw_data=chunk_data,
            pose_params=pose_params,
            reference_params=reference_params,  # Fixed geometry
            marker_names=self.config.topology.marker_names
        )

        # Compute rigidity metrics (should be good since geometry is fixed)
        logger.info("\nComputing rigidity metrics...")
        reference_distances = self.config.topology.compute_reference_distances(
            reference_geometry=reference_geometry
        )
        rigidity_metrics = result.compute_rigidity_metrics(
            rigid_edges=self.config.topology.rigid_edges,
            target_distances=reference_distances
        )
        logger.info(f"\n{json.dumps(rigidity_metrics, indent=2)}")
        self.solver_result = result  # Store for viewer generation
        self.generate_viewer()

        return result

    def _estimate_reference_geometry_from_median(
        self,
        *,
        chunk_data: TrajectoryDataset
    ) -> np.ndarray:
        """Estimate reference geometry with median distances between all markers.

        Computes the median distance between each pair of markers across all frames,
        then uses MDS to construct 3D positions that match those distances.

        Args:
            chunk_data: Trajectory data

        Returns:
            (n_markers, 3) reference positions (centered at origin)
        """

        # Get data array
        data_array = chunk_data.to_array(
            marker_names=self.config.topology.marker_names,
            fill_missing=False
        )  # (n_frames, n_markers, 3)

        n_frames, n_markers, _ = data_array.shape

        # Compute median distance matrix
        logger.info("  Computing median distances between all marker pairs...")
        median_distances = np.zeros((n_markers, n_markers))

        for i in range(n_markers):
            for j in range(i + 1, n_markers):
                # Compute distance in each frame
                frame_distances = np.linalg.norm(
                    data_array[:, i, :] - data_array[:, j, :],
                    axis=1
                )
                # Take median
                median_dist = np.nanmedian(frame_distances)
                median_distances[i, j] = median_dist
                median_distances[j, i] = median_dist

        # Use MDS to find 3D positions that match the median distance matrix
        logger.info("  Using MDS to construct reference geometry from median distances...")
        mds = MDS(
            n_components=3,
            dissimilarity='precomputed',
            random_state=42,
            max_iter=1000,
            eps=1e-9
        )
        reference_geometry = mds.fit_transform(X=median_distances)

        # Center at origin
        centroid = np.mean(reference_geometry, axis=0)
        reference_geometry = reference_geometry - centroid

        # Verify: log distances in reference vs median distances
        logger.info("\nReference geometry verification (median distance vs actual):")
        for i, j in self.config.topology.rigid_edges[:10]:  # Show first 10
            median_dist = median_distances[i, j]
            ref_dist = np.linalg.norm(reference_geometry[i] - reference_geometry[j])
            error_mm = abs(ref_dist - median_dist) * 1000

            logger.info(f"  Edge ({i},{j}): median={median_dist:.4f}m, "
                       f"ref={ref_dist:.4f}m, error={error_mm:.2f}mm")

        if len(self.config.topology.rigid_edges) > 10:
            logger.info(f"  ... and {len(self.config.topology.rigid_edges) - 10} more edges")

        # Log overall MDS stress (how well distances are preserved)
        logger.info(f"\n  MDS stress: {mds.stress_:.2f} (lower is better)")
        return reference_geometry




    def generate_viewer(self) -> None:
        """Generate interactive HTML viewer for rigid body tracking results."""
        if self.solver_result is None:
            logger.warning("No solver result available. Cannot generate viewer.")
            return

        try:


            logger.info("\n" + "=" * 80)
            logger.info("GENERATING INTERACTIVE VIEWER")
            logger.info("=" * 80)

            # Save topology to JSON
            topology_path = self.config.output_dir / "topology.json"
            self.config.topology.save_json(filepath=topology_path)
            logger.info(f"  Saved topology: {topology_path}")

            # Save raw data CSV
            raw_csv_path = self.config.output_dir / "raw_data.csv"
            save_trajectory_csv(dataset=self.input_data, filepath=raw_csv_path)
            logger.info(f"  Saved raw data: {raw_csv_path}")

            # Save optimized data CSV
            data_csv_path = self.config.output_dir / "optimized_data.csv"
            save_trajectory_csv(dataset=self.solver_result.optimized_data, filepath=data_csv_path)
            logger.info(f"  Saved optimized data: {data_csv_path}")

            # Generate viewer
            generator = MocapViewerGenerator()
            viewer_path = generator.generate(
                output_dir=self.config.output_dir,
                data_csv_path=data_csv_path,
                raw_csv_path=raw_csv_path,
                topology_json_path=topology_path,
                video_path=None  # Optional: add video support later
            )

            logger.info(f"\n✓ Interactive viewer generated: {viewer_path}")
            logger.info(f"  Open in browser to visualize results")
            logger.info("=" * 80)


        except Exception as e:
            logger.error(f"Error generating viewer: {e}", exc_info=True)
            logger.warning("Continuing without viewer generation.")