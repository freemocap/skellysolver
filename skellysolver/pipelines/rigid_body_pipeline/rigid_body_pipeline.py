"""Rigid body tracking pipeline - bundle adjustment for rigid marker sets.

This pipeline jointly optimizes:
1. Reference geometry: The canonical positions of markers in the body frame
2. Poses: Rotation and translation of the body for each frame

The optimization enforces:
- Data fitting: Transformed reference matches observations
- Rigidity: Fixed distances between markers in reference
- Smoothness: Gradual motion changes over time
"""

import logging
import time

import numpy as np
import pyceres
from pydantic import Field

from skellysolver.data.trajectory_dataset import TrajectoryDataset
from skellysolver.pipelines.base_pipeline import PipelineConfig, BasePipeline
from skellysolver.pipelines.rigid_body_pipeline.rigid_body_cost_builder import MeasurementFactorBA, RigidBodyFactorBA, RotationSmoothnessFactor, TranslationSmoothnessFactor, ReferenceAnchorFactor
from skellysolver.pipelines.rigid_body_pipeline.rigid_body_result import RigidBodySolverResult
from skellysolver.pipelines.rigid_body_pipeline.rigid_body_topology import RigidBodyTopology
from skellysolver.solvers.base_solver import PyceresProblemSolver

logger = logging.getLogger(__name__)


class RigidBodyPipelineConfig(PipelineConfig):
    """Configuration for rigid body tracking pipeline.
    
    Attributes:
        topology: Defines which markers form rigid body and connectivity
        measurement_weight: Weight for fitting to measured data
        rigidity_weight: Weight for maintaining fixed edge lengths
        smoothness_weight: Weight for temporal smoothness
        soft_weight: Weight for soft (flexible) edge constraints
        anchor_weight: Weight for preventing reference geometry drift
    """
    
    topology: RigidBodyTopology
    
    # Cost weights
    measurement_weight: float = 100.0
    rigidity_weight: float = 500.0
    smoothness_weight: float = 200.0
    soft_weight: float = 10.0
    anchor_weight: float = 10.0
    
    def __str__(self) -> str:
        """Human-readable config."""
        lines = [
            "=" * 80,
            "RIGID BODY PIPELINE CONFIGURATION",
            "=" * 80,
            f"Input:  {self.input_path}",
            f"Output: {self.output_dir}",
            "",
            f"Topology: {self.topology}",
            "",
            "Cost Weights:",
            f"  Measurement (data fitting):  {self.measurement_weight}",
            f"  Rigidity (edge constraints): {self.rigidity_weight}",
            f"  Smoothness (temporal):       {self.smoothness_weight}",
            f"  Soft edges (flexible):       {self.soft_weight}",
            f"  Anchor (prevent drift):      {self.anchor_weight}",
            "",
            str(self.solver_config),
            "",
            str(self.parallel),
            "=" * 80,
        ]
        return "\n".join(lines)


class RigidBodyPipeline(BasePipeline):
    """Pipeline for rigid body tracking using bundle adjustment.
    
    Flow:
    1. Load trajectory data (markers over time)
    2. Chunk data if needed (handled by base class)
    3. For each chunk:
       a. Initialize reference geometry (median frame)
       b. Initialize poses (identity rotations, centroids)
       c. Build cost functions (measurement, rigidity, smoothness)
       d. Solve bundle adjustment
       e. Extract optimized poses and geometry
    4. Stitch chunks with domain-aware blending (handled by base class)
    5. Save results
    """
    
    config: RigidBodyPipelineConfig
    
    def setup_problem_and_solve(
        self,
        chunk_data: TrajectoryDataset | None = None
    ) -> RigidBodySolverResult:
        """Setup solver and solve for a chunk of data.
        
        This is called for each chunk (or once for non-chunked data).
        
        Args:
            chunk_data: Trajectory data for this chunk
            
        Returns:
            RigidBodySolverResult with optimized poses and geometry
        """
        if chunk_data is None:
            chunk_data = self.input_data
        logger.info("=" * 80)
        logger.info("RIGID BODY BUNDLE ADJUSTMENT - SETUP AND SOLVE PYCERES PROBLEM")
        logger.info("=" * 80)
        logger.info(f"Chunk: {chunk_data.n_frames} frames, {chunk_data.n_markers} markers")
        logger.info("\nBuilding optimization problem...")

        # Validate that chunk has all required markers
        self.config.topology.validate_trajectory_data(
            trajectory_dict={name: traj.data for name, traj in chunk_data.data.items()}
        )
        
        # 1. Create solver
        logger.info("\nBuilding optimization problem...")
        problem = pyceres.Problem()
        
        # 2. Initialize reference geometry (use median frame as initial guess)
        logger.info("Initializing reference geometry...")
        reference_geometry = self._initialize_reference_geometry(chunk_data=chunk_data)
        reference_params = reference_geometry.flatten()

        # 3. Estimate initial edge distances for constraints
        input_data_array = chunk_data.to_array()  # (n_frames, n_markers, 3)
        rigid_edges = self.config.topology.rigid_edges
        reference_distances = self.estimate_initial_distances(
            noisy_data=input_data_array,
            edges=rigid_edges,
            edge_type="rigid"
        )

        # =========================================================================
        # INITIALIZE POSES
        # =========================================================================
        logger.info("Initializing poses...")
        poses: list[tuple[np.ndarray, np.ndarray]] = []
        input_data_array = chunk_data.to_array()  # (n_frames, n_markers, 3)
        n_frames, n_markers, _ = input_data_array.shape
        for frame_idx in range(n_frames):
            quat_ceres = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
            translation = np.mean(input_data_array[frame_idx], axis=0)
            poses.append((quat_ceres, translation))

        # =========================================================================
        # BUILD OPTIMIZATION PROBLEM
        # =========================================================================
        logger.info("\nBuilding optimization problem...")
        problem = pyceres.Problem()

        # Add reference geometry as parameters
        problem.add_parameter_block(reference_params, chunk_data.n_markers * 3)

        # Add pose parameters
        pose_params: list[tuple[np.ndarray, np.ndarray]] = []
        for quat, trans in poses:
            problem.add_parameter_block(quat, 4)
            problem.add_parameter_block(trans, 3)
            problem.set_manifold(quat, pyceres.QuaternionManifold())
            pose_params.append((quat, trans))

        # DATA FITTING
        logger.info("  Adding measurement factors...")
        for frame_idx in range(n_frames):
            quat, trans = pose_params[frame_idx]
            for point_idx in range(n_markers):
                cost = MeasurementFactorBA(
                    measured_point=input_data_array[frame_idx, point_idx],
                    marker_idx=point_idx,
                    n_markers=n_markers,
                    weight=self.config.measurement_weight
                )
                problem.add_residual_block(cost, None, [quat, trans, reference_params])

        # RIGID BODY CONSTRAINTS
        logger.info(f"  Adding {len(rigid_edges)} rigid body constraints...")
        for i, j in rigid_edges:
            cost = RigidBodyFactorBA(
                marker_i=i,
                marker_j=j,
                n_markers=n_markers,
                target_distance=float(reference_distances[i, j]),
                weight=self.config.rigidity_weight
            )
            problem.add_residual_block(cost, None, [reference_params])

        # # SOFT EDGES
        # if soft_edges is not None and soft_distances is not None:
        #     logger.info(f"  Adding {len(soft_edges)} soft constraints...")
        #     for frame_idx in range(n_frames):
        #         quat, trans = pose_params[frame_idx]
        #         for i, j in soft_edges:
        #             cost = SoftDistanceFactorBA(
        #                 measured_point=noisy_data[frame_idx, j],
        #                 marker_idx_on_body=i,
        #                 n_markers=n_markers,
        #                 median_distance=soft_distances[i, j],
        #                 weight=lambda_soft
        #             )
        #             problem.add_residual_block(cost, None, [quat, trans, reference_params])

        # SMOOTHNESS
        logger.info("  Adding smoothness factors...")
        for frame_idx in range(n_frames - 1):
            quat_t, trans_t = pose_params[frame_idx]
            quat_t1, trans_t1 = pose_params[frame_idx + 1]

            rot_cost = RotationSmoothnessFactor(weight=self.config.smoothness_weight)
            problem.add_residual_block(rot_cost, None, [quat_t, quat_t1])

            trans_cost = TranslationSmoothnessFactor(weight=self.config.smoothness_weight)
            problem.add_residual_block(trans_cost, None, [trans_t, trans_t1])

        # REFERENCE ANCHOR
        logger.info("  Adding reference anchor...")
        anchor_cost = ReferenceAnchorFactor(
            initial_reference=reference_params,
            weight=self.config.measurement_weight * 0.1 # Weak weight to prevent drift
        )
        problem.add_residual_block(anchor_cost, None, [reference_params])

        logger.info(f"\n  Total residual blocks: {problem.num_residual_blocks()}")
        logger.info(f"  Total parameters: {problem.num_parameters()}")

        # =========================================================================
        # SOLVE
        # =========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("SOLVING")
        logger.info("=" * 80)

        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        options.minimizer_progress_to_stdout = True
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
        logger.info(f"  Iterations: {summary.num_successful_steps}")
        logger.info(f"  Time: {summary.total_time_in_seconds:.2f}s")

        # 8. Create result (parameters were modified in-place by solver)
        logger.info("\nExtracting results...")
        result = RigidBodySolverResult.from_solver_and_params(
            summary=summary,
            solve_time_seconds=solve_time,
            raw_data=chunk_data,
            pose_params=pose_params,
            reference_params=reference_params,
            marker_names=self.config.topology.marker_names
        )
        
        # Compute metrics
        logger.info("\nComputing metrics...")
        rigidity_metrics = result.compute_rigidity_metrics(
            rigid_edges=self.config.topology.rigid_edges,
            target_distances=reference_distances
        )
        
        return result
    
    def _initialize_reference_geometry(
        self,
        *,
        chunk_data: TrajectoryDataset
    ) -> np.ndarray:
        """Initialize reference geometry from median frame.
        
        The median frame provides a robust initial estimate that's less
        affected by outliers than the mean.
        
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
        
        median_frame = np.nanmedian(data_array, axis=0)  # (n_markers, 3)
        
        # Center at origin (remove mean)
        centroid = np.nanmean(median_frame, axis=0)
        reference_geometry = median_frame - centroid
        
        logger.info(f"  Reference centroid: {centroid}")
        logger.info(f"  Reference span: {np.nanmax(reference_geometry) - np.nanmin(reference_geometry):.3f}m")
        
        return reference_geometry

    def estimate_initial_distances(
            self,
            *,
            noisy_data: np.ndarray,
            edges: list[tuple[int, int]],
            edge_type: str = "rigid"
    ) -> np.ndarray:
        """
        Estimate initial edge distances from noisy data using median.

        Args:
            noisy_data: (n_frames, n_markers, 3)
            edges: List of (i, j) pairs
            edge_type: "rigid" or "soft" (for logging)

        Returns:
            (n_markers, n_markers) distance matrix
        """
        n_markers = noisy_data.shape[1]
        distances = np.zeros((n_markers, n_markers))

        logger.info(f"Estimating {edge_type} edge distances from data...")

        for i, j in edges:
            frame_distances = np.linalg.norm(
                noisy_data[:, i, :] - noisy_data[:, j, :],
                axis=1
            )
            median_dist = np.median(frame_distances)
            std_dist = np.std(frame_distances)
            distances[i, j] = median_dist
            distances[j, i] = median_dist

            logger.info(f"  Edge ({i},{j}): {median_dist:.4f}m ± {std_dist * 1000:.1f}mm")

        return distances
    
    def _initialize_poses(
        self,
        *,
        chunk_data: TrajectoryDataset,
        solver: PyceresProblemSolver
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Initialize poses for all frames.
        
        Initializes:
        - Rotations: Identity (no rotation)
        - Translations: Centroid of markers in each frame
        
        Args:
            chunk_data: Trajectory data
            solver: Solver to add parameter blocks to
            
        Returns:
            List of (quaternion, translation) tuples
        """
        # Get data array
        data_array = chunk_data.to_array(
            marker_names=self.config.topology.marker_names,
            fill_missing=False
        )  # (n_frames, n_markers, 3)
        
        pose_params = []
        
        for frame_idx in range(chunk_data.n_frames):
            # Identity rotation (pyceres uses [w, x, y, z])
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Translation = centroid of markers in this frame
            frame_data = data_array[frame_idx]  # (n_markers, 3)
            
            translation = np.nanmean(frame_data, axis=0)
            
            # Handle all-NaN case
            if np.any(np.isnan(translation)):
                translation = np.zeros(3)
            
            # Add to solver with proper manifold for quaternion
            solver.add_quaternion_parameter(
                name=f"quat_frame_{frame_idx}",
                parameter=quat
            )
            
            solver.add_parameter_block(
                name=f"trans_frame_{frame_idx}",
                parameters=translation
            )
            
            pose_params.append((quat, translation))
        
        return pose_params
    
    def run(self) -> None:
        """Run the full pipeline with config logging."""
        logger.info("\n" + str(self.config))
        super().run()
