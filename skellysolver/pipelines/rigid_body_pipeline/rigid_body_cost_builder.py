"""Cost builder for rigid body tracking.

Converts RigidBodyTopology constraint into pyceres cost functions.
"""

import logging
import pyceres

import numpy as np
from scipy.spatial.transform import Rotation

from skellysolver.pipelines.rigid_body_pipeline.rigid_body_topology import RigidBodyTopology
from skellysolver.solvers.base_cost_builder import BaseCostBuilder
from skellysolver.solvers.costs.cost_info_models import (
    CostCollection,
    MeasurementCostInfo,
    RotationSmoothnessCostInfo,
    TranslationSmoothnessCostInfo,
    AnchorCostInfo,
)
from skellysolver.solvers.costs.measurement_costs import (
    RigidPoint3DMeasurementBundleAdjustment
)
from skellysolver.solvers.costs.edge_costs import (
    SoftEdgeCost,
    ReferenceAnchorCost,
)
from skellysolver.solvers.costs.smoothness_costs import (
    RotationSmoothnessCost,
    TranslationSmoothnessCost,
)
from skellysolver.data.trajectory_dataset import TrajectoryDataset

logger = logging.getLogger(__name__)


# =============================================================================
# COST FUNCTIONS
# =============================================================================

class MeasurementFactorBA(pyceres.CostFunction):
    """Data fitting: measured point should match transformed reference."""

    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        marker_idx: int,
        n_markers: int,
        weight: float = 100.0
    ) -> None:
        super().__init__()
        self.measured_point = measured_point.copy()
        self.marker_idx = marker_idx
        self.n_markers = n_markers
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 3, n_markers * 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]
        reference_flat = parameters[2]

        start_idx = self.marker_idx * 3
        reference_point = reference_flat[start_idx:start_idx + 3]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        predicted = R @ reference_point + translation
        residuals[:] = self.weight * (self.measured_point - predicted)

        if jacobians is not None:
            eps = 1e-8

            if jacobians[0] is not None:
                for j in range(4):
                    quat_plus = quat.copy()
                    quat_plus[j] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    predicted_plus = R_plus @ reference_point + translation
                    residual_plus = self.weight * (self.measured_point - predicted_plus)
                    for i in range(3):
                        jacobians[0][i * 4 + j] = (residual_plus[i] - residuals[i]) / eps

            if jacobians[1] is not None:
                for i in range(3):
                    for j in range(3):
                        jacobians[1][i * 3 + j] = -self.weight if i == j else 0.0

            if jacobians[2] is not None:
                jacobians[2][:] = 0.0
                start_idx = self.marker_idx * 3
                for i in range(3):
                    for j in range(3):
                        jacobians[2][i * (self.n_markers * 3) + (start_idx + j)] = -self.weight * R[i, j]

        return True


class RigidBodyFactorBA(pyceres.CostFunction):
    """Rigid body constraint: edge length in reference geometry."""

    def __init__(
        self,
        *,
        marker_i: int,
        marker_j: int,
        n_markers: int,
        target_distance: float,
        weight: float = 100.0
    ) -> None:
        super().__init__()
        self.marker_i = marker_i
        self.marker_j = marker_j
        self.n_markers = n_markers
        self.target_dist = target_distance
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([n_markers * 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        reference_flat = parameters[0]

        ref_i = reference_flat[self.marker_i * 3:(self.marker_i + 1) * 3]
        ref_j = reference_flat[self.marker_j * 3:(self.marker_j + 1) * 3]

        diff = ref_i - ref_j
        current_dist = np.linalg.norm(diff)
        residuals[0] = self.weight * (current_dist - self.target_dist)

        if jacobians is not None and jacobians[0] is not None:
            eps = 1e-8
            jacobians[0][:] = 0.0

            for k in range(3):
                ref_flat_plus = reference_flat.copy()
                ref_flat_plus[self.marker_i * 3 + k] += eps
                ref_i_plus = ref_flat_plus[self.marker_i * 3:(self.marker_i + 1) * 3]
                ref_j_plus = ref_flat_plus[self.marker_j * 3:(self.marker_j + 1) * 3]
                diff_plus = ref_i_plus - ref_j_plus
                dist_plus = np.linalg.norm(diff_plus)
                residual_plus = self.weight * (dist_plus - self.target_dist)
                jacobians[0][self.marker_i * 3 + k] = (residual_plus - residuals[0]) / eps

            for k in range(3):
                ref_flat_plus = reference_flat.copy()
                ref_flat_plus[self.marker_j * 3 + k] += eps
                ref_i_plus = ref_flat_plus[self.marker_i * 3:(self.marker_i + 1) * 3]
                ref_j_plus = ref_flat_plus[self.marker_j * 3:(self.marker_j + 1) * 3]
                diff_plus = ref_i_plus - ref_j_plus
                dist_plus = np.linalg.norm(diff_plus)
                residual_plus = self.weight * (dist_plus - self.target_dist)
                jacobians[0][self.marker_j * 3 + k] = (residual_plus - residuals[0]) / eps

        return True


class SoftDistanceFactorBA(pyceres.CostFunction):
    """Soft distance constraint between measured point and reference point."""

    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        marker_idx_on_body: int,
        n_markers: int,
        median_distance: float,
        weight: float = 10.0
    ) -> None:
        super().__init__()
        self.measured_point = measured_point.copy()
        self.marker_idx = marker_idx_on_body
        self.n_markers = n_markers
        self.median_dist = median_distance
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([4, 3, n_markers * 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat = parameters[0]
        translation = parameters[1]
        reference_flat = parameters[2]

        start_idx = self.marker_idx * 3
        ref_point = reference_flat[start_idx:start_idx + 3]

        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        transformed_ref = R @ ref_point + translation
        diff = self.measured_point - transformed_ref
        current_dist = np.linalg.norm(diff)
        residuals[0] = self.weight * (current_dist - self.median_dist)

        if jacobians is not None:
            eps = 1e-8

            if jacobians[0] is not None:
                for j in range(4):
                    quat_plus = quat.copy()
                    quat_plus[j] += eps
                    quat_plus = quat_plus / np.linalg.norm(quat_plus)
                    R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
                    transformed_plus = R_plus @ ref_point + translation
                    diff_plus = self.measured_point - transformed_plus
                    dist_plus = np.linalg.norm(diff_plus)
                    residual_plus = self.weight * (dist_plus - self.median_dist)
                    jacobians[0][j] = (residual_plus - residuals[0]) / eps

            if jacobians[1] is not None:
                if current_dist > 1e-10:
                    grad = -self.weight * diff / current_dist
                    jacobians[1][:] = grad
                else:
                    jacobians[1][:] = 0.0

            if jacobians[2] is not None:
                jacobians[2][:] = 0.0
                if current_dist > 1e-10:
                    start_idx = self.marker_idx * 3
                    grad_ref = self.weight * (diff / current_dist) @ R
                    for i in range(3):
                        jacobians[2][start_idx + i] = grad_ref[i]

        return True


class RotationSmoothnessFactor(pyceres.CostFunction):
    """Rotation smoothness between consecutive frames."""

    def __init__(self, *, weight: float = 500.0) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(4)
        self.set_parameter_block_sizes([4, 4])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        quat_t = parameters[0]
        quat_t1 = parameters[1]

        if np.dot(quat_t, quat_t1) < 0:
            quat_t1_corrected = -quat_t1.copy()
        else:
            quat_t1_corrected = quat_t1.copy()

        residuals[:] = self.weight * (quat_t1_corrected - quat_t)

        if jacobians is not None:
            if jacobians[0] is not None:
                for i in range(4):
                    for j in range(4):
                        jacobians[0][i * 4 + j] = -self.weight if i == j else 0.0
            if jacobians[1] is not None:
                sign = -1.0 if np.dot(quat_t, quat_t1) < 0 else 1.0
                for i in range(4):
                    for j in range(4):
                        jacobians[1][i * 4 + j] = self.weight * sign if i == j else 0.0

        return True


class TranslationSmoothnessFactor(pyceres.CostFunction):
    """Translation smoothness between consecutive frames."""

    def __init__(self, *, weight: float = 200.0) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        trans_t = parameters[0]
        trans_t1 = parameters[1]
        residuals[:] = self.weight * (trans_t1 - trans_t)

        if jacobians is not None:
            if jacobians[0] is not None:
                for i in range(3):
                    for j in range(3):
                        jacobians[0][i * 3 + j] = -self.weight if i == j else 0.0
            if jacobians[1] is not None:
                for i in range(3):
                    for j in range(3):
                        jacobians[1][i * 3 + j] = self.weight if i == j else 0.0

        return True


class ReferenceAnchorFactor(pyceres.CostFunction):
    """Soft anchor to prevent reference from drifting too far."""

    def __init__(self, *, initial_reference: np.ndarray, weight: float = 10.0) -> None:
        super().__init__()
        self.initial_ref = initial_reference.copy()
        self.weight = weight
        n_params = len(initial_reference)
        self.set_num_residuals(n_params)
        self.set_parameter_block_sizes([n_params])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        reference = parameters[0]
        residuals[:] = self.weight * (reference - self.initial_ref)

        if jacobians is not None and jacobians[0] is not None:
            n = len(residuals)
            for i in range(n):
                for j in range(n):
                    jacobians[0][i * n + j] = self.weight if i == j else 0.0

        return True


class RigidBodyCostBuilder(BaseCostBuilder[RigidBodyTopology]):
    """Builds cost functions for rigid body bundle adjustment.
    
    Creates:
    - Measurement costs: Fit transformed reference to observed markers
    - Rigid edge costs: Maintain fixed distances in reference geometry
    - Soft edge costs: Encourage (but don't require) distances
    - Smoothness costs: Penalize rapid motion changes
    - Anchor cost: Prevent reference geometry drift
    """
    
    constraint: RigidBodyTopology
    
    def build_all_costs(
        self,
        *,
        chunk_data: TrajectoryDataset,
        reference_params: np.ndarray,
        pose_params: list[tuple[np.ndarray, np.ndarray]],
        reference_distances: np.ndarray,
        measurement_weight: float,
        rigidity_weight: float,
        smoothness_weight: float,
        soft_weight: float = 10.0,
        anchor_weight: float = 1.0,
        **kwargs
    ) -> CostCollection:
        """Build all costs for rigid body optimization.
        
        Args:
            chunk_data: Trajectory data for this chunk
            reference_params: Flattened reference geometry (n_markers * 3,)
            pose_params: List of (quat, trans) for each frame
            reference_distances: (n_markers, n_markers) distance matrix
            measurement_weight: Weight for data fitting
            rigidity_weight: Weight for rigid edge constraints
            smoothness_weight: Weight for temporal smoothness
            soft_weight: Weight for soft edges
            anchor_weight: Weight for reference anchor
            
        Returns:
            CostCollection with all generated costs
        """
        collection = CostCollection()
        
        logger.info("Building rigid body costs...")
        logger.info(f"  Topology: {self.constraint}")
        logger.info(f"  Frames: {chunk_data.n_frames}")
        logger.info(f"  Weights: data={measurement_weight}, rigid={rigidity_weight}, "
                   f"smooth={smoothness_weight}, soft={soft_weight}")
        
        # 1. Measurement costs (data fitting)
        measurement_costs = self._build_measurement_costs(
            chunk_data=chunk_data,
            reference_params=reference_params,
            pose_params=pose_params,
            weight=measurement_weight
        )
        collection.extend(costs=measurement_costs)
        logger.info(f"  Added {len(measurement_costs)} measurement costs")
        
        # 2. Rigid edge costs (geometry constraints)
        rigid_costs = self._build_rigid_edge_costs(
            reference_params=reference_params,
            reference_distances=reference_distances,
            weight=rigidity_weight
        )
        collection.extend(costs=rigid_costs)
        logger.info(f"  Added {len(rigid_costs)} rigid edge costs")
        
        # 3. Soft edge costs (if any)
        if self.constraint.soft_edges:
            soft_costs = self._build_soft_edge_costs(
                chunk_data=chunk_data,
                reference_params=reference_params,
                pose_params=pose_params,
                reference_distances=reference_distances,
                weight=soft_weight
            )
            collection.extend(costs=soft_costs)
            logger.info(f"  Added {len(soft_costs)} soft edge costs")
        
        # 4. Smoothness costs (temporal)
        smoothness_costs = self._build_smoothness_costs(
            pose_params=pose_params,
            weight=smoothness_weight
        )
        collection.extend(costs=smoothness_costs)
        logger.info(f"  Added {len(smoothness_costs)} smoothness costs")
        
        # 5. Reference anchor (prevent drift)
        anchor_cost = self._build_anchor_cost(
            reference_params=reference_params,
            weight=anchor_weight
        )
        collection.add(cost_info=anchor_cost)
        logger.info(f"  Added anchor cost")
        
        logger.info(f"Total costs: {collection.total_costs}")
        
        return collection
    
    def _build_measurement_costs(
        self,
        *,
        chunk_data: TrajectoryDataset,
        reference_params: np.ndarray,
        pose_params: list[tuple[np.ndarray, np.ndarray]],
        weight: float
    ) -> list[MeasurementCostInfo]:
        """Build measurement fitting costs for all frames and markers."""
        costs = []
        n_markers = self.constraint.n_markers
        
        # Get data as array
        data_array = chunk_data.to_array(
            marker_names=self.constraint.marker_names,
            fill_missing=False
        )  # (n_frames, n_markers, 3)
        
        for frame_idx in range(chunk_data.n_frames):
            quat, trans = pose_params[frame_idx]
            
            for marker_idx in range(n_markers):
                measured_point = data_array[frame_idx, marker_idx]
                
                # Skip NaN measurements
                if np.any(np.isnan(measured_point)):
                    continue
                
                cost_fn = RigidPoint3DMeasurementBundleAdjustment(
                    measured_point=measured_point,
                    marker_idx=marker_idx,
                    n_markers=n_markers,
                    weight=weight
                )
                
                cost_info = MeasurementCostInfo(
                    cost=cost_fn,
                    parameters=[quat, trans, reference_params],
                    keypoint_name=self.constraint.marker_names[marker_idx],
                    frame_index=frame_idx,
                    weight=weight
                )
                
                costs.append(cost_info)
        
        return costs
    


    def _build_smoothness_costs(
        self,
        *,
        pose_params: list[tuple[np.ndarray, np.ndarray]],
        weight: float
    ) -> list[RotationSmoothnessCostInfo | TranslationSmoothnessCostInfo]:
        """Build temporal smoothness costs."""
        costs = []
        n_frames = len(pose_params)
        
        for frame_idx in range(n_frames - 1):
            quat_t, trans_t = pose_params[frame_idx]
            quat_t1, trans_t1 = pose_params[frame_idx + 1]
            
            # Rotation smoothness
            rot_cost = RotationSmoothnessCost(weight=weight)
            rot_info = RotationSmoothnessCostInfo(
                cost=rot_cost,
                parameters=[quat_t, quat_t1],
                frame_from=frame_idx,
                frame_to=frame_idx + 1,
                weight=weight
            )
            costs.append(rot_info)
            
            # Translation smoothness
            trans_cost = TranslationSmoothnessCost(weight=weight)
            trans_info = TranslationSmoothnessCostInfo(
                cost=trans_cost,
                parameters=[trans_t, trans_t1],
                frame_from=frame_idx,
                frame_to=frame_idx + 1,
                weight=weight
            )
            costs.append(trans_info)
        
        return costs
    
    def _build_anchor_cost(
        self,
        *,
        reference_params: np.ndarray,
        weight: float
    ) -> AnchorCostInfo:
        """Build reference anchor cost to prevent drift."""
        initial_reference = reference_params.copy()
        
        cost_fn = ReferenceAnchorCost(
            initial_reference=initial_reference,
            weight=weight
        )
        
        cost_info = AnchorCostInfo(
            cost=cost_fn,
            parameters=[reference_params],
            weight=weight
        )
        
        return cost_info




