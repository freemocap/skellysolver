"""Bundle adjustment approach: Optimize BOTH positions AND distances.

This is more flexible than precomputing distances. The optimizer jointly
estimates:
1. Keypoint trajectories (positions over time)
2. Skeleton structure (target distances between keypoints)

Benefits:
- Adapts to systematic measurement errors
- Can discover true skeleton structure from data
- More robust to initialization errors
- No need for perfect initial distance estimates

Tradeoffs:
- More parameters to optimize
- Needs anchoring to prevent degenerate solutions
- Slightly slower convergence
"""

import logging
import numpy as np
import pyceres
from itertools import combinations

from skellysolver.pipelines.skeleton_pipeline.skeleton_pipeline_config import SkeletonPipelineConfig
from skellysolver.solvers.base_cost_builder import BaseCostBuilder
from skellysolver.solvers.constraints.skeleton_constraint import SkeletonConstraint
from skellysolver.solvers.costs.cost_info_models import (
    SegmentRigidityCostInfo,
    LinkageStiffnessCostInfo,
    TranslationSmoothnessCostInfo,
    MeasurementCostInfo,
    CostCollection,
    AnchorCostInfo
)
from skellysolver.data.trajectory_dataset import TrajectoryDataset
from skellysolver.solvers.costs.edge_costs import RigidEdgeCost
from skellysolver.utilities.arbitrary_types_model import ABaseModel

logger = logging.getLogger(__name__)


class SkeletonCostBuildResult(ABaseModel):
    """Result of building costs with optimized distances.

    Attributes:
        costs: Collection of all generated costs
        distance_parameters: Dict of (kp_i, kp_j) -> distance parameter array
        initial_distances: Initial distance estimates for anchoring
    """
    costs: CostCollection
    distance_parameters: dict[tuple[str, str], np.ndarray]
    initial_distances: dict[tuple[str, str], float]


class OptimizedDistanceRigidEdgeCost(pyceres.CostFunction):
    """Enforce distance constraint with OPTIMIZED target distance.

    Model:
        current_distance = ||pos_i - pos_j||
        target_distance = distance_param[0]  # This is optimized!
        residual = current_distance - target_distance

    The target distance is now a parameter that gets optimized.
    """

    def __init__(self, *, weight: float) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([3, 3, 1])  # pos_i, pos_j, distance

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        """Evaluate edge constraint with optimized distance."""
        pos_i = parameters[0]
        pos_j = parameters[1]
        target_distance = parameters[2][0]  # The distance parameter

        # Current distance
        diff = pos_i - pos_j
        current_dist = np.linalg.norm(diff)

        # Residual
        residuals[0] = self.weight * (current_dist - target_distance)

        # Jacobians
        if jacobians is not None:
            eps = 1e-8

            # d/d(pos_i)
            if jacobians[0] is not None:
                for i in range(3):
                    pos_i_plus = pos_i.copy()
                    pos_i_plus[i] += eps
                    diff_plus = pos_i_plus - pos_j
                    dist_plus = np.linalg.norm(diff_plus)
                    jacobians[0][i] = self.weight * (dist_plus - current_dist) / eps

            # d/d(pos_j)
            if jacobians[1] is not None:
                for i in range(3):
                    pos_j_plus = pos_j.copy()
                    pos_j_plus[i] += eps
                    diff_plus = pos_i - pos_j_plus
                    dist_plus = np.linalg.norm(diff_plus)
                    jacobians[1][i] = self.weight * (dist_plus - current_dist) / eps

            # d/d(target_distance) = -weight (simple!)
            if jacobians[2] is not None:
                jacobians[2][0] = -self.weight

        return True


class DistanceAnchorCost(pyceres.CostFunction):
    """Anchor distance parameter to initial estimate.

    Model:
        residual = current_distance - initial_distance

    This prevents distances from drifting to arbitrary values.
    Use low weight to allow flexibility.
    """

    def __init__(
        self,
        *,
        initial_distance: float,
        weight: float
    ) -> None:
        super().__init__()
        self.initial_distance = initial_distance
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        """Evaluate distance anchor."""
        current_distance = parameters[0][0]

        # Residual
        residuals[0] = self.weight * (current_distance - self.initial_distance)

        # Jacobian is just weight
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][0] = self.weight

        return True


class DistanceLowerBoundCost(pyceres.CostFunction):
    """Prevent distances from collapsing to zero or negative.

    Model:
        if distance < min_distance:
            residual = (min_distance - distance)^2
        else:
            residual = 0

    This is a soft constraint that kicks in when distance gets too small.
    """

    def __init__(
        self,
        *,
        min_distance: float,
        weight: float
    ) -> None:
        super().__init__()
        self.min_distance = min_distance
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        """Evaluate lower bound constraint."""
        distance = parameters[0][0]

        if distance < self.min_distance:
            # Violation - penalize
            violation = self.min_distance - distance
            residuals[0] = self.weight * violation

            # Jacobian
            if jacobians is not None and jacobians[0] is not None:
                jacobians[0][0] = -self.weight
        else:
            # No violation
            residuals[0] = 0.0

            if jacobians is not None and jacobians[0] is not None:
                jacobians[0][0] = 0.0

        return True


class DirectKeypointMeasurementCost(pyceres.CostFunction):
    """Direct measurement fitting."""

    def __init__(self, *, measured_point: np.ndarray, weight: float) -> None:
        super().__init__()
        self.measured = measured_point.copy()
        self.weight = weight
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3])

    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        position = parameters[0]
        residual = self.measured - position
        residuals[:] = self.weight * residual

        if jacobians is not None and jacobians[0] is not None:
            for i in range(3):
                for j in range(3):
                    jacobians[0][i * 3 + j] = -self.weight if i == j else 0.0

        return True


class DirectKeypointSmoothnessCost(pyceres.CostFunction):
    """Temporal smoothness."""

    def __init__(self, *, weight: float) -> None:
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
        pos_t = parameters[0]
        pos_t1 = parameters[1]
        residual = pos_t1 - pos_t
        residuals[:] = self.weight * residual

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


class SkeletonCostBuilder(BaseCostBuilder[SkeletonConstraint]):
    @property
    def skeleton(self) -> SkeletonConstraint:
        return self.constraint



    def _initialize_segment_lengths(
        self,
        *,
        input_data: TrajectoryDataset,
        min_confidence: float,
        segment_rigidity_threshold: float
    ) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
        distance_params: dict[tuple[str, str], np.ndarray] = {}
        initial_distances: dict[tuple[str, str], float] = {}

        # Collect all constrained pairs
        constrained_pairs: set[tuple[str, str]] = set()

        # From segments
        for segment in self.skeleton.segments:
            if segment.rigidity < segment_rigidity_threshold:
                continue
            keypoint_names = [segment.parent.name] + [kp.name for kp in segment.children]
            for name_i, name_j in combinations(keypoint_names, 2):
                pair = tuple(sorted([name_i, name_j]))
                constrained_pairs.add(pair)



        # Initialize each distance parameter
        for pair in constrained_pairs:
            name_i, name_j = pair
            traj_i = input_data.data[name_i]
            traj_j = input_data.data[name_j]

            # Get valid frames
            valid_i = traj_i.is_valid(min_confidence=min_confidence)
            valid_j = traj_j.is_valid(min_confidence=min_confidence)
            valid_both = valid_i & valid_j

            if not np.any(valid_both):
                logger.warning(f"No valid frames for pair ({name_i}, {name_j})")
                continue
            else:
                # Compute median distance from data
                distances = []
                for frame_idx in np.where(valid_both)[0]:
                    pos_i = traj_i.data[frame_idx]
                    pos_j = traj_j.data[frame_idx]
                    dist = np.linalg.norm(pos_i - pos_j)
                    distances.append(dist)
                initial_dist = float(np.median(distances))

            # Create parameter (1D array)
            distance_params[pair] = np.array([initial_dist])
            initial_distances[pair] = initial_dist

            logger.debug(
                f"Distance parameter ({name_i}, {name_j}): "
                f"initialized to {initial_dist:.4f}m"
            )

        return distance_params, initial_distances

    def build_segment_rigidity_costs_target_segment_lengths(
        self,
        *,
        positions: np.ndarray,
        target_lengths: dict[tuple[str, str], float],
        base_weight: float
    ) -> list[SegmentRigidityCostInfo]:
        """Build rigidity costs with optimized distances."""
        costs: list[SegmentRigidityCostInfo] = []
        n_frames = positions.shape[0]

        for segment_pair, target_length in target_lengths.items():
            name_i, name_j = segment_pair
            idx_i = self.skeleton.get_keypoint_index(keypoint_name=name_i)
            idx_j = self.skeleton.get_keypoint_index(keypoint_name=name_j)
            weight = base_weight # * segment.rigidity # don't scale by rigidity, for now

            # Add cost for each frame
            for frame_idx in range(n_frames):
                cost = RigidEdgeCost(marker_i=1,
                                        marker_j=1,
                                        target_distance=target_length,
                                        weight=weight,
                                     )

                cost_info = SegmentRigidityCostInfo(
                    cost=cost,
                    parameters=[
                        positions[frame_idx, idx_i],
                        positions[frame_idx, idx_j],
                    ],
                    segment_name=f"{name_i}-{name_j}",
                    keypoint_i=name_i,
                    keypoint_j=name_j,
                    rigidity=1.0,  # Placeholder, all segments treated equally for now
                    target_distance=float(target_length),  #  target distance
                    weight=weight,
                )
                costs.append(cost_info)

        return costs


    def build_segment_rigidity_costs_optimized_segment_lengths(
        self,
        *,
        positions: np.ndarray,
        distance_parameters: dict[tuple[str, str], np.ndarray],
        base_weight: float
    ) -> list[SegmentRigidityCostInfo]:
        """Build rigidity costs with optimized distances."""
        costs: list[SegmentRigidityCostInfo] = []
        n_frames = positions.shape[0]

        for segment_pair in distance_parameters.keys():
            name_i, name_j = segment_pair
            distance_param = distance_parameters[segment_pair]
            idx_i = self.skeleton.get_keypoint_index(keypoint_name=name_i)
            idx_j = self.skeleton.get_keypoint_index(keypoint_name=name_j)
            weight = base_weight # * segment.rigidity # don't scale by rigidity, for now

            # Add cost for each frame (note - all frames share same distance parameter!)
            for frame_idx in range(n_frames):
                cost = OptimizedDistanceRigidEdgeCost(weight=weight)

                cost_info = SegmentRigidityCostInfo(
                    cost=cost,
                    parameters=[
                        positions[frame_idx, idx_i],
                        positions[frame_idx, idx_j],
                        distance_param  # Shared distance parameter!
                    ],
                    segment_name=f"{name_i}-{name_j}",
                    keypoint_i=name_i,
                    keypoint_j=name_j,
                    rigidity=1.0,  # Placeholder, all segments treated equally for now
                    target_distance=float(distance_param[0]),  # Current target distance
                    weight=weight,
                    description=f"Rigidity cost for segment ({name_i}, {name_j})"
                )
                costs.append(cost_info)

        return costs

    def build_linkage_stiffness_costs(
        self,
        *,
        positions: np.ndarray,
        distance_parameters: dict[tuple[str, str], np.ndarray],
        stiffness_threshold: float = 0.1,
        base_weight: float = 200.0
    ) -> list[LinkageStiffnessCostInfo]:
        """Build stiffness costs with optimized distances."""
        costs: list[LinkageStiffnessCostInfo] = []
        n_frames = positions.shape[0]

        for linkage in self.skeleton.linkages:
            if linkage.stiffness < stiffness_threshold:
                continue

            parent_names = [linkage.parent.parent.name] + [kp.name for kp in linkage.parent.children]

            for child_segment in linkage.children:
                child_names = [child_segment.parent.name] + [kp.name for kp in child_segment.children]

                for p_name in parent_names:
                    for c_name in child_names:
                        pair = tuple(sorted([p_name, c_name]))

                        if pair not in distance_parameters:
                            continue

                        distance_param = distance_parameters[pair]
                        idx_p = self.get_keypoint_index(keypoint_name=p_name)
                        idx_c = self.get_keypoint_index(keypoint_name=c_name)
                        weight = base_weight * linkage.stiffness

                        for frame_idx in range(n_frames):
                            cost = OptimizedDistanceRigidEdgeCost(weight=weight)

                            cost_info = LinkageStiffnessCostInfo(
                                cost=cost,
                                parameters=[
                                    positions[frame_idx, idx_p],
                                    positions[frame_idx, idx_c],
                                    distance_param
                                ],
                                linkage_name=linkage.name,
                                keypoint_i=p_name,
                                keypoint_j=c_name,
                                stiffness=linkage.stiffness,
                                target_distance=distance_param[0],
                                weight=weight
                            )
                            costs.append(cost_info)

        return costs

    def build_distance_anchor_costs(
        self,
        *,
        distance_parameters: dict[tuple[str, str], np.ndarray],
        initial_distances: dict[tuple[str, str], float],
        anchor_weight: float
    ) -> list[AnchorCostInfo]:
        """Build anchor costs for distance parameters.

        These prevent distances from drifting arbitrarily far from
        initial estimates. Use low weight for flexibility.
        """
        costs: list[AnchorCostInfo] = []

        for pair, distance_param in distance_parameters.items():
            initial_dist = initial_distances[pair]

            cost = DistanceAnchorCost(
                initial_distance=initial_dist,
                weight=anchor_weight
            )

            cost_info = AnchorCostInfo(
                cost=cost,
                parameters=[distance_param],
                weight=anchor_weight
            )
            costs.append(cost_info)

        return costs

    def build_distance_bound_costs(
        self,
        *,
        distance_parameters: dict[tuple[str, str], np.ndarray],
        min_distance_ratio: float,
        bound_weight: float
    ) -> list[AnchorCostInfo]:
        """Build lower bound costs for distances.

        Prevents degenerate solutions where distances collapse to zero.
        """
        costs: list[AnchorCostInfo] = []

        for pair, distance_param in distance_parameters.items():
            cost = DistanceLowerBoundCost(
                min_distance=float(min_distance_ratio * distance_param[0]),
                weight=bound_weight
            )

            cost_info = AnchorCostInfo(
                cost=cost,
                parameters=[distance_param],
                weight=bound_weight
            )
            costs.append(cost_info)

        return costs

    def build_temporal_smoothness_costs(
        self,
        *,
        positions: np.ndarray,
        smoothness_weight: float = 10.0
    ) -> list[TranslationSmoothnessCostInfo]:
        """Build temporal smoothness costs (unchanged)."""
        costs: list[TranslationSmoothnessCostInfo] = []
        n_frames, n_keypoints = positions.shape[:2]

        for frame_idx in range(n_frames - 1):
            for kp_idx in range(n_keypoints):
                cost = DirectKeypointSmoothnessCost(weight=smoothness_weight)

                cost_info = TranslationSmoothnessCostInfo(
                    cost=cost,
                    parameters=[
                        positions[frame_idx, kp_idx],
                        positions[frame_idx + 1, kp_idx]
                    ],
                    frame_from=frame_idx,
                    frame_to=frame_idx + 1,
                    weight=smoothness_weight
                )
                costs.append(cost_info)

        return costs

    def build_measurement_costs(
        self,
        *,
        positions: np.ndarray,
        input_data: TrajectoryDataset,
        measurement_weight: float
    ) -> list[MeasurementCostInfo]:
        """Build measurement costs (unchanged)."""
        costs: list[MeasurementCostInfo] = []
        n_frames = positions.shape[0]

        for trajectory_name, trajectory in  input_data.data.items():

            kp_idx = self.skeleton.get_keypoint_index(keypoint_name=trajectory_name)

            for frame_idx in range(n_frames):
                measured = trajectory.data[frame_idx]

                if np.isnan(measured).any():
                    continue

                if trajectory.confidence is not None:
                    weight = measurement_weight * trajectory.confidence[frame_idx]
                else:
                    weight = measurement_weight

                cost = DirectKeypointMeasurementCost(
                    measured_point=measured,
                    weight=weight
                )

                cost_info = MeasurementCostInfo(
                    cost=cost,
                    parameters=[positions[frame_idx, kp_idx]],
                    keypoint_name=trajectory_name,
                    frame_index=frame_idx,
                    weight=weight,
                    description="Measurement cost for keypoint {trajectory_name} at frame {frame_idx}"
                )
                costs.append(cost_info)

        return costs

    def build_all_costs(
        self,
        *,
        positions: np.ndarray,
        input_data: TrajectoryDataset,
        config: SkeletonPipelineConfig
    ) -> SkeletonCostBuildResult:
        """Build all costs with optimized distances.

        Args:
            positions: (n_frames, n_keypoints, 3) position parameters
            input_data: Measured trajectories
            config: Pipeline configuration

        Returns:
            SkeletonCostBuildResult with costs and distance parameters
        """
        logger.info("Building costs for bundle adjustment (positions + distances)...")

        # Initialize distance parameters from data
        logger.info("Initializing distance parameters from input data...")
        distance_params, initial_distances = self._initialize_segment_lengths(
            input_data=input_data,
            min_confidence=config.input_data_confidence_threshold,
            segment_rigidity_threshold=config.rigidity_threshold
        )
        logger.info(f"Initialized {len(distance_params)} distance parameters")


        # Build costs
        collection = CostCollection()

        # Segment rigidity (with optimized distances)
        logger.info("Building segment rigidity costs (optimized distances)...")
        # rigidity_costs = self.build_segment_rigidity_costs(
        #     positions=positions,
        #     distance_parameters=distance_params,
        #     base_weight=config.rigidity_weight
        # )
        rigidity_costs = self.build_segment_rigidity_costs_target_segment_lengths(
            positions=positions,
            target_lengths=initial_distances,
            base_weight=config.rigidity_weight
        )
        collection.extend(costs=rigidity_costs)
        logger.info(f"  Added {len(rigidity_costs)} rigidity costs")

        # Linkage stiffness (with optimized distances)
        # logger.info("Building linkage stiffness costs (optimized distances)...")
        # stiffness_costs = self.build_linkage_stiffness_costs(
        #     positions=positions,
        #     distance_parameters=distance_params,
        #     stiffness_threshold=stiffness_threshold
        # )
        # collection.extend(costs=stiffness_costs)
        # logger.info(f"  Added {len(stiffness_costs)} stiffness costs")

        # # Distance anchors (prevent drift)
        # logger.info("Building distance anchor costs...")
        # anchor_costs = self.build_distance_anchor_costs(
        #     distance_parameters=distance_params,
        #     initial_distances=initial_distances,
        #     anchor_weight=config.distance_anchor_weight
        # )
        # collection.extend(costs=anchor_costs)
        # logger.info(f"  Added {len(anchor_costs)} distance anchor costs")
        #
        # # Distance bounds (prevent collapse)
        # logger.info("Building distance bound costs...")
        # bound_costs = self.build_distance_bound_costs(
        #     distance_parameters=distance_params,
        #     min_distance_ratio=config.segment_length_change_threshold,
        #     bound_weight=config.rigidity_weight *1000 # very high cost to prevent collapse
        # )
        # collection.extend(costs=bound_costs)
        # logger.info(f"  Added {len(bound_costs)} distance bound costs")

        # Temporal smoothness
        logger.info("Building temporal smoothness costs...")
        smoothness_costs = self.build_temporal_smoothness_costs(
            positions=positions,
            smoothness_weight=config.smoothness_weight
        )
        collection.extend(costs=smoothness_costs)
        logger.info(f"  Added {len(smoothness_costs)} smoothness costs")

        # Measurements
        logger.info("Building measurement costs...")
        measurement_costs = self.build_measurement_costs(
            positions=positions,
            input_data=input_data,
            measurement_weight=config.measurement_weight
        )
        collection.extend(costs=measurement_costs)
        logger.info(f"  Added {len(measurement_costs)} measurement costs")

        # Summary
        logger.info("=" * 60)
        logger.info("Bundle adjustment cost summary:")
        collection.print_summary()
        logger.info(f"Distance parameters: {len(distance_params)}")
        logger.info("=" * 60)

        return SkeletonCostBuildResult(
            costs=collection,
            distance_parameters=distance_params,
            initial_distances=initial_distances
        )