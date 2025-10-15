"""Build pyceres cost functions from SkeletonConstraint constraints.

This module automatically generates cost functions based on the
constraints defined in a SkeletonConstraint model.

Now uses typed CostInfo models instead of raw dictionaries.
"""

import logging
import numpy as np
from itertools import combinations

from skellysolver.solvers.base_cost_builder import BaseCostBuilder
from skellysolver.solvers.constraints.keypoint_constraint import KeypointConstraint
from skellysolver.solvers.constraints.skeleton_constraint import SkeletonConstraint
from skellysolver.solvers.costs.constraint_costs import SegmentRigidityCostInfo, LinkageStiffnessCostInfo, \
    RotationSmoothnessCostInfo, TranslationSmoothnessCostInfo, AnchorCostInfo, \
    MeasurementCostInfo, CostCollection
from skellysolver.solvers.costs.edge_consts import RigidEdgeCost, SymmetryCostInfo, ReferenceAnchorCost
from skellysolver.solvers.costs.measurement_costs import RigidPoint3DMeasurementBundleAdjustment
from skellysolver.solvers.costs.smoothness_costs import RotationSmoothnessCost, TranslationSmoothnessCost
from skellysolver.pipelines.skeleton_pipeline.skeleton_definitions.ferret_skeleton_v1 import FERRET_SKELETON_V1
from skellysolver.data.trajectory_dataset import TrajectoryDataset
from skellysolver.utilities.arbitrary_types_model import ABaseModel

logger = logging.getLogger(__name__)


class CostBuildResult(ABaseModel):
    """Result of building costs with reference geometry.

    Attributes:
        costs: Collection of all generated costs
        reference_geometry: (n_keypoints * 3,) flattened reference positions
    """
    costs: CostCollection
    reference_geometry: np.ndarray


class SkeletonCostBuilder(BaseCostBuilder[SkeletonConstraint]):
    """Build cost functions from skeleton constraints.

    Automatically generates typed cost functions based on:
    - Segment rigidity constraints (intra-segment edges)
    - Linkage stiffness constraints (inter-segment edges)
    - Chain smoothness constraints (temporal continuity)
    - Symmetry constraints (bilateral anatomy)

    Usage:
        skeleton = FERRET_SKELETON_V1
        builder = SkeletonCostBuilder(constraint=skeleton)

        # Generate all constraint costs
        result = builder.build_all_costs(
            quaternions=quats,
            translations=trans,
            input_data=measurements,
            config=pipeline_config
        )

        # Add reference geometry as parameter
        solver.add_parameter_block(
            name="reference_geometry",
            parameters=result.reference_geometry
        )

        # Add costs to optimizer
        for cost_info in result.costs.costs:
            solver.add_residual_block(
                cost=cost_info.cost,
                parameters=cost_info.parameters
            )
    """

    # Convenience property for backward compatibility
    @property
    def skeleton(self) -> SkeletonConstraint:
        """Alias for constraint to maintain backward compatibility."""
        return self.constraint

    def get_keypoint_index(self, *, keypoint: KeypointConstraint) -> int:
        """Get index of keypoint in skeleton.keypoints list.

        Args:
            keypoint: KeypointConstraint to find

        Returns:
            Index in skeleton.keypoints
        """
        try:
            return self.skeleton.keypoints.index(keypoint)
        except ValueError:
            raise ValueError(f"KeypointConstraint {keypoint.name} not found in skeleton")

    def _compute_distance(
        self,
        *,
        reference_geometry: np.ndarray,
        idx_i: int,
        idx_j: int
    ) -> float:
        """Compute distance between two keypoints in reference geometry.

        Args:
            reference_geometry: (n_keypoints * 3,) flattened positions
            idx_i: First keypoint index
            idx_j: Second keypoint index

        Returns:
            Euclidean distance
        """
        pos_i = reference_geometry[idx_i*3:(idx_i+1)*3]
        pos_j = reference_geometry[idx_j*3:(idx_j+1)*3]
        return float(np.linalg.norm(pos_i - pos_j))

    def _build_reference_geometry_from_data(
        self,
        *,
        input_data: TrajectoryDataset,
        min_confidence: float = 0.3
    ) -> np.ndarray:
        """Build reference geometry from input trajectory data.

        Takes the centroid (mean of valid positions) for each keypoint
        as the reference pose.

        Args:
            input_data: TrajectoryDataset with measured marker positions
            min_confidence: Minimum confidence for valid data

        Returns:
            (n_keypoints * 3,) flattened reference positions
        """
        n_keypoints = len(self.skeleton.keypoints)
        reference_geometry = np.zeros(n_keypoints * 3)

        for keypoint in self.skeleton.keypoints:

            if keypoint.name not in input_data.data:
                raise ValueError(
                    f"Trajectory '{keypoint.name}' not found in input data. "
                    f"Available: {list(input_data.data.keys())}"
                )

            trajectory = input_data.data[keypoint.name]

            # Get centroid of valid values
            centroid = trajectory.get_centroid(min_confidence=min_confidence)

            # Store in reference geometry
            idx = self.get_keypoint_index(keypoint=keypoint)
            reference_geometry[idx*3:(idx+1)*3] = centroid

            logger.debug(
                f"Keypoint '{keypoint.name} centroid: {centroid}"
            )

        return reference_geometry

    def _extract_measurements_per_frame(
        self,
        *,
        input_data: TrajectoryDataset
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, float]]]:
        """Extract measured positions and confidence per frame.

        Args:
            input_data: TrajectoryDataset with measured marker positions

        Returns:
            Tuple of (measured_positions_per_frame, confidence_per_frame)
        """
        measured_positions_per_frame: list[dict[str, np.ndarray]] = []
        confidence_per_frame: list[dict[str, float]] = []

        for frame_idx in range(input_data.n_frames):
            measured_positions: dict[str, np.ndarray] = {}
            confidence_dict: dict[str, float] = {}

            for keypoint in self.skeleton.keypoints:
                trajectory = input_data.data[keypoint.name]

                # Get position at this frame
                position = trajectory.values[frame_idx]

                # Check if valid (not NaN)
                if not np.isnan(position).any():
                    measured_positions[keypoint.name] = position

                    # Get confidence if available
                    if trajectory.confidence is not None:
                        confidence_dict[keypoint.name] = float(trajectory.confidence[frame_idx])
                    else:
                        confidence_dict[keypoint.name] = 1.0

            measured_positions_per_frame.append(measured_positions)
            confidence_per_frame.append(confidence_dict)

        return measured_positions_per_frame, confidence_per_frame

    def build_segment_rigidity_costs(
        self,
        *,
        reference_geometry: np.ndarray,
        rigidity_threshold: float = 0.5,
        base_weight: float = 500.0
    ) -> list[SegmentRigidityCostInfo]:
        """Build RigidEdgeCost for segments with high rigidity.

        Args:
            reference_geometry: (n_keypoints * 3,) flattened reference positions
            rigidity_threshold: Minimum rigidity to enforce (0-1)
            base_weight: Base weight for rigid edges (higher = more rigid)

        Returns:
            List of SegmentRigidityCostInfo objects
        """
        costs: list[SegmentRigidityCostInfo] = []

        for segment in self.skeleton.segments:
            if segment.rigidity < rigidity_threshold:
                continue

            # Get all keypoints in this segment (root + keypoints)
            segment_indices = [
                self.get_keypoint_index(keypoint=kp)
                for kp in segment.keypoints
            ]

            # Create rigid edge cost for each pair
            for i, j in combinations(range(len(segment.keypoints)), 2):
                kp_i = segment.keypoints[i]
                kp_j = segment.keypoints[j]
                idx_i = segment_indices[i]
                idx_j = segment_indices[j]

                # Compute target distance from current reference geometry
                target_distance = self._compute_distance(
                    reference_geometry=reference_geometry,
                    idx_i=idx_i,
                    idx_j=idx_j
                )

                # Create cost with weight scaled by rigidity
                weight = base_weight * segment.rigidity
                cost = RigidEdgeCost(
                    marker_i=idx_i,
                    marker_j=idx_j,
                    n_markers=len(self.skeleton.keypoints),
                    target_distance=target_distance,
                    weight=weight
                )

                cost_info = SegmentRigidityCostInfo(
                    cost=cost,
                    parameters=[reference_geometry],
                    segment_name=segment.name,
                    keypoint_i=kp_i.name,
                    keypoint_j=kp_j.name,
                    rigidity=segment.rigidity,
                    target_distance=target_distance,
                    weight=weight
                )

                costs.append(cost_info)

        return costs

    def build_linkage_stiffness_costs(
        self,
        *,
        reference_geometry: np.ndarray,
        stiffness_threshold: float = 0.1,
        base_weight: float = 200.0
    ) -> list[LinkageStiffnessCostInfo]:
        """Build RigidEdgeCost for linkages with stiffness.

        Args:
            reference_geometry: (n_keypoints * 3,) reference positions
            stiffness_threshold: Minimum stiffness to enforce (0-1)
            base_weight: Base weight for linkage constraints

        Returns:
            List of LinkageStiffnessCostInfo objects
        """
        costs: list[LinkageStiffnessCostInfo] = []

        for linkage in self.skeleton.linkages:
            if linkage.stiffness < stiffness_threshold:
                continue

            # Get keypoints from parent segment
            parent_keypoints = [linkage.parent.parent] + linkage.parent.children
            parent_indices = [
                self.get_keypoint_index(keypoint=kp)
                for kp in parent_keypoints
            ]

            # For each child segment
            for child_segment in linkage.children:
                child_keypoints = [child_segment.parent] + child_segment.children
                child_indices = [
                    self.get_keypoint_index(keypoint=kp)
                    for kp in child_keypoints
                ]

                # Create rigid edges between parent and child keypoints
                for p_idx, p_kp in zip(parent_indices, parent_keypoints):
                    for c_idx, c_kp in zip(child_indices, child_keypoints):
                        # Compute current distance
                        target_distance = self._compute_distance(
                            reference_geometry=reference_geometry,
                            idx_i=p_idx,
                            idx_j=c_idx
                        )

                        # Weight by stiffness
                        weight = base_weight * linkage.stiffness

                        cost = RigidEdgeCost(
                            marker_i=p_idx,
                            marker_j=c_idx,
                            n_markers=len(self.skeleton.keypoints),
                            target_distance=target_distance,
                            weight=weight
                        )

                        cost_info = LinkageStiffnessCostInfo(
                            cost=cost,
                            parameters=[reference_geometry],
                            linkage_name=linkage.name,
                            keypoint_i=p_kp.name,
                            keypoint_j=c_kp.name,
                            stiffness=linkage.stiffness,
                            target_distance=target_distance,
                            weight=weight
                        )

                        costs.append(cost_info)

        return costs

    def build_temporal_smoothness_costs(
        self,
        *,
        quaternions: np.ndarray,
        translations: np.ndarray,
        rotation_weight: float = 10.0,
        translation_weight: float = 10.0
    ) -> list[RotationSmoothnessCostInfo | TranslationSmoothnessCostInfo]:
        """Build temporal smoothness costs for poses.

        Args:
            quaternions: (n_frames, 4) rotation quaternions
            translations: (n_frames, 3) translation vectors
            rotation_weight: Weight for rotation smoothness
            translation_weight: Weight for translation smoothness

        Returns:
            List of smoothness cost info objects
        """
        costs: list[RotationSmoothnessCostInfo | TranslationSmoothnessCostInfo] = []
        n_frames = len(quaternions)

        for frame_idx in range(n_frames - 1):
            # Rotation smoothness
            rot_cost = RotationSmoothnessCost(weight=rotation_weight)
            rot_info = RotationSmoothnessCostInfo(
                cost=rot_cost,
                parameters=[quaternions[frame_idx], quaternions[frame_idx + 1]],
                frame_from=frame_idx,
                frame_to=frame_idx + 1,
                weight=rotation_weight
            )
            costs.append(rot_info)

            # Translation smoothness
            trans_cost = TranslationSmoothnessCost(weight=translation_weight)
            trans_info = TranslationSmoothnessCostInfo(
                cost=trans_cost,
                parameters=[translations[frame_idx], translations[frame_idx + 1]],
                frame_from=frame_idx,
                frame_to=frame_idx + 1,
                weight=translation_weight
            )
            costs.append(trans_info)

        return costs

    def build_symmetry_costs(
        self,
        *,
        reference_geometry: np.ndarray,
        symmetry_pairs: list[tuple[str, str]],
        symmetry_plane: str = "yz",
        weight: float = 50.0
    ) -> list[SymmetryCostInfo]:
        """Build SymmetryCostInfo for bilateral symmetry.

        Args:
            reference_geometry: (n_keypoints * 3,) reference positions
            symmetry_pairs: List of (left_name, right_name) tuples
            symmetry_plane: Plane to mirror across ("yz", "xz", "xy")
            weight: Weight for symmetry constraint

        Returns:
            List of SymmetryCostInfo objects
        """
        costs: list[SymmetryCostInfo] = []

        # Build name to index mapping
        name_to_index = {
            kp.name: i
            for i, kp in enumerate(self.skeleton.keypoints)
        }

        for left_name, right_name in symmetry_pairs:
            if left_name not in name_to_index or right_name not in name_to_index:
                continue

            left_idx = name_to_index[left_name]
            right_idx = name_to_index[right_name]

            cost = SymmetryCostInfo(
                marker_idx=left_idx,
                symmetric_partner_idx=right_idx,
                n_markers=len(self.skeleton.keypoints),
                symmetry_plane=symmetry_plane,
                weight=weight
            )

            cost_info = SymmetryCostInfo(
                cost=cost,
                parameters=[reference_geometry],
                left_keypoint=left_name,
                right_keypoint=right_name,
                symmetry_plane=symmetry_plane,
                weight=weight
            )

            costs.append(cost_info)

        return costs

    def build_reference_anchor_cost(
        self,
        *,
        reference_geometry: np.ndarray,
        initial_reference: np.ndarray,
        weight: float = 1.0
    ) -> AnchorCostInfo:
        """Build anchor cost to prevent reference geometry drift.

        Args:
            reference_geometry: (n_keypoints * 3,) current reference
            initial_reference: (n_keypoints * 3,) initial reference
            weight: Weight for anchor (typically 0.1 to 10.0)

        Returns:
            AnchorCostInfo object
        """
        cost = ReferenceAnchorCost(
            initial_reference=initial_reference,
            weight=weight
        )

        cost_info = AnchorCostInfo(
            cost=cost,
            parameters=[reference_geometry],
            weight=weight
        )

        return cost_info

    def build_measurement_costs(
        self,
        *,
        quaternion: np.ndarray,
        translation: np.ndarray,
        reference_geometry: np.ndarray,
        measured_positions: dict[str, np.ndarray],
        frame_index: int | None = None,
        confidence_weights: dict[str, float] | None = None,
        base_weight: float = 1.0
    ) -> list[MeasurementCostInfo]:
        """Build measurement costs for observed keypoints.

        Args:
            quaternion: (4,) rotation quaternion
            translation: (3,) translation vector
            reference_geometry: (n_keypoints * 3,) reference positions
            measured_positions: Dict mapping keypoint names to (3,) positions
            frame_index: Optional frame index for this measurement
            confidence_weights: Optional dict mapping names to confidence (0-1)
            base_weight: Base weight for measurements

        Returns:
            List of MeasurementCostInfo objects
        """
        costs: list[MeasurementCostInfo] = []

        for keypoint in self.skeleton.keypoints:
            if keypoint.name not in measured_positions:
                continue

            measured_point = measured_positions[keypoint.name]
            marker_idx = self.get_keypoint_index(keypoint=keypoint)

            # Weight by confidence if available
            if confidence_weights and keypoint.name in confidence_weights:
                weight = base_weight * confidence_weights[keypoint.name]
            else:
                weight = base_weight

            cost = RigidPoint3DMeasurementBundleAdjustment(
                measured_point=measured_point,
                marker_idx=marker_idx,
                n_markers=len(self.skeleton.keypoints),
                weight=weight
            )

            cost_info = MeasurementCostInfo(
                cost=cost,
                parameters=[quaternion, translation, reference_geometry],
                keypoint_name=keypoint.name,
                frame_index=frame_index,
                weight=weight
            )

            costs.append(cost_info)

        return costs

    def build_all_costs(
        self,
        *,
        quaternions: np.ndarray,
        translations: np.ndarray,
        input_data: TrajectoryDataset,
        config: object
    ) -> CostBuildResult:
        """Build all costs from input data and config.

        This is the main entry point for cost generation. It handles all
        data wrangling internally:
        - Builds reference geometry from input data
        - Extracts measurements per frame
        - Extracts confidence per frame
        - Applies thresholds and weights from config
        - Builds all cost types

        Args:
            quaternions: (n_frames, 4) rotation quaternions
            translations: (n_frames, 3) translation vectors
            input_data: TrajectoryDataset with measured marker positions
            config: Pipeline configuration with weights and thresholds

        Returns:
            CostBuildResult containing costs and reference_geometry

        Example:
            result = builder.build_all_costs(
                quaternions=quats,
                translations=trans,
                input_data=dataset,
                config=pipeline_config
            )

            # Add reference geometry to solver
            solver.add_parameter_block(
                name="reference_geometry",
                parameters=result.reference_geometry
            )

            # Add all costs
            for cost_info in result.costs.costs:
                solver.add_residual_block(
                    cost=cost_info.cost,
                    parameters=cost_info.parameters
                )
        """
        logger.info("Building all costs from input data...")

        # 1. Build reference geometry from input data
        logger.info("Computing reference geometry from trajectory data...")
        reference_geometry = self._build_reference_geometry_from_data(
            input_data=input_data,
            min_confidence=0.3
        )
        initial_reference = reference_geometry.copy()

        logger.info(
            f"Reference geometry shape: {reference_geometry.shape} "
            f"({len(self.skeleton.keypoints)} keypoints)"
        )

        # 2. Extract measured positions and confidence per frame
        logger.info("Extracting measurements per frame...")
        measured_positions_per_frame, confidence_per_frame = self._extract_measurements_per_frame(
            input_data=input_data
        )

        n_frames = len(measured_positions_per_frame)
        avg_measurements = np.mean([len(m) for m in measured_positions_per_frame])
        logger.info(
            f"Extracted {n_frames} frames with avg {avg_measurements:.1f} measurements per frame"
        )

        # 3. Extract thresholds and weights from config
        rigidity_threshold = getattr(config, 'rigidity_threshold', 0.5)
        stiffness_threshold = getattr(config, 'stiffness_threshold', 0.1)
        rotation_smoothness_weight = getattr(config, 'rotation_smoothness_weight', 10.0)
        translation_smoothness_weight = getattr(config, 'translation_smoothness_weight', 10.0)
        anchor_weight = getattr(config, 'anchor_weight', 1.0)
        measurement_weight = getattr(config, 'measurement_weight', 1.0)

        # 4. Build cost collection
        collection = CostCollection()

        # Segment rigidity constraints
        logger.info("Building segment rigidity costs...")
        rigidity_costs = self.build_segment_rigidity_costs(
            reference_geometry=reference_geometry,
            rigidity_threshold=rigidity_threshold
        )
        collection.extend(costs=rigidity_costs)
        logger.info(f"  Added {len(rigidity_costs)} rigidity costs")

        # Linkage stiffness constraints
        logger.info("Building linkage stiffness costs...")
        stiffness_costs = self.build_linkage_stiffness_costs(
            reference_geometry=reference_geometry,
            stiffness_threshold=stiffness_threshold
        )
        collection.extend(costs=stiffness_costs)
        logger.info(f"  Added {len(stiffness_costs)} stiffness costs")

        # Symmetry constraints (if defined in config)
        symmetry_pairs = getattr(config, 'symmetry_pairs', None)
        if symmetry_pairs:
            logger.info("Building symmetry costs...")
            symmetry_costs = self.build_symmetry_costs(
                reference_geometry=reference_geometry,
                symmetry_pairs=symmetry_pairs
            )
            collection.extend(costs=symmetry_costs)
            logger.info(f"  Added {len(symmetry_costs)} symmetry costs")

        # Reference anchor (prevent drift)
        logger.info("Building anchor cost...")
        anchor_cost = self.build_reference_anchor_cost(
            reference_geometry=reference_geometry,
            initial_reference=initial_reference,
            weight=anchor_weight
        )
        collection.add(cost_info=anchor_cost)

        # Temporal smoothness
        logger.info("Building temporal smoothness costs...")
        smoothness_costs = self.build_temporal_smoothness_costs(
            quaternions=quaternions,
            translations=translations,
            rotation_weight=rotation_smoothness_weight,
            translation_weight=translation_smoothness_weight
        )
        collection.extend(costs=smoothness_costs)
        logger.info(f"  Added {len(smoothness_costs)} smoothness costs")

        # Measurement costs (per frame)
        logger.info("Building measurement costs...")
        measurement_cost_count = 0
        for frame_idx, measured_positions in enumerate(measured_positions_per_frame):
            if not measured_positions:
                continue

            confidence_weights = confidence_per_frame[frame_idx]

            frame_costs = self.build_measurement_costs(
                quaternion=quaternions[frame_idx],
                translation=translations[frame_idx],
                reference_geometry=reference_geometry,
                measured_positions=measured_positions,
                frame_index=frame_idx,
                confidence_weights=confidence_weights,
                base_weight=measurement_weight
            )
            collection.extend(costs=frame_costs)
            measurement_cost_count += len(frame_costs)

        logger.info(f"  Added {measurement_cost_count} measurement costs")

        # Summary
        logger.info("=" * 60)
        logger.info("Cost build summary:")
        collection.print_summary()
        logger.info("=" * 60)

        return CostBuildResult(
            costs=collection,
            reference_geometry=reference_geometry
        )


def build_ferret_skeleton_costs(
    *,
    quaternions: np.ndarray,
    translations: np.ndarray,
    input_data: TrajectoryDataset,
    config: object
) -> CostBuildResult:
    """Convenience function to build costs for ferret skeleton.

    Automatically includes appropriate symmetry pairs for ferret anatomy.

    Args:
        quaternions: (n_frames, 4) rotation quaternions
        translations: (n_frames, 3) translation vectors
        input_data: TrajectoryDataset with measured marker positions
        config: Pipeline configuration

    Returns:
        CostBuildResult with all costs and reference geometry
    """
    # Define symmetry pairs for ferret
    symmetry_pairs = [
        ("left_eye_camera", "right_eye_camera"),
        ("left_eye_inner", "right_eye_inner"),
        ("left_eye_center", "right_eye_center"),
        ("left_eye_outer", "right_eye_outer"),
        ("left_acoustic_meatus", "right_acoustic_meatus"),
    ]

    # Add symmetry pairs to config if not already present
    if not hasattr(config, 'symmetry_pairs'):
        setattr(config, 'symmetry_pairs', symmetry_pairs)

    builder = SkeletonCostBuilder(constraint=FERRET_SKELETON_V1)

    return builder.build_all_costs(
        quaternions=quaternions,
        translations=translations,
        input_data=input_data,
        config=config
    )