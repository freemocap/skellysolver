"""Build pyceres cost functions from Skeleton constraints.

This module automatically generates cost functions based on the
constraints defined in a Skeleton model.

Now uses typed CostInfo models instead of raw dictionaries.
"""

from typing import Any
import numpy as np
from itertools import combinations

from skellysolver.cost_primatives.cost_info_model import SegmentRigidityCostInfo, LinkageStiffnessCostInfo, \
    RotationSmoothnessCostInfo, TranslationSmoothnessCostInfo, SymmetryCostInfo, AnchorCostInfo, MeasurementCostInfo, \
    CostCollection
from skellysolver.data.primitive_objects.skeleton_model import Skeleton
from skellysolver.data.primitive_objects.keypoint_model import Keypoint
from skellysolver.cost_primatives.edge_consts import (
    RigidEdgeCost,
    SymmetryConstraintCost,
    ReferenceAnchorCost,
)
from skellysolver.cost_primatives.measurement_costs import (
    RigidPoint3DMeasurementBundleAdjustment,
)
from skellysolver.cost_primatives.smoothness_costs import (
    RotationSmoothnessCost,
    TranslationSmoothnessCost,
)

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.solvers.mocap_solver.mocap_models.ferret_skeleton_v1 import FERRET_SKELETON_V1


class SkeletonCostBuilder(ABaseModel):
    """Build cost functions from skeleton constraints.
    
    Automatically generates typed cost functions based on:
    - Segment rigidity constraints (intra-segment edges)
    - Linkage stiffness constraints (inter-segment edges)
    - Chain smoothness constraints (temporal continuity)
    - Symmetry constraints (bilateral anatomy)
    
    Usage:
        skeleton = FERRET_SKELETON_V1
        builder = SkeletonCostBuilder(skeleton=skeleton)
        
        # Generate all constraint costs
        costs = builder.build_all_constraint_costs(
            reference_geometry=ref_flat,
            quaternions=quats,
            translations=trans,
            measured_positions_per_frame=measurements
        )
        
        # Add to optimizer
        costs.to_optimizer(optimizer=optimizer)
        
        # Or manually
        for cost_info in costs.costs:
            optimizer.add_residual_block(
                cost=cost_info.cost,
                parameters=cost_info.parameters
            )
    """
    
    skeleton: Skeleton
    
    def get_keypoint_index(self, *, keypoint: Keypoint) -> int:
        """Get index of keypoint in skeleton.keypoints list.
        
        Args:
            keypoint: Keypoint to find
            
        Returns:
            Index in skeleton.keypoints
        """
        try:
            return self.skeleton.keypoints.index(keypoint)
        except ValueError:
            raise ValueError(f"Keypoint {keypoint.name} not found in skeleton")
    
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
            segment_keypoints = [segment.root] + segment.keypoints
            segment_indices = [
                self.get_keypoint_index(keypoint=kp) 
                for kp in segment_keypoints
            ]
            
            # Create rigid edge cost for each pair
            for i, j in combinations(range(len(segment_keypoints)), 2):
                kp_i = segment_keypoints[i]
                kp_j = segment_keypoints[j]
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
            parent_keypoints = [linkage.parent.root] + linkage.parent.keypoints
            parent_indices = [
                self.get_keypoint_index(keypoint=kp) 
                for kp in parent_keypoints
            ]
            
            # For each child segment
            for child_segment in linkage.children:
                child_keypoints = [child_segment.root] + child_segment.keypoints
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
        """Build SymmetryConstraintCost for bilateral symmetry.
        
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
            
            cost = SymmetryConstraintCost(
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
    
    def build_all_constraint_costs(
        self,
        *,
        reference_geometry: np.ndarray,
        initial_reference: np.ndarray | None = None,
        quaternions: np.ndarray | None = None,
        translations: np.ndarray | None = None,
        measured_positions_per_frame: list[dict[str, np.ndarray]] | None = None,
        confidence_per_frame: list[dict[str, float]] | None = None,
        symmetry_pairs: list[tuple[str, str]] | None = None,
        rigidity_threshold: float = 0.5,
        stiffness_threshold: float = 0.1,
        include_measurements: bool = True,
        include_temporal_smoothness: bool = True,
        include_anchor: bool = True,
        rotation_smoothness_weight: float = 10.0,
        translation_smoothness_weight: float = 10.0,
        anchor_weight: float = 1.0
    ) -> CostCollection:
        """Build all constraint costs from skeleton.
        
        This is the main entry point for generating costs.
        
        Args:
            reference_geometry: (n_keypoints * 3,) reference positions
            initial_reference: Optional initial reference (for anchor)
            quaternions: Optional (n_frames, 4) rotations (for smoothness)
            translations: Optional (n_frames, 3) translations (for smoothness)
            measured_positions_per_frame: Optional list of measurement dicts per frame
            confidence_per_frame: Optional list of confidence dicts per frame
            symmetry_pairs: Optional list of (left, right) name pairs
            rigidity_threshold: Minimum rigidity to enforce (0-1)
            stiffness_threshold: Minimum stiffness to enforce (0-1)
            include_measurements: Whether to include measurement costs
            include_temporal_smoothness: Whether to include smoothness costs
            include_anchor: Whether to anchor reference geometry
            rotation_smoothness_weight: Weight for rotation smoothness
            translation_smoothness_weight: Weight for translation smoothness
            anchor_weight: Weight for reference anchor
            
        Returns:
            CostCollection with all costs
        """
        collection = CostCollection()
        
        # 1. Segment rigidity constraints (reference geometry)
        rigidity_costs = self.build_segment_rigidity_costs(
            reference_geometry=reference_geometry,
            rigidity_threshold=rigidity_threshold
        )
        collection.extend(costs=rigidity_costs)
        
        # 2. Linkage stiffness constraints (reference geometry)
        stiffness_costs = self.build_linkage_stiffness_costs(
            reference_geometry=reference_geometry,
            stiffness_threshold=stiffness_threshold
        )
        collection.extend(costs=stiffness_costs)
        
        # 3. Symmetry constraints (reference geometry)
        if symmetry_pairs:
            symmetry_costs = self.build_symmetry_costs(
                reference_geometry=reference_geometry,
                symmetry_pairs=symmetry_pairs
            )
            collection.extend(costs=symmetry_costs)
        
        # 4. Reference anchor (prevent drift)
        if include_anchor and initial_reference is not None:
            anchor_cost = self.build_reference_anchor_cost(
                reference_geometry=reference_geometry,
                initial_reference=initial_reference,
                weight=anchor_weight
            )
            collection.add(cost_info=anchor_cost)
        
        # 5. Temporal smoothness (poses)
        if include_temporal_smoothness and quaternions is not None and translations is not None:
            smoothness_costs = self.build_temporal_smoothness_costs(
                quaternions=quaternions,
                translations=translations,
                rotation_weight=rotation_smoothness_weight,
                translation_weight=translation_smoothness_weight
            )
            collection.extend(costs=smoothness_costs)
        
        # 6. Measurement costs (per frame)
        if include_measurements and measured_positions_per_frame:
            for frame_idx, measured_positions in enumerate(measured_positions_per_frame):
                if not measured_positions:
                    continue
                
                # Get confidence for this frame if available
                confidence_weights = None
                if confidence_per_frame and frame_idx < len(confidence_per_frame):
                    confidence_weights = confidence_per_frame[frame_idx]
                
                # Get pose for this frame
                quat = quaternions[frame_idx] if quaternions is not None else None
                trans = translations[frame_idx] if translations is not None else None
                
                if quat is None or trans is None:
                    continue
                
                frame_costs = self.build_measurement_costs(
                    quaternion=quat,
                    translation=trans,
                    reference_geometry=reference_geometry,
                    measured_positions=measured_positions,
                    frame_index=frame_idx,
                    confidence_weights=confidence_weights
                )
                collection.extend(costs=frame_costs)
        
        return collection


def build_ferret_skeleton_costs(
    *,
    reference_geometry: np.ndarray,
    initial_reference: np.ndarray,
    quaternions: np.ndarray,
    translations: np.ndarray,
    measured_positions_per_frame: list[dict[str, np.ndarray]],
    confidence_per_frame: list[dict[str, float]] | None = None
) -> CostCollection:
    """Convenience function to build costs for ferret skeleton.
    
    Automatically includes appropriate symmetry pairs for ferret anatomy.
    
    Args:
        reference_geometry: (n_keypoints * 3,) reference positions
        initial_reference: (n_keypoints * 3,) initial reference (for anchor)
        quaternions: (n_frames, 4) rotation quaternions
        translations: (n_frames, 3) translation vectors
        measured_positions_per_frame: List of measurement dicts per frame
        confidence_per_frame: Optional list of confidence dicts per frame
        
    Returns:
        CostCollection with all costs
    """

    
    builder = SkeletonCostBuilder(skeleton=FERRET_SKELETON_V1)
    
    # Define symmetry pairs for ferret
    symmetry_pairs = [
        ("left_eye_camera", "right_eye_camera"),
        ("left_eye_inner", "right_eye_inner"),
        ("left_eye_center", "right_eye_center"),
        ("left_eye_outer", "right_eye_outer"),
        ("left_acoustic_meatus", "right_acoustic_meatus"),
    ]
    
    costs = builder.build_all_constraint_costs(
        reference_geometry=reference_geometry,
        initial_reference=initial_reference,
        quaternions=quaternions,
        translations=translations,
        measured_positions_per_frame=measured_positions_per_frame,
        confidence_per_frame=confidence_per_frame,
        symmetry_pairs=symmetry_pairs,
        rigidity_threshold=0.5,
        stiffness_threshold=0.1,
        include_measurements=True,
        include_temporal_smoothness=True,
        include_anchor=True
    )
    
    # Print summary
    costs.print_summary()
    
    return costs
