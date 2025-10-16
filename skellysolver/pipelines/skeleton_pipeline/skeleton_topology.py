"""Extract topology information from SkeletonConstraint for visualization.

This module extracts the structure (keypoints, segments, edges) from a
SkeletonConstraint definition to enable visualization of the skeleton.
"""

import logging
from typing import Any

from skellysolver.solvers.constraints.skeleton_constraint import SkeletonConstraint
from skellysolver.utilities.arbitrary_types_model import ABaseModel

logger = logging.getLogger(__name__)


class SkeletonTopology(ABaseModel):
    """Topology information for skeleton visualization.
    
    Attributes:
        name: Name of the skeleton
        keypoint_names: List of keypoint names in order
        rigid_edges: List of [i, j] pairs representing rigid connections
        flexible_edges: List of [i, j, rigidity] for semi-rigid connections
        segment_info: Dict mapping segment name to keypoint indices
        metadata: Additional information about the skeleton
    """
    name: str
    keypoint_names: list[str]
    rigid_edges: list[list[int]]
    flexible_edges: list[list[int | float]]
    segment_info: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def extract_skeleton_topology(
    *,
    skeleton: SkeletonConstraint,
    rigidity_threshold: float = 0.8
) -> SkeletonTopology:
    """Extract topology from SkeletonConstraint.
    
    Args:
        skeleton: The skeleton constraint definition
        rigidity_threshold: Threshold above which edges are considered rigid
        
    Returns:
        SkeletonTopology with connection information
    """
    logger.info(f"Extracting topology from skeleton: {skeleton.name}")
    
    # Get keypoint names in order
    keypoint_names = [kp.name for kp in skeleton.keypoints]
    
    # Create name -> index mapping
    name_to_idx = {name: idx for idx, name in enumerate(keypoint_names)}
    
    rigid_edges: list[list[int]] = []
    flexible_edges: list[list[int | float]] = []
    segment_info: dict[str, dict[str, Any]] = {}
    
    # Process segments
    for segment in skeleton.segments:
        parent_name = segment.parent.name
        parent_idx = name_to_idx[parent_name]
        
        # Get child indices
        child_indices = [name_to_idx[kp.name] for kp in segment.children]
        
        # Store segment info
        segment_info[segment.name] = {
            "parent_idx": parent_idx,
            "child_indices": child_indices,
            "rigidity": segment.rigidity,
            "keypoint_names": [parent_name] + [kp.name for kp in segment.children]
        }
        
        # Create edges between parent and all children
        for child_idx in child_indices:
            if segment.rigidity >= rigidity_threshold:
                rigid_edges.append([parent_idx, child_idx])
            else:
                flexible_edges.append([parent_idx, child_idx, segment.rigidity])
        
        # Create edges between all pairs of children (for rigid segments)
        if segment.rigidity >= rigidity_threshold and len(child_indices) > 1:
            for i, idx_i in enumerate(child_indices):
                for idx_j in child_indices[i+1:]:
                    rigid_edges.append([idx_i, idx_j])
    
    # Process linkages for cross-segment connections
    for linkage in skeleton.linkages:
        if linkage.stiffness < 0.01:  # Skip very weak linkages
            continue
            
        # Get keypoints from parent segment
        parent_kp_names = [linkage.parent.parent.name] + [kp.name for kp in linkage.parent.children]
        parent_indices = [name_to_idx[name] for name in parent_kp_names]
        
        # Get keypoints from child segments
        for child_segment in linkage.children:
            child_kp_names = [child_segment.parent.name] + [kp.name for kp in child_segment.children]
            child_indices = [name_to_idx[name] for name in child_kp_names]
            
            # Create edges between parent and child segment keypoints
            # Only connect the linked keypoint to nearby keypoints
            linked_idx = name_to_idx[linkage.linked_keypoint.name]
            
            for p_idx in parent_indices:
                for c_idx in child_indices:
                    # Only add if one of them is the linked keypoint
                    if p_idx == linked_idx or c_idx == linked_idx:
                        flexible_edges.append([p_idx, c_idx, linkage.stiffness])
    
    logger.info(f"  Keypoints: {len(keypoint_names)}")
    logger.info(f"  Rigid edges: {len(rigid_edges)}")
    logger.info(f"  Flexible edges: {len(flexible_edges)}")
    
    return SkeletonTopology(
        name=skeleton.name,
        keypoint_names=keypoint_names,
        rigid_edges=rigid_edges,
        flexible_edges=flexible_edges,
        segment_info=segment_info,
        metadata={
            "n_keypoints": len(keypoint_names),
            "n_segments": len(skeleton.segments),
            "n_linkages": len(skeleton.linkages),
            "n_chains": len(skeleton.chains)
        }
    )
