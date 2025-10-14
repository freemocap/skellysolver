"""Geometric constraint cost functions.

These enforce geometric properties like:
- Rigid edges (fixed distances)
- Soft edges (preferred distances)
- Anchoring (prevent drift)

Cost Functions:
- RigidEdgeCost: Enforce exact distance between points
- SoftEdgeCost: Encourage distance between points
- ReferenceAnchorCost: Prevent reference geometry drift
"""

import numpy as np
from scipy.spatial.transform import Rotation
import pyceres
from typing import Any

from .base_cost import BaseCostFunction


class RigidEdgeCost(BaseCostFunction):
    """Enforce rigid edge constraint in reference geometry.
    
    Used by rigid body tracking to maintain fixed distances
    between markers in the reference configuration.
    
    Model:
        current_distance = ||ref[i] - ref[j]||
        residual = current_distance - target_distance
    
    This operates on the reference geometry itself (not transformed).
    """
    
    def __init__(
        self,
        *,
        marker_i: int,
        marker_j: int,
        n_markers: int,
        target_distance: float,
        weight: float = 500.0
    ) -> None:
        """Initialize rigid edge cost.
        
        Args:
            marker_i: Index of first marker
            marker_j: Index of second marker
            n_markers: Total number of markers
            target_distance: Target distance to maintain
            weight: Weight for constraint (high = very rigid)
        """
        super().__init__(weight=weight)
        self.marker_i = marker_i
        self.marker_j = marker_j
        self.n_markers = n_markers
        self.target_dist = target_distance
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([n_markers * 3])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute edge length residual.
        
        Args:
            parameters: [reference_flat] - flattened reference positions
            
        Returns:
            1D residual (current_distance - target_distance)
        """
        reference_flat = parameters[0]
        
        # Extract marker positions
        ref_i = reference_flat[self.marker_i * 3:(self.marker_i + 1) * 3]
        ref_j = reference_flat[self.marker_j * 3:(self.marker_j + 1) * 3]
        
        # Compute current distance
        diff = ref_i - ref_j
        current_dist = np.linalg.norm(diff)
        
        # Residual
        residual = np.array([current_dist - self.target_dist])
        
        return residual


class SoftEdgeCost(BaseCostFunction):
    """Soft distance constraint between transformed point and measurement.
    
    Used for flexible connections (e.g., spine segments) where we want
    points to stay near each other but allow some deviation.
    
    Model:
        transformed = R @ ref[marker_idx] + translation
        distance = ||measured - transformed||
        residual = distance - target_distance
    """
    
    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        marker_idx: int,
        n_markers: int,
        target_distance: float,
        weight: float = 10.0
    ) -> None:
        """Initialize soft edge cost.
        
        Args:
            measured_point: (3,) observed position
            marker_idx: Index in reference geometry
            n_markers: Total number of markers
            target_distance: Preferred distance
            weight: Weight for constraint (low = flexible)
        """
        super().__init__(weight=weight)
        self.measured = measured_point.copy()
        self.marker_idx = marker_idx
        self.n_markers = n_markers
        self.target_dist = target_distance
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([4, 3, n_markers * 3])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute soft distance residual.
        
        Args:
            parameters: [quaternion, translation, reference_flat]
            
        Returns:
            1D residual (distance - target)
        """
        quat = parameters[0]
        translation = parameters[1]
        reference_flat = parameters[2]
        
        # Extract reference point
        start_idx = self.marker_idx * 3
        ref_point = reference_flat[start_idx:start_idx + 3]
        
        # Convert quaternion to rotation matrix
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        R = Rotation.from_quat(quat=quat_scipy).as_matrix()
        
        # Transform reference point
        transformed = R @ ref_point + translation
        
        # Compute distance
        diff = self.measured - transformed
        distance = np.linalg.norm(diff)
        
        # Residual
        residual = np.array([distance - self.target_dist])
        
        return residual


class ReferenceAnchorCost(BaseCostFunction):
    """Anchor reference geometry to prevent drift.
    
    Used in bundle adjustment to prevent the reference geometry
    from drifting to arbitrary positions (gauge freedom).
    
    Model:
        residual = current_reference - initial_reference
    
    Typically applied with low weight.
    """
    
    def __init__(
        self,
        *,
        initial_reference: np.ndarray,
        weight: float = 1.0
    ) -> None:
        """Initialize anchor cost.
        
        Args:
            initial_reference: (n_markers * 3,) initial reference geometry
            weight: Weight for anchor (low = soft anchor)
        """
        super().__init__(weight=weight)
        self.initial = initial_reference.copy()
        n_params = len(initial_reference)
        self.set_num_residuals(n_params)
        self.set_parameter_block_sizes([n_params])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute anchor residual.
        
        Args:
            parameters: [reference_flat]
            
        Returns:
            Residual (current - initial)
        """
        reference_flat = parameters[0]
        
        residual = reference_flat - self.initial
        
        return residual
    
    def _compute_jacobians(
        self,
        *,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray]
    ) -> None:
        """Compute analytic jacobian for anchor.
        
        Jacobian is simply the identity matrix scaled by weight.
        
        Args:
            parameters: [reference_flat]
            residuals: Current residual vector
            jacobians: List of jacobian matrices to fill
        """
        if jacobians[0] is not None:
            n_params = len(parameters[0])
            for i in range(n_params):
                for j in range(n_params):
                    jacobians[0][i * n_params + j] = self.weight if i == j else 0.0


class EdgeLengthVarianceCost(BaseCostFunction):
    """Penalize variance in edge length over time.
    
    Used to ensure edges maintain consistent length across all frames,
    not just in the reference geometry.
    
    Model:
        For edge (i,j):
        lengths_all_frames = [||pose_t[i] - pose_t[j]|| for all t]
        residual = variance(lengths_all_frames)
    
    This is more complex as it operates across multiple frames.
    Typically used less often than RigidEdgeCost.
    """
    
    def __init__(
        self,
        *,
        marker_i_positions: np.ndarray,
        marker_j_positions: np.ndarray,
        weight: float = 100.0
    ) -> None:
        """Initialize edge variance cost.
        
        Args:
            marker_i_positions: (n_frames, 3) positions of marker i
            marker_j_positions: (n_frames, 3) positions of marker j
            weight: Weight for constraint
        """
        super().__init__(weight=weight)
        self.pos_i = marker_i_positions.copy()
        self.pos_j = marker_j_positions.copy()
        self.n_frames = len(marker_i_positions)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute edge length variance.
        
        Args:
            parameters: Empty (operates on stored data)
            
        Returns:
            1D residual (standard deviation of lengths)
        """
        # Compute edge lengths for all frames
        lengths = np.zeros(self.n_frames)
        for t in range(self.n_frames):
            diff = self.pos_i[t] - self.pos_j[t]
            lengths[t] = np.linalg.norm(diff)
        
        # Residual is standard deviation
        residual = np.array([np.std(lengths)])
        
        return residual


class SymmetryConstraintCost(BaseCostFunction):
    """Enforce bilateral symmetry in reference geometry.
    
    Used when the tracked object has bilateral symmetry
    (e.g., left/right eyes, left/right ears).
    
    Model:
        mirror_point = [-x, y, z] of reference point
        residual = symmetric_partner - mirror_point
    """
    
    def __init__(
        self,
        *,
        marker_idx: int,
        symmetric_partner_idx: int,
        n_markers: int,
        symmetry_plane: str = "yz",
        weight: float = 50.0
    ) -> None:
        """Initialize symmetry constraint.
        
        Args:
            marker_idx: Index of marker
            symmetric_partner_idx: Index of symmetric partner
            n_markers: Total number of markers
            symmetry_plane: Which plane to mirror across ("yz", "xz", "xy")
            weight: Weight for constraint
        """
        super().__init__(weight=weight)
        self.marker_idx = marker_idx
        self.partner_idx = symmetric_partner_idx
        self.n_markers = n_markers
        self.symmetry_plane = symmetry_plane
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([n_markers * 3])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute symmetry residual.
        
        Args:
            parameters: [reference_flat]
            
        Returns:
            3D residual (partner - mirrored_point)
        """
        reference_flat = parameters[0]
        
        # Extract positions
        point = reference_flat[self.marker_idx * 3:(self.marker_idx + 1) * 3]
        partner = reference_flat[self.partner_idx * 3:(self.partner_idx + 1) * 3]
        
        # Mirror point across symmetry plane
        mirrored = point.copy()
        if self.symmetry_plane == "yz":
            mirrored[0] = -mirrored[0]
        elif self.symmetry_plane == "xz":
            mirrored[1] = -mirrored[1]
        elif self.symmetry_plane == "xy":
            mirrored[2] = -mirrored[2]
        
        # Residual
        residual = partner - mirrored
        
        return residual
