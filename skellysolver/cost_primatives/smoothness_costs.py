"""Temporal smoothness cost functions.

Used by both rigid body tracking and eye tracking to enforce
smooth motion over time.

Cost Functions:
- RotationSmoothnessCost: Smooth rotation changes (quaternions)
- TranslationSmoothnessCost: Smooth translation changes (3D vectors)
- ScalarSmoothnessCost: Smooth scalar parameter changes (e.g., pupil dilation)
"""

import numpy as np

from .base_cost import BaseCostFunction


class RotationSmoothnessCost(BaseCostFunction):
    """Temporal smoothness for rotation parameters (quaternions).
    
    Penalizes large changes in rotation between consecutive frames.
    Used by:
    - Rigid body tracking: smooth head/body rotation
    - Eye tracking: smooth gaze direction changes
    
    The residual is the difference between consecutive quaternions,
    accounting for the double-cover property (q and -q represent same rotation).
    """
    
    def __init__(self, *, weight: float) -> None:
        """Initialize rotation smoothness cost.
        
        Args:
            weight: Weight for this cost term (higher = smoother motion)
        """
        super().__init__(weight=weight)
        self.set_num_residuals(4)
        self.set_parameter_block_sizes([4, 4])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute residual as quaternion difference.
        
        Args:
            parameters: [quat_t, quat_t1] - two consecutive quaternions
            
        Returns:
            4D residual vector (quat_t1 - quat_t)
        """
        quat_t = parameters[0]
        quat_t1 = parameters[1]
        
        # Ensure quaternions are in same hemisphere (account for double cover)
        # If dot product is negative, quaternions point to opposite hemispheres
        if np.dot(quat_t, quat_t1) < 0.0:
            quat_t1_corrected = -quat_t1
        else:
            quat_t1_corrected = quat_t1
        
        # Residual is difference
        residual = quat_t1_corrected - quat_t
        
        return residual
    
    def _compute_jacobians(
        self,
        *,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray]
    ) -> None:
        """Compute analytic jacobians for rotation smoothness.
        
        Jacobian is simple for this cost:
        - d/d(quat_t) = -I * weight (negative identity)
        - d/d(quat_t1) = ±I * weight (identity, sign depends on hemisphere)
        
        Args:
            parameters: [quat_t, quat_t1]
            residuals: Current residual vector
            jacobians: List of jacobian matrices to fill
        """
        quat_t = parameters[0]
        quat_t1 = parameters[1]
        
        # Determine sign based on hemisphere
        sign = -1.0 if np.dot(quat_t, quat_t1) < 0.0 else 1.0
        
        # Jacobian w.r.t. quat_t
        if jacobians[0] is not None:
            for i in range(4):
                for j in range(4):
                    jacobians[0][i * 4 + j] = -self.weight if i == j else 0.0
        
        # Jacobian w.r.t. quat_t1
        if jacobians[1] is not None:
            for i in range(4):
                for j in range(4):
                    jacobians[1][i * 4 + j] = self.weight * sign if i == j else 0.0


class TranslationSmoothnessCost(BaseCostFunction):
    """Temporal smoothness for translation parameters (3D vectors).
    
    Penalizes large changes in position between consecutive frames.
    Used by:
    - Rigid body tracking: smooth position changes
    - Eye tracking: smooth eyeball position (if optimizing)
    
    The residual is simply the difference between consecutive translations.
    """
    
    def __init__(self, *, weight: float) -> None:
        """Initialize translation smoothness cost.
        
        Args:
            weight: Weight for this cost term (higher = smoother motion)
        """
        super().__init__(weight=weight)
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 3])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute residual as translation difference.
        
        Args:
            parameters: [trans_t, trans_t1] - two consecutive translations
            
        Returns:
            3D residual vector (trans_t1 - trans_t)
        """
        trans_t = parameters[0]
        trans_t1 = parameters[1]
        
        residual = trans_t1 - trans_t
        
        return residual
    
    def _compute_jacobians(
        self,
        *,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray]
    ) -> None:
        """Compute analytic jacobians for translation smoothness.
        
        Jacobian is simple:
        - d/d(trans_t) = -I * weight (negative identity)
        - d/d(trans_t1) = I * weight (identity)
        
        Args:
            parameters: [trans_t, trans_t1]
            residuals: Current residual vector
            jacobians: List of jacobian matrices to fill
        """
        # Jacobian w.r.t. trans_t
        if jacobians[0] is not None:
            for i in range(3):
                for j in range(3):
                    jacobians[0][i * 3 + j] = -self.weight if i == j else 0.0
        
        # Jacobian w.r.t. trans_t1
        if jacobians[1] is not None:
            for i in range(3):
                for j in range(3):
                    jacobians[1][i * 3 + j] = self.weight if i == j else 0.0


class ScalarSmoothnessCost(BaseCostFunction):
    """Temporal smoothness for scalar parameters.
    
    Penalizes large changes in a scalar value between consecutive frames.
    Used by:
    - Eye tracking: smooth pupil dilation changes
    - Any time-varying scalar parameter
    
    The residual is the difference between consecutive scalar values.
    """
    
    def __init__(self, *, weight: float = 10.0) -> None:
        """Initialize scalar smoothness cost.
        
        Args:
            weight: Weight for this cost term (higher = smoother changes)
        """
        super().__init__(weight=weight)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute residual as scalar difference.
        
        Args:
            parameters: [scalar_t, scalar_t1] - two consecutive scalars
            
        Returns:
            1D residual (scalar_t1 - scalar_t)
        """
        scalar_t = parameters[0][0]
        scalar_t1 = parameters[1][0]
        
        residual = np.array([scalar_t1 - scalar_t])
        
        return residual
    
    def _compute_jacobians(
        self,
        *,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray]
    ) -> None:
        """Compute analytic jacobians for scalar smoothness.
        
        Jacobian is trivial:
        - d/d(scalar_t) = -weight
        - d/d(scalar_t1) = weight
        
        Args:
            parameters: [scalar_t, scalar_t1]
            residuals: Current residual vector
            jacobians: List of jacobian matrices to fill
        """
        # Jacobian w.r.t. scalar_t
        if jacobians[0] is not None:
            jacobians[0][0] = -self.weight
        
        # Jacobian w.r.t. scalar_t1
        if jacobians[1] is not None:
            jacobians[1][0] = self.weight
