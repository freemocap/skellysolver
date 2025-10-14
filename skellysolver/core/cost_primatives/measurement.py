"""Measurement fitting cost functions.

These cost functions fit model predictions to observed measurements.
Used for data fidelity terms in optimization.

Cost Functions:
- Point3DMeasurementCost: Fit 3D point to transformed reference
- Point2DProjectionCost: Fit 2D observation to projected 3D point
"""

import numpy as np
from scipy.spatial.transform import Rotation
import pyceres
from typing import Any

from .base_cost import BaseCostFunction


class Point3DMeasurementCost(BaseCostFunction):
    """Fit measured 3D point to transformed reference point.
    
    Used by rigid body tracking to fit noisy marker measurements
    to a rigid body model.
    
    Model:
        predicted = R @ reference + translation
        residual = measured - predicted
    
    Where R is rotation matrix from quaternion.
    """
    
    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        reference_point: np.ndarray,
        weight: float = 100.0
    ) -> None:
        """Initialize 3D measurement cost.
        
        Args:
            measured_point: (3,) observed 3D position
            reference_point: (3,) reference position in body frame
            weight: Weight for this measurement
        """
        super().__init__(weight=weight)
        self.measured = measured_point.copy()
        self.reference = reference_point.copy()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 3])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute residual as measurement error.
        
        Args:
            parameters: [quaternion, translation]
            
        Returns:
            3D residual (measured - predicted)
        """
        quat = parameters[0]
        translation = parameters[1]
        
        # Convert quaternion to rotation matrix
        # pyceres uses [w, x, y, z] but scipy uses [x, y, z, w]
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        R = Rotation.from_quat(quat=quat_scipy).as_matrix()
        
        # Transform reference point
        predicted = R @ self.reference + translation
        
        # Residual
        residual = self.measured - predicted
        
        return residual


class Point2DProjectionCost(BaseCostFunction):
    """Fit 2D observation to projection of 3D point.
    
    Used by eye tracking and camera calibration to fit
    observed image points to projections of 3D points.
    
    Model:
        point_3d = transform(reference_point)
        projected = project(point_3d, camera)
        residual = observed - projected
    
    This is a base class - subclasses specify how to compute point_3d.
    """
    
    def __init__(
        self,
        *,
        observed_px: np.ndarray,
        camera_fx: float,
        camera_fy: float,
        camera_cx: float,
        camera_cy: float,
        weight: float = 1.0
    ) -> None:
        """Initialize 2D projection cost.
        
        Args:
            observed_px: (2,) observed pixel coordinates [u, v]
            camera_fx: Focal length in x (pixels)
            camera_fy: Focal length in y (pixels)
            camera_cx: Principal point x (pixels)
            camera_cy: Principal point y (pixels)
            weight: Weight for this measurement
        """
        super().__init__(weight=weight)
        self.observed = observed_px.copy()
        self.fx = camera_fx
        self.fy = camera_fy
        self.cx = camera_cx
        self.cy = camera_cy
        self.set_num_residuals(2)
    
    def _project_point(
        self,
        *,
        point_3d: np.ndarray
    ) -> np.ndarray:
        """Project 3D point to 2D using pinhole camera model.
        
        Args:
            point_3d: (3,) point in camera frame [x, y, z]
            
        Returns:
            (2,) pixel coordinates [u, v]
        """
        x = point_3d[0]
        y = point_3d[1]
        z = point_3d[2]
        
        # Normalize by depth
        x_norm = x / z
        y_norm = y / z
        
        # Apply camera intrinsics
        u = self.fx * x_norm + self.cx
        v = self.fy * y_norm + self.cy
        
        return np.array([u, v])


class RigidPoint3DMeasurementBundleAdjustment(BaseCostFunction):
    """Fit 3D measurement with joint optimization of pose AND geometry.
    
    Used in bundle adjustment where both the reference geometry and
    poses are being optimized simultaneously.
    
    Model:
        predicted = R @ reference[marker_idx] + translation
        residual = measured - predicted
    
    Parameters:
        - quaternion (4): rotation
        - translation (3): translation
        - reference_flat (n_markers * 3): all reference points flattened
    """
    
    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        marker_idx: int,
        n_markers: int,
        weight: float = 100.0
    ) -> None:
        """Initialize bundle adjustment measurement cost.
        
        Args:
            measured_point: (3,) observed position
            marker_idx: Index of this marker in reference geometry
            n_markers: Total number of markers
            weight: Weight for this measurement
        """
        super().__init__(weight=weight)
        self.measured = measured_point.copy()
        self.marker_idx = marker_idx
        self.n_markers = n_markers
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 3, n_markers * 3])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute residual with reference geometry as parameter.
        
        Args:
            parameters: [quaternion, translation, reference_flat]
            
        Returns:
            3D residual (measured - predicted)
        """
        quat = parameters[0]
        translation = parameters[1]
        reference_flat = parameters[2]
        
        # Extract this marker's reference position
        start_idx = self.marker_idx * 3
        reference_point = reference_flat[start_idx:start_idx + 3]
        
        # Convert quaternion to rotation matrix
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        R = Rotation.from_quat(quat=quat_scipy).as_matrix()
        
        # Transform reference point
        predicted = R @ reference_point + translation
        
        # Residual
        residual = self.measured - predicted
        
        return residual


class SimpleDistanceCost(BaseCostFunction):
    """Simple distance penalty between measured point and reference.
    
    Used for soft constraints where we want points to be near
    a reference but allow some flexibility.
    
    Model:
        distance = ||measured - (R @ reference + translation)||
        residual = distance - target_distance
    """
    
    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        reference_point: np.ndarray,
        target_distance: float,
        weight: float = 10.0
    ) -> None:
        """Initialize distance cost.
        
        Args:
            measured_point: (3,) observed position
            reference_point: (3,) reference position
            target_distance: Desired distance
            weight: Weight for this constraint
        """
        super().__init__(weight=weight)
        self.measured = measured_point.copy()
        self.reference = reference_point.copy()
        self.target_dist = target_distance
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([4, 3])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute distance residual.
        
        Args:
            parameters: [quaternion, translation]
            
        Returns:
            1D residual (distance - target)
        """
        quat = parameters[0]
        translation = parameters[1]
        
        # Convert quaternion to rotation matrix
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        R = Rotation.from_quat(quat=quat_scipy).as_matrix()
        
        # Transform reference point
        transformed = R @ self.reference + translation
        
        # Compute distance
        diff = self.measured - transformed
        distance = np.linalg.norm(diff)
        
        # Residual
        residual = np.array([distance - self.target_dist])
        
        return residual
