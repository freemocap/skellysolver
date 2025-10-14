"""Eye tracking specific cost functions.

Cost functions for fitting eye model to 2D pupil observations.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from skellysolver.core.cost_primatives.base_cost import BaseCostFunction


class PupilPointProjectionCost(BaseCostFunction):
    """Project 3D pupil point to 2D and compare with observation.
    
    Model:
        1. Create 3D pupil ellipse in eye frame
        2. Rotate by gaze direction (quaternion)
        3. Scale by pupil dilation
        4. Project to 2D using pinhole camera
        5. Compare with observed 2D point
    
    Parameters:
        - quaternion (4): gaze direction
        - pupil_scale (1): pupil dilation factor
        - eyeball_center (3): 3D position of eyeball center
        - pupil_shape (2): [semi_major, semi_minor] in mm
    """
    
    def __init__(
        self,
        *,
        observed_px: np.ndarray,
        point_index: int,
        n_pupil_points: int,
        camera_fx: float,
        camera_fy: float,
        camera_cx: float,
        camera_cy: float,
        base_semi_major_mm: float,
        base_semi_minor_mm: float,
        weight: float = 1.0
    ) -> None:
        """Initialize pupil projection cost.
        
        Args:
            observed_px: (2,) observed pixel coordinates [u, v]
            point_index: Index of this point on pupil ellipse [0, n_pupil_points)
            n_pupil_points: Total number of points on pupil ellipse
            camera_fx: Focal length in x (pixels)
            camera_fy: Focal length in y (pixels)
            camera_cx: Principal point x (pixels)
            camera_cy: Principal point y (pixels)
            base_semi_major_mm: Base pupil semi-major axis (mm)
            base_semi_minor_mm: Base pupil semi-minor axis (mm)
            weight: Weight for this measurement
        """
        super().__init__(weight=weight)
        self.observed = observed_px.copy()
        self.point_idx = point_index
        self.n_points = n_pupil_points
        self.fx = camera_fx
        self.fy = camera_fy
        self.cx = camera_cx
        self.cy = camera_cy
        self.base_major = base_semi_major_mm
        self.base_minor = base_semi_minor_mm
        
        # Pre-compute angle for this point on ellipse
        self.theta = 2.0 * np.pi * point_index / n_pupil_points
        
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([4, 1, 3]) # [quaternion, pupil_scale, eyeball_center]
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute projection residual.
        
        Args:
            parameters: [quaternion, pupil_scale, eyeball_center]
            
        Returns:
            2D residual (observed - projected)
        """
        quat = parameters[0]
        scale = parameters[1][0]
        eyeball_center = parameters[2]
        
        # 1. Generate point on unit ellipse in eye frame
        # Ellipse in eye frame: facing forward (z-axis)
        x_local = self.base_major * scale * np.cos(self.theta)
        y_local = self.base_minor * scale * np.sin(self.theta)
        z_local = 0.0  # Pupil is flat in eye frame
        
        point_local = np.array([x_local, y_local, z_local])
        
        # 2. Rotate by gaze direction
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        R = Rotation.from_quat(quat=quat_scipy).as_matrix()
        
        point_rotated = R @ point_local
        
        # 3. Translate to world position (relative to eyeball center)
        point_3d = point_rotated + eyeball_center
        
        # 4. Project to 2D using pinhole model
        x = point_3d[0]
        y = point_3d[1]
        z = point_3d[2]
        
        # Avoid division by zero
        if z < 0.1:
            z = 0.1
        
        x_norm = x / z
        y_norm = y / z
        
        u = self.fx * x_norm + self.cx
        v = self.fy * y_norm + self.cy
        
        projected = np.array([u, v])
        
        # 5. Residual
        residual = self.observed - projected
        
        return residual


class TearDuctProjectionCost(BaseCostFunction):
    """Project 3D tear duct to 2D and compare with observation.
    
    The tear duct is a fixed anatomical landmark on the eye.
    It doesn't rotate with gaze and stays fixed relative to the eyeball center and eye camera reference frame.
    
    Parameters:
        - eyeball_center (3): 3D position of eyeball center
        - tear_duct_offset (3): Offset from eyeball center to tear duct
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
        """Initialize tear duct projection cost.
        
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
        self.set_parameter_block_sizes([3, 3])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute projection residual.
        
        Args:
            parameters: [eyeball_center, tear_duct_offset]
            
        Returns:
            2D residual (observed - projected)
        """
        eyeball_center = parameters[0]
        tear_duct_offset = parameters[1]
        
        # 1. Compute 3D position
        point_3d = eyeball_center + tear_duct_offset
        
        # 2. Project to 2D
        x = point_3d[0]
        y = point_3d[1]
        z = point_3d[2]
        
        # Avoid division by zero
        if z < 0.01:
            z = 0.01
        
        x_norm = x / z
        y_norm = y / z
        
        u = self.fx * x_norm + self.cx
        v = self.fy * y_norm + self.cy
        
        projected = np.array([u, v])
        
        # 3. Residual
        residual = self.observed - projected
        
        return residual
