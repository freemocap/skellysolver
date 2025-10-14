"""Manifold definitions for constrained parameter spaces.

Manifolds define the geometry of parameter spaces that have
constraints (e.g., unit quaternions).

This module provides convenient wrappers around pyceres manifolds.
"""

import numpy as np
import pyceres


def get_quaternion_manifold() -> pyceres.QuaternionManifold:
    """Get quaternion manifold for rotation parameters.

    Quaternions must maintain unit length ||q|| = 1.
    The manifold ensures this constraint during optimization.

    Used by:
    - Rigid body tracking (body orientation)
    - Eye tracking (gaze direction)

    Returns:
        pyceres.QuaternionManifold instance
    """
    return


def get_sphere_manifold(*, size: int) -> pyceres.SphereManifold:
    """Get sphere manifold for unit-length vectors.
    
    Constrains parameter vector to lie on unit sphere.
    
    Args:
        size: Dimension of the ambient space
        
    Returns:
        pyceres.SphereManifold instance
    """
    return pyceres.SphereManifold(size)



def normalize_quaternion(*, quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length.
    
    Args:
        quat: (4,) quaternion [w, x, y, z] or [x, y, z, w]
        
    Returns:
        (4,) normalized quaternion
    """
    norm = np.linalg.norm(quat)
    if norm < 1e-10:
        # Degenerate case - return identity
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def quaternion_distance(*, q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute geodesic distance between quaternions.
    
    The distance is the angle of rotation between the two orientations.
    
    Args:
        q1: (4,) first quaternion
        q2: (4,) second quaternion
        
    Returns:
        Distance in radians [0, π]
    """
    # Account for double cover (q and -q represent same rotation)
    dot = np.abs(np.dot(q1, q2))
    
    # Clamp to avoid numerical issues with arccos
    dot = np.clip(dot, -1.0, 1.0)
    
    # Geodesic distance
    distance = 2.0 * np.arccos(dot)
    
    return distance


def quaternion_slerp(
    *,
    q1: np.ndarray,
    q2: np.ndarray,
    t: float
) -> np.ndarray:
    """Spherical linear interpolation between quaternions.
    
    Args:
        q1: (4,) start quaternion
        q2: (4,) end quaternion
        t: Interpolation parameter [0, 1]
        
    Returns:
        (4,) interpolated quaternion
    """
    # Account for double cover
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Clamp to avoid numerical issues
    dot = np.clip(dot, -1.0, 1.0)
    
    # Compute angle
    theta = np.arccos(dot)
    
    # Handle near-parallel case
    if np.abs(theta) < 1e-10:
        return normalize_quaternion(quat=(1.0 - t) * q1 + t * q2)
    
    # SLERP formula
    sin_theta = np.sin(theta)
    w1 = np.sin((1.0 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result = w1 * q1 + w2 * q2
    
    return normalize_quaternion(quat=result)


def check_quaternion_valid(*, quat: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if quaternion is valid (unit length).
    
    Args:
        quat: (4,) quaternion to check
        tol: Tolerance for unit length check
        
    Returns:
        True if valid
    """
    norm = np.linalg.norm(quat)
    return np.abs(norm - 1.0) < tol
