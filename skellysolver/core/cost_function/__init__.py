"""Cost functions module for SkellySolver.

This module provides all cost functions used across different pipelines.
Eliminates code duplication between rigid body tracking and eye tracking.

Modules:
- base: Abstract base classes
- smoothness: Temporal smoothness costs
- measurement: Data fitting costs
- constraints: Geometric constraint costs
- manifolds: Parameter space manifolds

Usage:
    from skellysolver.core.cost_functions import RotationSmoothnessCost
    from skellysolver.core.cost_functions import Point3DMeasurementCost
"""

from .base import BaseCostFunction
from .smoothness import (
    RotationSmoothnessCost,
    TranslationSmoothnessCost,
    ScalarSmoothnessCost,
)
from .measurement import (
    Point3DMeasurementCost,
    Point2DProjectionCost,
    RigidPoint3DMeasurementBundleAdjustment,
    SimpleDistanceCost,
)
from .constraints import (
    RigidEdgeCost,
    SoftEdgeCost,
    ReferenceAnchorCost,
    EdgeLengthVarianceCost,
    SymmetryConstraintCost,
)
from .manifolds import (
    get_quaternion_manifold,
    get_sphere_manifold,
    normalize_quaternion,
    quaternion_distance,
    quaternion_slerp,
    check_quaternion_valid,
)

__all__ = [
    # Base
    "BaseCostFunction",
    # Smoothness
    "RotationSmoothnessCost",
    "TranslationSmoothnessCost",
    "ScalarSmoothnessCost",
    # Measurement
    "Point3DMeasurementCost",
    "Point2DProjectionCost",
    "RigidPoint3DMeasurementBundleAdjustment",
    "SimpleDistanceCost",
    # Constraints
    "RigidEdgeCost",
    "SoftEdgeCost",
    "ReferenceAnchorCost",
    "EdgeLengthVarianceCost",
    "SymmetryConstraintCost",
    # Manifolds
    "get_quaternion_manifold",
    "get_sphere_manifold",
    "normalize_quaternion",
    "quaternion_distance",
    "quaternion_slerp",
    "check_quaternion_valid",
]
