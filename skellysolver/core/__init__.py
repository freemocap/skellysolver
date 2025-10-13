"""Core optimization infrastructure for SkellySolver.

This module provides unified optimization components used by all pipelines:

- config: Optimization and parallel configuration
- result: Optimization result structures
- optimizer: High-level optimizer wrapper
- cost_functions: Library of cost functions
- topology: Rigid body topology (existing)
- geometry: Geometric utilities (existing)
- metrics: Evaluation metrics (existing)
- chunking: Chunking for long sequences (existing)
- parallel: Parallel optimization (existing)

Usage:
    from skellysolver.core import OptimizationConfig, Optimizer
    from skellysolver.core import RotationSmoothnessCost
    
    config = OptimizationConfig(max_iterations=300)
    optimizer = Optimizer(config=config)
"""

# Configuration
from .config import (
    OptimizationConfig,
    ParallelConfig,
    WeightConfig,
    RigidBodyWeightConfig,
    EyeTrackingWeightConfig,
)

# Results
from .result import (
    OptimizationResult,
    RigidBodyResult,
    EyeTrackingResult,
    ChunkedResult,
)

# Optimizer
from .optimizer import (
    Optimizer,
    BatchOptimizer,
)

# Cost functions (import everything from cost_functions module)
from .cost_functions import (
    BaseCostFunction,
    RotationSmoothnessCost,
    TranslationSmoothnessCost,
    ScalarSmoothnessCost,
    Point3DMeasurementCost,
    Point2DProjectionCost,
    RigidPoint3DMeasurementBundleAdjustment,
    SimpleDistanceCost,
    RigidEdgeCost,
    SoftEdgeCost,
    ReferenceAnchorCost,
    EdgeLengthVarianceCost,
    SymmetryConstraintCost,
    get_quaternion_manifold,
    get_sphere_manifold,
    normalize_quaternion,
    quaternion_distance,
    quaternion_slerp,
    check_quaternion_valid,
)

# Keep existing modules (not part of Phase 1)
# These are imported for convenience but not re-exported
# topology, geometry, metrics, chunking, parallel

__all__ = [
    # Configuration
    "OptimizationConfig",
    "ParallelConfig",
    "WeightConfig",
    "RigidBodyWeightConfig",
    "EyeTrackingWeightConfig",
    # Results
    "OptimizationResult",
    "RigidBodyResult",
    "EyeTrackingResult",
    "ChunkedResult",
    # Optimizer
    "Optimizer",
    "BatchOptimizer",
    # Cost functions - Base
    "BaseCostFunction",
    # Cost functions - Smoothness
    "RotationSmoothnessCost",
    "TranslationSmoothnessCost",
    "ScalarSmoothnessCost",
    # Cost functions - Measurement
    "Point3DMeasurementCost",
    "Point2DProjectionCost",
    "RigidPoint3DMeasurementBundleAdjustment",
    "SimpleDistanceCost",
    # Cost functions - Constraints
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
