"""Unified optimization configuration for all SkellySolver pipelines.

This module provides configuration classes used by ALL pipelines:
- OptimizationConfig: pyceres solver parameters
- ParallelConfig: Parallel processing parameters
- WeightConfig: Cost function weights

Replaces:
- rigid_body_optimization.py::OptimizationConfig
- eye_pyceres_bundle_adjustment.py::OptimizationConfig
"""
from skellysolver.data.arbitrary_types_model import ABaseModel


class EyeTrackingWeightConfig(ABaseModel):
    """Weights specifically for eye tracking.
    
    Provides sensible defaults for eye tracking optimization.
    """
    
    lambda_data: float = 1.0
    lambda_rigid: float = 0.0
    lambda_soft: float = 0.0
    lambda_rot_smooth: float = 10.0
    lambda_trans_smooth: float = 0.0
    lambda_scalar_smooth: float = 5.0
    lambda_anchor: float = 0.0
    
    # Eye tracking specific weights
    lambda_pupil: float = 1.0
    lambda_tear_duct: float = 1.0
