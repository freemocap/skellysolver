"""Pipelines module for SkellySolver.

This module provides pipeline implementations for different optimization tasks:

Base:
- BasePipeline: Abstract base class for all pipelines
- PipelineConfig: Base configuration class
- PipelineRunner: Utility for running multiple pipelines

Rigid Body Tracking:
- RigidBodyPipeline: Rigid body pose optimization
- RigidBodyConfig: Configuration for rigid body tracking

Eye Tracking:
- EyeTrackingPipeline: Eye orientation and pupil dilation optimization
- EyeTrackingConfig: Configuration for eye tracking
- CameraIntrinsics: Camera model
- EyeModel: Eye model parameters

Usage:
    # Rigid body tracking
    from skellysolver.pipelines.rigid_body import (
        RigidBodyPipeline,
        RigidBodyConfig,
    )
    
    config = RigidBodyConfig(...)
    pipeline = RigidBodyPipeline(config=config)
    result = pipeline.run()
    
    # Eye tracking
    from skellysolver.pipelines.eye_tracking import (
        EyeTrackingPipeline,
        EyeTrackingConfig,
    )
    
    config = EyeTrackingConfig(...)
    pipeline = EyeTrackingPipeline(config=config)
    result = pipeline.run()
"""

# Base pipeline infrastructure
from .base_pipeline import (
    BasePipeline,
    PipelineConfig,
    PipelineRunner,
)

# Rigid body tracking
from .rigid_body_pipeline import (
    RigidBodyPipeline,
    RigidBodyConfig,
)

# Eye tracking
from .eye_pipeline import (
    EyeTrackingPipeline,
    EyeTrackingConfig,
    CameraIntrinsics,
)
__all__ = [
    # Base
    "BasePipeline",
    "PipelineConfig",
    "PipelineRunner",
    # Rigid body
    "RigidBodyPipeline",
    "RigidBodyConfig",
    # Eye tracking
    "EyeTrackingPipeline",
    "EyeTrackingConfig",
    "EyeModel",
    "CameraIntrinsics",
]
