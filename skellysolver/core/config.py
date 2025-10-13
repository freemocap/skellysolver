"""Unified optimization configuration for all SkellySolver pipelines.

This module provides configuration classes used by ALL pipelines:
- OptimizationConfig: pyceres solver parameters
- ParallelConfig: Parallel processing parameters
- WeightConfig: Cost function weights

Replaces:
- rigid_body_optimization.py::OptimizationConfig
- eye_pyceres_bundle_adjustment.py::OptimizationConfig
"""

import os
import pyceres
from dataclasses import dataclass
from typing import Literal


@dataclass
class OptimizationConfig:
    """Configuration for pyceres nonlinear optimization.
    
    Used by ALL pipelines - rigid body tracking, eye tracking, future pipelines.
    
    Attributes:
        max_iterations: Maximum number of optimization iterations
        function_tolerance: Convergence tolerance for cost function
        gradient_tolerance: Convergence tolerance for gradient
        parameter_tolerance: Convergence tolerance for parameters
        use_robust_loss: Whether to use robust loss function
        robust_loss_type: Type of robust loss (huber, cauchy, soft_l1)
        robust_loss_param: Parameter for robust loss (e.g., huber delta)
        linear_solver: Linear solver type
        trust_region_strategy: Trust region strategy
        num_threads: Number of threads (None = auto-detect)
        minimizer_progress_to_stdout: Print optimization progress
    """
    
    # Convergence criteria
    max_iterations: int = 300
    function_tolerance: float = 1e-9
    gradient_tolerance: float = 1e-11
    parameter_tolerance: float = 1e-10
    
    # Robust loss function
    use_robust_loss: bool = True
    robust_loss_type: Literal["huber", "cauchy", "soft_l1"] = "huber"
    robust_loss_param: float = 2.0
    
    # Linear solver
    linear_solver: Literal["dense_qr", "sparse_normal_cholesky", "sparse_schur"] = "sparse_normal_cholesky"
    
    # Trust region strategy
    trust_region_strategy: Literal["levenberg_marquardt", "dogleg"] = "levenberg_marquardt"
    
    # Parallelization
    num_threads: int | None = None
    
    # Logging
    minimizer_progress_to_stdout: bool = True
    
    def to_solver_options(self) -> pyceres.SolverOptions:
        """Convert to pyceres SolverOptions.
        
        Returns:
            pyceres.SolverOptions configured with these settings
        """
        options = pyceres.SolverOptions()
        
        # Convergence
        options.max_num_iterations = self.max_iterations
        options.function_tolerance = self.function_tolerance
        options.gradient_tolerance = self.gradient_tolerance
        options.parameter_tolerance = self.parameter_tolerance
        
        # Linear solver
        if self.linear_solver == "dense_qr":
            options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        elif self.linear_solver == "sparse_normal_cholesky":
            options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        elif self.linear_solver == "sparse_schur":
            options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        
        # Trust region
        if self.trust_region_strategy == "levenberg_marquardt":
            options.trust_region_strategy_type = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
        elif self.trust_region_strategy == "dogleg":
            options.trust_region_strategy_type = pyceres.TrustRegionStrategyType.DOGLEG
        
        # Threading
        if self.num_threads is None:
            cpu_count = os.cpu_count()
            options.num_threads = max(cpu_count - 1 if cpu_count else 1, 1)
        else:
            options.num_threads = self.num_threads
        
        # Logging
        options.minimizer_progress_to_stdout = self.minimizer_progress_to_stdout
        
        return options
    
    def get_loss_function(self) -> pyceres.LossFunction | None:
        """Get robust loss function.
        
        Returns:
            pyceres loss function or None if robust loss disabled
        """
        if not self.use_robust_loss:
            return None
        
        if self.robust_loss_type == "huber":
            return pyceres.HuberLoss(self.robust_loss_param)
        elif self.robust_loss_type == "cauchy":
            return pyceres.CauchyLoss(self.robust_loss_param)
        elif self.robust_loss_type == "soft_l1":
            return pyceres.SoftLOneLoss(self.robust_loss_param)
        else:
            return None


@dataclass
class ParallelConfig:
    """Configuration for parallel chunked optimization.
    
    Used when processing long recordings that need to be split
    into chunks and processed in parallel.
    
    Attributes:
        enabled: Whether to use parallel processing
        chunk_size: Number of frames per chunk
        overlap_size: Number of overlapping frames between chunks
        blend_window: Size of blending window in overlap region
        min_chunk_size: Minimum frames to process as separate chunk
        num_workers: Number of parallel workers (None = auto-detect)
    """
    
    enabled: bool = True
    chunk_size: int = 500
    overlap_size: int = 50
    blend_window: int = 25
    min_chunk_size: int = 100
    num_workers: int | None = None
    
    def get_num_workers(self) -> int:
        """Get number of workers.
        
        Returns:
            Number of workers to use
        """
        if self.num_workers is None:
            cpu_count = os.cpu_count()
            return cpu_count if cpu_count else 1
        return self.num_workers
    
    def should_use_parallel(self, *, n_frames: int) -> bool:
        """Determine if parallel processing should be used.
        
        Args:
            n_frames: Total number of frames
            
        Returns:
            True if should use parallel processing
        """
        if not self.enabled:
            return False
        
        # Only use parallel if we have enough frames
        return n_frames > (self.chunk_size + self.min_chunk_size)


@dataclass
class WeightConfig:
    """Configuration for cost function weights.
    
    These weights control the relative importance of different
    terms in the optimization objective.
    
    Higher weight = more importance
    
    Common patterns:
    - Data fitting: 100.0
    - Rigid constraints: 500.0 (very high)
    - Soft constraints: 10.0 (low)
    - Smoothness: 100-200
    """
    
    # Data fitting
    lambda_data: float = 100.0
    
    # Geometric constraints
    lambda_rigid: float = 500.0
    lambda_soft: float = 10.0
    
    # Temporal smoothness
    lambda_rot_smooth: float = 100.0
    lambda_trans_smooth: float = 100.0
    lambda_scalar_smooth: float = 10.0
    
    # Anchoring
    lambda_anchor: float = 1.0
    
    def scale_all(self, *, factor: float) -> None:
        """Scale all weights by a factor.
        
        Useful for globally increasing or decreasing constraint strength.
        
        Args:
            factor: Multiplicative factor
        """
        self.lambda_data *= factor
        self.lambda_rigid *= factor
        self.lambda_soft *= factor
        self.lambda_rot_smooth *= factor
        self.lambda_trans_smooth *= factor
        self.lambda_scalar_smooth *= factor
        self.lambda_anchor *= factor


@dataclass
class RigidBodyWeightConfig(WeightConfig):
    """Weights specifically for rigid body tracking.
    
    Provides sensible defaults for rigid body optimization.
    """
    
    lambda_data: float = 100.0
    lambda_rigid: float = 500.0
    lambda_soft: float = 10.0
    lambda_rot_smooth: float = 200.0
    lambda_trans_smooth: float = 200.0
    lambda_anchor: float = 10.0


@dataclass
class EyeTrackingWeightConfig(WeightConfig):
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
