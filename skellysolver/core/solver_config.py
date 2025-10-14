import os
from dataclasses import dataclass
from typing import Literal

import pyceres


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
    function_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-8
    parameter_tolerance: float = 1e-7

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
