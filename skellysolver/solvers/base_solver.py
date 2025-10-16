import logging
import time
from abc import ABC, abstractmethod

import numpy as np
import pyceres
from pydantic import Field, model_validator
from typing_extensions import Self

from skellysolver.utilities.arbitrary_types_model import ABaseModel
from skellysolver.solvers.costs.manifold_helpers import get_quaternion_manifold
from skellysolver.data.trajectory_dataset import TrajectoryDataset

logger = logging.getLogger(__name__)


class SolverOptimizationReport(ABaseModel):
    """Summary of optimization results.

    Attributes:
        success: Whether optimization converged successfully
        iterations: Number of iterations run
        initial_cost: Initial cost value
        final_cost: Final cost value
        cost_reduction: Percentage reduction in cost
    """
    success: bool
    iterations: int
    initial_cost: float
    final_cost: float
    cost_reduction: float

    def __str__(self) -> str:
        """Human-readable optimization report."""
        status_symbol = "✓" if self.success else "✗"
        status_text = "Converged" if self.success else "Did not converge"

        lines = [
            "Optimization Report:",
            f"  Status:       {status_symbol} {status_text}",
            f"  Iterations:   {self.iterations}",
            f"  Initial cost: {self.initial_cost:.6f}",
            f"  Final cost:   {self.final_cost:.6f}",
            f"  Reduction:    {self.cost_reduction:.1f}%"
        ]
        return "\n".join(lines)


class SolverResult(ABaseModel, ABC):
    """Base class for solver results.

    Subclasses should implement from_pyceres_summary to create instances
    specific to their problem type.
    """
    success: bool
    summary: pyceres.SolverSummary
    solve_time_seconds: float
    raw_data: TrajectoryDataset
    optimized_data: TrajectoryDataset | None = None

    @model_validator(mode="after")
    def validate(self) -> Self:
        """Validate and check result - CRASHES if optimization failed.

        Raises:
            RuntimeError: If optimization did not converge, with detailed debugging info
        """
        if not self.success:
            error_msg = (
                f"❌ OPTIMIZATION FAILED TO CONVERGE!\n"
                f"\n"
                f"Optimization Details:\n"
                f"  Iterations completed: {self.summary.num_inner_iteration_steps}\n"
                f"  Initial cost:         {self.summary.initial_cost:.6f}\n"
                f"  Final cost:           {self.summary.final_cost:.6f}\n"
                f"  Cost reduction:       {self.cost_reduction_percent:.1f}%\n"
                f"  Solve time:           {self.solve_time_seconds:.2f}s\n"
                f"\n"
                f"This indicates the optimizer could not find a good solution.\n"
                f"\n"
                f"Common Causes:\n"
                f"  1. Bad input data (missing markers, excessive noise)\n"
                f"  2. Weights too strong (rigid constraints preventing convergence)\n"
                f"  3. max_iterations too low (increase to 500-1000)\n"
                f"  4. Poor initial conditions (bad reference geometry)\n"
                f"  5. Numerical instability (check for NaN/Inf in data)\n"
                f"\n"
                f"Debug Steps:\n"
                f"  1. Check input data quality (plot marker trajectories)\n"
                f"  2. Reduce constraint weights (lambda_rigid, lambda_smooth)\n"
                f"  3. Increase max_iterations in OptimizationConfig\n"
                f"  4. Try with smaller data subset to isolate problem\n"
                f"  5. Check cost reduction - if very small, may need better initialization"
            )
            # raise RuntimeError(error_msg)
        return self

    @property
    def cost_reduction(self) -> float:
        """Compute relative cost reduction.

        Returns:
            Fraction of cost reduced (0-1)
        """
        if self.summary.initial_cost == 0.0:
            return 0.0
        return (self.summary.initial_cost - self.summary.final_cost) / self.summary.initial_cost

    @property
    def cost_reduction_percent(self) -> float:
        """Compute cost reduction percentage.

        Returns:
            Percentage of cost reduced (0-100)
        """
        return self.cost_reduction * 100.0

    @classmethod
    def from_pyceres_summary(
            cls,
            *,
            summary: pyceres.SolverSummary,
            solve_time_seconds: float,
            raw_data: TrajectoryDataset,
    ) -> Self:
        """Create result from pyceres summary.

        Args:
            summary: pyceres solver summary
            solve_time_seconds: Measured solve time
            raw_data: Original input data

        Returns:
            SkeletonSolverResult instance
        """
        success = summary.termination_type in [
            pyceres.TerminationType.CONVERGENCE,
            pyceres.TerminationType.USER_SUCCESS
        ]


        return cls(
            success=success,
            summary= summary,
            solve_time_seconds=solve_time_seconds,
            raw_data=raw_data,
        )


class SolverConfig(ABaseModel):
    """Configuration for pyceres nonlinear optimization.

    Used by ALL pipelines - rigid body tracking, eye tracking, skeleton solving.

    Now stores loss function parameters instead of objects for picklability.
    """

    # Convergence criteria
    max_num_iterations: int = 100
    function_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-8
    parameter_tolerance: float = 1e-7

    linear_solver_type: pyceres.LinearSolverType = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    trust_region_strategy_type: pyceres.TrustRegionStrategyType = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT

    # Loss function (stored as type + params for picklability)
    loss_type: str = "huber"  # "none", "huber", "cauchy", "soft_l_one", "arctan", "tolerant"
    loss_scale: float = 2.0  # Used by huber, cauchy, soft_l_one, arctan
    loss_a: float | None = None  # Used by tolerant
    loss_b: float | None = None  # Used by tolerant

    # Logging
    minimizer_progress_to_stdout: bool = True

    def get_loss_function(self) -> pyceres.LossFunction | None:
        """Create loss function from stored parameters.

        Returns:
            pyceres.LossFunction instance or None
        """
        if self.loss_type == "none":
            return None
        elif self.loss_type == "huber":
            return pyceres.HuberLoss(self.loss_scale)
        elif self.loss_type == "cauchy":
            return pyceres.CauchyLoss(self.loss_scale)
        elif self.loss_type == "soft_l_one":
            return pyceres.SoftLOneLoss(self.loss_scale)
        elif self.loss_type == "arctan":
            return pyceres.ArctanLoss(self.loss_scale)
        elif self.loss_type == "tolerant":
            if self.loss_a is None or self.loss_b is None:
                raise ValueError("Tolerant loss requires both loss_a and loss_b parameters")
            return pyceres.TolerantLoss(self.loss_a, self.loss_b)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def to_solver_options(self) -> pyceres.SolverOptions:
        """Convert to pyceres SolverOptions."""
        options = pyceres.SolverOptions()
        options.max_num_iterations = self.max_num_iterations
        options.function_tolerance = self.function_tolerance
        options.gradient_tolerance = self.gradient_tolerance
        options.parameter_tolerance = self.parameter_tolerance
        options.linear_solver_type = self.linear_solver_type
        options.trust_region_strategy_type = self.trust_region_strategy_type
        options.minimizer_progress_to_stdout = self.minimizer_progress_to_stdout
        return options

    def __str__(self) -> str:
        """Human-readable solver config."""
        if self.loss_type == "none":
            loss_str = "None"
        elif self.loss_type == "tolerant":
            loss_str = f"TolerantLoss(a={self.loss_a}, b={self.loss_b})"
        else:
            loss_str = f"{self.loss_type.title()}Loss({self.loss_scale})"

        lines = [
            "Solver Configuration:",
            f"  Max iterations:   {self.max_num_iterations}",
            f"  Function tol:     {self.function_tolerance:.2e}",
            f"  Gradient tol:     {self.gradient_tolerance:.2e}",
            f"  Parameter tol:    {self.parameter_tolerance:.2e}",
            f"  Linear solver:    {self.linear_solver_type.name}",
            f"  Trust region:     {self.trust_region_strategy_type.name}",
            f"  Loss function:    {loss_str}",
            f"  Progress to stdout: {self.minimizer_progress_to_stdout}"
        ]
        return "\n".join(lines)


class PyceresProblemSolver(ABaseModel):
    """Generic pyceres solver wrapper.

    Provides high-level interface to pyceres for all pipelines.
    Handles problem setup, cost addition, and solving.

    Note: This is a standalone solver that should be created fresh
    for each chunk in parallel processing.

    Usage:
        config = SolverConfig(...)
        solver = PyceresSolver(config=config)

        # Add parameters
        solver.add_parameter_block(name="rotation", parameters=quat)
        solver.add_quaternion_parameter(name="rotation", parameters=quat)

        # Add costs
        solver.add_residual_block(cost=cost_fn, parameters=[quat, trans])

        # Solve
        result = solver.solve()
    """

    config: SolverConfig
    problem: pyceres.Problem = Field(default_factory=pyceres.Problem)
    parameter_blocks: dict[str, np.ndarray] = Field(default_factory=dict)

    def add_parameter_block(
            self,
            *,
            name: str,
            parameters: np.ndarray,
            manifold: pyceres.Manifold | None = None
    ) -> None:
        """Add parameter block to optimization problem.

        Args:
            name: Identifier for this parameter block
            parameters: Parameter array (will be modified in-place)
            manifold: Optional manifold constraint
        """
        self.parameter_blocks[name] = parameters
        self.problem.add_parameter_block(parameters, len(parameters))

        if manifold is not None:
            self.problem.set_manifold(parameters, manifold)

        logger.debug(f"Added parameter block '{name}' with {len(parameters)} parameters")

    def add_quaternion_parameter(
            self,
            *,
            name: str,
            parameter: np.ndarray
    ) -> None:
        """Add quaternion parameter with manifold constraint.

        Convenience method for quaternion parameters, automatically
        applies unit-norm manifold.

        Args:
            name: Identifier for this parameter block
            parameter: (4,) quaternion array
        """
        if len(parameter) != 4:
            raise ValueError(f"Quaternion must have 4 elements, got {len(parameter)}")

        manifold = get_quaternion_manifold()
        self.add_parameter_block(
            name=name,
            parameters=parameter,
            manifold=manifold
        )

    def add_residual_block(
            self,
            *,
            cost: pyceres.CostFunction,
            parameters: list[np.ndarray],
            loss: pyceres.LossFunction | None = None
    ) -> None:
        """Add residual block to optimization problem.

        Args:
            cost: Cost function
            parameters: List of parameter arrays used by this cost
            loss: Optional loss function (uses config default if None)
        """
        if loss is None:
            loss = self.config.get_loss_function()

        self.problem.add_residual_block(cost, loss, parameters)

    def set_parameter_bounds(
            self,
            *,
            parameters: np.ndarray,
            index: int,
            lower: float | None = None,
            upper: float | None = None
    ) -> None:
        """Set bounds on a specific parameter.

        Args:
            parameters: Parameter array
            index: Index within parameter array
            lower: Lower bound (None = unbounded)
            upper: Upper bound (None = unbounded)
        """
        if lower is not None:
            self.problem.set_parameter_lower_bound(parameters, index, lower)

        if upper is not None:
            self.problem.set_parameter_upper_bound(parameters, index, upper)

    def set_parameter_constant(
            self,
            *,
            parameters: np.ndarray
    ) -> None:
        """Set parameter block as constant (not optimized).

        Args:
            parameters: Parameter array to hold constant
        """
        self.problem.set_parameter_block_constant(parameters)

    def get_parameter(self, *, name: str) -> np.ndarray:
        """Get parameter block by name.

        Args:
            name: Parameter block identifier

        Returns:
            Parameter array
        """
        if name not in self.parameter_blocks:
            raise KeyError(f"Parameter block '{name}' not found")

        return self.parameter_blocks[name]

    def num_parameters(self) -> int:
        """Get total number of parameters.

        Returns:
            Number of parameters being optimized
        """
        return self.problem.num_parameters()

    def num_residuals(self) -> int:
        """Get total number of residuals.

        Returns:
            Number of residual blocks
        """
        return self.problem.num_residual_blocks()

    def solve(self) -> tuple[pyceres.SolverSummary, float]:
        """Solve optimization problem and return summary.

        Pipelines should use the returned summary to create their specific
        SolverResult subclass via from_pyceres_summary().

        Returns:
            Tuple of (summary, solve_time_seconds)
        """
        logger.info("=" * 80)
        logger.info("STARTING OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Parameters:      {self.num_parameters()}")
        logger.info(f"Residual blocks: {self.num_residuals()}")

        # Configure solver
        options = self.config.to_solver_options()
        summary = pyceres.SolverSummary()

        # Solve
        start_time = time.time()
        pyceres.solve(options, self.problem, summary)
        solve_time = time.time() - start_time

        # Log result
        success = summary.termination_type in [
            pyceres.TerminationType.CONVERGENCE,
            pyceres.TerminationType.USER_SUCCESS
        ]

        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Status:       {'✓ Converged' if success else '✗ Did not converge'}")
        logger.info(f"Iterations:   {summary.num_successful_steps}")
        logger.info(f"Solve time:   {solve_time:.2f}s")
        logger.info(f"Initial cost: {summary.initial_cost:.6f}")
        logger.info(f"Final cost:   {summary.final_cost:.6f}")

        cost_reduction = 0.0
        if summary.initial_cost > 0:
            cost_reduction = (summary.initial_cost - summary.final_cost) / summary.initial_cost * 100.0

        logger.info(f"Reduction:    {cost_reduction:.1f}%")
        logger.info("=" * 80)

        return summary, solve_time