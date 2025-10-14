"""Generic optimizer wrapper for pyceres.

This module provides a high-level interface to pyceres optimization
that handles:
- Problem setup
- Parameter block management
- Cost function addition
- Manifold specification
- Bounds enforcement
- Solving
- Result extraction

Used by ALL pipelines to eliminate duplicated optimization code.
"""

import numpy as np
import pyceres
import time
import logging
from typing import Any

from .config import OptimizationConfig
from .cost_primatives import get_quaternion_manifold
from .result import OptimizationResult

logger = logging.getLogger(__name__)


class Optimizer:
    """Generic pyceres optimizer wrapper.
    
    Provides high-level interface to pyceres for all pipelines.
    Handles problem setup, cost addition, and solving.
    
    Usage:
        config = OptimizationConfig(...)
        optimizer = Optimizer(config=config)
        
        # Add parameters
        optimizer.add_parameter_block(name="rotation", parameters=quat)
        optimizer.add_quaternion_parameter(name="rotation", parameters=quat)
        
        # Add costs
        optimizer.add_residual_block(cost=cost_fn, parameters=[quat, trans])
        
        # Solve
        result = optimizer.solve()
    """
    
    def __init__(self, *, config: OptimizationConfig) -> None:
        """Initialize optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.problem = pyceres.Problem()
        self.parameter_blocks: dict[str, np.ndarray] = {}
        self.loss_function = config.get_loss_function()
    
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
        parameters: np.ndarray
    ) -> None:
        """Add quaternion parameter with manifold constraint.
        
        Convenience method for quaternion parameters.
        
        Args:
            name: Identifier for this parameter block
            parameters: (4,) quaternion array
        """
        if len(parameters) != 4:
            raise ValueError(f"Quaternion must have 4 elements, got {len(parameters)}")
        
        manifold = get_quaternion_manifold()
        self.add_parameter_block(
            name=name,
            parameters=parameters,
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
            loss = self.loss_function
        
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
    
    def solve(self) -> OptimizationResult:
        """Solve optimization problem.
        
        Returns:
            OptimizationResult with optimization results
        """
        logger.info("="*80)
        logger.info("STARTING OPTIMIZATION")
        logger.info("="*80)
        logger.info(f"Parameters:      {self.num_parameters()}")
        logger.info(f"Residual blocks: {self.num_residuals()}")
        
        # Configure solver
        options = self.config.to_solver_options()
        summary = pyceres.SolverSummary()
        
        # Solve
        start_time = time.time()
        pyceres.solve(options, self.problem, summary)
        solve_time = time.time() - start_time
        
        # Create result
        result = OptimizationResult.from_pyceres_summary(
            summary=summary,
            solve_time_seconds=solve_time
        )
        
        # Log result
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Status:       {'✓ Converged' if result.success else '✗ Did not converge'}")
        logger.info(f"Iterations:   {result.num_iterations}")
        logger.info(f"Solve time:   {result.solve_time_seconds:.2f}s")
        logger.info(f"Initial cost: {result.initial_cost:.6f}")
        logger.info(f"Final cost:   {result.final_cost:.6f}")
        logger.info(f"Reduction:    {result.cost_reduction_percent:.1f}%")
        logger.info("="*80)
        
        return result
    
    def solve_with_callback(
        self,
        *,
        callback: Any
    ) -> OptimizationResult:
        """Solve with iteration callback.
        
        The callback is called after each iteration with:
        - iteration: int
        - cost: float
        - parameters: dict[str, np.ndarray]
        
        Args:
            callback: Function to call each iteration
            
        Returns:
            OptimizationResult
        """
        # TODO: Implement callback support
        # pyceres doesn't directly support callbacks,
        # would need to implement via IterationCallback
        logger.warning("Callback support not yet implemented")
        return self.solve()


class BatchOptimizer:
    """Optimize multiple independent problems in batch.
    
    Useful when you have many similar optimization problems
    that can be solved in parallel.
    
    Usage:
        batch = BatchOptimizer(config=config, num_problems=10)
        
        for i in range(10):
            batch.add_parameter_block(problem_idx=i, name="rotation", ...)
            batch.add_residual_block(problem_idx=i, cost=..., ...)
        
        results = batch.solve_all()
    """
    
    def __init__(
        self,
        *,
        config: OptimizationConfig,
        num_problems: int
    ) -> None:
        """Initialize batch optimizer.
        
        Args:
            config: Optimization configuration
            num_problems: Number of independent problems
        """
        self.config = config
        self.num_problems = num_problems
        self.optimizers = [
            Optimizer(config=config)
            for _ in range(num_problems)
        ]
    
    def get_optimizer(self, *, problem_idx: int) -> Optimizer:
        """Get optimizer for specific problem.
        
        Args:
            problem_idx: Index of problem
            
        Returns:
            Optimizer instance
        """
        if problem_idx < 0 or problem_idx >= self.num_problems:
            raise IndexError(f"Problem index {problem_idx} out of range")
        
        return self.optimizers[problem_idx]
    
    def add_parameter_block(
        self,
        *,
        problem_idx: int,
        name: str,
        parameters: np.ndarray,
        manifold: pyceres.Manifold | None = None
    ) -> None:
        """Add parameter block to specific problem.
        
        Args:
            problem_idx: Index of problem
            name: Parameter identifier
            parameters: Parameter array
            manifold: Optional manifold
        """
        optimizer = self.get_optimizer(problem_idx=problem_idx)
        optimizer.add_parameter_block(
            name=name,
            parameters=parameters,
            manifold=manifold
        )
    
    def add_residual_block(
        self,
        *,
        problem_idx: int,
        cost: pyceres.CostFunction,
        parameters: list[np.ndarray],
        loss: pyceres.LossFunction | None = None
    ) -> None:
        """Add residual block to specific problem.
        
        Args:
            problem_idx: Index of problem
            cost: Cost function
            parameters: Parameter list
            loss: Optional loss function
        """
        optimizer = self.get_optimizer(problem_idx=problem_idx)
        optimizer.add_residual_block(
            cost=cost,
            parameters=parameters,
            loss=loss
        )
    
    def solve_all(self) -> list[OptimizationResult]:
        """Solve all problems.
        
        Returns:
            List of optimization results
        """
        logger.info(f"Solving {self.num_problems} problems...")
        
        results = []
        for i, optimizer in enumerate(self.optimizers):
            logger.info(f"\nProblem {i+1}/{self.num_problems}")
            result = optimizer.solve()
            results.append(result)
        
        logger.info(f"\n✓ Completed {self.num_problems} problems")
        
        return results
    
    def solve_all_parallel(self) -> list[OptimizationResult]:
        """Solve all problems in parallel.
        
        Uses multiprocessing to solve problems simultaneously.
        
        Returns:
            List of optimization results
        """
        import multiprocessing as mp
        
        logger.info(f"Solving {self.num_problems} problems in parallel...")
        
        # Create worker function
        def solve_worker(idx: int) -> OptimizationResult:
            return self.optimizers[idx].solve()
        
        # Solve in parallel
        with mp.Pool() as pool:
            results = pool.map(solve_worker, range(self.num_problems))
        
        logger.info(f"\n✓ Completed {self.num_problems} problems in parallel")
        
        return results
