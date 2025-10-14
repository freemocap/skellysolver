"""Tests for core optimizer (Phase 1).

Tests the Optimizer wrapper and configuration classes.
"""

import numpy as np
import pytest

from skellysolver.core.config import OptimizationConfig, ParallelConfig, WeightConfig
from skellysolver.core.result import OptimizationResult
from skellysolver.core.optimizer import Optimizer
from skellysolver.core.cost_primatives import (
    RotationSmoothnessCost,
    TranslationSmoothnessCost,
)


class TestOptimizationConfig:
    """Test optimization configuration."""
    
    def test_create_default_config(self) -> None:
        """Should create config with defaults."""
        config = OptimizationConfig()
        
        assert config.max_iterations == 300
        assert config.use_robust_loss is True
        assert config.robust_loss_type == "huber"
    
    def test_create_custom_config(self) -> None:
        """Should create config with custom values."""
        config = OptimizationConfig(
            max_iterations=500,
            use_robust_loss=False,
            robust_loss_type="cauchy"
        )
        
        assert config.max_iterations == 500
        assert config.use_robust_loss is False
        assert config.robust_loss_type == "cauchy"
    
    def test_to_solver_options(self) -> None:
        """Should convert to pyceres SolverOptions."""
        config = OptimizationConfig(max_iterations=100)
        options = config.to_solver_options()
        
        import pyceres
        assert isinstance(options, pyceres.SolverOptions)
        assert options.max_num_iterations == 100
    
    def test_get_loss_function(self) -> None:
        """Should return appropriate loss function."""
        config = OptimizationConfig(
            use_robust_loss=True,
            robust_loss_type="huber",
            robust_loss_param=2.0
        )
        
        loss = config.get_loss_function()
        
        import pyceres
        assert isinstance(loss, pyceres.HuberLoss)


class TestParallelConfig:
    """Test parallel configuration."""
    
    def test_create_default_config(self) -> None:
        """Should create config with defaults."""
        config = ParallelConfig()
        
        assert config.enabled is True
        assert config.chunk_size == 500
        assert config.overlap_size == 50
    
    def test_should_use_parallel(self) -> None:
        """Should correctly determine if parallel is needed."""
        config = ParallelConfig(enabled=True, chunk_size=100, min_chunk_size=50)
        
        # Small dataset - no parallel
        assert not config.should_use_parallel(n_frames=120)
        
        # Large dataset - use parallel
        assert config.should_use_parallel(n_frames=200)
    
    def test_get_num_workers(self) -> None:
        """Should get number of workers."""
        config = ParallelConfig()
        n_workers = config.get_num_workers()
        
        assert n_workers >= 1


class TestOptimizer:
    """Test Optimizer wrapper."""
    
    def test_create_optimizer(self) -> None:
        """Should create optimizer instance."""
        config = OptimizationConfig(max_iterations=10)
        optimizer = Optimizer(config=config)
        
        assert optimizer.config == config
        assert optimizer.num_parameters() == 0
    
    def test_add_parameter_block(self) -> None:
        """Should add parameter block."""
        config = OptimizationConfig()
        optimizer = Optimizer(config=config)
        
        params = np.array([1.0, 2.0, 3.0])
        optimizer.add_parameter_block(name="test", parameters=params)
        
        assert optimizer.num_parameters() == 3
        assert "test" in optimizer.parameter_blocks
    
    def test_add_quaternion_parameter(self) -> None:
        """Should add quaternion with manifold."""
        config = OptimizationConfig()
        optimizer = Optimizer(config=config)
        
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        optimizer.add_quaternion_parameter(name="rotation", parameters=quat)
        
        assert optimizer.num_parameters() == 4
    
    def test_add_residual_block(self) -> None:
        """Should add residual block."""
        config = OptimizationConfig()
        optimizer = Optimizer(config=config)
        
        # Add parameters
        quat_1 = np.array([1.0, 0.0, 0.0, 0.0])
        quat_2 = np.array([0.9, 0.1, 0.0, 0.0])
        
        optimizer.add_quaternion_parameter(name="q1", parameters=quat_1)
        optimizer.add_quaternion_parameter(name="q2", parameters=quat_2)
        
        # Add cost
        cost = RotationSmoothnessCost(weight=100.0)
        optimizer.add_residual_block(cost=cost, parameters=[quat_1, quat_2])
        
        assert optimizer.num_residuals() >= 1
    
    def test_simple_optimization(self) -> None:
        """Should solve simple optimization problem."""
        config = OptimizationConfig(max_iterations=50)
        optimizer = Optimizer(config=config)
        
        # Create parameters that should converge
        quat_1 = np.array([1.0, 0.0, 0.0, 0.0])
        quat_2 = np.array([0.8, 0.2, 0.0, 0.0])
        quat_2 = quat_2 / np.linalg.norm(quat_2)
        
        optimizer.add_quaternion_parameter(name="q1", parameters=quat_1)
        optimizer.add_quaternion_parameter(name="q2", parameters=quat_2)
        
        # Add smoothness cost
        cost = RotationSmoothnessCost(weight=100.0)
        optimizer.add_residual_block(cost=cost, parameters=[quat_1, quat_2])
        
        # Solve
        result = optimizer.solve()
        
        assert isinstance(result, OptimizationResult)
        assert result.final_cost <= result.initial_cost
    
    def test_set_parameter_bounds(self) -> None:
        """Should set parameter bounds."""
        config = OptimizationConfig()
        optimizer = Optimizer(config=config)
        
        params = np.array([1.0])
        optimizer.add_parameter_block(name="scale", parameters=params)
        
        # Set bounds
        optimizer.set_parameter_bounds(
            parameters=params,
            index=0,
            lower=0.5,
            upper=2.0
        )
        
        # Should not raise
        assert True


class TestOptimizationResult:
    """Test optimization result."""
    
    def test_create_result(self) -> None:
        """Should create result."""
        result = OptimizationResult(
            success=True,
            num_iterations=100,
            initial_cost=10.0,
            final_cost=1.0,
            solve_time_seconds=5.0
        )
        
        assert result.success
        assert result.num_iterations == 100
    
    def test_cost_reduction(self) -> None:
        """Should compute cost reduction."""
        result = OptimizationResult(
            success=True,
            num_iterations=100,
            initial_cost=10.0,
            final_cost=5.0,
            solve_time_seconds=5.0
        )
        
        assert result.cost_reduction == 0.5
        assert result.cost_reduction_percent == 50.0
    
    def test_summary(self) -> None:
        """Should generate summary string."""
        result = OptimizationResult(
            success=True,
            num_iterations=100,
            initial_cost=10.0,
            final_cost=1.0,
            solve_time_seconds=5.0
        )
        
        summary = result.summary()
        
        assert "OPTIMIZATION RESULT" in summary
        assert "Converged" in summary
        assert "100" in summary


class TestWeightConfig:
    """Test weight configuration."""
    
    def test_create_default_weights(self) -> None:
        """Should create default weights."""
        weights = WeightConfig()
        
        assert weights.lambda_data == 100.0
        assert weights.lambda_rigid == 500.0
    
    def test_scale_all_weights(self) -> None:
        """Should scale all weights."""
        weights = WeightConfig(
            lambda_data=100.0,
            lambda_rigid=500.0
        )
        
        weights.scale_all(factor=2.0)
        
        assert weights.lambda_data == 200.0
        assert weights.lambda_rigid == 1000.0
