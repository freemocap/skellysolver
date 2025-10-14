"""Base classes for all cost functions in SkellySolver.

This module provides abstract base classes that all cost functions inherit from.
Provides common functionality like numeric jacobian computation.
"""

import numpy as np
import pyceres
from abc import ABC, abstractmethod
from typing import Any


class BaseCostFunction(pyceres.CostFunction):
    """Abstract base class for all SkellySolver cost functions.
    
    Provides:
    - Automatic weight application
    - Numeric jacobian computation (override for analytic)
    - Consistent interface across all cost functions
    
    Subclasses must implement:
    - _compute_residual(): Compute the residual vector
    - Set num_residuals and parameter_block_sizes in __init__
    """
    
    def __init__(self, *, weight: float = 1.0) -> None:
        """Initialize base cost function.
        
        Args:
            weight: Weight for this cost term in the optimization
        """
        super().__init__()
        self.weight = weight
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute residual vector (unweighted).
        
        Must be implemented by subclasses.
        
        Args:
            parameters: List of parameter blocks
            
        Returns:
            Residual vector (will be weighted automatically)
        """
        pass
    
    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        """Evaluate cost function (pyceres interface).
        
        This is called by pyceres during optimization.
        Computes weighted residual and optionally jacobians.
        
        Args:
            parameters: List of parameter blocks
            residuals: Output residual vector
            jacobians: Optional list of jacobian matrices
            
        Returns:
            True if evaluation succeeded
        """
        # Compute residual (unweighted)
        residual_unweighted = self._compute_residual(parameters=parameters)
        
        # Apply weight
        residuals[:] = self.weight * residual_unweighted
        
        # Compute jacobians if requested
        if jacobians is not None:
            self._compute_jacobians(
                parameters=parameters,
                residuals=residuals,
                jacobians=jacobians
            )
        
        return True
    
    def _compute_jacobians(
        self,
        *,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray]
    ) -> None:
        """Compute jacobians (numeric by default, override for analytic).
        
        Args:
            parameters: List of parameter blocks
            residuals: Current residual vector
            jacobians: List of jacobian matrices to fill
        """
        # Default: numeric differentiation
        self._compute_jacobians_numeric(
            parameters=parameters,
            residuals=residuals,
            jacobians=jacobians,
            eps=1e-8
        )
    
    def _compute_jacobians_numeric(
        self,
        *,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray],
        eps: float = 1e-8
    ) -> None:
        """Compute jacobians using finite differences.
        
        This is a fallback for when analytic jacobians are not provided.
        Can be slow but is always correct.
        
        Args:
            parameters: List of parameter blocks
            residuals: Current residual vector
            jacobians: List of jacobian matrices to fill
            eps: Step size for finite differences
        """
        n_residuals = len(residuals)
        
        # For each parameter block
        for param_idx, param in enumerate(parameters):
            if jacobians[param_idx] is None:
                continue
            
            n_params = len(param)
            
            # For each parameter in this block
            for i in range(n_params):
                # Perturb parameter
                param_plus = param.copy()
                param_plus[i] += eps
                
                # Compute residual with perturbed parameter
                params_plus = parameters.copy()
                params_plus[param_idx] = param_plus
                residual_plus = self.weight * self._compute_residual(parameters=params_plus)
                
                # Finite difference
                for j in range(n_residuals):
                    jacobians[param_idx][j * n_params + i] = (residual_plus[j] - residuals[j]) / eps
