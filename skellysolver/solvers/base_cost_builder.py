"""Base class for cost builders.

CostBuilders convert high-level constraints into low-level cost functions.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from skellysolver.utilities.arbitrary_types_model import ABaseModel
from skellysolver.solvers.costs.constraint_costs import CostCollection
from skellysolver.solvers.constraints.base_constraint import BaseConstraint

# Generic type for the constraint being built from
TConstraint = TypeVar('TConstraint', bound=BaseConstraint)


class BaseCostBuilder(ABaseModel, ABC, Generic[TConstraint]):
    """Base class for building costs from constraints.
    
    Flow:
        Constraint (structure definition)
          ↓
        CostBuilder (converts structure to costs)
          ↓  
        CostInfo (metadata + cost function + parameters)
          ↓
        BaseCostFunction (actual pyceres math)
    
    Subclasses should implement methods to build specific cost types
    from their constraint type.
    
    Example:
        class SkeletonCostBuilder(BaseCostBuilder[Skeleton]):
            constraint: Skeleton
            
            def build_segment_costs(...) -> list[CostInfo]:
                # Build costs from skeleton.segments
                ...
    """
    
    constraint: TConstraint
    
    @abstractmethod
    def build_all_costs(
        self,
        *,
        reference_geometry: np.ndarray,
        **kwargs: object
    ) -> CostCollection:
        """Build all costs from the constraint.
        
        This is the main entry point for cost generation.
        Subclasses should implement this to generate all relevant
        cost functions from their constraint type.
        
        Args:
            reference_geometry: Reference configuration
            **kwargs: Additional builder-specific parameters
            
        Returns:
            CostCollection with all generated costs
        """
        pass
