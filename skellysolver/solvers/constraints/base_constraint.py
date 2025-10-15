"""Base class for all constraint models.

Constraints define the structure and rules of the system being optimized.
They are converted to cost functions by CostBuilders.
"""

from pydantic import BaseModel


class BaseConstraint(BaseModel):
    """Base class for all constraint definitions.
    
    Constraints define structural relationships that should be maintained
    during optimization. They are declarative - they describe what should
    be true, not how to enforce it.
    
    CostBuilders convert constraints into CostInfo objects containing
    the actual pyceres cost functions that enforce these constraints.
    
    Examples of constraints:
    - Keypoint: A point that should exist in the model
    - Segment: A rigid connection between keypoints  
    - Linkage: A semi-rigid connection between segments
    - Chain: A sequence of linked segments
    - Skeleton: A complete hierarchical constraint structure
    """
    
    name: str
