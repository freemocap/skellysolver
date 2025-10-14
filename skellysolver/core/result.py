"""Unified optimization result for all SkellySolver pipelines.

This module provides result classes that store optimization outputs.
Used by ALL pipelines to maintain consistent result structure.

Replaces:
- rigid_body_optimization.py::OptimizationResult
- eye_pyceres_bundle_adjustment.py::OptimizationResult
"""

import numpy as np
import pyceres
from dataclasses import dataclass, field
from typing import Any

from pydantic import model_validator, Field
from typing_extensions import Self

from skellysolver.data.arbitrary_types_model import ArbitraryTypesModel


class OptimizationResult(ArbitraryTypesModel):
    """Results from pyceres optimization.
    
    Core results that are always present:
    - success: Whether optimization converged
    - num_iterations: Number of iterations performed
    - initial_cost: Initial cost function value
    - final_cost: Final cost function value
    - solve_time_seconds: Time spent in solver
    
    Optional results (depending on pipeline):
    - reconstructed: Reconstructed 3D positions
    - rotations: Rotation matrices or quaternions
    - translations: Translation vectors
    - reference_geometry: Reference configuration
    - gaze_directions: Gaze direction vectors (eye tracking)
    - pupil_scales: Pupil dilation scales (eye tracking)
    - metadata: Additional pipeline-specific data
    """
    
    # Core results (always present)
    success: bool
    num_iterations: int
    initial_cost: float
    final_cost: float
    solve_time_seconds: float
    
    # Optional results (depending on pipeline)
    reconstructed: np.ndarray | None = None
    rotations: np.ndarray | None = None
    translations: np.ndarray | None = None
    reference_geometry: np.ndarray | None = None
    
    # Eye tracking specific
    gaze_directions: np.ndarray | None = None
    pupil_scales: np.ndarray | None = None
    
    # Additional data
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode="after")
    def validate(self) -> Self:
        """Validate and log result."""
        if not self.success:
            print(f"⚠ Warning: Optimization did not converge")
            print(f"  Iterations: {self.num_iterations}")
            print(f"  Final cost: {self.final_cost:.6f}")
        return self

    @property
    def cost_reduction(self) -> float:
        """Compute relative cost reduction.
        
        Returns:
            Fraction of cost reduced (0-1)
        """
        if self.initial_cost == 0.0:
            return 0.0
        return (self.initial_cost - self.final_cost) / self.initial_cost
    
    @property
    def cost_reduction_percent(self) -> float:
        """Compute cost reduction percentage.
        
        Returns:
            Percentage of cost reduced (0-100)
        """
        return self.cost_reduction * 100.0
    
    def summary(self) -> str:
        """Generate human-readable summary.
        
        Returns:
            Multi-line summary string
        """
        lines = [
            "="*80,
            "OPTIMIZATION RESULT",
            "="*80,
            f"Status:       {'✓ Converged' if self.success else '✗ Did not converge'}",
            f"Iterations:   {self.num_iterations}",
            f"Solve time:   {self.solve_time_seconds:.2f}s",
            f"Initial cost: {self.initial_cost:.6f}",
            f"Final cost:   {self.final_cost:.6f}",
            f"Reduction:    {self.cost_reduction_percent:.1f}%",
            "="*80,
        ]
        
        # Add optional result info
        if self.reconstructed is not None:
            n_frames = self.reconstructed.shape[0]
            n_markers = self.reconstructed.shape[1]
            lines.append(f"Reconstructed: {n_frames} frames × {n_markers} markers")
        
        if self.rotations is not None:
            n_frames = self.rotations.shape[0]
            lines.append(f"Rotations:     {n_frames} frames")
        
        if self.translations is not None:
            n_frames = self.translations.shape[0]
            lines.append(f"Translations:  {n_frames} frames")
        
        if self.reference_geometry is not None:
            n_markers = self.reference_geometry.shape[0]
            lines.append(f"Reference:     {n_markers} markers")
        
        if self.gaze_directions is not None:
            n_frames = self.gaze_directions.shape[0]
            lines.append(f"Gaze:          {n_frames} frames")
        
        if self.pupil_scales is not None:
            n_frames = len(self.pupil_scales)
            mean_scale = np.mean(self.pupil_scales)
            std_scale = np.std(self.pupil_scales)
            lines.append(f"Pupil scale:   {mean_scale:.3f} ± {std_scale:.3f}")
        
        if self.metadata:
            lines.append(f"Metadata:      {len(self.metadata)} entries")
        
        return "\n".join(lines)
    
    @classmethod
    def from_pyceres_summary(
        cls,
        *,
        summary: pyceres.SolverSummary,
        solve_time_seconds: float
    ) -> "OptimizationResult":
        """Create result from pyceres SolverSummary.
        
        Args:
            summary: pyceres solver summary
            solve_time_seconds: Measured solve time
            
        Returns:
            OptimizationResult with core fields filled
        """
        success = (
            summary.termination_type == pyceres.TerminationType.CONVERGENCE or
            summary.termination_type == pyceres.TerminationType.USER_SUCCESS
        )
        
        return cls(
            success=success,
            num_iterations=summary.num_successful_steps + summary.num_unsuccessful_steps,
            initial_cost=summary.initial_cost,
            final_cost=summary.final_cost,
            solve_time_seconds=solve_time_seconds
        )



class RigidBodyResult(OptimizationResult):
    """Specialized result for rigid body tracking.
    
    Extends OptimizationResult with rigid body specific fields.
    All fields from OptimizationResult are inherited.
    """
    
    # Override to make required
    reconstructed: np.ndarray = None
    rotations: np.ndarray = None
    translations: np.ndarray = None
    reference_geometry: np.ndarray = None
    
    @model_validator(mode="after")
    def validate(self) -> Self:
        """Validate rigid body result."""

        
        # Check required fields
        if self.reconstructed is None:
            raise ValueError("RigidBodyResult requires reconstructed positions")
        if self.rotations is None:
            raise ValueError("RigidBodyResult requires rotations")
        if self.translations is None:
            raise ValueError("RigidBodyResult requires translations")
        if self.reference_geometry is None:
            raise ValueError("RigidBodyResult requires reference_geometry")
        return self
    
    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.reconstructed.shape[0]
    
    @property
    def n_markers(self) -> int:
        """Number of markers."""
        return self.reconstructed.shape[1]


@dataclass
class EyeTrackingResult(OptimizationResult):
    """Specialized result for eye tracking.
    
    Extends OptimizationResult with eye tracking specific fields.
    All fields from OptimizationResult are inherited.
    """
    
    # Override to make required
    rotations: np.ndarray = None
    gaze_directions: np.ndarray = None
    pupil_scales: np.ndarray = None
    
    # Eye tracking specific
    pupil_centers_3d: np.ndarray | None = None
    tear_ducts_3d: np.ndarray | None = None
    projected_pupil_centers: np.ndarray | None = None
    projected_tear_ducts: np.ndarray | None = None
    pupil_errors: np.ndarray | None = None
    tear_duct_errors: np.ndarray | None = None

    @model_validator(mode="after")
    def validate(self) -> Self:


        # Check required fields
        if self.rotations is None:
            raise ValueError("EyeTrackingResult requires rotations (eye orientations)")
        if self.gaze_directions is None:
            raise ValueError("EyeTrackingResult requires gaze_directions")
        if self.pupil_scales is None:
            raise ValueError("EyeTrackingResult requires pupil_scales")
        return self
    
    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.gaze_directions)
    
    @property
    def mean_pupil_error(self) -> float:
        """Mean pupil reprojection error in pixels."""
        if self.pupil_errors is None:
            return 0.0
        return float(np.mean(self.pupil_errors))
    
    @property
    def mean_tear_duct_error(self) -> float:
        """Mean tear duct reprojection error in pixels."""
        if self.tear_duct_errors is None:
            return 0.0
        return float(np.mean(self.tear_duct_errors))



class ChunkedResult(ArbitraryTypesModel):
    """Result from chunked parallel optimization.
    
    Contains results from individual chunks plus stitched result.
    """
    
    # Individual chunk results
    chunk_results: list[OptimizationResult]
    
    # Stitched result
    stitched_result: OptimizationResult
    
    # Timing info
    total_time_seconds: float
    chunk_times_seconds: list[float]
    
    @property
    def n_chunks(self) -> int:
        """Number of chunks."""
        return len(self.chunk_results)
    
    @property
    def average_chunk_time(self) -> float:
        """Average time per chunk in seconds."""
        return float(np.mean(self.chunk_times_seconds))
    
    @property
    def speedup(self) -> float:
        """Speedup from parallelization.
        
        Returns:
            Speedup factor (>1 = faster with parallel)
        """
        sequential_time = sum(self.chunk_times_seconds)
        if self.total_time_seconds == 0.0:
            return 1.0
        return sequential_time / self.total_time_seconds
    
    def summary(self) -> str:
        """Generate summary of chunked optimization.
        
        Returns:
            Multi-line summary string
        """
        lines = [
            "="*80,
            "CHUNKED OPTIMIZATION RESULT",
            "="*80,
            f"Chunks:          {self.n_chunks}",
            f"Total time:      {self.total_time_seconds:.2f}s",
            f"Avg chunk time:  {self.average_chunk_time:.2f}s",
            f"Speedup:         {self.speedup:.1f}x",
            "="*80,
        ]
        
        # Add stitched result summary
        lines.append("\nStitched Result:")
        lines.append(self.stitched_result.summary())
        
        return "\n".join(lines)
