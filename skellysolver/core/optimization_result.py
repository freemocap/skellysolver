"""Unified optimization result for all SkellySolver pipelines.

This module provides result classes that store optimization outputs.
Used by ALL pipelines to maintain consistent result structure.

Replaces:
- rigid_body_optimization.py::OptimizationResult
- eye_pyceres_bundle_adjustment.py::OptimizationResult
"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pyceres
from pydantic import model_validator, Field
from typing_extensions import Self

from skellysolver.data.arbitrary_types_model import ABaseModel


class OptimizationResult(ABaseModel, ABC):
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
    

    # Additional data
    metadata: dict[str, Any] = Field(default_factory=dict)
    
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
                f"  Iterations completed: {self.num_iterations}\n"
                f"  Initial cost:         {self.initial_cost:.6f}\n"
                f"  Final cost:           {self.final_cost:.6f}\n"
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
            raise RuntimeError(error_msg)
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

    @abstractmethod
    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Multi-line summary string
        """
        pass

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
        """Validate rigid body result - CRASHES if required fields missing.

        Raises:
            ValueError: If any required field is missing, with detailed explanation
        """
        # Build detailed error message
        missing_fields = []
        if self.reconstructed is None:
            missing_fields.append("reconstructed: (n_frames, n_markers, 3) 3D marker positions")
        if self.rotations is None:
            missing_fields.append("rotations: (n_frames, 3, 3) rotation matrices for each frame")
        if self.translations is None:
            missing_fields.append("translations: (n_frames, 3) translation vectors for each frame")
        if self.reference_geometry is None:
            missing_fields.append("reference_geometry: (n_markers, 3) rigid body reference shape")

        if missing_fields:
            error_msg = (
                f"❌ INVALID RigidBodyResult - Missing Required Fields!\n"
                f"\n"
                f"The following required fields are None:\n" +
                "\n".join(f"  • {field}" for field in missing_fields) +
                f"\n\n"
                f"This is a BUG in the optimization code!\n"
                f"RigidBodyResult objects MUST have all these fields populated.\n"
                f"\n"
                f"Likely causes:\n"
                f"  1. Optimization function did not return all required outputs\n"
                f"  2. Result assembly code has a bug\n"
                f"  3. Data conversion/extraction failed silently\n"
                f"\n"
                f"Debug steps:\n"
                f"  1. Check the optimization function return value\n"
                f"  2. Verify result assembly in the pipeline code\n"
                f"  3. Add logging before result creation to see what's None"
            )
            raise ValueError(error_msg)

        return self

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.reconstructed.shape[0]

    @property
    def n_markers(self) -> int:
        """Number of markers."""
        return self.reconstructed.shape[1]
    def summary(self) -> str:
        """Generate summary of rigid body optimization.

        Returns:
            Multi-line summary string
        """
        lines = [
            "="*80,
            "RIGID BODY OPTIMIZATION RESULT",
            "="*80,
            f"Success:         {'YES' if self.success else 'NO'}",
            f"Iterations:      {self.num_iterations}",
            f"Initial cost:    {self.initial_cost:.6f}",
            f"Final cost:      {self.final_cost:.6f}",
            f"Cost reduction:  {self.cost_reduction_percent:.1f}%",
            f"Solve time:      {self.solve_time_seconds:.2f}s",
            f"Frames:          {self.n_frames}",
            f"Markers:         {self.n_markers}",
            "="*80,
        ]
        return "\n".join(lines)


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
        """Validate eye tracking result - CRASHES if required fields missing.

        Raises:
            ValueError: If any required field is missing, with detailed explanation
        """
        # Build detailed error message
        missing_fields = []
        if self.rotations is None:
            missing_fields.append("rotations: (n_frames, 3, 3) or (n_frames, 4) - eye orientations as matrices or quaternions")
        if self.gaze_directions is None:
            missing_fields.append("gaze_directions: (n_frames, 3) - 3D gaze direction vectors")
        if self.pupil_scales is None:
            missing_fields.append("pupil_scales: (n_frames,) - pupil dilation scale factors")

        if missing_fields:
            error_msg = (
                f"❌ INVALID EyeTrackingResult - Missing Required Fields!\n"
                f"\n"
                f"The following required fields are None:\n" +
                "\n".join(f"  • {field}" for field in missing_fields) +
                f"\n\n"
                f"This is a BUG in the eye tracking optimization code!\n"
                f"EyeTrackingResult objects MUST have all these fields populated.\n"
                f"\n"
                f"Likely causes:\n"
                f"  1. Eye tracking optimizer did not return all required outputs\n"
                f"  2. Result extraction from optimization variables failed\n"
                f"  3. Data conversion from pyceres parameters failed silently\n"
                f"\n"
                f"Debug steps:\n"
                f"  1. Check the eye tracking optimization function return value\n"
                f"  2. Verify parameter extraction after optimization\n"
                f"  3. Add logging to see which fields are None and why\n"
                f"  4. Check if optimization actually ran (success flag)"
            )
            raise ValueError(error_msg)

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


class ChunkedResult(ABaseModel):
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