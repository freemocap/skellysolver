# SkellySolver Refactoring Implementation Guide

## Phase 1: Core Consolidation

### Step 1.1: Create Unified Cost Functions

#### File: `skellysolver/core/cost_functions/base.py`

```python
"""Base classes for all cost functions."""
from abc import ABC, abstractmethod
import pyceres
import numpy as np

class BaseCostFunction(pyceres.CostFunction, ABC):
    """Base class for all SkellySolver cost functions."""
    
    def __init__(self, *, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
    
    @abstractmethod
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        """Compute residual - must be implemented by subclasses."""
        pass
    
    def Evaluate(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray] | None
    ) -> bool:
        """Standard evaluate method with automatic jacobian computation."""
        residual = self._compute_residual(parameters)
        residuals[:] = self.weight * residual
        
        if jacobians is not None:
            self._compute_jacobians_numeric(parameters, residuals, jacobians)
        
        return True
    
    def _compute_jacobians_numeric(
        self,
        parameters: list[np.ndarray],
        residuals: np.ndarray,
        jacobians: list[np.ndarray],
        eps: float = 1e-8
    ) -> None:
        """Numeric jacobian computation (override for analytic)."""
        # Standard finite difference implementation
        pass
```

#### File: `skellysolver/core/cost_functions/smoothness.py`

```python
"""Temporal smoothness cost functions - used by BOTH pipelines."""
import numpy as np
from scipy.spatial.transform import Rotation
import pyceres
from .base import BaseCostFunction

class RotationSmoothnessCost(BaseCostFunction):
    """Temporal smoothness for rotation (quaternions).
    
    Used by:
    - Rigid body tracking (smooth head rotation)
    - Eye tracking (smooth gaze changes)
    """
    
    def __init__(self, *, weight: float = 100.0) -> None:
        super().__init__(weight=weight)
        self.set_num_residuals(4)
        self.set_parameter_block_sizes([4, 4])  # Two quaternions
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        quat_t = parameters[0]
        quat_t1 = parameters[1]
        
        # Ensure same hemisphere
        if np.dot(quat_t, quat_t1) < 0:
            quat_t1 = -quat_t1
        
        return quat_t1 - quat_t


class TranslationSmoothnessCost(BaseCostFunction):
    """Temporal smoothness for translation.
    
    Used by:
    - Rigid body tracking (smooth position changes)
    - Eye tracking (smooth eyeball position - if optimizing)
    """
    
    def __init__(self, *, weight: float = 100.0) -> None:
        super().__init__(weight=weight)
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 3])  # Two translations
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        trans_t = parameters[0]
        trans_t1 = parameters[1]
        return trans_t1 - trans_t


class ScalarSmoothnessCost(BaseCostFunction):
    """Temporal smoothness for scalar parameters.
    
    Used by:
    - Eye tracking (pupil dilation)
    - Any scalar time series
    """
    
    def __init__(self, *, weight: float = 10.0) -> None:
        super().__init__(weight=weight)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1])
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        scalar_t = parameters[0][0]
        scalar_t1 = parameters[1][0]
        return np.array([scalar_t1 - scalar_t])
```

#### File: `skellysolver/core/cost_functions/measurement.py`

```python
"""Data fitting cost functions."""
import numpy as np
from scipy.spatial.transform import Rotation
from .base import BaseCostFunction

class Point3DMeasurementCost(BaseCostFunction):
    """Fit measured 3D point to transformed reference point.
    
    Used by:
    - Rigid body tracking (fit markers to rigid body)
    """
    
    def __init__(
        self,
        *,
        measured_point: np.ndarray,
        reference_point: np.ndarray,
        weight: float = 100.0
    ) -> None:
        super().__init__(weight=weight)
        self.measured = measured_point.copy()
        self.reference = reference_point.copy()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 3])  # quat, translation
    
    def _compute_residual(
        self,
        parameters: list[np.ndarray]
    ) -> np.ndarray:
        quat = parameters[0]
        translation = parameters[1]
        
        R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        predicted = R @ self.reference + translation
        
        return self.measured - predicted


class ProjectionMeasurementCost(BaseCostFunction):
    """Fit observed 2D point to projection of 3D point.
    
    Used by:
    - Eye tracking (fit pupil points to projected eyeball model)
    - Camera calibration
    """
    
    def __init__(
        self,
        *,
        observed_px: np.ndarray,
        camera_matrix: np.ndarray,
        weight: float = 1.0
    ) -> None:
        super().__init__(weight=weight)
        self.observed = observed_px.copy()
        self.camera_matrix = camera_matrix
        self.set_num_residuals(2)
        # Parameter blocks defined by subclass
    
    def _project_point(
        self,
        point_3d: np.ndarray
    ) -> np.ndarray:
        """Project 3D point to 2D using camera matrix."""
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        x_norm = point_3d[0] / point_3d[2]
        y_norm = point_3d[1] / point_3d[2]
        
        u = fx * x_norm + cx
        v = fy * y_norm + cy
        
        return np.array([u, v])
```

### Step 1.2: Create Unified Configuration

#### File: `skellysolver/core/config.py`

```python
"""Unified optimization configuration."""
from dataclasses import dataclass
from typing import Literal

@dataclass
class OptimizationConfig:
    """Configuration for pyceres optimization.
    
    Used by ALL pipelines - rigid body, eye tracking, future pipelines.
    """
    
    # Solver parameters
    max_iterations: int = 300
    function_tolerance: float = 1e-9
    gradient_tolerance: float = 1e-11
    parameter_tolerance: float = 1e-10
    
    # Loss function
    use_robust_loss: bool = True
    robust_loss_type: Literal["huber", "cauchy", "soft_l1"] = "huber"
    robust_loss_param: float = 2.0
    
    # Linear solver
    linear_solver: Literal["dense_qr", "sparse_normal_cholesky", "sparse_schur"] = "sparse_normal_cholesky"
    
    # Trust region
    trust_region_strategy: Literal["levenberg_marquardt", "dogleg"] = "levenberg_marquardt"
    
    # Parallelization
    num_threads: int | None = None  # None = auto-detect
    
    # Logging
    minimizer_progress_to_stdout: bool = True
    
    def to_solver_options(self) -> "pyceres.SolverOptions":
        """Convert to pyceres SolverOptions."""
        import pyceres
        import os
        
        options = pyceres.SolverOptions()
        options.max_num_iterations = self.max_iterations
        options.function_tolerance = self.function_tolerance
        options.gradient_tolerance = self.gradient_tolerance
        options.parameter_tolerance = self.parameter_tolerance
        options.minimizer_progress_to_stdout = self.minimizer_progress_to_stdout
        
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
        
        # Threads
        if self.num_threads is None:
            options.num_threads = max(os.cpu_count() - 1 if os.cpu_count() else 1, 1)
        else:
            options.num_threads = self.num_threads
        
        return options


@dataclass
class ParallelConfig:
    """Configuration for parallel chunked optimization."""
    
    enabled: bool = True
    chunk_size: int = 500
    overlap_size: int = 50
    blend_window: int = 25
    min_chunk_size: int = 100
    num_workers: int | None = None  # None = auto-detect
```

#### File: `skellysolver/core/result.py`

```python
"""Unified optimization result."""
from dataclasses import dataclass
import numpy as np
from typing import Any

@dataclass
class OptimizationResult:
    """Results from optimization - used by ALL pipelines.
    
    Domain-specific pipelines can extend this with additional fields.
    """
    
    # Core results (always present)
    success: bool
    num_iterations: int
    initial_cost: float
    final_cost: float
    solve_time_seconds: float
    
    # Optional results (depending on pipeline)
    reconstructed: np.ndarray | None = None  # (n_frames, n_markers, 3) for rigid body
    rotations: np.ndarray | None = None      # (n_frames, 3, 3) or quaternions
    translations: np.ndarray | None = None   # (n_frames, 3)
    
    # For rigid body
    reference_geometry: np.ndarray | None = None  # (n_markers, 3)
    
    # For eye tracking
    gaze_directions: np.ndarray | None = None     # (n_frames, 3)
    pupil_scales: np.ndarray | None = None        # (n_frames,)
    
    # Additional data
    metadata: dict[str, Any] | None = None
    
    def __post_init__(self) -> None:
        """Validate result."""
        if not self.success:
            print(f"Warning: Optimization did not converge after {self.num_iterations} iterations")
```

### Step 1.3: Create Generic Optimizer Wrapper

#### File: `skellysolver/core/optimizer.py`

```python
"""Generic optimizer wrapper for pyceres."""
import numpy as np
import pyceres
import time
import logging
from typing import Callable, Any
from .config import OptimizationConfig
from .result import OptimizationResult

logger = logging.getLogger(__name__)

class Optimizer:
    """Generic pyceres optimizer wrapper.
    
    Handles:
    - Problem setup
    - Cost function addition
    - Constraint application
    - Solving
    - Result extraction
    
    Used by ALL pipelines.
    """
    
    def __init__(self, *, config: OptimizationConfig) -> None:
        self.config = config
        self.problem = pyceres.Problem()
        self.parameter_blocks: dict[str, np.ndarray] = {}
    
    def add_parameter_block(
        self,
        *,
        name: str,
        parameters: np.ndarray,
        manifold: pyceres.Manifold | None = None
    ) -> None:
        """Add parameter block to optimization."""
        self.parameter_blocks[name] = parameters
        self.problem.add_parameter_block(parameters, len(parameters))
        
        if manifold is not None:
            self.problem.set_manifold(parameters, manifold)
    
    def add_residual_block(
        self,
        *,
        cost: pyceres.CostFunction,
        parameters: list[np.ndarray],
        loss: pyceres.LossFunction | None = None
    ) -> None:
        """Add residual block to optimization."""
        self.problem.add_residual_block(cost, loss, parameters)
    
    def set_bounds(
        self,
        *,
        parameters: np.ndarray,
        index: int,
        lower: float | None = None,
        upper: float | None = None
    ) -> None:
        """Set parameter bounds."""
        if lower is not None:
            self.problem.set_parameter_lower_bound(parameters, index, lower)
        if upper is not None:
            self.problem.set_parameter_upper_bound(parameters, index, upper)
    
    def solve(self) -> OptimizationResult:
        """Solve optimization problem."""
        logger.info(f"Starting optimization with {self.problem.num_residual_blocks()} residual blocks")
        
        options = self.config.to_solver_options()
        summary = pyceres.SolverSummary()
        
        start_time = time.time()
        pyceres.solve(options, self.problem, summary)
        solve_time = time.time() - start_time
        
        success = (
            summary.termination_type == pyceres.TerminationType.CONVERGENCE or
            summary.termination_type == pyceres.TerminationType.USER_SUCCESS
        )
        
        logger.info(f"Optimization {'succeeded' if success else 'failed'} after {solve_time:.2f}s")
        logger.info(f"  Initial cost: {summary.initial_cost:.6f}")
        logger.info(f"  Final cost: {summary.final_cost:.6f}")
        logger.info(f"  Iterations: {summary.num_successful_steps + summary.num_unsuccessful_steps}")
        
        return OptimizationResult(
            success=success,
            num_iterations=summary.num_successful_steps + summary.num_unsuccessful_steps,
            initial_cost=summary.initial_cost,
            final_cost=summary.final_cost,
            solve_time_seconds=solve_time
        )
```

## Phase 2: Data Layer Consolidation

### Step 2.1: Create Unified Data Structures

#### File: `skellysolver/data/base.py`

```python
"""Base data structures used across all pipelines."""
from dataclasses import dataclass
import numpy as np
from typing import Any

@dataclass
class Trajectory3D:
    """3D trajectory data - used by rigid body tracking."""
    
    marker_name: str
    positions: np.ndarray  # (n_frames, 3)
    confidence: np.ndarray | None = None  # (n_frames,) if available
    
    @property
    def n_frames(self) -> int:
        return len(self.positions)
    
    def is_valid(self, *, min_confidence: float = 0.3) -> np.ndarray:
        """Return mask of valid frames."""
        if self.confidence is None:
            return ~np.isnan(self.positions[:, 0])
        return self.confidence >= min_confidence


@dataclass
class Observation2D:
    """2D observation data - used by eye tracking."""
    
    point_name: str
    positions: np.ndarray  # (n_frames, 2)
    confidence: np.ndarray | None = None  # (n_frames,) if available
    
    @property
    def n_frames(self) -> int:
        return len(self.positions)
    
    def is_valid(self, *, min_confidence: float = 0.3) -> np.ndarray:
        """Return mask of valid frames."""
        if self.confidence is None:
            return ~np.isnan(self.positions[:, 0])
        return self.confidence >= min_confidence


@dataclass
class TrajectoryDataset:
    """Collection of trajectories with metadata."""
    
    trajectories: dict[str, Trajectory3D | Observation2D]
    frame_indices: np.ndarray
    metadata: dict[str, Any] | None = None
    
    @property
    def n_frames(self) -> int:
        return len(self.frame_indices)
    
    @property
    def marker_names(self) -> list[str]:
        return list(self.trajectories.keys())
    
    def to_array(self, *, marker_names: list[str] | None = None) -> np.ndarray:
        """Convert to numpy array (n_frames, n_markers, 3 or 2)."""
        if marker_names is None:
            marker_names = self.marker_names
        
        first_traj = self.trajectories[marker_names[0]]
        n_dims = first_traj.positions.shape[1]
        
        array = np.stack(
            [self.trajectories[name].positions for name in marker_names],
            axis=1
        )
        
        return array
```

### Step 2.2: Consolidate Loaders

#### File: `skellysolver/data/formats.py`

```python
"""CSV format detection - consolidates from multiple files."""
from pathlib import Path
from typing import Literal

CSVFormat = Literal["tidy", "wide", "dlc"]

def detect_csv_format(*, filepath: Path) -> CSVFormat:
    """Auto-detect CSV format.
    
    Replaces:
    - loaders.py::detect_csv_format
    - load_trajectories.py::detect_csv_format
    """
    with open(filepath, 'r') as f:
        lines = [f.readline().strip() for _ in range(3)]
    
    if len(lines) < 1:
        raise ValueError(f"CSV file is empty: {filepath}")
    
    # Check for DLC format (3-row header)
    if len(lines) >= 3:
        row3_values = lines[2].split(',')
        if any(val.strip() in ['x', 'y', 'likelihood', 'coords'] for val in row3_values):
            return 'dlc'
    
    # Check for tidy vs wide
    import csv
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        if headers is None:
            raise ValueError("CSV has no headers")
        
        # Tidy format
        if 'keypoint' in headers and 'x' in headers and 'y' in headers:
            return 'tidy'
        
        # Wide format
        if any(h.endswith('_x') for h in headers) and any(h.endswith('_y') for h in headers):
            return 'wide'
    
    raise ValueError("Unknown CSV format")
```

## Phase 3: Pipeline Framework

### Step 3.1: Create Base Pipeline

#### File: `skellysolver/pipelines/base.py`

```python
"""Base pipeline class - ALL pipelines inherit from this."""
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Any

from ..core.config import OptimizationConfig, ParallelConfig
from ..core.result import OptimizationResult
from ..data.base import TrajectoryDataset

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Base configuration for all pipelines."""
    
    input_path: Path
    output_dir: Path
    optimization: OptimizationConfig
    parallel: ParallelConfig | None = None
    
    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


class BasePipeline(ABC):
    """Abstract base class for all pipelines.
    
    Defines standard pipeline interface:
    1. Load data
    2. Preprocess/validate
    3. Optimize
    4. Evaluate
    5. Save results
    6. Generate visualizations
    """
    
    def __init__(self, *, config: PipelineConfig) -> None:
        self.config = config
        self.data: TrajectoryDataset | None = None
        self.result: OptimizationResult | None = None
    
    def run(self) -> OptimizationResult:
        """Run complete pipeline."""
        logger.info(f"="*80)
        logger.info(f"{self.__class__.__name__} PIPELINE")
        logger.info(f"="*80)
        
        # Standard pipeline steps
        self.data = self.load_data()
        self.data = self.preprocess_data(self.data)
        self.result = self.optimize(self.data)
        metrics = self.evaluate(self.result)
        self.save_results(self.result, metrics)
        self.generate_viewer(self.result)
        
        logger.info(f"\n{'='*80}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'='*80}")
        
        return self.result
    
    @abstractmethod
    def load_data(self) -> TrajectoryDataset:
        """Load data from input file."""
        pass
    
    @abstractmethod
    def preprocess_data(self, *, data: TrajectoryDataset) -> TrajectoryDataset:
        """Preprocess and validate data."""
        pass
    
    @abstractmethod
    def optimize(self, *, data: TrajectoryDataset) -> OptimizationResult:
        """Run optimization."""
        pass
    
    @abstractmethod
    def evaluate(self, *, result: OptimizationResult) -> dict[str, Any]:
        """Evaluate results and compute metrics."""
        pass
    
    @abstractmethod
    def save_results(
        self,
        *,
        result: OptimizationResult,
        metrics: dict[str, Any]
    ) -> None:
        """Save results to disk."""
        pass
    
    @abstractmethod
    def generate_viewer(self, *, result: OptimizationResult) -> None:
        """Generate interactive HTML viewer."""
        pass
```

### Step 3.2: Refactor Rigid Body Pipeline

#### File: `skellysolver/pipelines/rigid_body/pipeline.py`

```python
"""Rigid body tracking pipeline - inherits from BasePipeline."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from ..base import BasePipeline, PipelineConfig
from ...core.topology import RigidBodyTopology
from ...core.result import OptimizationResult
from ...data.base import TrajectoryDataset
from ...data.loaders import load_trajectories
from .optimizer import RigidBodyOptimizer

@dataclass
class RigidBodyConfig(PipelineConfig):
    """Configuration for rigid body tracking."""
    
    topology: RigidBodyTopology
    soft_edges: list[tuple[int, int]] | None = None
    lambda_soft: float = 10.0


class RigidBodyPipeline(BasePipeline):
    """Rigid body tracking pipeline.
    
    Usage:
        config = RigidBodyConfig(...)
        pipeline = RigidBodyPipeline(config=config)
        result = pipeline.run()
    """
    
    config: RigidBodyConfig  # Type hint for IDE
    
    def load_data(self) -> TrajectoryDataset:
        """Load trajectory data from CSV."""
        trajectory_dict = load_trajectories(filepath=self.config.input_path)
        # Convert to TrajectoryDataset
        # ...
    
    def preprocess_data(self, *, data: TrajectoryDataset) -> TrajectoryDataset:
        """Validate and filter data."""
        # Check for missing markers
        # Interpolate if needed
        # ...
    
    def optimize(self, *, data: TrajectoryDataset) -> OptimizationResult:
        """Run rigid body optimization."""
        optimizer = RigidBodyOptimizer(
            config=self.config.optimization,
            topology=self.config.topology
        )
        return optimizer.optimize(data=data)
    
    def evaluate(self, *, result: OptimizationResult) -> dict[str, float]:
        """Compute reconstruction metrics."""
        from ...core.metrics import evaluate_reconstruction
        return evaluate_reconstruction(...)
    
    def save_results(
        self,
        *,
        result: OptimizationResult,
        metrics: dict[str, float]
    ) -> None:
        """Save CSV, JSON, etc."""
        from ...io.writers import ResultsWriter
        writer = ResultsWriter(output_dir=self.config.output_dir)
        writer.save(result=result, metrics=metrics)
    
    def generate_viewer(self, *, result: OptimizationResult) -> None:
        """Generate HTML viewer."""
        from ...io.viewers import RigidBodyViewer
        viewer = RigidBodyViewer()
        viewer.generate(
            output_dir=self.config.output_dir,
            result=result
        )
```

## Migration Checklist

### Phase 1 Tasks
- [ ] Create `core/cost_functions/` module
  - [ ] `base.py`
  - [ ] `smoothness.py` 
  - [ ] `measurement.py`
  - [ ] `constraints.py`
- [ ] Create `core/config.py`
- [ ] Create `core/result.py`
- [ ] Create `core/optimizer.py`
- [ ] Update existing pipelines to use new cost functions
- [ ] Test that examples still work

### Phase 2 Tasks
- [ ] Create `data/` module
  - [ ] `base.py`
  - [ ] `formats.py`
  - [ ] `loaders.py`
- [ ] Migrate loaders to new structure
- [ ] Update pipelines to use new loaders
- [ ] Test data loading with all formats

### Phase 3 Tasks
- [ ] Create `pipelines/base.py`
- [ ] Refactor rigid body pipeline
- [ ] Refactor eye tracking pipeline
- [ ] Move examples to `pipelines/*/examples/`
- [ ] Test both pipelines end-to-end

### Phase 4 Tasks (IO)
- [ ] Create `io/readers/` module
- [ ] Create `io/writers/` module
- [ ] Create `io/viewers/` module
- [ ] Migrate existing IO code
- [ ] Test viewers generate correctly

### Phase 5 Tasks (Batch)
- [ ] Create `batch/` module
- [ ] Implement batch processor
- [ ] Create batch examples
- [ ] Documentation

### Final Cleanup
- [ ] Delete old duplicate files
- [ ] Update all imports
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create migration guide
