"""Base solver class for all SkellySolver solvers.

All solvers (rigid body, eye tracking, future solvers) inherit from BaseSolver.
Provides standard interface and workflow:
1. Load data
2. Preprocess/validate
3. Optimize
4. Evaluate
5. Save results
6. Generate visualizations

This eliminates code duplication and ensures consistent API.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from skellysolver.core import OptimizationConfig, ChunkingConfig, OptimizationResult
from skellysolver.data import TrajectoryDataset
from skellysolver.data.arbitrary_types_model import ABaseModel

logger = logging.getLogger(__name__)


class SolverTimingReport(ABaseModel):
    """Timing information for solver steps.

    Attributes:
        load: Time to load data
        preprocess: Time to preprocess data
        optimize: Time to run optimization
        evaluate: Time to evaluate results
        save: Time to save results
        viewer: Time to generate viewer
        total: Total time for entire solver run
    """
    load: float = 0.0
    preprocess: float = 0.0
    optimize: float = 0.0
    evaluate: float = 0.0
    save: float = 0.0
    viewer: float = 0.0
    total: float = 0.0

class SolverEvaluationReport(ABaseModel):
    """Evaluation metrics for solver results.

    Attributes:
        reconstruction_error: Average reprojection error
        constraint_satisfaction: Measure of constraint satisfaction
        smoothness: Temporal smoothness metric
        custom_metrics: Dictionary of additional custom metrics
    """
    reconstruction_error: float | None = None
    constraint_satisfaction: float | None = None
    smoothness: float | None = None
    custom_metrics: dict[str, Any] = Field(default_factory=dict)

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

class SolverSummary(ABaseModel):
    """Summary of solver execution.

    Attributes:
        solver: Name of solver class
        status: 'complete' or 'incomplete'
        timing: Timing report
        evaluation: Evaluation metrics report
        optimization: Optimization summary report

    """
    solver: str
    status: str
    timing: SolverTimingReport
    evaluation: SolverEvaluationReport
    optimization: SolverOptimizationReport


class SolverConfig(ABaseModel):
    """Base configuration for all solvers.

    All solver configs inherit from this.

    Attributes:
        input_path: Path to input data file
        output_dir: Directory for output files
        config: Optimization configuration
        parallel: Parallel processing configuration (optional)
        metadata: Additional solver-specific metadata
    """
    input_path: Path
    output_dir: Path
    config: OptimizationConfig
    parallel: ChunkingConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def ensure_paths_and_create_output_dir(self) -> 'SolverConfig':
        """Ensure paths are Path objects and output dir exists."""
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


class BaseSolver(ABaseModel, ABC):
    """Abstract base class for all SkellySolver solvers.

    Defines standard solver interface that all solvers must implement.
    Provides common functionality like timing, logging, and workflow management.

    Subclasses must implement:
    - load_data(): Load input data
    - preprocess_data(): Validate and preprocess
    - optimize(): Run optimization
    - evaluate(): Compute metrics
    - save_results(): Save outputs
    - generate_viewer(): Create visualization

    Usage:
        class MySolver(BaseSolver):
            def load_data(self) -> TrajectoryDataset:
                # Implementation

            def preprocess_data(self, *, data: TrajectoryDataset) -> TrajectoryDataset:
                # Implementation

            # ... implement other methods

        config = MySolverConfig(...)
        solver = MySolver(config=config)
        result = solver.run()
    """
    config: SolverConfig
    data: TrajectoryDataset | None = None
    result: OptimizationResult | None = None
    metrics: SolverEvaluationReport | None = None
    timing: SolverTimingReport = Field(default_factory=SolverTimingReport)


    def run(self) -> OptimizationResult:
        """Run complete solver.

        Executes all solver steps in order:
        1. Load data
        2. Preprocess
        3. Optimize
        4. Evaluate
        5. Save results
        6. Generate viewer

        Returns:
            OptimizationResult with optimization results
        """
        logger.info("="*80)
        logger.info(f"{self.__class__.__name__.upper()} PIPELINE")
        logger.info("="*80)
        logger.info(f"Input:  {self.config.input_path}")
        logger.info(f"Output: {self.config.output_dir}")

        solver_start = time.time()

        # Step 1: Load data
        step_start = time.time()
        logger.info("\n" + "="*80)
        logger.info("STEP 1: LOAD DATA")
        logger.info("="*80)
        self.data = self.load_data()
        self.timing["load"] = time.time() - step_start
        logger.info(f"  Time: {self.timing['load']:.2f}s")

        # Step 2: Preprocess
        step_start = time.time()
        logger.info("\n" + "="*80)
        logger.info("STEP 2: PREPROCESS DATA")
        logger.info("="*80)
        self.data = self.preprocess_data(data=self.data)
        self.timing["preprocess"] = time.time() - step_start
        logger.info(f"  Time: {self.timing['preprocess']:.2f}s")

        # Step 3: Optimize
        step_start = time.time()
        logger.info("\n" + "="*80)
        logger.info("STEP 3: OPTIMIZE")
        logger.info("="*80)
        self.result = self.optimize(data=self.data)
        self.timing["optimize"] = time.time() - step_start
        logger.info(f"  Time: {self.timing['optimize']:.2f}s")

        # Step 4: Evaluate
        step_start = time.time()
        logger.info("\n" + "="*80)
        logger.info("STEP 4: EVALUATE")
        logger.info("="*80)
        self.metrics = self.evaluate(result=self.result)
        self.timing["evaluate"] = time.time() - step_start
        logger.info(f"  Time: {self.timing['evaluate']:.2f}s")

        # Step 5: Save results
        step_start = time.time()
        logger.info("\n" + "="*80)
        logger.info("STEP 5: SAVE RESULTS")
        logger.info("="*80)
        self.save_results(result=self.result, metrics=self.metrics)
        self.timing["save"] = time.time() - step_start
        logger.info(f"  Time: {self.timing['save']:.2f}s")

        # Step 6: Generate viewer
        step_start = time.time()
        logger.info("\n" + "="*80)
        logger.info("STEP 6: GENERATE VIEWER")
        logger.info("="*80)
        self.generate_viewer(result=self.result)
        self.timing["viewer"] = time.time() - step_start
        logger.info(f"  Time: {self.timing['viewer']:.2f}s")

        # Done
        self.timing["total"] = time.time() - solver_start

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total time: {self.timing['total']:.2f}s")
        logger.info(f"  Load:       {self.timing['load']:.2f}s")
        logger.info(f"  Preprocess: {self.timing['preprocess']:.2f}s")
        logger.info(f"  Optimize:   {self.timing['optimize']:.2f}s")
        logger.info(f"  Evaluate:   {self.timing['evaluate']:.2f}s")
        logger.info(f"  Save:       {self.timing['save']:.2f}s")
        logger.info(f"  Viewer:     {self.timing['viewer']:.2f}s")
        logger.info("="*80)

        return self.result

    @abstractmethod
    def load_data(self) -> TrajectoryDataset:
        """Load input data from file.

        Must be implemented by subclasses.

        Returns:
            TrajectoryDataset with loaded data
        """
        pass

    @abstractmethod
    def preprocess_data(self, *, data: TrajectoryDataset) -> TrajectoryDataset:
        """Preprocess and validate data.

        Must be implemented by subclasses.
        Typical operations:
        - Validate required markers present
        - Filter low-confidence frames
        - Interpolate missing data
        - Smooth trajectories

        Args:
            data: Raw loaded data

        Returns:
            Preprocessed data
        """
        pass

    @abstractmethod
    def optimize(self, *, data: TrajectoryDataset) -> OptimizationResult:
        """Run optimization.

        Must be implemented by subclasses.
        This is the core solver step where pyceres optimization happens.

        Args:
            data: Preprocessed data

        Returns:
            OptimizationResult with optimized parameters
        """
        pass

    @abstractmethod
    def evaluate(self, *, result: OptimizationResult) -> SolverSummary:
        """Evaluate results and compute metrics.

        Must be implemented by subclasses.
        Typical metrics:
        - Reconstruction error
        - Constraint satisfaction
        - Temporal smoothness

        Args:
            result: Optimization result

        Returns:
            Dictionary with metric name -> value
        """
        pass

    @abstractmethod
    def save_results(
        self,
        *,
        result: OptimizationResult,
        metrics: dict[str, Any]
    ) -> None:
        """Save results to disk.

        Must be implemented by subclasses.
        Typical outputs:
        - CSV with trajectories
        - JSON with metrics
        - NPY files with arrays

        Args:
            result: Optimization result
            metrics: Evaluation metrics
        """
        pass

    @abstractmethod
    def generate_viewer(self, *, result: OptimizationResult) -> None:
        """Generate interactive visualization.

        Must be implemented by subclasses.
        Typically generates HTML viewer for results.

        Args:
            result: Optimization result
        """
        pass

    def get_summary(self) -> dict[str, Any]:
        """Get solver execution summary.

        Returns:
            Dictionary with timing, metrics, and status
        """
        summary = {
            "solver": self.__class__.__name__,
            "status": "complete" if self.result is not None else "incomplete",
            "timing": self.timing,
            "metrics": self.metrics,
        }

        if self.result is not None:
            summary["optimization"] = {
                "success": self.result.success,
                "iterations": self.result.num_iterations,
                "initial_cost": self.result.initial_cost,
                "final_cost": self.result.final_cost,
                "cost_reduction": self.result.cost_reduction_percent,
            }

        if self.data is not None:
            summary["data"] = {
                "n_frames": self.data.n_frames,
                "n_markers": self.data.n_markers,
                "marker_names": self.data.marker_names,
            }

        return summary

    def print_summary(self) -> None:
        """Print solver summary to console."""
        summary = self.get_summary()

        print("\n" + "="*80)
        print(f"{summary['solver']} SUMMARY")
        print("="*80)
        print(f"Status: {summary['status']}")

        if "data" in summary:
            print(f"\nData:")
            print(f"  Frames:  {summary['data']['n_frames']}")
            print(f"  Markers: {summary['data']['n_markers']}")

        if "optimization" in summary:
            print(f"\nOptimization:")
            print(f"  Success:     {summary['optimization']['success']}")
            print(f"  Iterations:  {summary['optimization']['iterations']}")
            print(f"  Cost:        {summary['optimization']['initial_cost']:.2f} → {summary['optimization']['final_cost']:.2f}")
            print(f"  Reduction:   {summary['optimization']['cost_reduction']:.1f}%")

        if summary["timing"]:
            print(f"\nTiming:")
            print(f"  Total:   {summary['timing']['total']:.2f}s")
            for step, duration in summary["timing"].items():
                if step != "total":
                    print(f"  {step.capitalize():12} {duration:.2f}s")

        if summary["metrics"]:
            print(f"\nMetrics:")
            for name, value in summary["metrics"].items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")

        print("="*80)


class SolverRunner:
    """Utility class for running multiple solvers.

    Useful for batch processing or comparing different configurations.
    """

    def __init__(self) -> None:
        """Initialize solver runner."""
        self.solvers: list[BaseSolver] = []
        self.results: list[OptimizationResult] = []

    def add_solver(self, *, solver: BaseSolver) -> None:
        """Add solver to run.

        Args:
            solver: Solver instance
        """
        self.solvers.append(solver)

    def run_all(self) -> list[OptimizationResult]:
        """Run all solvers sequentially.

        Returns:
            List of optimization results
        """
        logger.info(f"Running {len(self.solvers)} solvers...")

        for i, solver in enumerate(self.solvers, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"PIPELINE {i}/{len(self.solvers)}")
            logger.info(f"{'='*80}")

            result = solver.run()
            self.results.append(result)

        logger.info(f"\n✓ All solvers complete!")
        return self.results

    def run_all_parallel(self) -> list[OptimizationResult]:
        """Run all solvers in parallel.

        Returns:
            List of optimization results
        """
        import multiprocessing as mp

        logger.info(f"Running {len(self.solvers)} solvers in parallel...")

        def run_solver(solver: BaseSolver) -> OptimizationResult:
            return solver.run()

        with mp.Pool() as pool:
            self.results = pool.map(run_solver, self.solvers)

        logger.info(f"\n✓ All solvers complete!")
        return self.results

    def compare_results(self) -> dict[str, Any]:
        """Compare results across solvers.

        Returns:
            Comparison dictionary
        """
        if not self.results:
            raise ValueError("No results to compare - run solvers first")

        comparison = {
            "n_solvers": len(self.results),
            "success_rate": sum(r.success for r in self.results) / len(self.results),
            "avg_iterations": sum(r.num_iterations for r in self.results) / len(self.results),
            "avg_cost_reduction": sum(r.cost_reduction_percent for r in self.results) / len(self.results),
            "best_solver": None,
            "worst_solver": None,
        }

        # Find best (lowest final cost)
        best_idx = min(range(len(self.results)), key=lambda i: self.results[i].final_cost)
        comparison["best_solver"] = best_idx

        # Find worst (highest final cost)
        worst_idx = max(range(len(self.results)), key=lambda i: self.results[i].final_cost)
        comparison["worst_solver"] = worst_idx

        return comparison