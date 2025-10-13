"""Batch processing configuration for SkellySolver.

Configuration classes for processing multiple datasets in batch.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Literal

from ..pipelines.base import PipelineConfig


@dataclass
class BatchJobConfig:
    """Configuration for a single batch job.
    
    Each job processes one dataset with one configuration.
    
    Attributes:
        job_id: Unique identifier for this job
        job_name: Human-readable name
        pipeline_config: Configuration for pipeline
        priority: Job priority (higher = runs first)
        metadata: Additional job metadata
    """
    
    job_id: str
    job_name: str
    pipeline_config: PipelineConfig
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate job config."""
        if not self.job_id:
            raise ValueError("job_id cannot be empty")


@dataclass
class BatchConfig:
    """Configuration for batch processing.
    
    Defines how to process multiple jobs.
    
    Attributes:
        batch_name: Name for this batch
        jobs: List of job configurations
        output_root: Root directory for all outputs
        parallel_mode: How to run jobs ("sequential", "parallel", "auto")
        max_workers: Maximum parallel workers (None = auto-detect)
        continue_on_error: Whether to continue if a job fails
        save_intermediate: Save results after each job
        generate_summary_report: Generate batch summary report
    """
    
    batch_name: str
    jobs: list[BatchJobConfig]
    output_root: Path
    parallel_mode: Literal["sequential", "parallel", "auto"] = "auto"
    max_workers: int | None = None
    continue_on_error: bool = True
    save_intermediate: bool = True
    generate_summary_report: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate batch config and setup output."""
        if not self.jobs:
            raise ValueError("Batch must contain at least one job")
        
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
    
    @property
    def n_jobs(self) -> int:
        """Number of jobs in batch."""
        return len(self.jobs)
    
    def get_sorted_jobs(self) -> list[BatchJobConfig]:
        """Get jobs sorted by priority (highest first).
        
        Returns:
            List of jobs sorted by priority
        """
        return sorted(self.jobs, key=lambda j: j.priority, reverse=True)
    
    def should_use_parallel(self) -> bool:
        """Determine if parallel execution should be used.
        
        Returns:
            True if should use parallel execution
        """
        if self.parallel_mode == "sequential":
            return False
        elif self.parallel_mode == "parallel":
            return True
        else:  # auto
            # Use parallel if we have multiple jobs
            return self.n_jobs > 1
    
    def get_num_workers(self) -> int:
        """Get number of parallel workers.
        
        Returns:
            Number of workers to use
        """
        if self.max_workers is not None:
            return self.max_workers
        
        import os
        cpu_count = os.cpu_count()
        if cpu_count is None:
            return 1
        
        # Use all but one CPU
        return max(cpu_count - 1, 1)


@dataclass
class ParameterSweepConfig:
    """Configuration for parameter sweep.
    
    Automatically generates batch jobs with different parameter values.
    Useful for hyperparameter optimization.
    
    Attributes:
        base_config: Base pipeline config (will be copied and modified)
        parameter_grid: Dictionary mapping parameter paths to value lists
        output_root: Root directory for sweep results
        sweep_name: Name for this sweep
    """
    
    base_config: PipelineConfig
    parameter_grid: dict[str, list[Any]]
    output_root: Path
    sweep_name: str = "parameter_sweep"
    
    def generate_batch_config(self) -> BatchConfig:
        """Generate BatchConfig from parameter grid.
        
        Creates one job for each combination of parameters.
        
        Returns:
            BatchConfig with jobs for all parameter combinations
        """
        import itertools
        from copy import deepcopy
        
        # Get all parameter combinations
        param_names = list(self.parameter_grid.keys())
        param_values = list(self.parameter_grid.values())
        
        combinations = list(itertools.product(*param_values))
        
        print(f"Generating {len(combinations)} jobs from parameter grid...")
        
        # Create job for each combination
        jobs = []
        for i, combo in enumerate(combinations):
            # Create unique job ID
            job_id = f"{self.sweep_name}_{i:04d}"
            
            # Create parameter string for name
            param_str = "_".join(
                f"{name}={value}"
                for name, value in zip(param_names, combo)
            )
            job_name = f"{self.sweep_name}_{param_str}"
            
            # Copy base config
            config = deepcopy(self.base_config)
            
            # Set parameters
            for param_name, param_value in zip(param_names, combo):
                self._set_nested_attribute(
                    obj=config,
                    path=param_name,
                    value=param_value
                )
            
            # Update output directory
            config.output_dir = self.output_root / job_id
            
            # Create job
            jobs.append(BatchJobConfig(
                job_id=job_id,
                job_name=job_name,
                pipeline_config=config,
                metadata=dict(zip(param_names, combo))
            ))
        
        # Create batch config
        return BatchConfig(
            batch_name=self.sweep_name,
            jobs=jobs,
            output_root=self.output_root,
            parallel_mode="auto",
            generate_summary_report=True
        )
    
    def _set_nested_attribute(
        self,
        *,
        obj: Any,
        path: str,
        value: Any
    ) -> None:
        """Set nested attribute using dot notation.
        
        Args:
            obj: Object to modify
            path: Attribute path (e.g., "optimization.max_iterations")
            value: Value to set
        """
        parts = path.split('.')
        
        # Navigate to parent
        current = obj
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Set final attribute
        setattr(current, parts[-1], value)
