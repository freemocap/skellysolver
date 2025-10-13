"""Utility functions for batch processing.

Helper functions for:
- Creating batch jobs from file lists
- Parameter sweeps
- Dataset organization
"""

import numpy as np
from pathlib import Path
from typing import Any, Callable
from copy import deepcopy

from skellysolver.batch.batch_config import BatchConfig, BatchJobConfig, ParameterSweepConfig
from skellysolver.pipelines import PipelineConfig


def create_batch_from_files(
    *,
    file_paths: list[Path],
    config_factory: Callable[[Path], PipelineConfig],
    output_root: Path,
    batch_name: str = "file_batch"
) -> BatchConfig:
    """Create batch config from list of files.
    
    Processes each file with same pipeline configuration.
    
    Args:
        file_paths: List of input file paths
        config_factory: Function that creates config from filepath
        output_root: Root directory for outputs
        batch_name: Name for this batch
        
    Returns:
        BatchConfig with jobs for each file
        
    Example:
        def make_config(filepath: Path) -> RigidBodyConfig:
            return RigidBodyConfig(
                input_path=filepath,
                output_dir=output_root / filepath.stem,
                topology=my_topology,
                optimization=OptimizationConfig()
            )
        
        batch_config = create_batch_from_files(
            file_paths=[Path("data1.csv"), Path("data2.csv")],
            config_factory=make_config,
            output_root=Path("batch_output/")
        )
    """
    print(f"Creating batch from {len(file_paths)} files...")
    
    jobs = []
    
    for i, filepath in enumerate(file_paths):
        # Create config for this file
        config = config_factory(filepath)
        
        # Create job
        job = BatchJobConfig(
            job_id=f"{batch_name}_{i:04d}",
            job_name=filepath.stem,
            pipeline_config=config,
            metadata={"source_file": str(filepath)}
        )
        
        jobs.append(job)
    
    print(f"  ✓ Created {len(jobs)} jobs")
    
    return BatchConfig(
        batch_name=batch_name,
        jobs=jobs,
        output_root=output_root
    )


def create_parameter_sweep(
    *,
    base_config: PipelineConfig,
    parameter_grid: dict[str, list[Any]],
    output_root: Path,
    sweep_name: str = "parameter_sweep"
) -> BatchConfig:
    """Create batch config for parameter sweep.
    
    Generates all combinations of parameters and creates jobs.
    
    Args:
        base_config: Base pipeline configuration
        parameter_grid: Dictionary mapping parameter paths to value lists
        output_root: Root directory for sweep results
        sweep_name: Name for this sweep
        
    Returns:
        BatchConfig with jobs for all parameter combinations
        
    Example:
        base_config = RigidBodyConfig(...)
        
        parameter_grid = {
            "optimization.max_iterations": [100, 200, 300],
            "weights.lambda_rigid": [100.0, 500.0, 1000.0],
        }
        
        batch_config = create_parameter_sweep(
            base_config=base_config,
            parameter_grid=parameter_grid,
            output_root=Path("sweep_output/")
        )
        
        # This creates 3 × 3 = 9 jobs
    """
    sweep_config = ParameterSweepConfig(
        base_config=base_config,
        parameter_grid=parameter_grid,
        output_root=output_root,
        sweep_name=sweep_name
    )
    
    return sweep_config.generate_batch_config()


def create_cross_validation_batch(
    *,
    file_paths: list[Path],
    config_factory: Callable[[list[Path], list[Path]], PipelineConfig],
    n_folds: int,
    output_root: Path,
    batch_name: str = "cross_validation"
) -> BatchConfig:
    """Create batch config for k-fold cross-validation.
    
    Splits files into train/test sets and creates jobs.
    
    Args:
        file_paths: List of all data files
        config_factory: Function that creates config from (train_files, test_files)
        n_folds: Number of CV folds
        output_root: Root directory for outputs
        batch_name: Name for this batch
        
    Returns:
        BatchConfig with jobs for each fold
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    
    if len(file_paths) < n_folds:
        raise ValueError(f"Not enough files ({len(file_paths)}) for {n_folds} folds")
    
    print(f"Creating {n_folds}-fold cross-validation batch...")
    
    # Split into folds
    fold_size = len(file_paths) // n_folds
    
    jobs = []
    
    for fold_idx in range(n_folds):
        # Determine test set for this fold
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < n_folds - 1 else len(file_paths)
        
        test_files = file_paths[test_start:test_end]
        train_files = file_paths[:test_start] + file_paths[test_end:]
        
        # Create config
        config = config_factory(train_files, test_files)
        
        # Create job
        job = BatchJobConfig(
            job_id=f"{batch_name}_fold_{fold_idx:02d}",
            job_name=f"Fold {fold_idx + 1}/{n_folds}",
            pipeline_config=config,
            metadata={
                "fold": fold_idx,
                "n_train": len(train_files),
                "n_test": len(test_files)
            }
        )
        
        jobs.append(job)
    
    print(f"  ✓ Created {len(jobs)} CV jobs")
    
    return BatchConfig(
        batch_name=batch_name,
        jobs=jobs,
        output_root=output_root
    )


def group_files_by_pattern(
    *,
    directory: Path,
    pattern: str = "*.csv"
) -> dict[str, list[Path]]:
    """Group files by pattern matching.
    
    Useful for organizing datasets by session, subject, etc.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern
        
    Returns:
        Dictionary mapping group names to file lists
        
    Example:
        # Group files like "subject_01_session_1.csv", "subject_01_session_2.csv"
        groups = group_files_by_pattern(
            directory=Path("data/"),
            pattern="subject_*_session_*.csv"
        )
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all matching files
    files = list(directory.glob(pattern))
    
    if not files:
        return {}
    
    # Group by prefix (everything before the number)
    groups = {}
    
    for filepath in files:
        # Simple grouping: use stem up to first digit
        stem = filepath.stem
        
        # Find first digit
        group_name = stem
        for i, char in enumerate(stem):
            if char.isdigit():
                group_name = stem[:i].rstrip('_')
                break
        
        if group_name not in groups:
            groups[group_name] = []
        
        groups[group_name].append(filepath)
    
    # Sort files within each group
    for group_name in groups:
        groups[group_name] = sorted(groups[group_name])
    
    return groups


def create_batch_from_directory(
    *,
    directory: Path,
    pattern: str,
    config_factory: Callable[[Path], PipelineConfig],
    output_root: Path,
    batch_name: str | None = None,
    recursive: bool = False
) -> BatchConfig:
    """Create batch config from all files in directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.csv", "**/*.csv" for recursive)
        config_factory: Function that creates config from filepath
        output_root: Root directory for outputs
        batch_name: Batch name (None = use directory name)
        recursive: Whether to search recursively
        
    Returns:
        BatchConfig with jobs for all matching files
    """
    directory = Path(directory)
    
    if batch_name is None:
        batch_name = directory.name
    
    # Find files
    if recursive and '**' not in pattern:
        pattern = f"**/{pattern}"
    
    file_paths = sorted(directory.glob(pattern))
    
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    print(f"Found {len(file_paths)} files in {directory}")
    
    return create_batch_from_files(
        file_paths=file_paths,
        config_factory=config_factory,
        output_root=output_root,
        batch_name=batch_name
    )


def estimate_batch_time(
    *,
    batch_config: BatchConfig,
    time_per_job_seconds: float
) -> dict[str, float]:
    """Estimate total batch processing time.
    
    Args:
        batch_config: Batch configuration
        time_per_job_seconds: Estimated time per job
        
    Returns:
        Dictionary with time estimates
    """
    n_jobs = batch_config.n_jobs
    
    # Sequential time
    sequential_time = n_jobs * time_per_job_seconds
    
    # Parallel time
    if batch_config.should_use_parallel():
        n_workers = batch_config.get_num_workers()
        parallel_time = (n_jobs / n_workers) * time_per_job_seconds * 1.1  # 10% overhead
        speedup = sequential_time / parallel_time
    else:
        parallel_time = sequential_time
        speedup = 1.0
    
    return {
        "n_jobs": n_jobs,
        "time_per_job_seconds": time_per_job_seconds,
        "sequential_time_seconds": sequential_time,
        "sequential_time_minutes": sequential_time / 60,
        "parallel_time_seconds": parallel_time,
        "parallel_time_minutes": parallel_time / 60,
        "speedup": speedup,
        "time_saved_minutes": (sequential_time - parallel_time) / 60,
    }
