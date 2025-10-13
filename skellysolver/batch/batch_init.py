"""Batch processing module for SkellySolver.

Process multiple datasets or run parameter sweeps in batch.

Components:
- BatchConfig: Configuration for batch processing
- BatchProcessor: Execute batch jobs
- BatchReportGenerator: Generate reports from results
- Utilities: Helper functions for creating batches

Usage:
    # Process multiple files
    from skellysolver.batch import create_batch_from_files, BatchProcessor
    
    batch_config = create_batch_from_files(
        file_paths=[Path("data1.csv"), Path("data2.csv")],
        config_factory=make_config,
        output_root=Path("batch_output/")
    )
    
    processor = BatchProcessor(config=batch_config)
    result = processor.run()
    
    # Parameter sweep
    from skellysolver.batch import create_parameter_sweep
    
    batch_config = create_parameter_sweep(
        base_config=my_config,
        parameter_grid={
            "optimization.max_iterations": [100, 200, 300],
            "weights.lambda_rigid": [100, 500, 1000],
        },
        output_root=Path("sweep_output/")
    )
    
    processor = BatchProcessor(config=batch_config)
    result = processor.run()
"""

# Configuration
from .config import (
    BatchConfig,
    BatchJobConfig,
    ParameterSweepConfig,
)

# Processor
from .processor import (
    BatchProcessor,
    BatchJobResult,
    BatchResult,
    ProgressTracker,
)

# Reporting
from .report import (
    BatchReportGenerator,
    compare_parameter_sweep_results,
    find_best_parameters,
)

# Utilities
from .utils import (
    create_batch_from_files,
    create_parameter_sweep,
    create_cross_validation_batch,
    group_files_by_pattern,
    create_batch_from_directory,
    estimate_batch_time,
)

__all__ = [
    # Configuration
    "BatchConfig",
    "BatchJobConfig",
    "ParameterSweepConfig",
    # Processor
    "BatchProcessor",
    "BatchJobResult",
    "BatchResult",
    "ProgressTracker",
    # Reporting
    "BatchReportGenerator",
    "compare_parameter_sweep_results",
    "find_best_parameters",
    # Utilities
    "create_batch_from_files",
    "create_parameter_sweep",
    "create_cross_validation_batch",
    "group_files_by_pattern",
    "create_batch_from_directory",
    "estimate_batch_time",
]
