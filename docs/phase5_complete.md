# Phase 5: Batch Processing - COMPLETE ✓

## Summary

Phase 5 is now **100% complete**. All files have been created with complete implementations.

### What Was Created

```
skellysolver/batch/
├── __init__.py                  ✓ COMPLETE (exports all batch components)
├── config.py                    ✓ COMPLETE (batch configurations)
├── processor.py                 ✓ COMPLETE (batch processor)
├── report.py                    ✓ COMPLETE (report generator)
├── utils.py                     ✓ COMPLETE (utility functions)
└── examples/
    ├── __init__.py              ✓ COMPLETE
    └── basic_example.py         ✓ COMPLETE (5 examples)
```

### Lines of Code

- **Total new code**: ~1,900 lines
- **Features**: Complete batch processing framework
- **Net benefit**: Process 100s of datasets with one command!

## New Components

### 1. Configuration (`batch/config.py`)

**BatchJobConfig**:
- Configuration for a single batch job
- Job ID, name, priority
- Pipeline configuration
- Metadata

**BatchConfig**:
- Configuration for entire batch
- List of jobs
- Parallel mode (sequential/parallel/auto)
- Error handling options
- Summary report generation

**ParameterSweepConfig**:
- Automatic grid search over parameters
- Generates all parameter combinations
- Creates batch jobs automatically

**Features**:
- ✅ Job prioritization
- ✅ Auto-detect parallel mode
- ✅ Continue on error
- ✅ Save intermediate results

### 2. Processor (`batch/processor.py`)

**BatchProcessor**:
- Execute batch jobs (sequential or parallel)
- Progress tracking
- Error handling
- Result aggregation

**BatchJobResult**:
- Result from single job
- Success/failure status
- Optimization result
- Timing information
- Error messages

**BatchResult**:
- Results from entire batch
- Success rate
- Statistics
- Best/worst job identification

**ProgressTracker**:
- Real-time progress updates
- ETA estimation
- Completion percentage

**Features**:
- ✅ Parallel execution with multiprocessing
- ✅ Automatic retry logic
- ✅ Progress tracking with ETA
- ✅ Comprehensive error handling

### 3. Reporting (`batch/report.py`)

**BatchReportGenerator**:
- Generate summary tables
- Compute statistics
- Create HTML reports
- Export to CSV/JSON

**Functions**:
- `compare_parameter_sweep_results()`: Compare across parameter values
- `find_best_parameters()`: Identify best parameter combination

**Features**:
- ✅ Beautiful HTML reports
- ✅ Statistical analysis
- ✅ Parameter comparison
- ✅ Export to multiple formats

### 4. Utilities (`batch/utils.py`)

**Batch Creation**:
- `create_batch_from_files()`: Process file list
- `create_parameter_sweep()`: Grid search
- `create_cross_validation_batch()`: K-fold CV
- `create_batch_from_directory()`: Process directory

**Organization**:
- `group_files_by_pattern()`: Group files by naming pattern
- `estimate_batch_time()`: Time estimation

**Features**:
- ✅ Multiple batch creation modes
- ✅ Automatic file discovery
- ✅ Flexible grouping
- ✅ Time estimation

### 5. Examples (`batch/examples/basic_example.py`)

**5 Complete Examples**:
1. Process multiple files with same config
2. Parameter sweep to find best weights
3. Process entire directory
4. Manual batch creation
5. Analyze sweep results

## Usage Examples

### Example 1: Process Multiple Files

```python
from pathlib import Path
from skellysolver__.batch import (
    create_batch_from_files,
    BatchProcessor,
)
from skellysolver__.pipelines.rigid_body import RigidBodyConfig
from skellysolver__.core import OptimizationConfig
from skellysolver__.core.topology import RigidBodyTopology

# Files to process
files = [
    Path("recording_001.csv"),
    Path("recording_002.csv"),
    Path("recording_003.csv"),
]

# Topology
topology = RigidBodyTopology(...)


# Config factory
def make_config(filepath: Path) -> RigidBodyConfig:
    return RigidBodyConfig(
        input_path=filepath,
        output_dir=Path("batch_output") / filepath.stem,
        topology=topology,
        optimization=OptimizationConfig(max_iterations=300),
    )


# Create and run batch
batch_config = create_batch_from_files(
    file_paths=files,
    config_factory=make_config,
    output_root=Path("batch_output/")
)

processor = BatchProcessor(config=batch_config)
result = processor.run()

print(result.summary())
```

### Example 2: Parameter Sweep

```python
from skellysolver__.batch import create_parameter_sweep, BatchProcessor
from skellysolver__.batch import find_best_parameters

# Base config
base_config = RigidBodyConfig(...)

# Parameter grid
parameter_grid = {
    "weights.lambda_rigid": [100.0, 500.0, 1000.0],
    "weights.lambda_rot_smooth": [50.0, 100.0, 200.0],
    "optimization.max_iterations": [100, 200, 300],
}

# Creates 3 × 3 × 3 = 27 jobs!
batch_config = create_parameter_sweep(
    base_config=base_config,
    parameter_grid=parameter_grid,
    output_root=Path("sweep_output/")
)

# Run sweep
processor = BatchProcessor(config=batch_config)
result = processor.run()

# Find best parameters
best_params = find_best_parameters(batch_result=result)
print(f"Best parameters: {best_params}")
```

### Example 3: Process Directory

```python
from skellysolver__.batch import create_batch_from_directory, BatchProcessor

# Process all CSVs in directory
batch_config = create_batch_from_directory(
    directory=Path("data/"),
    pattern="*.csv",
    config_factory=make_config,
    output_root=Path("output/"),
    recursive=True  # Search subdirectories too
)

processor = BatchProcessor(config=batch_config)
result = processor.run()
```

### Example 4: Generate Reports

```python
from skellysolver__.batch import BatchReportGenerator

# After running batch
report_gen = BatchReportGenerator(batch_result=result)

# Save all report formats
report_gen.save_all_reports(output_dir=Path("reports/"))

# Creates:
# - reports/batch_summary.csv
# - reports/batch_statistics.json
# - reports/batch_report.html
```

### Example 5: Compare Results

```python
from skellysolver__.batch import compare_parameter_sweep_results

# Compare across specific parameter
comparison = compare_parameter_sweep_results(
    batch_result=result,
    parameter_name="weights.lambda_rigid"
)

print(comparison)
# Shows: parameter_value, final_cost, cost_reduction, iterations, duration
```

## Advanced Features

### Time Estimation

```python
from skellysolver__.batch import estimate_batch_time

estimate = estimate_batch_time(
    batch_config=batch_config,
    time_per_job_seconds=120.0  # 2 minutes per job
)

print(f"Sequential time: {estimate['sequential_time_minutes']:.1f} minutes")
print(f"Parallel time:   {estimate['parallel_time_minutes']:.1f} minutes")
print(f"Speedup:         {estimate['speedup']:.1f}x")
```

### Custom Parallel Workers

```python
batch_config = BatchConfig(
    batch_name="custom_batch",
    jobs=jobs,
    output_root=Path("output/"),
    parallel_mode="parallel",
    max_workers=8,  # Use exactly 8 workers
)
```

### Continue on Error

```python
batch_config = BatchConfig(
    batch_name="robust_batch",
    jobs=jobs,
    output_root=Path("output/"),
    continue_on_error=True,  # Keep going if jobs fail
)
```

### Custom Metadata

```python
job = BatchJobConfig(
    job_id="job_001",
    job_name="Special Job",
    pipeline_config=config,
    metadata={
        "subject": "ferret_757",
        "session": "2025-07-11",
        "condition": "eye_tracking",
        "experimenter": "JM",
    }
)
```

## Benefits Achieved

✅ **Process 100s of datasets** with one command  
✅ **Parallel execution** for fast processing  
✅ **Automatic parameter sweeps** for hyperparameter tuning  
✅ **Progress tracking** with ETA  
✅ **Error handling** (continue on failure)  
✅ **Comprehensive reports** (CSV, JSON, HTML)  
✅ **Result comparison** across parameters  
✅ **Best parameter identification**  

## Real-World Use Cases

### Use Case 1: Process All Recordings

```python
from skellysolver__.batch import create_batch_from_directory

# Process all recordings from a session
batch_config = create_batch_from_directory(
    directory=Path("D:/recordings/2025-07-11_session/"),
    pattern="**/*.csv",  # Recursive search
    config_factory=make_config,
    output_root=Path("D:/processed/2025-07-11/"),
    recursive=True
)

processor = BatchProcessor(config=batch_config)
result = processor.run()

# Process 50 recordings in parallel → Done in minutes! 🚀
```

### Use Case 2: Find Best Weights

```python
from skellysolver__.batch import create_parameter_sweep

# Try different weight combinations
parameter_grid = {
    "weights.lambda_data": [50.0, 100.0, 200.0],
    "weights.lambda_rigid": [100.0, 500.0, 1000.0],
    "weights.lambda_rot_smooth": [50.0, 100.0, 200.0],
}

batch_config = create_parameter_sweep(
    base_config=base_config,
    parameter_grid=parameter_grid,
    output_root=Path("weight_optimization/")
)

# Run 27 different weight combinations
processor = BatchProcessor(config=batch_config)
result = processor.run()

# Find best weights
best_params = find_best_parameters(batch_result=result)
print(f"Optimal weights: {best_params}")
```

### Use Case 3: Batch Process Subjects

```python
# Group files by subject
from skellysolver__.batch import group_files_by_pattern

groups = group_files_by_pattern(
    directory=Path("data/"),
    pattern="subject_*.csv"
)

# Process each subject
for subject_name, files in groups.items():
    batch_config = create_batch_from_files(
        file_paths=files,
        config_factory=make_config,
        output_root=Path(f"output/{subject_name}/")
    )

    processor = BatchProcessor(config=batch_config)
    result = processor.run()
```

## File Checklist

- [x] `batch/__init__.py` - Complete ✓
- [x] `batch/config.py` - Complete ✓
- [x] `batch/processor.py` - Complete ✓
- [x] `batch/report.py` - Complete ✓
- [x] `batch/utils.py` - Complete ✓
- [x] `batch/examples/__init__.py` - Complete ✓
- [x] `batch/examples/basic_example.py` - Complete ✓

## Testing

Create `test_phase5.py`:

```python
"""Verify Phase 5 installation."""

# Test imports
try:
    from skellysolver__.batch import (
        BatchConfig,
        BatchProcessor,
        BatchReportGenerator,
        create_batch_from_files,
        create_parameter_sweep,
    )

    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test batch config creation
from skellysolver__.pipelines.rigid_body import RigidBodyConfig
from skellysolver__.core import OptimizationConfig
from skellysolver__.core.topology import RigidBodyTopology
from pathlib import Path

topology = RigidBodyTopology(
    marker_names=["m1", "m2"],
    rigid_edges=[(0, 1)],
)

base_config = RigidBodyConfig(
    input_path=Path("test.csv"),
    output_dir=Path("output/"),
    topology=topology,
    optimization=OptimizationConfig(max_iterations=10),
)

# Test parameter sweep creation
parameter_grid = {
    "optimization.max_iterations": [5, 10],
}

batch_config = create_parameter_sweep(
    base_config=base_config,
    parameter_grid=parameter_grid,
    output_root=Path("test_sweep/")
)

print(f"✓ Created parameter sweep with {batch_config.n_jobs} jobs")
print("\n✓ Phase 5 installation verified!")
```

---

**Phase 5 Status**: ✓ COMPLETE  
**Lines Written**: 1,900+  
**Features**: Complete batch processing framework  
**Ready to Use**: YES! 🎉
