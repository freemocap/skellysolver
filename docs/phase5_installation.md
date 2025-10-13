# Phase 5 Installation Instructions

## Quick Start

Copy all Phase 5 files into your project:

```bash
cd skellysolver/

# Create batch directory
mkdir -p batch/examples

# Copy all files (shown in artifacts above)
# - batch/__init__.py
# - batch/config.py
# - batch/processor.py
# - batch/report.py
# - batch/utils.py
# - batch/examples/__init__.py
# - batch/examples/basic_example.py
```

## Detailed Installation Steps

### Step 1: Create Directory Structure

```bash
cd skellysolver/
mkdir -p batch/examples
```

### Step 2: Copy Files

I've created 7 complete files in the artifacts above:

1. **batch/__init__.py** - Module exports (110 lines)
2. **batch/config.py** - Batch configurations (280 lines)
3. **batch/processor.py** - Batch processor (380 lines)
4. **batch/report.py** - Report generator (320 lines)
5. **batch/utils.py** - Utility functions (340 lines)
6. **batch/examples/__init__.py** - Examples module (10 lines)
7. **batch/examples/basic_example.py** - Complete examples (460 lines)

Total: **~1,900 lines** of batch processing infrastructure!

### Step 3: Verify Installation

Create `test_phase5.py`:

```python
"""Verify Phase 5 installation."""

# Test imports
try:
    from skellysolver__.batch import (
        BatchConfig,
        BatchJobConfig,
        BatchProcessor,
        BatchReportGenerator,
        create_batch_from_files,
        create_parameter_sweep,
    )

    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

print("\n✓ Phase 5 installation verified!")
```

Run verification:
```bash
python test_phase5.py
```

## Complete Usage Examples

### Example 1: Process Multiple Recordings

```python
from pathlib import Path
from skellysolver__.batch import create_batch_from_directory, BatchProcessor
from skellysolver__.pipelines.rigid_body import RigidBodyConfig
from skellysolver__.core import OptimizationConfig
from skellysolver__.core.topology import RigidBodyTopology

# Define topology
topology = RigidBodyTopology(
    marker_names=["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    rigid_edges=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)],
    name="ferret_head"
)


# Config factory
def make_config(filepath: Path) -> RigidBodyConfig:
    return RigidBodyConfig(
        input_path=filepath,
        output_dir=Path("batch_output") / filepath.stem,
        topology=topology,
        optimization=OptimizationConfig(max_iterations=300),
    )


# Create batch from entire directory
batch_config = create_batch_from_directory(
    directory=Path("D:/recordings/2025-07-11/"),
    pattern="**/*_rigid_3d_xyz.csv",
    config_factory=make_config,
    output_root=Path("D:/processed/2025-07-11/"),
    recursive=True
)

print(f"Created batch with {batch_config.n_jobs} jobs")

# Run batch in parallel
processor = BatchProcessor(config=batch_config)
result = processor.run()

# Results!
print(f"Success rate: {result.success_rate * 100:.1f}%")
print(f"Total time: {result.total_duration_seconds / 60:.1f} minutes")

# Generate reports
from skellysolver__.batch import BatchReportGenerator

report_gen = BatchReportGenerator(batch_result=result)
report_gen.save_all_reports(output_dir=Path("D:/processed/2025-07-11/reports/"))

print("✓ Done! Open reports/batch_report.html to view results")
```

### Example 2: Find Optimal Weights

```python
from skellysolver__.batch import create_parameter_sweep, BatchProcessor, find_best_parameters

# Base configuration
base_config = RigidBodyConfig(
    input_path=Path("test_data.csv"),
    output_dir=Path("sweep/"),
    topology=topology,
    optimization=OptimizationConfig(max_iterations=200),
)

# Try different weights
parameter_grid = {
    "weights.lambda_data": [50.0, 100.0, 200.0],
    "weights.lambda_rigid": [100.0, 500.0, 1000.0, 2000.0],
    "weights.lambda_rot_smooth": [50.0, 100.0, 200.0],
}

# Creates 3 × 4 × 3 = 36 jobs
batch_config = create_parameter_sweep(
    base_config=base_config,
    parameter_grid=parameter_grid,
    output_root=Path("weight_sweep/")
)

# Run sweep
processor = BatchProcessor(config=batch_config)
result = processor.run()

# Find best
best_params = find_best_parameters(batch_result=result)

print("\n" + "=" * 80)
print("OPTIMAL WEIGHTS FOUND!")
print("=" * 80)
for param, value in best_params.items():
    print(f"{param}: {value}")
```

### Example 3: Cross-Validation

```python
from skellysolver__.batch import create_cross_validation_batch

# All data files
all_files = list(Path("data/").glob("*.csv"))


# Config factory (takes train and test files)
def make_cv_config(train_files: list[Path], test_files: list[Path]) -> RigidBodyConfig:
    # Use first train file for input (or concatenate)
    return RigidBodyConfig(
        input_path=train_files[0],
        output_dir=Path("cv_output/"),
        topology=topology,
        optimization=OptimizationConfig(max_iterations=300),
    )


# Create 5-fold CV batch
batch_config = create_cross_validation_batch(
    file_paths=all_files,
    config_factory=make_cv_config,
    n_folds=5,
    output_root=Path("cv_output/")
)

# Run CV
processor = BatchProcessor(config=batch_config)
result = processor.run()
```

## Integration with Phases 1-4

Batch processing uses ALL previous phases:

**Phase 1 (Core)**: Optimization config and results  
**Phase 2 (Data)**: Data loading and preprocessing  
**Phase 3 (Pipelines)**: Pipeline execution  
**Phase 4 (IO)**: Results saving and viewer generation  
**Phase 5 (Batch)**: Orchestrates everything in batch! 🎯

## Performance Tips

### Tip 1: Estimate Time First

```python
from skellysolver__.batch import estimate_batch_time

estimate = estimate_batch_time(
    batch_config=batch_config,
    time_per_job_seconds=300  # 5 minutes per job
)

print(f"Estimated completion: {estimate['parallel_time_minutes']:.1f} minutes")

# Decide if you want to run it
if estimate['parallel_time_minutes'] < 60:
    # Less than 1 hour - go for it!
    processor = BatchProcessor(config=batch_config)
    result = processor.run()
else:
    print("This will take too long. Consider:")
    print("  - Reducing max_iterations")
    print("  - Using more workers")
    print("  - Processing subset of files")
```

### Tip 2: Use Parallel Mode

```python
# For many jobs, parallel is much faster
batch_config = BatchConfig(
    ...,
    parallel_mode="parallel",  # Force parallel
    max_workers=None,  # Use all CPUs
)

# Speedup example:
# 50 jobs × 5 min each = 250 minutes sequential
# With 10 workers = 25 minutes parallel
# Speedup: 10x! 🚀
```

### Tip 3: Monitor Progress

```python
# BatchProcessor logs progress automatically
# You'll see:
# ✓ [1/50] recording_001 - ETA: 24.5m
# ✓ [2/50] recording_002 - ETA: 24.0m
# ...
```

## Common Issues

### Issue 1: Jobs Failing Silently

**Problem**: Some jobs fail but batch continues

**Solution**: Check batch report to see which jobs failed:
```python
report_gen = BatchReportGenerator(batch_result=result)
report_gen.save_summary_csv(filepath=Path("summary.csv"))

# Open summary.csv and filter by success=False
```

### Issue 2: Out of Memory

**Problem**: Running too many parallel jobs causes OOM

**Solution**: Reduce number of workers:
```python
batch_config = BatchConfig(
    ...,
    max_workers=4,  # Limit parallel workers
)
```

### Issue 3: Parameter Sweep Too Large

**Problem**: Parameter grid creates thousands of jobs

**Solution**: Reduce grid size or use staged sweeps:
```python
# Instead of full grid:
# parameter_grid = {
#     "param1": [1, 2, 3, 4, 5],  # 5 values
#     "param2": [1, 2, 3, 4, 5],  # 5 values
#     "param3": [1, 2, 3, 4, 5],  # 5 values
# }
# = 5 × 5 × 5 = 125 jobs! Too many!

# Do staged sweep:
# Stage 1: Find best param1
parameter_grid_1 = {"param1": [1, 2, 3, 4, 5]}

# Stage 2: Fix param1, find best param2
parameter_grid_2 = {"param1": [best_value], "param2": [1, 2, 3, 4, 5]}

# Stage 3: Fix param1 and param2, find best param3
# ...
```

## Dependencies

Phase 5 requires (already installed from previous phases):
- `numpy`
- `pandas`
- `multiprocessing` (built-in)
- `pathlib` (built-in)
- `json` (built-in)

No new dependencies! ✓

---

**Phase 5 Status**: ✓ COMPLETE  
**Installation Time**: ~15 minutes  
**Testing Time**: ~10 minutes  
**Total**: ~25 minutes  

**Ready to process hundreds of datasets!** 🚀🚀🚀
