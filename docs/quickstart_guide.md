# SkellySolver Quick Start Guide 🚀

## Installation

### Step 1: Copy All Files

Copy all 45 files from the artifacts into your `skellysolver/` directory following the structure shown in `SKELLYSOLVER_REFACTORING_COMPLETE.md`.

### Step 2: Install Dependencies

```bash
pip install numpy scipy pyceres pandas
```

### Step 3: Verify Installation

```bash
python test_phase1.py  # Core optimization
python test_phase2.py  # Data layer
python test_phase3.py  # Pipelines
python test_phase4.py  # IO
python test_phase5.py  # Batch processing
```

All tests should pass! ✓

---

## Quick Start: Process ONE Dataset

```python
from pathlib import Path
from skellysolver__.pipelines.rigid_body import (
    RigidBodyPipeline,
    RigidBodyConfig,
)
from skellysolver__.core import OptimizationConfig, RigidBodyWeightConfig
from skellysolver__.core.topology import RigidBodyTopology

# 1. Define topology
topology = RigidBodyTopology(
    marker_names=["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    rigid_edges=[
        (0, 1), (0, 2), (1, 2),  # Face
        (1, 3), (2, 4),  # Eyes to ears
        (3, 4),  # Between ears
    ],
    name="ferret_head"
)

# 2. Configure pipeline
config = RigidBodyConfig(
    input_path=Path("mocap_data.csv"),
    output_dir=Path("output/"),
    topology=topology,
    optimization=OptimizationConfig(max_iterations=300),
    weights=RigidBodyWeightConfig(),
)

# 3. Run pipeline
pipeline = RigidBodyPipeline(config=config)
result = pipeline.run()

# 4. Done! Results automatically saved to output/
print(result.summary())
```

**That's it! 15 lines to process one dataset!** ✨

---

## Quick Start: Process MANY Datasets

```python
from pathlib import Path
from skellysolver__.batch import (
    create_batch_from_directory,
    BatchProcessor,
    BatchReportGenerator,
)


# 1. Create config factory
def make_config(filepath: Path) -> RigidBodyConfig:
    return RigidBodyConfig(
        input_path=filepath,
        output_dir=Path("batch_output") / filepath.stem,
        topology=topology,  # Same topology for all
        optimization=OptimizationConfig(max_iterations=300),
    )


# 2. Create batch from directory
batch_config = create_batch_from_directory(
    directory=Path("D:/recordings/2025-07-11/"),
    pattern="**/*_rigid_3d_xyz.csv",
    config_factory=make_config,
    output_root=Path("D:/processed/2025-07-11/"),
    recursive=True
)

print(f"Found {batch_config.n_jobs} datasets to process")

# 3. Run batch processor
processor = BatchProcessor(config=batch_config)
result = processor.run()

# 4. Generate reports
report_gen = BatchReportGenerator(batch_result=result)
report_gen.save_all_reports(output_dir=Path("D:/processed/2025-07-11/reports/"))

# 5. Done! All datasets processed!
print(f"Processed {result.n_jobs_successful}/{result.n_jobs_total} datasets")
print(f"Success rate: {result.success_rate * 100:.1f}%")
print(f"Total time: {result.total_duration_seconds / 60:.1f} minutes")
```

**Process 100 datasets in ~20 lines!** 🔥

---

## Quick Start: Find Optimal Weights

```python
from skellysolver__.batch import (
    create_parameter_sweep,
    BatchProcessor,
    find_best_parameters,
)

# 1. Define parameter grid
parameter_grid = {
    "weights.lambda_data": [50.0, 100.0, 200.0],
    "weights.lambda_rigid": [100.0, 500.0, 1000.0],
    "weights.lambda_rot_smooth": [50.0, 100.0, 200.0],
}

# 2. Create parameter sweep
batch_config = create_parameter_sweep(
    base_config=base_config,
    parameter_grid=parameter_grid,
    output_root=Path("weight_optimization/")
)

print(f"Testing {batch_config.n_jobs} parameter combinations")

# 3. Run sweep
processor = BatchProcessor(config=batch_config)
result = processor.run()

# 4. Find best parameters
best_params = find_best_parameters(batch_result=result)

print("\n" + "=" * 80)
print("OPTIMAL WEIGHTS FOUND!")
print("=" * 80)
for param, value in best_params.items():
    print(f"  {param}: {value}")

# 5. Use optimal weights for production!
optimal_config = RigidBodyConfig(
    ...,
    weights=RigidBodyWeightConfig(**best_params)
)
```

**Find optimal weights automatically!** 🎯

---

## Common Workflows

### Workflow 1: New Dataset → Results

```python
# Load data
from skellysolver__.data import load_trajectories, validate_dataset

dataset = load_trajectories(filepath=Path("new_data.csv"))

# Validate
report = validate_dataset(dataset=dataset)
if not report["valid"]:
    print("Data has issues!")
    for error in report["errors"]:
        print(f"  - {error}")
else:
    # Process
    pipeline = RigidBodyPipeline(config=config)
    result = pipeline.run()
    print("✓ Done!")
```

### Workflow 2: Directory → Batch Results

```python
from skellysolver__.batch import create_batch_from_directory, BatchProcessor

batch_config = create_batch_from_directory(
    directory=Path("recordings/"),
    pattern="*.csv",
    config_factory=make_config,
    output_root=Path("processed/")
)

processor = BatchProcessor(config=batch_config)
result = processor.run()

print(f"Processed {result.n_jobs_successful} datasets!")
```

### Workflow 3: Hyperparameter Tuning → Production

```python
# Step 1: Run parameter sweep
batch_config = create_parameter_sweep(
    base_config=base_config,
    parameter_grid=parameter_grid,
    output_root=Path("tuning/")
)

processor = BatchProcessor(config=batch_config)
result = processor.run()

# Step 2: Find best
best_params = find_best_parameters(batch_result=result)

# Step 3: Use in production
production_config = RigidBodyConfig(
    weights=RigidBodyWeightConfig(
        lambda_rigid=best_params["weights.lambda_rigid"],
        lambda_rot_smooth=best_params["weights.lambda_rot_smooth"],
    )
)
```

---

## Tips & Tricks

### Tip 1: Start Small

```python
# Test with one file first
config = RigidBodyConfig(...)
pipeline = RigidBodyPipeline(config=config)
result = pipeline.run()

# If it works, scale to batch!
```

### Tip 2: Use Progress Tracking

```python
# BatchProcessor automatically shows progress:
# ✓ [1/50] recording_001 - ETA: 24.5m
# ✓ [2/50] recording_002 - ETA: 24.0m
# ...
```

### Tip 3: Check Reports

```python
# After batch processing, check HTML report:
# batch_output/reports/batch_report.html

# Shows:
# - Success rate
# - Job details
# - Performance metrics
# - Failed jobs with errors
```

### Tip 4: Handle Errors Gracefully

```python
batch_config = BatchConfig(
    ...,
    continue_on_error=True,  # Don't stop on failures
)

# After running, check which failed:
failed_jobs = [j for j in result.job_results if not j.success]
for job in failed_jobs:
    print(f"Failed: {job.job_name} - {job.error}")
```

### Tip 5: Estimate Before Running

```python
from skellysolver__.batch import estimate_batch_time

estimate = estimate_batch_time(
    batch_config=batch_config,
    time_per_job_seconds=300  # 5 min per job
)

print(f"Estimated time: {estimate['parallel_time_minutes']:.1f} minutes")

# Decide if you want to run it now or overnight
```

---

## Troubleshooting

### Issue: Batch Processing is Slow

**Solution**: Make sure parallel mode is enabled:
```python
batch_config = BatchConfig(
    ...,
    parallel_mode="parallel",
    max_workers=10,  # Use 10 cores
)
```

### Issue: Running Out of Memory

**Solution**: Reduce number of workers:
```python
batch_config = BatchConfig(
    ...,
    max_workers=4,  # Fewer workers = less memory
)
```

### Issue: Some Jobs Fail

**Solution**: Check error messages in batch report:
```python
report_gen = BatchReportGenerator(batch_result=result)
report_gen.save_summary_csv(filepath=Path("summary.csv"))

# Open summary.csv and look at error column
```

---

## You're Ready! 🎉

You now have **everything you need** to:

1. ✅ Process single datasets with clean pipelines
2. ✅ Process multiple datasets in batch
3. ✅ Run parameter sweeps for optimization
4. ✅ Generate beautiful reports
5. ✅ Extend with custom pipelines
6. ✅ Scale to 100s or 1000s of datasets

**Start processing your data and enjoy the clean architecture!** 🚀

---

## Quick Reference

### Import Cheat Sheet

```python
# Core optimization
from skellysolver__.core import (
    OptimizationConfig,
    Optimizer,
    RotationSmoothnessCost,
)

# Data handling
from skellysolver__.data import (
    load_trajectories,
    validate_dataset,
    interpolate_missing,
)

# Pipelines
from skellysolver__.pipelines.rigid_body import (
    RigidBodyPipeline,
    RigidBodyConfig,
)

# Batch processing
from skellysolver__.batch import (
    create_batch_from_directory,
    BatchProcessor,
    BatchReportGenerator,
)
```

### File Locations

- **Input data**: Any CSV (tidy/wide/DLC)
- **Output results**: Configured per pipeline
- **Batch reports**: `batch_output/reports/`
- **Viewers**: `output/rigid_body_viewer.html`

### Getting Help

- Check phase documentation: `PHASE_X_COMPLETE.md`
- Run examples: `python -m skellysolver.batch.examples.basic_example`
- Read docstrings: All functions fully documented

---

**Happy optimizing!** 🎯
