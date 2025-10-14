# SkellySolver Documentation

## Overview

SkellySolver is a Python framework for optimizing motion capture trajectories using nonlinear least squares optimization. The package provides unified infrastructure for rigid body tracking, eye tracking, and custom optimization pipelines.

## Architecture

```
skellysolver/
├── core/           # Optimization infrastructure
├── data/           # Data loading and preprocessing
├── pipelines/      # Pipeline implementations
├── io/             # File I/O operations
└── batch/          # Batch processing
```

### Core Components

**core/**: Optimization primitives
- `cost_functions/`: 12 cost function implementations
- `config.py`: Configuration classes
- `optimizer.py`: pyceres wrapper
- `result.py`: Result structures

**data/**: Data handling
- `loaders.py`: CSV loading (tidy, wide, DLC formats)
- `formats.py`: Format detection
- `validators.py`: Data validation
- `preprocessing.py`: Interpolation, smoothing, outlier removal

**pipelines/**: Pipeline implementations
- `base.py`: Abstract pipeline class
- `rigid_body/`: Rigid body tracking
- `eye_tracking/`: Eye tracking

**io/**: File operations
- `readers/`: Input file readers
- `writers/`: Output file writers
- `viewers/`: HTML viewer generation

**batch/**: Batch processing
- `processor.py`: Parallel batch execution
- `config.py`: Batch configuration
- `report.py`: Report generation

## Installation

### Requirements

```bash
pip install numpy scipy pyceres pandas
```

### Package Structure

Copy all files maintaining the directory structure shown above. Total: 45 files, approximately 10,700 lines of code.

## Usage

### Single Dataset Processing

```python
from pathlib import Path
from skellysolver.pipelines.rigid_body import (
    RigidBodyPipeline,
    RigidBodyConfig,
)
from skellysolver.core import OptimizationConfig
from skellysolver.core.topology import RigidBodyTopology

# Define topology
topology = RigidBodyTopology(
    marker_names=["nose", "left_eye", "right_eye"],
    rigid_edges=[(0, 1), (1, 2), (2, 0)],
    name="example"
)

# Configure
config = RigidBodyConfig(
    input_path=Path("data.csv"),
    output_dir=Path("output/"),
    topology=topology,
    optimization=OptimizationConfig(max_iterations=300),
)

# Execute
pipeline = RigidBodyPipeline(config=config)
result = pipeline.run()
```

### Batch Processing

```python
from skellysolver.batch import (
    create_batch_from_directory,
    BatchProcessor,
)

def make_config(filepath: Path) -> RigidBodyConfig:
    return RigidBodyConfig(
        input_path=filepath,
        output_dir=Path("output") / filepath.stem,
        topology=topology,
        optimization=OptimizationConfig(max_iterations=300),
    )

batch_config = create_batch_from_directory(
    directory=Path("data/"),
    pattern="*.csv",
    config_factory=make_config,
    output_root=Path("output/"),
)

processor = BatchProcessor(config=batch_config)
result = processor.run()
```

### Parameter Optimization

```python
from skellysolver.batch import create_parameter_sweep

parameter_grid = {
    "weights.lambda_rigid": [100.0, 500.0, 1000.0],
    "weights.lambda_rot_smooth": [50.0, 100.0, 200.0],
}

batch_config = create_parameter_sweep(
    base_config=base_config,
    parameter_grid=parameter_grid,
    output_root=Path("sweep/"),
)

processor = BatchProcessor(config=batch_config)
result = processor.run()
```

## Core API

### OptimizationConfig

Configuration for pyceres solver.

```python
from skellysolver.core import OptimizationConfig

config = OptimizationConfig(
    max_iterations: int = 300,
    function_tolerance: float = 1e-9,
    gradient_tolerance: float = 1e-11,
    parameter_tolerance: float = 1e-10,
    use_robust_loss: bool = True,
    robust_loss_type: str = "huber",
    robust_loss_param: float = 2.0,
    linear_solver: str = "sparse_normal_cholesky",
    trust_region_strategy: str = "levenberg_marquardt",
    num_threads: int | None = None,
)
```

### Cost Functions

All cost functions inherit from `BaseCostFunction` and accept a `weight` parameter.

**Smoothness costs:**
- `RotationSmoothnessCost`: Temporal smoothness for quaternions
- `TranslationSmoothnessCost`: Temporal smoothness for translations
- `ScalarSmoothnessCost`: Temporal smoothness for scalar values

**Measurement costs:**
- `Point3DMeasurementCost`: Fit 3D measurements
- `Point2DProjectionCost`: Fit 2D projections
- `RigidPoint3DMeasurementBundleAdjustment`: Bundle adjustment variant
- `SimpleDistanceCost`: Distance penalties

**Constraint costs:**
- `RigidEdgeCost`: Enforce fixed distances
- `SoftEdgeCost`: Encourage distances with flexibility
- `ReferenceAnchorCost`: Prevent geometry drift
- `EdgeLengthVarianceCost`: Minimize distance variance
- `SymmetryConstraintCost`: Enforce bilateral symmetry

### Optimizer

Wrapper around pyceres for problem setup and solving.

```python
from skellysolver.core import Optimizer, OptimizationConfig

optimizer = Optimizer(config=OptimizationConfig())

# Add parameters
optimizer.add_parameter_block(
    name="rotation",
    parameters=quaternion_array,
    manifold=get_quaternion_manifold(),
)

# Add costs
optimizer.add_residual_block(
    cost=cost_function,
    parameters=[param1, param2],
)

# Solve
result = optimizer.solve()
```

## Data API

### Loading Data

```python
from skellysolver.data import load_trajectories

# Automatically detects format (tidy, wide, or DLC)
dataset = load_trajectories(
    filepath=Path("data.csv"),
    scale_factor: float = 1.0,
    z_value: float | None = None,
    likelihood_threshold: float = 0.0,
)
```

### Data Validation

```python
from skellysolver.data import validate_dataset

report = validate_dataset(
    dataset=dataset,
    required_markers: list[str] | None = None,
    min_valid_frames: int = 0,
    min_confidence: float = 0.0,
    check_outliers: bool = False,
)

if not report["valid"]:
    print("Errors:", report["errors"])
```

### Preprocessing

```python
from skellysolver.data import (
    filter_by_confidence,
    interpolate_missing,
    smooth_trajectories,
    remove_outliers,
)

# Filter low-confidence data
dataset = filter_by_confidence(
    dataset=dataset,
    min_confidence=0.5,
)

# Interpolate missing values
dataset = interpolate_missing(
    dataset=dataset,
    method="cubic",  # or "linear"
    max_gap=10,
)

# Smooth trajectories
dataset = smooth_trajectories(
    dataset=dataset,
    window_size=5,
    poly_order=2,
)

# Remove outliers
dataset = remove_outliers(
    dataset=dataset,
    threshold=5.0,
    method="velocity",  # or "position"
)
```

## Pipeline API

### BasePipeline

All pipelines inherit from `BasePipeline` and implement these methods:

- `load_data() -> TrajectoryDataset`
- `preprocess_data(data: TrajectoryDataset) -> TrajectoryDataset`
- `optimize(data: TrajectoryDataset) -> OptimizationResult`
- `evaluate(result: OptimizationResult) -> dict[str, Any]`
- `save_results(result: OptimizationResult, metrics: dict[str, Any]) -> None`
- `generate_viewer(result: OptimizationResult) -> None`

### RigidBodyPipeline

```python
from skellysolver.pipelines.rigid_body import (
    RigidBodyPipeline,
    RigidBodyConfig,
)
from skellysolver.core import RigidBodyWeightConfig

config = RigidBodyConfig(
    input_path=Path("data.csv"),
    output_dir=Path("output/"),
    topology=topology,
    optimization=OptimizationConfig(),
    weights=RigidBodyWeightConfig(
        lambda_data=100.0,
        lambda_rigid=500.0,
        lambda_rot_smooth=200.0,
        lambda_trans_smooth=200.0,
    ),
    soft_edges: list[tuple[int, int]] | None = None,
    lambda_soft: float = 10.0,
)

pipeline = RigidBodyPipeline(config=config)
result = pipeline.run()
```

### EyeTrackingPipeline

```python
from skellysolver.pipelines.eye_tracking import (
    EyeTrackingPipeline,
    EyeTrackingConfig,
    CameraIntrinsics,
    EyeModel,
)

camera = CameraIntrinsics.create_pupil_labs_camera()
eye_model = EyeModel.create_initial_guess()

config = EyeTrackingConfig(
    input_path=Path("data.csv"),
    output_dir=Path("output/"),
    camera=camera,
    initial_eye_model=eye_model,
    optimization=OptimizationConfig(),
    min_confidence=0.3,
    min_pupil_points=6,
)

pipeline = EyeTrackingPipeline(config=config)
result = pipeline.run()
```

## Batch API

### BatchConfig

```python
from skellysolver.batch import BatchConfig, BatchJobConfig

job = BatchJobConfig(
    job_id="job_001",
    job_name="Dataset 1",
    pipeline_config=pipeline_config,
    priority: int = 0,
    metadata: dict[str, Any] | None = None,
)

batch_config = BatchConfig(
    batch_name="My Batch",
    jobs=[job1, job2, job3],
    output_root=Path("output/"),
    parallel_mode="auto",  # "sequential", "parallel", or "auto"
    max_workers: int | None = None,
    continue_on_error: bool = False,
)
```

### Batch Creation Utilities

```python
from skellysolver.batch import (
    create_batch_from_files,
    create_batch_from_directory,
    create_parameter_sweep,
    create_cross_validation_batch,
)

# From file list
batch_config = create_batch_from_files(
    file_paths=[Path("f1.csv"), Path("f2.csv")],
    config_factory=make_config,
    output_root=Path("output/"),
)

# From directory
batch_config = create_batch_from_directory(
    directory=Path("data/"),
    pattern="*.csv",
    config_factory=make_config,
    output_root=Path("output/"),
    recursive: bool = False,
)

# Parameter sweep
batch_config = create_parameter_sweep(
    base_config=base_config,
    parameter_grid={"param1": [1, 2, 3]},
    output_root=Path("output/"),
)

# Cross-validation
batch_config = create_cross_validation_batch(
    file_paths=all_files,
    config_factory=make_cv_config,
    n_folds=5,
    output_root=Path("output/"),
)
```

### BatchProcessor

```python
from skellysolver.batch import BatchProcessor

processor = BatchProcessor(config=batch_config)
result = processor.run()

# Access results
print(f"Success rate: {result.success_rate}")
print(f"Total time: {result.total_duration_seconds}s")
print(f"Successful jobs: {result.n_jobs_successful}/{result.n_jobs_total}")

# Individual job results
for job_result in result.job_results:
    print(f"{job_result.job_name}: {job_result.success}")
```

### Report Generation

```python
from skellysolver.batch import BatchReportGenerator

report_gen = BatchReportGenerator(batch_result=result)

# Save all reports
report_gen.save_all_reports(output_dir=Path("reports/"))

# Individual reports
report_gen.save_summary_csv(filepath=Path("summary.csv"))
report_gen.save_statistics_json(filepath=Path("stats.json"))
report_gen.save_html_report(filepath=Path("report.html"))
```

### Finding Best Parameters

```python
from skellysolver.batch import find_best_parameters

best_params = find_best_parameters(
    batch_result=result,
    metric="final_cost",  # or "cost_reduction"
)

print(f"Best parameters: {best_params}")
```

## IO API

### Readers

```python
from skellysolver.io.readers import (
    TidyCSVReader,
    WideCSVReader,
    DLCCSVReader,
    JSONReader,
    NPYReader,
)

reader = TidyCSVReader()
data = reader.read(filepath=Path("data.csv"))
```

### Writers

```python
from skellysolver.io.writers import (
    ResultsWriter,
    TrajectoryCSVWriter,
    JSONWriter,
    NPYWriter,
)

# Unified results writer
writer = ResultsWriter(output_dir=Path("output/"))
writer.save_rigid_body_results(
    result=result,
    noisy_data=noisy_data,
    marker_names=marker_names,
    topology_dict=topology.to_dict(),
    metrics=metrics,
    copy_viewer: bool = True,
)

# Individual format writers
csv_writer = TrajectoryCSVWriter()
csv_writer.write(
    filepath=Path("output.csv"),
    data={"noisy_data": noisy, "optimized_data": optimized},
)
```

### Viewers

```python
from skellysolver.io.viewers import (
    generate_rigid_body_viewer,
    generate_eye_tracking_viewer,
)

viewer_path = generate_rigid_body_viewer(
    output_dir=Path("output/"),
    data_csv_path=Path("output/trajectory_data.csv"),
    topology_json_path=Path("output/topology.json"),
    video_path: Path | None = None,
)
```

## Migration Guide

### From Old Rigid Body Code

**Before:**
```python
from python_code.rigid_body_tracker.io.loaders import load_trajectories
from python_code.rigid_body_tracker.process_rigid_body_trajectories import (
    run_ferret_skull_solver
)

trajectory_dict = load_trajectories(filepath=csv_path)
result = run_ferret_skull_solver(trajectory_dict, ...)
```

**After:**
```python
from skellysolver.pipelines.rigid_body import RigidBodyPipeline, RigidBodyConfig

config = RigidBodyConfig(input_path=csv_path, ...)
pipeline = RigidBodyPipeline(config=config)
result = pipeline.run()
```

### From Old Eye Tracking Code

**Before:**
```python
from python_code.eye_tracking.eye_data_loader import EyeTrackingData
from python_code.eye_tracking.eye_tracking_main import run_eye_tracking

data = EyeTrackingData.load_from_dlc_csv(filepath=csv_path)
result = run_eye_tracking(data, ...)
```

**After:**
```python
from skellysolver.pipelines.eye_tracking import EyeTrackingPipeline, EyeTrackingConfig

config = EyeTrackingConfig(input_path=csv_path, ...)
pipeline = EyeTrackingPipeline(config=config)
result = pipeline.run()
```

## Testing

Run verification scripts after installation:

```bash
python test_phase1.py  # Core optimization
python test_phase2.py  # Data layer
python test_phase3.py  # Pipelines
python test_phase4.py  # IO
python test_phase5.py  # Batch processing
```

## Performance Considerations

### Parallel Processing

Batch processing automatically uses multiprocessing when `parallel_mode="auto"` or `parallel_mode="parallel"`. Number of workers defaults to CPU count minus 1.

```python
batch_config = BatchConfig(
    ...,
    parallel_mode="parallel",
    max_workers=8,  # Explicit worker count
)
```

### Memory Management

For large datasets, reduce parallel workers:

```python
batch_config = BatchConfig(
    ...,
    max_workers=4,  # Lower memory usage
)
```

### Optimization Speed

Reduce iterations for faster processing:

```python
config = OptimizationConfig(
    max_iterations=100,  # Default: 300
)
```

## Troubleshooting

### Import Errors

Ensure all `__init__.py` files are present in the directory structure.

### Format Detection Failures

Manually specify format:
```python
from skellysolver.io.readers import TidyCSVReader

reader = TidyCSVReader()
data = reader.read(filepath=Path("data.csv"))
```

### Optimization Not Converging

Adjust solver parameters:
```python
config = OptimizationConfig(
    max_iterations=500,
    function_tolerance=1e-8,
    use_robust_loss=True,
)
```

### Batch Jobs Failing

Enable error continuation:
```python
batch_config = BatchConfig(
    ...,
    continue_on_error=True,
)
```

Check error messages in batch report:
```python
report_gen = BatchReportGenerator(batch_result=result)
report_gen.save_summary_csv(filepath=Path("summary.csv"))
```