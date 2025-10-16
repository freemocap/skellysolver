# Phase 2 Installation Instructions

## Quick Start

Copy all Phase 2 files into your project:

```bash
cd skellysolver/

# Create data directory
mkdir -p data

# Copy all files (shown in artifacts above)
# - data/__init__.py
# - data/base.py
# - data/formats.py
# - data/loaders.py
# - data/validators.py
# - data/preprocessing.py
```

## Detailed Installation Steps

### Step 1: Create Directory

```bash
cd skellysolver/
mkdir -p data
```

### Step 2: Copy Files

I've created 6 complete files in the artifacts above:

1. **data/__init__.py** - Module exports (120 lines)
2. **data/base.py** - Data structures (420 lines)
3. **data/formats.py** - Format detection (320 lines)
4. **data/loaders.py** - CSV loaders (560 lines)
5. **data/validators.py** - Validation (430 lines)
6. **data/preprocessing.py** - Preprocessing (550 lines)

Total: **~2,400 lines** of unified data handling code!

### Step 3: Verify Installation

Create `test_phase2.py`:

```python
"""Verify Phase 2 installation."""

# Test imports
try:
    from skellysolver__.data import (
        Trajectory3D,
        Observation2D,
        TrajectoryDataset,
        load_trajectories,
        validate_dataset,
        interpolate_missing,
        smooth_trajectories,
    )

    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test basic functionality
import numpy as np

# Create test trajectory
positions = np.random.randn(100, 3)
traj = Trajectory3D(
    marker_name="test",
    positions=positions
)

print(f"✓ Created trajectory with {traj.n_frames} frames")

# Create test dataset
from pathlib import Path

data = {
    "marker1": traj,
    "marker2": Trajectory3D(
        marker_name="marker2",
        positions=np.random.randn(100, 3)
    )
}

dataset = TrajectoryDataset(
    data=data,
    frame_indices=np.arange(100)
)

print(f"✓ Created dataset with {dataset.n_markers} markers")

# Test validation
report = validate_dataset(
    dataset=dataset,
    min_valid_frames=10
)

print(f"✓ Validation: {'PASSED' if report['valid'] else 'FAILED'}")
print("\n✓ Phase 2 installation verified!")
```

Run verification:
```bash
python test_phase2.py
```

### Step 4: Test with Real Data

Create `test_loading.py`:

```python
"""Test loading real data."""

from pathlib import Path
from skellysolver__.data import (
    load_trajectories,
    detect_csv_format,
    validate_dataset,
)

# Test with your actual CSV file
csv_path = Path("path/to/your/data.csv")

if csv_path.exists():
    # Detect format
    format_type = detect_csv_format(filepath=csv_path)
    print(f"Detected format: {format_type}")

    # Load
    dataset = load_trajectories(filepath=csv_path)
    print(f"Loaded: {dataset.n_markers} markers × {dataset.n_frames} frames")
    print(f"Markers: {dataset.marker_names}")

    # Validate
    report = validate_dataset(dataset=dataset)
    print(f"Valid: {report['valid']}")

    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  ⚠ {warning}")
else:
    print(f"Test file not found: {csv_path}")
    print("Create a test CSV or update the path")
```

## Complete Example

Here's a complete example showing Phase 1 + Phase 2 integration:

```python
"""Complete example: Load data and optimize with SkellySolver."""

import numpy as np
from pathlib import Path

# Phase 2: Load and preprocess data
from skellysolver__.data import (
    load_trajectories,
    validate_dataset,
    filter_by_confidence,
    interpolate_missing,
    smooth_trajectories,
)

# Phase 1: Optimize
from skellysolver__.core import (
    OptimizationConfig,
    RigidBodyWeightConfig,
    Optimizer,
    Point3DMeasurementCost,
    RigidEdgeCost,
)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("Loading data...")
dataset = load_trajectories(
    filepath=Path("mocap_data.csv"),
    scale_factor=0.001,  # mm to m
)

print(f"Loaded {dataset.n_markers} markers × {dataset.n_frames} frames")

# ============================================================================
# STEP 2: VALIDATE
# ============================================================================
print("\nValidating...")
report = validate_dataset(
    dataset=dataset,
    min_valid_frames=100,
    check_outliers=True
)

if not report["valid"]:
    print("⚠ Validation failed:")
    for error in report["errors"]:
        print(f"  {error}")
    exit(1)

print("✓ Validation passed")

# ============================================================================
# STEP 3: PREPROCESS
# ============================================================================
print("\nPreprocessing...")

# Filter low confidence
dataset = filter_by_confidence(
    dataset=dataset,
    min_confidence=0.5
)

# Interpolate missing
dataset = interpolate_missing(
    dataset=dataset,
    method="cubic",
    max_gap=10
)

# Smooth
dataset = smooth_trajectories(
    dataset=dataset,
    window_size=5,
    poly_order=2
)

print(f"✓ Preprocessed: {dataset.n_frames} frames remaining")

# ============================================================================
# STEP 4: PREPARE FOR OPTIMIZATION
# ============================================================================
print("\nPreparing for optimization...")

# Convert to array
marker_names = dataset.marker_names
noisy_data = dataset.to_array(marker_names=marker_names)
n_frames, n_markers, _ = noisy_data.shape

print(f"Data shape: {noisy_data.shape}")

# Initialize optimization parameters
reference_geometry = np.mean(noisy_data, axis=0)
reference_geometry -= np.mean(reference_geometry, axis=0)

quaternions = np.zeros((n_frames, 4))
quaternions[:, 0] = 1.0  # w=1 for identity

translations = np.zeros((n_frames, 3))

# ============================================================================
# STEP 5: OPTIMIZE (using Phase 1)
# ============================================================================
print("\nOptimizing...")

weights = RigidBodyWeightConfig()
config = OptimizationConfig(max_iterations=100)
optimizer = Optimizer(config=config)

# Add parameters
for i in range(n_frames):
    optimizer.add_quaternion_parameter(
        name=f"quat_{i}",
        parameter=quaternions[i]
    )
    optimizer.add_parameter_block(
        name=f"trans_{i}",
        parameters=translations[i]
    )

# Add measurement costs
for i in range(n_frames):
    for j in range(n_markers):
        cost = Point3DMeasurementCost(
            measured_point=noisy_data[i, j],
            reference_point=reference_geometry[j],
            weight=weights.lambda_data
        )
        optimizer.add_residual_block(
            cost=cost,
            parameters=[quaternions[i], translations[i]]
        )

# Solve
result = optimizer.solve()

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(result.summary())
```

## Common Migration Patterns

### Pattern 1: Replace Old Loader

**Before**:
```python
from python_code.rigid_body_tracker.io.loaders import load_trajectories

trajectory_dict = load_trajectories(
    filepath=input_csv,
    scale_factor=1.0,
    z_value=0.0
)

# Extract to array
marker_names = list(trajectory_dict.keys())
noisy_data = np.stack(
    [trajectory_dict[name] for name in marker_names],
    axis=1
)
```

**After**:

```python
from skellysolver__.data import load_trajectories

dataset = load_trajectories(
    filepath=input_csv,
    scale_factor=1.0,
    z_value=0.0
)

marker_names = dataset.marker_names
noisy_data = dataset.to_array()
```

### Pattern 2: Replace Eye Data Loader

**Before**:
```python
from eye_data_loader import EyeTrackingData

data = EyeTrackingData.load_from_dlc_csv(
    filepath=csv_path,
    min_confidence=0.3
)

data = data.filter_bad_frames(
    min_pupil_points=6,
    require_tear_duct=True
)

pupil_points = data.pupil_points_px
```

**After**:

```python
from skellysolver__.data import (
    load_trajectories,
    filter_by_confidence,
)

dataset = load_trajectories(
    filepath=csv_path,
    likelihood_threshold=0.3
)

dataset = filter_by_confidence(
    dataset=dataset,
    min_confidence=0.3,
    min_valid_markers=6
)

# Extract pupil points (assuming they're named p1-p8)
pupil_point_names = [f"p{i}" for i in range(1, 9)]
pupil_points = dataset.to_array(marker_names=pupil_point_names)
```

### Pattern 3: Add Validation

**Before**:
```python
# No validation - just hope data is good!
trajectory_dict = load_data()
```

**After**:

```python
from skellysolver__.data import load_trajectories, validate_dataset

dataset = load_trajectories(filepath=csv_path)

# Validate
report = validate_dataset(
    dataset=dataset,
    required_markers=["nose", "left_eye", "right_eye"],
    min_valid_frames=100
)

if not report["valid"]:
    raise ValueError(f"Data validation failed: {report['errors']}")
```

## Troubleshooting

### Issue 1: Format Not Detected

**Problem**: `ValueError: Unknown CSV format`

**Solution**: Check your CSV structure. Should be one of:
- Tidy: `frame, keypoint, x, y, z`
- Wide: `frame, marker1_x, marker1_y, marker1_z, ...`
- DLC: 3-row header with scorer/bodyparts/coords

### Issue 2: Wrong Data Type

**Problem**: `ValueError: Operations require 3D data`

**Solution**: Some operations (like `remove_outliers`) only work on 3D data. Check:
```python
if dataset.is_3d:
    dataset = remove_outliers(dataset=dataset)
else:
    print("Skipping outlier removal for 2D data")
```

### Issue 3: Interpolation Fails

**Problem**: `ValueError: Not enough points to interpolate`

**Solution**: Need at least 2 valid points. Check data quality:
```python
quality = check_data_quality(dataset=dataset)
for marker, stats in quality["marker_validity"].items():
    if stats["n_valid"] < 2:
        print(f"⚠ {marker} has insufficient valid points")
```

## What's Next?

After Phase 2 is installed and tested:

1. ✅ **Phase 1**: Core optimization (DONE)
2. ✅ **Phase 2**: Data layer (DONE)
3. **Phase 3**: Pipeline framework (NEXT)
4. **Phase 4**: IO refactoring
5. **Phase 5**: Batch processing

Ready for Phase 3? Let me know! 🚀

---

**Phase 2 Status**: ✓ COMPLETE
**Installation Time**: ~15 minutes
**Testing Time**: ~15 minutes
**Total**: ~30 minutes to full integration
