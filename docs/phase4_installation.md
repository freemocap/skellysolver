# Phase 4 Installation Instructions

## Quick Start

Copy all Phase 4 files into your project:

```bash
cd skellysolver/io/

# Create subdirectories
mkdir -p readers writers viewers
mkdir -p viewers/templates

# Copy all files (shown in artifacts above)
# - __init__.py (UPDATE existing)
# - legacy.py
# - readers/__init__.py
# - readers/base.py
# - readers/csv_reader.py
# - writers/__init__.py
# - writers/base.py
# - writers/csv_writer.py
# - writers/results_writer.py
# - viewers/__init__.py
# - viewers/base_viewer.py
# - viewers/rigid_body_viewer.py
```

## Detailed Installation Steps

### Step 1: Create Directory Structure

```bash
cd skellysolver/io/
mkdir -p readers writers viewers/templates
```

### Step 2: Copy Files

I've created 12 complete files in the artifacts above:

**Main IO:**
1. **io/__init__.py** - Module exports (110 lines)
2. **io/legacy.py** - Legacy compatibility (180 lines)

**Readers:**
3. **io/readers/__init__.py** - Module exports (40 lines)
4. **io/readers/base.py** - Abstract readers (280 lines)
5. **io/readers/csv_reader.py** - CSV readers (320 lines)

**Writers:**
6. **io/writers/__init__.py** - Module exports (60 lines)
7. **io/writers/base.py** - Abstract writers (240 lines)
8. **io/writers/csv_writer.py** - CSV writers (380 lines)
9. **io/writers/results_writer.py** - Unified writer (280 lines)

**Viewers:**
10. **io/viewers/__init__.py** - Module exports (50 lines)
11. **io/viewers/base_viewer.py** - Abstract viewers (220 lines)
12. **io/viewers/rigid_body_viewer.py** - Specialized viewers (200 lines)

Total: **~2,360 lines** of organized IO code!

### Step 3: Move Existing Templates

If you have existing viewer HTML files, move them:

```bash
# Move existing viewers to templates/
cd skellysolver/io/

# From pipelines/rigid_body/
mv ../pipelines/rigid_body/rigid_body_viewer.html viewers/templates/

# From pipelines/eye_tracking/
mv ../pipelines/eye_tracking/eye_tracking_viewer.html viewers/templates/
```

### Step 4: Verify Installation

Create `test_phase4.py`:

```python
"""Verify Phase 4 installation."""

# Test imports
try:
    from skellysolver__.io.readers import (
        TidyCSVReader,
        WideCSVReader,
        DLCCSVReader,
        JSONReader,
    )
    from skellysolver__.io.writers import (
        ResultsWriter,
        TrajectoryCSVWriter,
        JSONWriter,
    )
    from skellysolver__.io.viewers import (
        RigidBodyViewerGenerator,
        generate_rigid_body_viewer,
    )

    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test reader
import numpy as np
from pathlib import Path

print("\nTesting readers...")
reader = JSONReader()
print(f"✓ Created JSONReader: {reader}")

# Test writer
print("\nTesting writers...")
writer = ResultsWriter(output_dir=Path("test_output/"))
print(f"✓ Created ResultsWriter: {writer}")

# Test viewer generator
print("\nTesting viewers...")
viewer_gen = RigidBodyViewerGenerator()
print(f"✓ Created RigidBodyViewerGenerator")

print("\n✓ Phase 4 installation verified!")
```

Run verification:
```bash
python test_phase4.py
```

## Complete Integration Example

Here's **ALL 4 PHASES working together**:

```python
"""Complete example using all 4 phases."""

from pathlib import Path
import numpy as np

# Phase 2: Data
from skellysolver__.data import load_trajectories

# Phase 3: Pipeline
from skellysolver__.pipelines.rigid_body import (
    RigidBodyPipeline,
    RigidBodyConfig,
)

# Phase 1: Core
from skellysolver__.core import (
    OptimizationConfig,
    RigidBodyWeightConfig,
)
from skellysolver__.core.topology import RigidBodyTopology

# Phase 4: IO (used internally by pipeline, but can use directly)
from skellysolver__.io.viewers import generate_rigid_body_viewer

# ============================================================================
# DEFINE TOPOLOGY
# ============================================================================
topology = RigidBodyTopology(
    marker_names=["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    rigid_edges=[
        (0, 1), (0, 2), (1, 2),  # Face
        (1, 3), (2, 4),  # Eyes to ears
        (3, 4),  # Between ears
    ],
    name="ferret_head"
)

# ============================================================================
# CONFIGURE PIPELINE
# ============================================================================
config = RigidBodyConfig(
    input_path=Path("mocap_data.csv"),
    output_dir=Path("output/"),
    topology=topology,
    optimization=OptimizationConfig(
        max_iterations=300,
        use_robust_loss=True,
    ),
    weights=RigidBodyWeightConfig(),
    min_confidence=0.5,
    interpolate_missing_data=True,
)

# ============================================================================
# RUN PIPELINE
# ============================================================================
pipeline = RigidBodyPipeline(config=config)
result = pipeline.run()

# Everything is done!
# - Data loaded (Phase 2)
# - Optimized (Phase 1)
# - Saved (Phase 4)
# - Viewer generated (Phase 4)

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(result.summary())
pipeline.print_summary()
print(f"\n→ Open {config.output_dir / 'rigid_body_viewer.html'} to visualize!")
```

## Testing

### Unit Tests

Create `tests/io/test_readers.py`:

```python
from skellysolver__.io.readers import TidyCSVReader, JSONReader
from pathlib import Path
import tempfile


def test_tidy_reader(tmp_path):
    """Test tidy CSV reader."""
    # Create test CSV
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "frame,keypoint,x,y,z\n"
        "0,m1,1.0,2.0,3.0\n"
        "0,m2,4.0,5.0,6.0\n"
    )

    # Read
    reader = TidyCSVReader()
    data = reader.read(filepath=csv_path)

    assert data["format"] == "tidy"
    assert data["n_markers"] == 2
    assert "m1" in data["trajectories"]


def test_json_reader(tmp_path):
    """Test JSON reader."""
    import json

    # Create test JSON
    json_path = tmp_path / "test.json"
    json_path.write_text('{"key": "value"}')

    # Read
    reader = JSONReader()
    data = reader.read(filepath=json_path)

    assert data["key"] == "value"
```

Create `tests/io/test_writers.py`:

```python
from skellysolver__.io.writers import TrajectoryCSVWriter, JSONWriter
import numpy as np


def test_trajectory_writer(tmp_path):
    """Test trajectory CSV writer."""
    writer = TrajectoryCSVWriter()

    noisy = np.random.randn(10, 3, 3)
    optimized = np.random.randn(10, 3, 3)
    names = ["m1", "m2", "m3"]

    output_path = tmp_path / "output.csv"
    writer.write(
        filepath=output_path,
        data={
            "noisy_data": noisy,
            "optimized_data": optimized,
            "marker_names": names
        }
    )

    assert output_path.exists()


def test_json_writer(tmp_path):
    """Test JSON writer."""
    writer = JSONWriter()

    data = {"key": "value", "number": 42}
    output_path = tmp_path / "output.json"

    writer.write(filepath=output_path, data=data)

    assert output_path.exists()
```

## Common Issues

### Issue 1: Template Not Found

**Problem**: `FileNotFoundError: Template not found`

**Solution**: Make sure viewer templates are in `io/viewers/templates/`:
```bash
ls io/viewers/templates/
# Should show:
# - rigid_body_viewer.html
# - eye_tracking_viewer.html
```

If templates are missing, viewers will use fallback HTML generation.

### Issue 2: Import Errors

**Problem**: `ImportError: cannot import name 'ResultsWriter'`

**Solution**: Make sure all __init__.py files are in place:
```bash
ls io/
ls io/readers/
ls io/writers/
ls io/viewers/
# Each should have __init__.py
```

### Issue 3: Pandas Not Found

**Problem**: `ModuleNotFoundError: No module named 'pandas'`

**Solution**: Install pandas:
```bash
pip install pandas
```

## Dependencies

Phase 4 requires:
- `numpy` (already installed)
- `pandas` (for DataFrame operations in CSV writing)
- `pathlib` (built-in)
- `json` (built-in)
- `shutil` (built-in)

Install any missing:
```bash
pip install pandas
```

## What's Next?

After Phase 4:

1. ✅ **Phase 1**: Core optimization - DONE
2. ✅ **Phase 2**: Data layer - DONE
3. ✅ **Phase 3**: Pipeline framework - DONE
4. ✅ **Phase 4**: IO refactoring - DONE
5. **Phase 5**: Batch processing - OPTIONAL

You now have a **complete, production-ready optimization framework**! 🎉

---

**Phase 4 Status**: ✓ COMPLETE  
**Installation Time**: ~20 minutes  
**Testing Time**: ~10 minutes  
**Total**: ~30 minutes to organized IO  

**Ready to install!** 🚀
