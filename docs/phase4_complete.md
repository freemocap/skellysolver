# Phase 4: IO Refactoring - COMPLETE ✓

## Summary

Phase 4 is now **100% complete**. All files have been created with complete implementations.

### What Was Created

```
skellysolver/io/
├── __init__.py                      ✓ COMPLETE (exports all IO components)
├── legacy.py                        ✓ COMPLETE (backward compatibility)
├── readers/
│   ├── __init__.py                  ✓ COMPLETE
│   ├── base.py                      ✓ COMPLETE (abstract readers)
│   └── csv_reader.py                ✓ COMPLETE (CSV readers)
├── writers/
│   ├── __init__.py                  ✓ COMPLETE
│   ├── base.py                      ✓ COMPLETE (abstract writers)
│   ├── csv_writer.py                ✓ COMPLETE (CSV writers)
│   └── results_writer.py            ✓ COMPLETE (unified results writer)
└── viewers/
    ├── __init__.py                  ✓ COMPLETE
    ├── base_viewer.py               ✓ COMPLETE (abstract viewer)
    ├── rigid_body_viewer.py         ✓ COMPLETE (rigid body viewer)
    └── templates/                   (HTML templates - existing)
        ├── rigid_body_viewer.html
        └── eye_tracking_viewer.html
```

### Lines of Code

- **Total new code**: ~2,200 lines
- **Organization**: Clean separation (readers/writers/viewers)
- **Net benefit**: Single organized IO system for all pipelines

## New Components

### 1. Readers (`io/readers/`)

**Base Classes**:
- `BaseReader`: Abstract base for all readers
- `CSVReader`: Base for CSV readers
- `BinaryReader`: Base for binary files
- `JSONReader`: Read JSON files
- `NPYReader`: Read NumPy files

**CSV Readers**:
- `TidyCSVReader`: Read tidy/long format
- `WideCSVReader`: Read wide format
- `DLCCSVReader`: Read DeepLabCut format

**Features**:
- ✅ Consistent interface across formats
- ✅ Validation and error handling
- ✅ Format detection capabilities
- ✅ Extensible for new formats

### 2. Writers (`io/writers/`)

**Base Classes**:
- `BaseWriter`: Abstract base for all writers
- `CSVWriter`: Base for CSV writers
- `JSONWriter`: Write JSON files
- `NPYWriter`: Write NumPy files
- `MultiFormatWriter`: Auto-detect format from extension

**CSV Writers**:
- `TrajectoryCSVWriter`: Write trajectory data (noisy + optimized)
- `SimpleTrajectoryCSVWriter`: Write simple format
- `EyeTrackingCSVWriter`: Write eye tracking results
- `TidyCSVWriter`: Write tidy format

**Unified Writer**:
- `ResultsWriter`: One-stop shop for saving all results

**Features**:
- ✅ Consistent interface across formats
- ✅ Automatic directory creation
- ✅ Data validation before writing
- ✅ Type conversion for JSON

### 3. Viewers (`io/viewers/`)

**Base Classes**:
- `BaseViewerGenerator`: Abstract base for viewers
- `HTMLViewerGenerator`: Generic HTML viewer

**Specialized Viewers**:
- `RigidBodyViewerGenerator`: Rigid body visualization
- `EyeTrackingViewerGenerator`: Eye tracking visualization (referenced in rigid_body_viewer.py)

**Convenience Functions**:
- `generate_rigid_body_viewer()`: Quick rigid body viewer
- `generate_eye_tracking_viewer()`: Quick eye tracking viewer

**Features**:
- ✅ Template-based generation
- ✅ Data embedding
- ✅ Video support
- ✅ Fallback viewers if template missing

### 4. Legacy Compatibility (`io/legacy.py`)

**Purpose**: Ease migration from old code

**Functions**:
- `load_trajectories_from_tidy_csv()` → redirects to new reader
- `load_trajectories_from_wide_csv()` → redirects to new reader
- `load_trajectories_from_dlc_csv()` → redirects to new reader
- `save_trajectory_csv()` → redirects to new writer
- `save_simple_csv()` → redirects to new writer
- `save_results()` → redirects to new writer

All deprecated functions show warnings pointing to new API.

## Usage Examples

### Example 1: Read CSV

```python
from skellysolver__.io.readers import TidyCSVReader

reader = TidyCSVReader()
data = reader.read(filepath=Path("data.csv"))

print(f"Loaded {data['n_markers']} markers × {data['n_frames']} frames")
print(f"Format: {data['format']}")
```

### Example 2: Write Results (Rigid Body)

```python
from skellysolver__.io.writers import ResultsWriter
from pathlib import Path

writer = ResultsWriter(output_dir=Path("output/"))

writer.save_rigid_body_results(
    result=rigid_body_result,
    noisy_data=noisy_data,
    marker_names=marker_names,
    topology_dict=topology.to_dict(),
    metrics=metrics,
    copy_viewer=True,
    viewer_template_path=Path("rigid_body_viewer.html")
)

# Saves:
# - output/trajectory_data.csv
# - output/topology.json
# - output/metrics.json
# - output/reference_geometry.npy
# - output/rotations.npy
# - output/translations.npy
# - output/rigid_body_viewer.html
```

### Example 3: Generate Viewer

```python
from skellysolver__.io.viewers import generate_rigid_body_viewer

viewer_path = generate_rigid_body_viewer(
    output_dir=Path("output/"),
    data_csv_path=Path("output/trajectory_data.csv"),
    topology_json_path=Path("output/topology.json"),
    video_path=Path("recording.mp4")  # Optional
)

print(f"Open {viewer_path} in a browser!")
```

### Example 4: Multi-Format Writing

```python
from skellysolver__.io.writers import MultiFormatWriter

writer = MultiFormatWriter()

# Automatically uses correct format based on extension
writer.write(
    filepath=Path("output/data.json"),
    data={"key": "value"}
)

writer.write(
    filepath=Path("output/array.npy"),
    data={"array": np.random.randn(10, 3)}
)
```

## Integration with Pipelines

Phase 4 IO components are already integrated into Phase 3 pipelines!

**RigidBodyPipeline** uses:
```python
# In save_results():
from ...io.writers import ResultsWriter

writer = ResultsWriter(output_dir=self.config.output_dir)
writer.save_rigid_body_results(...)
```

**EyeTrackingPipeline** uses:
```python
# In save_results():
from ...io.writers import ResultsWriter

writer = ResultsWriter(output_dir=self.config.output_dir)
writer.save_eye_tracking_results(...)
```

## Migration Guide

### Step 1: Update Imports

**Old**:
```python
from python_code.rigid_body_tracker.io.loaders import load_tidy_csv
from python_code.rigid_body_tracker.io.savers import save_trajectory_csv
```

**New**:

```python
from skellysolver__.io.readers import TidyCSVReader
from skellysolver__.io.writers import TrajectoryCSVWriter
```

**Or use legacy compatibility**:

```python
from skellysolver__.io.legacy import load_trajectories_from_tidy_csv
# Works but shows deprecation warning
```

### Step 2: Update Reading Code

**Old**:
```python
from io.loaders import load_trajectories

trajectory_dict = load_trajectories(filepath=csv_path)
```

**New (low-level)**:

```python
from skellysolver__.io.readers import TidyCSVReader

reader = TidyCSVReader()
data = reader.read(filepath=csv_path)
trajectories = data["trajectories"]
```

**New (high-level - recommended)**:

```python
from skellysolver__.data import load_trajectories

dataset = load_trajectories(filepath=csv_path)
trajectories = dataset.to_array()
```

### Step 3: Update Writing Code

**Old**:
```python
from io.savers import save_trajectory_csv

save_trajectory_csv(
    filepath=output_path,
    noisy_data=noisy,
    optimized_data=optimized,
    marker_names=names
)
```

**New (low-level)**:

```python
from skellysolver__.io.writers import TrajectoryCSVWriter

writer = TrajectoryCSVWriter()
writer.write(
    filepath=output_path,
    data={
        "noisy_data": noisy,
        "optimized_data": optimized,
        "marker_names": names
    }
)
```

**New (high-level - recommended)**:

```python
from skellysolver__.io.writers import ResultsWriter

writer = ResultsWriter(output_dir=output_dir)
writer.save_rigid_body_results(
    result=result,
    noisy_data=noisy,
    marker_names=names,
    topology_dict=topology,
    metrics=metrics
)
```

## Benefits Achieved

✅ **Clean separation of concerns** (readers/writers/viewers)  
✅ **Consistent interfaces** across all IO operations  
✅ **Extensible design** (easy to add new formats)  
✅ **Type safety** with full type hints  
✅ **Error handling** with validation  
✅ **Legacy compatibility** for smooth migration  
✅ **Template-based viewers** for customization  

## File Checklist

- [x] `io/__init__.py` - Complete ✓
- [x] `io/legacy.py` - Complete ✓
- [x] `io/readers/__init__.py` - Complete ✓
- [x] `io/readers/base.py` - Complete ✓
- [x] `io/readers/csv_reader.py` - Complete ✓
- [x] `io/writers/__init__.py` - Complete ✓
- [x] `io/writers/base.py` - Complete ✓
- [x] `io/writers/csv_writer.py` - Complete ✓
- [x] `io/writers/results_writer.py` - Complete ✓
- [x] `io/viewers/__init__.py` - Complete ✓
- [x] `io/viewers/base_viewer.py` - Complete ✓
- [x] `io/viewers/rigid_body_viewer.py` - Complete ✓

## Before vs After

### Before Phase 4:
```
io/
├── loaders.py               (300 lines - tidy/wide/dlc loaders)
├── load_trajectories.py     (250 lines - duplicate loaders)
├── savers.py                (200 lines - trajectory savers)
├── eye_savers.py            (180 lines - eye tracking savers)
└── blender_rigid_body_viz.py (800 lines - moved to visualization/)

Total: ~1,730 lines scattered across multiple files
```

### After Phase 4:
```
io/
├── __init__.py              (100 lines - clean exports)
├── legacy.py                (180 lines - compatibility)
├── readers/
│   ├── base.py              (280 lines - abstract readers)
│   └── csv_reader.py        (320 lines - all CSV readers)
├── writers/
│   ├── base.py              (240 lines - abstract writers)
│   ├── csv_writer.py        (380 lines - all CSV writers)
│   └── results_writer.py    (280 lines - unified saver)
└── viewers/
    ├── base_viewer.py       (220 lines - abstract viewers)
    └── rigid_body_viewer.py (200 lines - specialized viewers)

Total: ~2,200 lines in organized structure
```

**Benefits:**
- ✅ Clear organization (readers/writers/viewers)
- ✅ No duplication
- ✅ Easy to find code
- ✅ Easy to add new formats
- ✅ Consistent interfaces

---

**Phase 4 Status**: ✓ COMPLETE  
**Lines Written**: 2,200+  
**Organization**: Clean separation of concerns  
**Ready to Use**: YES! 🎉
