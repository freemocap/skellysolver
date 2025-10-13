# SkellySolver Architecture Plan

## Directory Structure

```
skellysolver/
├── core/
│   ├── __init__.py
│   ├── cost_functions/              # NEW: Unified cost function library
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract base classes for costs
│   │   ├── measurement.py           # Data fitting costs (both pipelines)
│   │   ├── smoothness.py            # Temporal smoothness (rotation, translation, scalar)
│   │   ├── constraints.py           # Geometric constraints (rigid, soft distance)
│   │   └── manifolds.py             # Quaternion and other manifolds
│   ├── config.py                    # REFACTORED: Unified OptimizationConfig
│   ├── result.py                    # REFACTORED: Unified OptimizationResult
│   ├── optimizer.py                 # NEW: Generic pyceres optimizer wrapper
│   ├── topology.py                  # KEEP: Rigid body specific (unchanged)
│   ├── geometry.py                  # KEEP: Geometric utilities
│   ├── metrics.py                   # KEEP: Evaluation metrics
│   ├── chunking.py                  # KEEP: Chunking for long sequences
│   └── parallel.py                  # REFACTORED: Renamed from parallel_opt.py, made generic
│
├── models/                          # NEW: Domain-specific models
│   ├── __init__.py
│   ├── camera.py                    # Camera models (moved from eye_tracking/camera_model.py)
│   ├── eye_model.py                 # Eye model (moved from eye_tracking/)
│   └── rigid_body.py                # Rigid body model (extracted from topology.py)
│
├── data/                            # NEW: Unified data handling
│   ├── __init__.py
│   ├── base.py                      # Base data classes (Trajectory, Observation)
│   ├── formats.py                   # CSV format detection (consolidates loaders)
│   ├── loaders.py                   # REFACTORED: Single unified loader
│   ├── validators.py                # Data validation and filtering
│   └── preprocessing.py             # Common preprocessing (interpolation, etc.)
│
├── io/                              # REFACTORED: Clean separation
│   ├── __init__.py
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract reader interface
│   │   ├── csv_reader.py            # Generic CSV reading
│   │   ├── dlc_reader.py            # DeepLabCut CSV format
│   │   └── trajectory_reader.py     # Trajectory formats (tidy, wide)
│   ├── writers/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract writer interface
│   │   ├── csv_writer.py            # CSV output
│   │   ├── json_writer.py           # JSON output (topology, configs)
│   │   └── results_writer.py        # Unified results writing
│   └── viewers/
│       ├── __init__.py
│       ├── templates/
│       │   ├── rigid_body_viewer.html
│       │   └── eye_tracking_viewer.html
│       ├── base_viewer.py           # Base viewer generator
│       ├── rigid_body_viewer.py     # Rigid body specific
│       └── eye_tracking_viewer.py   # Eye tracking specific
│
├── pipelines/
│   ├── __init__.py
│   ├── base.py                      # NEW: BasePipeline abstract class
│   ├── rigid_body/
│   │   ├── __init__.py
│   │   ├── costs.py                 # Rigid body specific cost functions
│   │   ├── optimizer.py             # Rigid body optimization (refactored from rigid_body_optimization.py)
│   │   ├── pipeline.py              # REFACTORED: Main pipeline class
│   │   ├── config.py                # TrackingConfig (from process_rigid_body_trajectories.py)
│   │   └── examples/
│   │       ├── __init__.py
│   │       ├── ferret_head.py       # MOVED from root
│   │       └── synthetic_demo.py    # MOVED from root
│   └── eye_tracking/
│       ├── __init__.py
│       ├── costs.py                 # Eye tracking specific cost functions
│       ├── optimizer.py             # Eye tracking optimization (refactored)
│       ├── pipeline.py              # REFACTORED: Main pipeline class
│       ├── config.py                # Eye tracking configs
│       └── examples/
│           ├── __init__.py
│           └── basic_example.py     # NEW: Simple example
│
├── visualization/                   # NEW: Visualization tools
│   ├── __init__.py
│   └── blender/
│       ├── __init__.py
│       └── rigid_body_viz.py        # MOVED from io/blender_rigid_body_viz.py
│
└── batch/                           # NEW: Batch processing
    ├── __init__.py
    ├── processor.py                 # Batch processor framework
    ├── config.py                    # Batch configuration
    └── examples/
        └── __init__.py
```

## Key Changes & Rationale

### 1. Core Consolidation

**Before:**
- `rigid_body_optimization.py` - rigid body specific optimization
- `eye_pyceres_bundle_adjustment.py` - eye tracking optimization
- Duplicated cost functions, configs, results

**After:**
- `core/cost_functions/` - Unified cost function library
- `core/config.py` - Single `OptimizationConfig` class
- `core/result.py` - Single `OptimizationResult` class
- `core/optimizer.py` - Generic optimizer wrapper

**Benefits:**
- ✅ Eliminate duplicate cost functions (MeasurementFactor, RotationSmoothnessF, TranslationSmoothnessF)
- ✅ Single source of truth for optimization parameters
- ✅ Easier to add new constraints

### 2. Data Layer Unification

**Before:**
- `io/loaders.py` - trajectory loading
- `io/load_trajectories.py` - alternative trajectory loading
- `io/eye_data_loader.py` - eye tracking data loading
- Duplicated CSV parsing logic

**After:**
- `data/formats.py` - Format detection (tidy, wide, DLC)
- `data/loaders.py` - Single unified loader
- `data/base.py` - Common data structures
- `io/readers/` - Clean reader abstractions

**Benefits:**
- ✅ Single CSV parsing implementation
- ✅ Consistent data structures across pipelines
- ✅ Easy to add new formats

### 3. IO Separation

**Before:**
- `io/savers.py` - rigid body saving
- `io/eye_savers.py` - eye tracking saving
- Mixed responsibilities (reading, writing, viewer generation)

**After:**
- `io/readers/` - Reading only
- `io/writers/` - Writing only
- `io/viewers/` - Viewer generation only

**Benefits:**
- ✅ Clear single responsibility
- ✅ Easier testing
- ✅ Reusable components

### 4. Pipeline Framework

**Before:**
- `pipelines/rigid_body_tracker/process_rigid_body_trajectories.py` - monolithic
- `pipelines/eye_tracking/eye_tracking_main.py` - monolithic
- Similar patterns but duplicated code

**After:**
- `pipelines/base.py` - Abstract `BasePipeline` class
- `pipelines/rigid_body/pipeline.py` - Inherits from base
- `pipelines/eye_tracking/pipeline.py` - Inherits from base

**Benefits:**
- ✅ Shared pipeline infrastructure
- ✅ Consistent API across pipelines
- ✅ Easy to create new pipelines

## Implementation Phases

### Phase 1: Core Consolidation (HIGH PRIORITY)
1. Create `core/cost_functions/` with unified cost functions
2. Consolidate `OptimizationConfig` into `core/config.py`
3. Consolidate `OptimizationResult` into `core/result.py`
4. Create `core/optimizer.py` wrapper

**Files affected:** 15+ files
**Duplicates removed:** ~500 lines

### Phase 2: Data Layer (HIGH PRIORITY)
1. Create `data/` module structure
2. Consolidate CSV loading into `data/loaders.py`
3. Extract format detection to `data/formats.py`
4. Create common data structures in `data/base.py`

**Files affected:** 5+ files
**Duplicates removed:** ~300 lines

### Phase 3: IO Refactoring (MEDIUM PRIORITY)
1. Split readers into `io/readers/`
2. Split writers into `io/writers/`
3. Extract viewers to `io/viewers/`
4. Create base abstractions

**Files affected:** 10+ files
**Organization improved:** Clear separation of concerns

### Phase 4: Pipeline Framework (MEDIUM PRIORITY)
1. Create `pipelines/base.py`
2. Refactor rigid body pipeline
3. Refactor eye tracking pipeline
4. Extract examples

**Files affected:** 4+ files
**New features enabled:** Easy pipeline creation

### Phase 5: Batch Processing (LOW PRIORITY)
1. Create `batch/` module
2. Implement batch processor
3. Add batch configuration
4. Create examples

**Files affected:** New files
**New capability:** Process multiple datasets easily

## Migration Strategy

1. **Keep old files during refactoring** - Don't delete until new version works
2. **Create parallel structure** - Build new modules alongside old ones
3. **Gradual migration** - Move one component at a time
4. **Test each phase** - Ensure examples still work after each phase
5. **Update imports last** - Once everything works, update imports and remove old files

## API Design Principles

### Consistent Configuration
```python
# All pipelines use similar config structure
config = PipelineConfig(
    input_data=Path("data.csv"),
    output_dir=Path("output/"),
    optimization=OptimizationConfig(...),
    parallel=ParallelConfig(...),
)
```

### Consistent Pipeline Interface
```python
# All pipelines have same interface
pipeline = RigidBodyPipeline(config)
result = pipeline.run()
pipeline.save_results()
pipeline.generate_viewer()
```

### Consistent Data Flow
```python
# Standard pipeline steps
data = load_data(config.input_data)
validated = validate_data(data)
result = optimize(validated, config)
metrics = evaluate(result)
save(result, metrics, config.output_dir)
```

## Testing Strategy

1. **Unit tests** for each core module
2. **Integration tests** for each pipeline
3. **Regression tests** using existing examples
4. **Performance tests** for optimization speed

## Documentation Plan

1. **Architecture docs** (this document)
2. **API reference** (auto-generated from docstrings)
3. **User guide** with examples
4. **Migration guide** for existing users
5. **Developer guide** for contributors
