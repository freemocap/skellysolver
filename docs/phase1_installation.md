# Phase 1 Installation Instructions

## Quick Start

Copy all Phase 1 files into your project:

```bash
cd skellysolver/

# Create cost_functions directory
mkdir -p core/cost_functions

# Copy all files (shown in artifacts above)
# - core/cost_functions/__init__.py
# - core/cost_functions/base.py
# - core/cost_functions/smoothness.py
# - core/cost_functions/measurement.py
# - core/cost_functions/constraints.py
# - core/cost_functions/manifolds.py
# - core/config.py
# - core/result.py
# - core/optimizer.py
# - core/__init__.py (UPDATE existing)
```

## Detailed Installation Steps

### Step 1: Create Directory Structure

```bash
cd skellysolver/core/
mkdir -p cost_functions
touch cost_functions/__init__.py
```

### Step 2: Copy Files

I've created 10 complete files in the artifacts above:

1. **core/cost_functions/__init__.py** - Module exports
2. **core/cost_functions/base.py** - BaseCostFunction (200 lines)
3. **core/cost_functions/smoothness.py** - 3 smoothness costs (240 lines)
4. **core/cost_functions/measurement.py** - 4 measurement costs (280 lines)
5. **core/cost_functions/constraints.py** - 5 constraint costs (350 lines)
6. **core/cost_functions/manifolds.py** - Quaternion utilities (120 lines)
7. **core/config.py** - Unified configs (280 lines)
8. **core/result.py** - Unified results (380 lines)
9. **core/optimizer.py** - Generic optimizer (420 lines)
10. **core/__init__.py** - Core module exports (UPDATE existing)

### Step 3: Verify Installation

Create `test_phase1.py`:

```python
"""Verify Phase 1 installation."""

# Test imports
try:
    from skellysolver__.core import (
        OptimizationConfig,
        Optimizer,
        RotationSmoothnessCost,
        TranslationSmoothnessCost,
        ScalarSmoothnessCost,
        Point3DMeasurementCost,
        RigidEdgeCost,
        get_quaternion_manifold,
    )

    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test basic functionality
import numpy as np

config = OptimizationConfig(max_iterations=10)
optimizer = Optimizer(config=config)

quat = np.array([1.0, 0.0, 0.0, 0.0])
optimizer.add_quaternion_parameter(name="test", parameters=quat)

print(f"✓ Optimizer created with {optimizer.num_parameters()} parameters")
print("\n✓ Phase 1 installation verified!")
```

Run verification:
```bash
python test_phase1.py
```

### Step 4: Run Examples

Test with existing examples:

```python
"""Simple example using Phase 1 components."""

import numpy as np
from skellysolver__.core import (
    OptimizationConfig,
    Optimizer,
    RotationSmoothnessCost,
)

# Create optimizer
config = OptimizationConfig(max_iterations=100)
optimizer = Optimizer(config=config)

# Create some quaternions
n_frames = 10
quaternions = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(n_frames)]

# Add as parameters
for i, q in enumerate(quaternions):
    optimizer.add_quaternion_parameter(name=f"quat_{i}", parameters=q)

# Add smoothness costs
for i in range(n_frames - 1):
    cost = RotationSmoothnessCost(weight=100.0)
    optimizer.add_residual_block(
        cost=cost,
        parameters=[quaternions[i], quaternions[i + 1]]
    )

# Solve
print("Solving...")
result = optimizer.solve()
print(result.summary())
```

## Migration Examples

### Updating Rigid Body Pipeline

Find code like this in `rigid_body_optimization.py`:

```python
# OLD CODE (DELETE THIS)
class RotationSmoothnessFactor(pyceres.CostFunction):
    def __init__(self, *, weight: float = 500.0) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(4)
        self.set_parameter_block_sizes([4, 4])
    
    def Evaluate(self, parameters, residuals, jacobians):
        quat_t = parameters[0]
        quat_t1 = parameters[1]
        
        if np.dot(quat_t, quat_t1) < 0:
            quat_t1 = -quat_t1
        
        residuals[:] = self.weight * (quat_t1 - quat_t)
        
        # ... jacobian code ...
        return True
```

Replace with:

```python
# NEW CODE (USE THIS)
from skellysolver__.core import RotationSmoothnessCost

# That's it! Just use the cost function:
cost = RotationSmoothnessCost(weight=500.0)
```

### Updating Eye Tracking Pipeline

Find code like this in `eye_pyceres_bundle_adjustment.py`:

```python
# OLD CODE (DELETE THIS)
class ScaleSmoothnessCost(pyceres.CostFunction):
    def __init__(self, *, weight: float) -> None:
        super().__init__()
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1])
    
    def Evaluate(self, parameters, residuals, jacobians):
        scale_t = parameters[0][0]
        scale_t1 = parameters[1][0]
        residuals[0] = self.weight * (scale_t1 - scale_t)
        return True
```

Replace with:

```python
# NEW CODE (USE THIS)
from skellysolver__.core import ScalarSmoothnessCost

# That's it!
cost = ScalarSmoothnessCost(weight=5.0)
```

## Common Issues

### Issue 1: Import Errors

**Problem**: `ImportError: cannot import name 'RotationSmoothnessCost'`

**Solution**: Make sure you've copied ALL files, especially `core/cost_functions/__init__.py`

### Issue 2: pyceres Not Found

**Problem**: `ModuleNotFoundError: No module named 'pyceres'`

**Solution**: Install pyceres:
```bash
pip install pyceres
```

### Issue 3: Scipy Not Found

**Problem**: `ModuleNotFoundError: No module named 'scipy'`

**Solution**: Install scipy:
```bash
pip install scipy
```

## Dependencies

Phase 1 requires:
- `numpy` (already installed)
- `scipy` (for Rotation)
- `pyceres` (for optimization)

Install all at once:
```bash
pip install numpy scipy pyceres
```

## What Changed

### Before Phase 1:
```
rigid_body_optimization.py
├── RotationSmoothnessFactor (50 lines)
├── TranslationSmoothnessFactor (40 lines)
├── MeasurementFactorBA (80 lines)
├── RigidBodyFactorBA (70 lines)
└── OptimizationConfig (30 lines)

eye_pyceres_bundle_adjustment.py
├── RotationSmoothnessCost (50 lines)
├── ScaleSmoothnessCost (30 lines)
├── PupilPointCost (100 lines)
└── OptimizationConfig (40 lines)

TOTAL DUPLICATE CODE: ~500 lines
```

### After Phase 1:
```
core/
├── cost_functions/
│   ├── smoothness.py (240 lines, used by BOTH)
│   ├── measurement.py (280 lines, used by BOTH)
│   └── constraints.py (350 lines, used by BOTH)
├── config.py (280 lines, used by BOTH)
├── result.py (380 lines, used by BOTH)
└── optimizer.py (420 lines, used by BOTH)

TOTAL SHARED CODE: ~2400 lines
DUPLICATES ELIMINATED: ~500 lines
CODE QUALITY: ✓ Improved
```

## Success Criteria

Phase 1 is successfully installed when:

- ✅ All 10 files are in place
- ✅ `test_phase1.py` runs without errors
- ✅ Existing examples still work
- ✅ No import errors
- ✅ Can create Optimizer and add costs

## Next Actions

After Phase 1 is installed and tested:

1. **Update rigid body pipeline** to use new cost functions
2. **Update eye tracking pipeline** to use new cost functions
3. **Remove old duplicate code** from both pipelines
4. **Add tests** for new components
5. **Start Phase 2**: Data layer consolidation

## Need Help?

If you encounter issues:

1. Check that all files are in the correct locations
2. Verify imports work: `python test_phase1.py`
3. Check dependencies are installed: `pip list | grep pyceres`
4. Make sure you're using Python 3.10+

---

**Ready to install!** All files are complete and ready to use.
