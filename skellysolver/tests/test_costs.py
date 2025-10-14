"""Tests for core cost functions (Phase 1).

Tests all cost functions in skellysolver.core.cost_functions.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from skellysolver.core.cost_primatives import (
    RotationSmoothnessCost,
    TranslationSmoothnessCost,
    ScalarSmoothnessCost,
    Point3DMeasurementCost,
    RigidEdgeCost,
    SoftEdgeCost,
    ReferenceAnchorCost,
)


class TestRotationSmoothnessCost:
    """Test rotation smoothness cost function."""
    
    def test_identical_quaternions_zero_residual(self) -> None:
        """Identical quaternions should give zero residual."""
        quat_1 = np.array([1.0, 0.0, 0.0, 0.0])
        quat_2 = np.array([1.0, 0.0, 0.0, 0.0])
        
        cost = RotationSmoothnessCost(weight=100.0)
        residuals = np.zeros(4)
        
        cost.Evaluate([quat_1, quat_2], residuals, None)
        
        assert np.allclose(residuals, 0.0)
    
    def test_different_quaternions_nonzero_residual(self) -> None:
        """Different quaternions should give non-zero residual."""
        quat_1 = np.array([1.0, 0.0, 0.0, 0.0])
        quat_2 = np.array([0.9, 0.1, 0.0, 0.0])
        quat_2 = quat_2 / np.linalg.norm(quat_2)  # Normalize
        
        cost = RotationSmoothnessCost(weight=100.0)
        residuals = np.zeros(4)
        
        cost.Evaluate([quat_1, quat_2], residuals, None)
        
        assert not np.allclose(residuals, 0.0)
    
    def test_weight_applied(self) -> None:
        """Weight should be applied to residual."""
        quat_1 = np.array([1.0, 0.0, 0.0, 0.0])
        quat_2 = np.array([0.9, 0.1, 0.0, 0.0])
        
        # Test with weight=1
        cost_1 = RotationSmoothnessCost(weight=1.0)
        residuals_1 = np.zeros(4)
        cost_1.Evaluate([quat_1, quat_2], residuals_1, None)
        
        # Test with weight=100
        cost_100 = RotationSmoothnessCost(weight=100.0)
        residuals_100 = np.zeros(4)
        cost_100.Evaluate([quat_1, quat_2], residuals_100, None)
        
        # Residual should scale with weight
        assert np.allclose(residuals_100, residuals_1 * 100.0)
    
    def test_hemisphere_correction(self) -> None:
        """Should handle quaternion double cover (q and -q)."""
        quat_1 = np.array([1.0, 0.0, 0.0, 0.0])
        quat_2 = np.array([-1.0, 0.0, 0.0, 0.0])  # Opposite hemisphere
        
        cost = RotationSmoothnessCost(weight=1.0)
        residuals = np.zeros(4)
        
        cost.Evaluate([quat_1, quat_2], residuals, None)
        
        # Should correct to same hemisphere
        # Residual should be small (quaternions represent same rotation)
        assert np.linalg.norm(residuals) < 0.1


class TestTranslationSmoothnessCost:
    """Test translation smoothness cost function."""
    
    def test_identical_translations_zero_residual(self) -> None:
        """Identical translations should give zero residual."""
        trans_1 = np.array([1.0, 2.0, 3.0])
        trans_2 = np.array([1.0, 2.0, 3.0])
        
        cost = TranslationSmoothnessCost(weight=100.0)
        residuals = np.zeros(3)
        
        cost.Evaluate([trans_1, trans_2], residuals, None)
        
        assert np.allclose(residuals, 0.0)
    
    def test_different_translations_nonzero_residual(self) -> None:
        """Different translations should give non-zero residual."""
        trans_1 = np.array([1.0, 2.0, 3.0])
        trans_2 = np.array([1.5, 2.5, 3.5])
        
        cost = TranslationSmoothnessCost(weight=1.0)
        residuals = np.zeros(3)
        
        cost.Evaluate([trans_1, trans_2], residuals, None)
        
        expected = trans_2 - trans_1
        assert np.allclose(residuals, expected)
    
    def test_weight_applied(self) -> None:
        """Weight should be applied to residual."""
        trans_1 = np.array([1.0, 2.0, 3.0])
        trans_2 = np.array([1.1, 2.1, 3.1])
        
        cost = TranslationSmoothnessCost(weight=50.0)
        residuals = np.zeros(3)
        
        cost.Evaluate([trans_1, trans_2], residuals, None)
        
        expected = 50.0 * (trans_2 - trans_1)
        assert np.allclose(residuals, expected)


class TestScalarSmoothnessCost:
    """Test scalar smoothness cost function."""
    
    def test_identical_scalars_zero_residual(self) -> None:
        """Identical scalars should give zero residual."""
        scalar_1 = np.array([1.0])
        scalar_2 = np.array([1.0])
        
        cost = ScalarSmoothnessCost(weight=10.0)
        residuals = np.zeros(1)
        
        cost.Evaluate([scalar_1, scalar_2], residuals, None)
        
        assert np.allclose(residuals, 0.0)
    
    def test_different_scalars_nonzero_residual(self) -> None:
        """Different scalars should give non-zero residual."""
        scalar_1 = np.array([1.0])
        scalar_2 = np.array([1.5])
        
        cost = ScalarSmoothnessCost(weight=1.0)
        residuals = np.zeros(1)
        
        cost.Evaluate([scalar_1, scalar_2], residuals, None)
        
        expected = np.array([0.5])
        assert np.allclose(residuals, expected)
    
    def test_weight_applied(self) -> None:
        """Weight should be applied to residual."""
        scalar_1 = np.array([1.0])
        scalar_2 = np.array([2.0])
        
        cost = ScalarSmoothnessCost(weight=5.0)
        residuals = np.zeros(1)
        
        cost.Evaluate([scalar_1, scalar_2], residuals, None)
        
        expected = np.array([5.0])
        assert np.allclose(residuals, expected)


class TestPoint3DMeasurementCost:
    """Test 3D measurement cost function."""
    
    def test_perfect_measurement_zero_residual(self) -> None:
        """Perfect measurement (measured = predicted) gives zero residual."""
        measured = np.array([1.0, 2.0, 3.0])
        reference = np.array([0.0, 0.0, 0.0])
        
        # Identity rotation, translation = measured
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        translation = measured.copy()
        
        cost = Point3DMeasurementCost(
            measured_point=measured,
            reference_point=reference,
            weight=1.0
        )
        
        residuals = np.zeros(3)
        cost.Evaluate([quat, translation], residuals, None)
        
        assert np.allclose(residuals, 0.0, atol=1e-10)
    
    def test_measurement_error(self) -> None:
        """Measurement error should equal (measured - predicted)."""
        measured = np.array([2.0, 3.0, 4.0])
        reference = np.array([1.0, 0.0, 0.0])
        
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        translation = np.array([0.0, 0.0, 0.0])
        
        # Predicted = R @ reference + translation = [1, 0, 0]
        # Residual = measured - predicted = [2, 3, 4] - [1, 0, 0] = [1, 3, 4]
        
        cost = Point3DMeasurementCost(
            measured_point=measured,
            reference_point=reference,
            weight=1.0
        )
        
        residuals = np.zeros(3)
        cost.Evaluate([quat, translation], residuals, None)
        
        expected = np.array([1.0, 3.0, 4.0])
        assert np.allclose(residuals, expected)


class TestRigidEdgeCost:
    """Test rigid edge cost function."""
    
    def test_correct_distance_zero_residual(self) -> None:
        """Correct distance should give zero residual."""
        n_markers = 3
        reference_flat = np.array([
            0.0, 0.0, 0.0,  # Marker 0
            1.0, 0.0, 0.0,  # Marker 1
            0.0, 1.0, 0.0,  # Marker 2
        ])
        
        # Distance from marker 0 to marker 1 is 1.0
        cost = RigidEdgeCost(
            marker_i=0,
            marker_j=1,
            n_markers=n_markers,
            target_distance=1.0,
            weight=1.0
        )
        
        residuals = np.zeros(1)
        cost.Evaluate([reference_flat], residuals, None)
        
        assert np.allclose(residuals, 0.0, atol=1e-10)
    
    def test_incorrect_distance_nonzero_residual(self) -> None:
        """Incorrect distance should give non-zero residual."""
        n_markers = 2
        reference_flat = np.array([
            0.0, 0.0, 0.0,  # Marker 0
            2.0, 0.0, 0.0,  # Marker 1
        ])
        
        # Actual distance is 2.0, target is 1.0
        # Residual should be 2.0 - 1.0 = 1.0
        
        cost = RigidEdgeCost(
            marker_i=0,
            marker_j=1,
            n_markers=n_markers,
            target_distance=1.0,
            weight=1.0
        )
        
        residuals = np.zeros(1)
        cost.Evaluate([reference_flat], residuals, None)
        
        expected = np.array([1.0])
        assert np.allclose(residuals, expected, atol=1e-10)


class TestReferenceAnchorCost:
    """Test reference anchor cost function."""
    
    def test_no_drift_zero_residual(self) -> None:
        """No drift from initial should give zero residual."""
        initial = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        current = initial.copy()
        
        cost = ReferenceAnchorCost(initial_reference=initial, weight=1.0)
        residuals = np.zeros(len(initial))
        
        cost.Evaluate([current], residuals, None)
        
        assert np.allclose(residuals, 0.0)
    
    def test_drift_nonzero_residual(self) -> None:
        """Drift from initial should give non-zero residual."""
        initial = np.array([1.0, 2.0, 3.0])
        current = np.array([1.5, 2.5, 3.5])
        
        cost = ReferenceAnchorCost(initial_reference=initial, weight=1.0)
        residuals = np.zeros(len(initial))
        
        cost.Evaluate([current], residuals, None)
        
        expected = current - initial
        assert np.allclose(residuals, expected)


class TestCostFunctionIntegration:
    """Integration tests for cost functions."""
    
    def test_all_costs_importable(self) -> None:
        """All cost functions should be importable."""
        from skellysolver.core.cost_primatives import (
            BaseCostFunction,
            RotationSmoothnessCost,
            TranslationSmoothnessCost,
            ScalarSmoothnessCost,
            Point3DMeasurementCost,
            Point2DProjectionCost,
            RigidPoint3DMeasurementBundleAdjustment,
            SimpleDistanceCost,
            RigidEdgeCost,
            SoftEdgeCost,
            ReferenceAnchorCost,
        )
        
        # All should be classes
        assert callable(RotationSmoothnessCost)
        assert callable(TranslationSmoothnessCost)
        assert callable(ScalarSmoothnessCost)
    
    def test_cost_functions_have_evaluate(self) -> None:
        """All cost functions should have Evaluate method."""
        costs = [
            RotationSmoothnessCost(weight=1.0),
            TranslationSmoothnessCost(weight=1.0),
            ScalarSmoothnessCost(weight=1.0),
        ]
        
        for cost in costs:
            assert hasattr(cost, 'Evaluate')
            assert callable(cost.Evaluate)
