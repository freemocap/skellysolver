"""Typed wrapper for cost function information.

Replaces raw dictionaries with proper Pydantic models for type safety.
"""

from typing import Any, Literal
import numpy as np
import pyceres
from pydantic import Field, field_validator, model_validator

from skellysolver.data.arbitrary_types_model import ABaseModel


CostType = Literal[
    "segment_rigidity",
    "linkage_stiffness",
    "symmetry",
    "measurement",
    "rotation_smoothness",
    "translation_smoothness",
    "anchor",
    "chain_smoothness",
]


class CostInfo(ABaseModel):
    """Information about a single cost function in the optimization.

    This replaces raw dictionaries with a typed, validated model.

    Attributes:
        cost: The pyceres cost function
        parameters: List of parameter arrays this cost operates on
        description: Human-readable description
        cost_type: Category of cost (segment_rigidity, measurement, etc.)
        weight: Weight applied to this cost
        metadata: Additional type-specific metadata
    """

    cost: pyceres.CostFunction
    parameters: list[np.ndarray]
    description: str
    cost_type: CostType
    weight: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        """String representation."""
        return f"CostInfo[{self.cost_type}]: {self.description}"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"CostInfo(type={self.cost_type}, weight={self.weight:.2f}, "
            f"params={len(self.parameters)}, desc='{self.description}')"
        )

    @property
    def num_parameters(self) -> int:
        """Number of parameter blocks."""
        return len(self.parameters)

    @property
    def total_parameter_size(self) -> int:
        """Total size of all parameter blocks."""
        return sum(len(p) for p in self.parameters)


class SegmentRigidityCostInfo(CostInfo):
    """Cost info for segment rigidity constraints."""

    cost_type: Literal["segment_rigidity"] = "segment_rigidity"
    segment_name: str
    keypoint_i: str
    keypoint_j: str
    rigidity: float
    target_distance: float

    @model_validator(mode='before')
    @classmethod
    def set_description(cls, data: Any) -> Any:
        """Set description before validation."""
        if isinstance(data, dict):
            if 'description' not in data:
                data['description'] = (
                    f"RigidEdge[{data['segment_name']}]: {data['keypoint_i']} ↔ {data['keypoint_j']} "
                    f"(rigidity={data['rigidity']:.2f}, dist={data['target_distance']:.3f}m)"
                )
            if 'metadata' not in data:
                data['metadata'] = {
                    "segment_name": data['segment_name'],
                    "keypoint_i": data['keypoint_i'],
                    "keypoint_j": data['keypoint_j'],
                    "rigidity": data['rigidity'],
                    "target_distance": data['target_distance'],
                }
        return data


class LinkageStiffnessCostInfo(CostInfo):
    """Cost info for linkage stiffness constraints."""

    cost_type: Literal["linkage_stiffness"] = "linkage_stiffness"
    linkage_name: str
    keypoint_i: str
    keypoint_j: str
    stiffness: float
    target_distance: float

    @model_validator(mode='before')
    @classmethod
    def set_description(cls, data: Any) -> Any:
        """Set description before validation."""
        if isinstance(data, dict):
            if 'description' not in data:
                data['description'] = (
                    f"LinkageEdge[{data['linkage_name']}]: {data['keypoint_i']} ↔ {data['keypoint_j']} "
                    f"(stiffness={data['stiffness']:.2f}, dist={data['target_distance']:.3f}m)"
                )
            if 'metadata' not in data:
                data['metadata'] = {
                    "linkage_name": data['linkage_name'],
                    "keypoint_i": data['keypoint_i'],
                    "keypoint_j": data['keypoint_j'],
                    "stiffness": data['stiffness'],
                    "target_distance": data['target_distance'],
                }
        return data


class SymmetryCostInfo(CostInfo):
    """Cost info for symmetry constraints."""

    cost_type: Literal["symmetry"] = "symmetry"
    left_keypoint: str
    right_keypoint: str
    symmetry_plane: str

    @model_validator(mode='before')
    @classmethod
    def set_description(cls, data: Any) -> Any:
        """Set description before validation."""
        if isinstance(data, dict):
            if 'description' not in data:
                data['description'] = (
                    f"Symmetry: {data['left_keypoint']} ↔ {data['right_keypoint']} "
                    f"(plane={data['symmetry_plane']})"
                )
            if 'metadata' not in data:
                data['metadata'] = {
                    "left_keypoint": data['left_keypoint'],
                    "right_keypoint": data['right_keypoint'],
                    "symmetry_plane": data['symmetry_plane'],
                }
        return data


class MeasurementCostInfo(CostInfo):
    """Cost info for measurement fitting."""

    cost_type: Literal["measurement"] = "measurement"
    keypoint_name: str
    frame_index: int | None = None

    @model_validator(mode='before')
    @classmethod
    def set_description(cls, data: Any) -> Any:
        """Set description before validation."""
        if isinstance(data, dict):
            if 'description' not in data:
                frame_str = f" (frame {data['frame_index']})" if data.get('frame_index') is not None else ""
                data['description'] = f"Measurement: {data['keypoint_name']}{frame_str} (w={data['weight']:.2f})"
            if 'metadata' not in data:
                data['metadata'] = {
                    "keypoint_name": data['keypoint_name'],
                    "frame_index": data.get('frame_index'),
                }
        return data


class RotationSmoothnessCostInfo(CostInfo):
    """Cost info for rotation smoothness."""

    cost_type: Literal["rotation_smoothness"] = "rotation_smoothness"
    frame_from: int
    frame_to: int

    @model_validator(mode='before')
    @classmethod
    def set_description(cls, data: Any) -> Any:
        """Set description before validation."""
        if isinstance(data, dict):
            if 'description' not in data:
                data['description'] = f"RotationSmooth: frame {data['frame_from']} → {data['frame_to']}"
            if 'metadata' not in data:
                data['metadata'] = {
                    "frame_from": data['frame_from'],
                    "frame_to": data['frame_to'],
                }
        return data


class TranslationSmoothnessCostInfo(CostInfo):
    """Cost info for translation smoothness."""

    cost_type: Literal["translation_smoothness"] = "translation_smoothness"
    frame_from: int
    frame_to: int

    @model_validator(mode='before')
    @classmethod
    def set_description(cls, data: Any) -> Any:
        """Set description before validation."""
        if isinstance(data, dict):
            if 'description' not in data:
                data['description'] = f"TranslationSmooth: frame {data['frame_from']} → {data['frame_to']}"
            if 'metadata' not in data:
                data['metadata'] = {
                    "frame_from": data['frame_from'],
                    "frame_to": data['frame_to'],
                }
        return data


class AnchorCostInfo(CostInfo):
    """Cost info for reference anchor."""

    cost_type: Literal["anchor"] = "anchor"

    @model_validator(mode='before')
    @classmethod
    def set_description(cls, data: Any) -> Any:
        """Set description before validation."""
        if isinstance(data, dict):
            if 'description' not in data:
                data['description'] = f"ReferenceAnchor (weight={data['weight']:.2f})"
        return data


class CostCollection(ABaseModel):
    """Collection of cost functions with utility methods.

    Provides convenient filtering, grouping, and summary operations.
    """

    costs: list[CostInfo] = Field(default_factory=list)

    def add(self, *, cost_info: CostInfo) -> None:
        """Add a cost to the collection."""
        self.costs.append(cost_info)

    def extend(self, *, costs: list[CostInfo]) -> None:
        """Add multiple costs to the collection."""
        self.costs.extend(costs)

    def filter_by_type(self, *, cost_type: CostType) -> list[CostInfo]:
        """Get all costs of a specific type."""
        return [c for c in self.costs if c.cost_type == cost_type]

    def filter_by_frame(self, *, frame_index: int) -> list[CostInfo]:
        """Get all costs associated with a specific frame."""
        return [
            c for c in self.costs
            if isinstance(c, MeasurementCostInfo) and c.frame_index == frame_index
        ] + [
            c for c in self.costs
            if isinstance(c, (RotationSmoothnessCostInfo, TranslationSmoothnessCostInfo))
            and (c.frame_from == frame_index or c.frame_to == frame_index)
        ]

    def get_segment_costs(self, *, segment_name: str) -> list[SegmentRigidityCostInfo]:
        """Get all segment rigidity costs for a specific segment."""
        return [
            c for c in self.costs
            if isinstance(c, SegmentRigidityCostInfo) and c.segment_name == segment_name
        ]

    def get_linkage_costs(self, *, linkage_name: str) -> list[LinkageStiffnessCostInfo]:
        """Get all linkage stiffness costs for a specific linkage."""
        return [
            c for c in self.costs
            if isinstance(c, LinkageStiffnessCostInfo) and c.linkage_name == linkage_name
        ]

    @property
    def total_costs(self) -> int:
        """Total number of costs."""
        return len(self.costs)

    @property
    def total_weight(self) -> float:
        """Sum of all weights."""
        return sum(c.weight for c in self.costs)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        summary: dict[str, Any] = {
            "total_costs": self.total_costs,
            "total_weight": self.total_weight,
            "by_type": {},
        }

        for cost in self.costs:
            cost_type = cost.cost_type
            if cost_type not in summary["by_type"]:
                summary["by_type"][cost_type] = {
                    "count": 0,
                    "total_weight": 0.0,
                    "avg_weight": 0.0,
                }

            summary["by_type"][cost_type]["count"] += 1
            summary["by_type"][cost_type]["total_weight"] += cost.weight

        # Compute averages
        for cost_type, info in summary["by_type"].items():
            info["avg_weight"] = info["total_weight"] / info["count"]

        return summary

    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.get_summary()

        print("\n" + "="*80)
        print("COST SUMMARY")
        print("="*80)
        print(f"Total costs: {summary['total_costs']}")
        print(f"Total weight: {summary['total_weight']:.1f}")
        print("\nBy type:")

        for cost_type, info in summary["by_type"].items():
            print(
                f"  {cost_type:25s}: {info['count']:4d} costs, "
                f"weight={info['total_weight']:8.1f} "
                f"(avg={info['avg_weight']:.2f})"
            )
        print("="*80)

    def to_optimizer(self, *, optimizer: Any) -> None:
        """Add all costs to a pyceres optimizer.

        Args:
            optimizer: PyceresOptimizer instance
        """
        for cost_info in self.costs:
            optimizer.add_residual_block(
                cost=cost_info.cost,
                parameters=cost_info.parameters
            )