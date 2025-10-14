from abc import ABC, abstractmethod
from typing import List, Optional, Any, Self

import numpy as np
from pydantic import model_validator, Field

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.keypoint_abc import KeypointABC
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.keypoint_mapping_abc import TrackedToKeypointMapping


class SegmentABC(ABaseModel, ABC):
    """
    A Segment is a collection of keypoints that are linked together, such that the distance between them is constrained.
    A Segment has one keypoint that defines its origin, and at least one other keypoint that defines the orientation of the segment in 3D space.
    """
    keypoints: list[KeypointABC]
    rigid: float = Field(gt=0, le=1, default=1.0)  # 1.0 = fully rigid, 0.0 = fully flexible
    name: str

    @abstractmethod
    @property
    def origin(self) -> KeypointABC | TrackedToKeypointMapping:
        pass

    @abstractmethod
    @property
    def x_axis_reference(self) -> KeypointABC | TrackedToKeypointMapping | None:
        pass

    @abstractmethod
    @property
    def y_axis_reference(self) -> KeypointABC | TrackedToKeypointMapping | None:
        pass

    @abstractmethod
    @property
    def z_axis_reference(self) -> KeypointABC | TrackedToKeypointMapping | None:
        pass

    @model_validator(mode="after")
    def validate(self) -> Self:
        if len(self.keypoints) < 2:
            raise ValueError("A segment must have at least two keypoints")
        if self.origin not in self.keypoints:
            raise ValueError("The origin keypoint must be one of the segment's keypoints")
        if self.x_axis_reference and self.x_axis_reference not in self.keypoints:
            raise ValueError("The x_axis_reference keypoint must be one of the segment's keypoints")
        if self.y_axis_reference and self.y_axis_reference not in self.keypoints:
            raise ValueError("The y_axis_reference keypoint must be one of the segment's keypoints")
        if self.z_axis_reference and self.z_axis_reference not in self.keypoints:
            raise ValueError("The z_axis_reference keypoint must be one of the segment's keypoints")
        if not any([self.x_axis_reference, self.y_axis_reference, self.z_axis_reference]):
            raise ValueError("At least one of the axis reference keypoints must be defined to establish orientation")






class SimpleSegmentABC(SegmentABC):
    """
    A simple rigid body is a Segment consisting of Two and Only Two keypoints that are linked together, the distance between them is constant.
    The parent keypoint defines the origin of the rigid body, and the child keypoint is the end of the rigid body.
    The primary axis (+X) of the rigid body is the vector from the parent to the child, the secondary and tertiary axes (+Y, +Z) are undefined (i.e. we have enough information to define the pitch and yaw, but not the roll).
    """
    origin: KeypointABC
    z_axis_reference: KeypointABC

    @model_validator(mode="after")
    def validate(self) -> Self:

        if self.origin == self.z_axis_reference:
            raise ValueError("Parent and child keypoints must be different")
        print(f"SimpleSegment: {self.name} instantiated with parent {self.origin} and child {self.z_axis_reference}")
        return self


    def get_children(self) -> List[KeypointABC]:
        return [self.z_axis_reference]

    def __str__(self):
        out_str = f"Segment: {self.name}"
        out_str += f"\n\tParent: {self.origin}"
        return out_str



class CompoundSegmentABC(SimpleSegmentABC):
    """
    A composite rigid body is a collection of keypoints that are linked together, such that the distance between all keypoints is constant.
    The parent keypoint is the origin of the rigid body
    The primary and secondary axes must be defined in the class, and will be used to calculate the orthonormal basis of the rigid body
    """
    segments: List[SegmentABC]
    # origin: KeypointABC # This is inherited from SimpleSegmentABC
    # z_axis_reference: Optional[KeypointABC] # This is inherited from SimpleSegmentABC
    x_axis_reference: Optional[KeypointABC]
    y_axis_reference: Optional[KeypointABC]

    def __post_init__(self):
        if not np.sum([self.z_axis_reference, self.x_axis_reference, self.y_axis_reference]) >= 2:
            raise ValueError(
                "At least two of the reference keypoints must be provided to define a compound rigid body")

        print(f"CompoundSegment: {self.name} instantiated with parent {self.parent} and children {self.segments}")

    @classmethod
    def get_children(cls) -> List[KeypointABC]:
        children = []
        for segment in cls.segments:
            children.extend(segment.value.get_children())
        return children
