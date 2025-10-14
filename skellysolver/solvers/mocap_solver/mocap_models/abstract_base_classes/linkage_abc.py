from abc import ABC
from dataclasses import dataclass

from pydantic import model_validator
from typing_extensions import Self

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.keypoint_abc import KeypointABC
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.segments_abc import SegmentABC, \
    CompoundSegmentABC, SimpleSegmentABC



class LinkageABC(ABaseModel, ABC):
    """
    A linkage comprises two or more Segments that share a common Keypoint. One segment is the 'parent' segment,
    and the others are 'children' segments. The parent segment is the segment closest to the root of the kinematic tree.

    The distance from the linked keypoint is constrained relative to the keypoints in the same segment in accordance to the rigidity of the segment,
     but the distances between the unlinked keypoints may vary on the range from 0 to the sum of the lengths of the segments.

     #for now these are all 'universal' (ball) joints. Later we can add different constraints
    """
    parent: SegmentABC
    children: list[SegmentABC]
    linked_keypoint: list[KeypointABC]

    def get_name(self) -> str:
        return self.__class__.__name__

    @property
    def root(self) -> KeypointABC:
        return self.parent.root

    @model_validator(mode="after")
    def validate(self) -> Self:
        for segment in [self.parent] + self.children:
            if isinstance(segment, SimpleSegmentABC):
                if self.linked_keypoint not in [segment.origin, segment.z_axis_reference]:
                    raise ValueError(
                        f"Error instantiation Linkage: {self.get_name()} - Common keypoint {self.linked_keypoint} not found in segment {segment}")
            elif isinstance(segment, CompoundSegmentABC):
                if self.linked_keypoint not in [segment.parent] + [child for child in segment.children]:
                    raise ValueError(
                        f"Error instantiation Linkage: {self.get_name()} - Common keypoint {self.linked_keypoint} not found in segment {segment}")
            else:
                raise ValueError(f"Body {segment} is not a valid rigid segment type")
        print(f"Linkage: {self.get_name()} instantiated with parent {self.parent} and children {self.children}")
        return self
    @property
    def get_segments(self) -> list[SegmentABC]:
        segments = [self.parent] + self.children
        return segments

    @property
    def get_keypoints(self) -> list[SegmentABC]:
        keypoints = self.parent.get_keypoints()
        for linkage in self.children:
            keypoints.extend(linkage.get_keypoints())
        return keypoints

    def __str__(self) -> str:
        out_str = super().__str__()
        out_str += "\n\t".join(f"Common Keypoints: {self.linked_keypoint}\n")
        return out_str
