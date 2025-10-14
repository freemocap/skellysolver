from pydantic import model_validator, Field, BaseModel
from typing_extensions import Self

from skellysolver.data.primitive_objects.keypoint_model import Keypoint
from skellysolver.data.primitive_objects.segment_model import Segment


class Linkage(BaseModel):
    """
    A linkage comprises two or more Segments that share a common Keypoint. The segment closest to the root of the kinematic tree is called the 'parent' segment,
    and the others are 'children' segments.

    The distance from the linked keypoint is constrained relative to the keypoints in the same segment in accordance to the rigidity of the segment,

     but the distances between any two unlinked keypoints may vary on the range from 0 to the sum of the lengths of the involved segments.

     the 'stiffness' term defines how rigidly the segments are linked together, with 0 meaning no stiffness (the segments can move independently) and 1 meaning full rigidity (the segments move as a single unit, degenerate to a fully rigid Segment).

    """
    name: str
    parent: Segment
    children: list[Segment]
    linked_keypoint: Keypoint
    stiffness: float = Field(ge=0.0, lt=1.0, default=0.0, description="Stiffness of the linkage, from 0 (no stiffness) to <1 (almost fully rigid - cannot be 1, as that would degenerate to a fully rigid Segment)")


    @property
    def root(self) -> Keypoint:
        return self.parent.root

    @property
    def segments(self) -> list[Segment]:
        return [self.parent] + self.children

    @model_validator(mode="after")
    def validate(self) -> Self:
        for segment in self.segments:
            # Check if linked_keypoint is either the root OR in the keypoints list
            if self.linked_keypoint != segment.root and self.linked_keypoint not in segment.keypoints:
                raise ValueError(
                    f"Linked keypoint {self.linked_keypoint} not found in segment {segment}. "
                    f"It must be either the segment's root or in its keypoints list."
                )
        return self