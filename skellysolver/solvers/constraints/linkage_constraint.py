from pydantic import model_validator, Field
from typing_extensions import Self

from skellysolver.solvers.constraints.base_constraint import BaseConstraint
from skellysolver.solvers.constraints.keypoint_constraint import KeypointConstraint
from skellysolver.solvers.constraints.segment_constraint import SegmentConstraint


class LinkageConstraint(BaseConstraint):
    """
    A linkage comprises two or more SegmentConstraints that share a common KeypointConstraint. The segment closest to the root of the kinematic tree is called the 'parent' segment,
    and the others are 'children' segments.

    The distance from the linked keypoint is constrained relative to the keypoints in the same segment in accordance to the rigidity of the segment,

     but the distances between any two unlinked keypoints may vary on the range from 0 to the sum of the lengths of the involved segments.

     the 'stiffness' term defines how rigidly the segments are linked together, with 0 meaning no stiffness (the segments can move independently) and 1 meaning full rigidity (the segments move as a single unit, degenerate to a fully rigid SegmentConstraint).

    """
    parent: SegmentConstraint
    children: list[SegmentConstraint]
    linked_keypoint: KeypointConstraint
    stiffness: float = Field(ge=0.0, lt=1.0, default=0.0, description="Stiffness of the linkage, from 0 (no stiffness) to <1 (almost fully rigid - cannot be 1, as that would degenerate to a fully rigid SegmentConstraint)")


    @property
    def root(self) -> KeypointConstraint:
        return self.parent.parent

    @property
    def segments(self) -> list[SegmentConstraint]:
        return [self.parent] + self.children

    @model_validator(mode="after")
    def validate(self) -> Self:
        for segment in self.segments:
            # Check if linked_keypoint is either the root OR in the keypoints list
            if self.linked_keypoint != segment.parent and self.linked_keypoint not in segment.children:
                raise ValueError(
                    f"Linked keypoint {self.linked_keypoint} not found in segment {segment}. "
                    f"It must be either the segment's root or in its keypoints list."
                )
        return self