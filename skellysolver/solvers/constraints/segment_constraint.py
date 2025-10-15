from typing import Self

from pydantic import model_validator, Field

from skellysolver.solvers.constraints.base_constraint import BaseConstraint
from skellysolver.solvers.constraints.keypoint_constraint import KeypointConstraint


class SegmentConstraint(BaseConstraint):
    """
    A Segment is a collection of keypoints that are linked together, such that the distance between them is constrained.
    A Segment has one keypoint that defines its root, and at least one other keypoint that defines the orientation of the segment in 3D space.
    The rigidity parameter defines how rigidly the distances between the keypoints are constrained, with 1.0 being fully rigid (distances are constant) and 0.0 being fully flexible (no constraint on distances).
    """
    children: list[KeypointConstraint]
    parent: KeypointConstraint
    rigidity: float = Field(ge=0, le=1,
                            default=1.0)  # 1.0 = fully rigid (distance between keypoints constant), 0.0 = fully flexible (no constraint on distance between keypoints)

    @property
    def keypoints(self) -> list[KeypointConstraint]:
        return [self.parent] + self.children

    @model_validator(mode="after")
    def validate(self) -> Self:
        if len(self.children) < 1:
            raise ValueError("A segment must have at least one keypoint")
        if self.parent in self.children:
            raise ValueError("The root keypoint should not be included in the keypoints list")
        return self