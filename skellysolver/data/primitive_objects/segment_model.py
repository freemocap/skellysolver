from typing import Self

from pydantic import model_validator, Field, BaseModel

from skellysolver.data.primitive_objects.keypoint_model import Keypoint


class Segment(BaseModel):
    """
    A Segment is a collection of keypoints that are linked together, such that the distance between them is constrained.
    A Segment has one keypoint that defines its root, and at least one other keypoint that defines the orientation of the segment in 3D space.
    """
    keypoints: list[Keypoint]
    root: Keypoint
    rigidity: float = Field(gt=0, le=1, default=1.0)  # 1.0 = fully rigid (distance between keypoints constant), 0.0 = fully flexible (no constraint on distance between keypoints)
    name: str

    @model_validator(mode="after")
    def validate(self) -> Self:
        if len(self.keypoints) < 1:
            raise ValueError("A segment must have at least one keypoint")
        if self.root  in self.keypoints:
            raise ValueError("The root keypoint should not be included in the keypoints list")
        return self
