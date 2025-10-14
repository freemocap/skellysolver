import numpy as np
from pydantic import model_validator, BaseModel

from skellysolver.types.generic_types import KeypointNameString


class Keypoint(BaseModel):
    """
    A Keypoint is a named "key" location on a skeleton, used to define the position of a rigid body or linkage.
    In marker-based motion capture, keypoints could correspond to markers placed on the body.
    In markerless motion capture, keypoints could correspond to a tracked point in the image.
    When a Keypoint is hydrated with data, it becomes a Trajectory.

    `definition` is a human-oriented description of the keypoint's location (e.g. an anatomical
    description of a landmark on a bone).
    """
    name: str
    definition: str



class TrackedPoint(Keypoint):
    """
    A TrackedPoint is a Keypoint that corresponds directly to an observed point in the motion capture data.
    It defined by its name, and its definition in terms of a weighted combination of other keypoints.
    For example, a TrackedPoint could be defined as the midpoint between two other keypoints, or as a 1:1 mapping of
    a single keypoint (i.e. weight=1 for that keypoint for all others).

    'definition' is a human-oriented description of how the tracked point is computed from other keypoints.
    """
    name: str
    definition: str
    weights: dict[KeypointNameString, float]  # Mapping from other keypoint names to their weights in the computation

    @model_validator(mode="after")
    def validate(self):
        if np.sum(list(self.weights.values())) != 1:
            raise ValueError("The sum of the weights must be 1")

        return self
