from abc import ABC

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.types.generic_types import KeypointNameString


class KeypointABC(ABaseModel, ABC):
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

    def __str__(self):
        return f"Keypoint: {self.name}"

class VirtualKeypoint(KeypointABC):
    """
    A VirtualKeypoint is a Keypoint that is not directly observed, but is instead computed from other keypoints.
    For example, a VirtualKeypoint could be the midpoint between two observed keypoints.
    """
    weights: dict[KeypointNameString, Key]  # Mapping from other keypoint names to their weights in the computation


