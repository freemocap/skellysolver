from dataclasses import dataclass
from typing import List

import numpy as np

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.types.generic_types import TrackedPointNameString, KeypointMappingType


class TrackedToKeypointMapping(ABaseModel):
    """
    A KeypointMapping provides information on how to map a keypoint to data from a TrackingDataSource trajectory.
    It can represent:
     a single keypoint (maps to the keypoint)
     a list of keypoints (maps to the geometric mean of the keypoints),
     a dictionary of keypoints with weights (maps to the weighted sum of the tracked points), or
     a dictionary of keypoints with offsets (maps to the tracked point with an offset defined in the local reference frame of the Segment).

    params:
    tracked_points: List[TrackedPointName] - The list of tracked points that will be combined to create the mapping to a keypoint.
    weights: List[float] - The weights of the tracked points (must sum to 1 and have the same length as tracked_points).
    offset: Tuple[float, float, float] - The offset of the keypoint in the local reference from of the Segment. #TODO - implement and double check logic
    """
    tracked_points: List[TrackedPointNameString]
    weights: List[float]

    @classmethod
    def create(cls, mapping: KeypointMappingType):

        if isinstance(mapping, str):
            tracked_points = [mapping]
            weights = [1]
        elif isinstance(mapping, list):
            tracked_points = mapping
            weights = [1 / len(mapping)] * len(mapping)

        elif isinstance(mapping, dict):
            tracked_points = list(mapping.keys())
            weights = list(mapping.values())
        else:
            raise ValueError("Mapping must be a TrackedPointName, TrackedPointList, or WeightedTrackedPoints")

        if np.sum(weights) != 1:
            raise ValueError("The sum of the weights must be 1")
        if len(tracked_points) != len(weights):
            raise ValueError("The number of tracked points must match the number of weights")

        return cls(tracked_points=tracked_points, weights=weights)

    def calculate_trajectory(self, data_fr_name_xyz: np.ndarray, names: List[TrackedPointNameString]) -> np.ndarray:
        """
        Calculate a trajectory from a mapping of tracked points and their weights.
        """

        if data_fr_name_xyz.shape[1] != len(names):
            raise ValueError("Data shape does not match trajectory names length")
        if not all(tracked_point_name in names for tracked_point_name in self.tracked_points):
            raise ValueError("Not all tracked points in mapping found in trajectory names")

        number_of_frames = data_fr_name_xyz.shape[0]
        number_of_dimensions = data_fr_name_xyz.shape[2]
        trajectories_frame_xyz = np.zeros((number_of_frames, number_of_dimensions), dtype=np.float32)

        for tracked_point_name, weight in zip(self.tracked_points, self.weights):
            if tracked_point_name not in names:
                raise ValueError(f"Key {tracked_point_name} not found in trajectory names")

            keypoint_index = names.index(tracked_point_name)
            keypoint_fr_xyz = data_fr_name_xyz[:, keypoint_index, :]  # slice out the relevant tracked point
            trajectories_frame_xyz += keypoint_fr_xyz * weight  # scale the tracked point by the weight and add to the trajectory

        if np.sum(np.isnan(trajectories_frame_xyz)) == trajectories_frame_xyz.size:
            raise ValueError(f"Trajectory calculation resulted in all NaNs")

        return trajectories_frame_xyz
