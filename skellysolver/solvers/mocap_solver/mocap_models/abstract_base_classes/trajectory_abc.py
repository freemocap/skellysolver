from abc import ABC
from dataclasses import dataclass

import numpy as np

from skellysolver.data.arbitrary_types_model import ABaseModel


@dataclass
class Trajectory(ABaseModel):
    """
    A Trajectory is a time series of 3D coordinates that represents the movement of a point over time
    """
    name: str
    trajectory_fr_xyz: np.ndarray

    def __post_init__(self):
        if not len(self.trajectory_fr_xyz.shape) == 2:
            raise ValueError("Data shape should be (frame, xyz)")
        if not self.trajectory_fr_xyz.shape[1] == 3:
            raise ValueError("Trajectory data should be 3D (xyz)")

        print(f"Created {self}")

    def __str__(self):
        out_str = f"Trajectory: {self.name} (shape: {self.trajectory_fr_xyz.shape})"
        return out_str

