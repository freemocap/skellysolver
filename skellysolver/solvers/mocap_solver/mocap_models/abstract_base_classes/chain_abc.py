from abc import ABC
from enum import Enum

from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.linkage_abc import LinkageABC
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.segments_abc import SegmentABC


class ChainABC(ABC):
    """
    A Chain is a set of linkages that are connected via shared Segments.
    """
    parent: LinkageABC
    children: list[LinkageABC]
    shared_segments: list[SegmentABC]

    @property
    def root(self) -> KeypointABC:
        # Chain -> Linkage -> Segment -> Keypoint
        return self.parent.value.root

    def get_name(self):
        return self.__class__.__name__

    def __post_init__(self):
        for body in self.shared_segments:
            if not any(body == linkage.value.origin for linkage in self.children):
                raise ValueError(f"Shared segment {body.name} not found in children {self.children}")
        print(
            f"Chain: {self.get_name()} instantiated with parent {self.parent} and children {[child.name for child in self.children]}")

    @classmethod
    def get_keypoints(cls) -> list[KeypointEnum]:
        keypoints = cls.parent.value.get_keypoints()
        for linkage in cls.children:
            keypoints.extend(linkage.value.get_keypoints())
        return keypoints

    @classmethod
    def get_segments(cls) -> list[Enum]:
        segments = cls.parent.value.get_segments()
        for linkage in cls.children:
            segments.extend(linkage.value.get_segments())
        return segments

    def get_linkages(self) -> list[Enum]:
        linkages = [self.parent]
        linkages.extend(self.children)
        return linkages
