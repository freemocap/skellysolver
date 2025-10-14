from abc import ABC
from dataclasses import dataclass

from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.chain_abc import ChainABC
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.keypoint_abc import KeypointABC
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.linkage_abc import LinkageABC
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.segments_abc import SegmentABC


@dataclass
class SkeletonABC(ABC):
    """
    A Skeleton is composed of chains with connecting KeyPoints.
    """
    parent: ChainABC
    children: list[ChainABC]

    def get_name(self) -> str:
        return self.__class__.__name__

    @property
    def root(self) -> KeypointABC:
        # Skeleton -> Chain -> Linkage -> Segment -> Keypoint
        return self.parent.root

    @classmethod
    def get_linkages(cls) -> list[LinkageABC]:
        linkages = []
        linkages.extend(cls.parent.get_linkages())
        for chain in cls.children:
            linkages.extend(chain.get_linkages())
        return list(set(linkages))

    @classmethod
    def get_segments(cls) -> list[SegmentABC]:

        segments = []
        segments.extend(cls.parent.get_segments())
        for chain in cls.children:
            segments.extend(chain.get_segments())
        return list(set(segments))

    @classmethod
    def get_keypoints(cls) -> list[KeypointABC]:
        keypoints = []
        for chain in cls.children:
            keypoints.extend(chain.get_keypoints())
        return keypoints


    def get_keypoint_children(self, keypoint_name: str) -> list[KeypointABC]:
        """
        Recursively get all children keypoints for a given keypoint name.

        Parameters
        ----------
        keypoint_name : str
            The name of the keypoint to find children for.

        Returns
        -------
        Set[KeypointABC]
            A set of all children keypoints.
        """

        def recursive_find_children(name: str,
                                    segments: list[SegmentABC],
                                    found_keypoint_children: set[KeypointABC]) -> None:
            for segment in segments:
                if segment.origin.lower() == name:
                    children = segment.get_children()
                    for child in children:
                        if child not in found_keypoint_children:  # Avoid infinite recursion
                            found_keypoint_children.add(child)
                            recursive_find_children(name=child.lower(),
                                                    segments=segments,
                                                    found_keypoint_children=found_keypoint_children)

        found_keypoint_children = set()
        recursive_find_children(name=keypoint_name,
                                segments=self.get_segments(),
                                found_keypoint_children=found_keypoint_children)
        return list(found_keypoint_children)
