from pydantic import model_validator

from skellysolver.solvers.constraints.base_constraint import BaseConstraint
from skellysolver.solvers.constraints.chain_constraint import ChainConstraint
from skellysolver.solvers.constraints.keypoint_constraint import KeypointConstraint
from skellysolver.solvers.constraints.linkage_constraint import LinkageConstraint
from skellysolver.solvers.constraints.segment_constraint import SegmentConstraint


class SkeletonConstraint(BaseConstraint):
    """
    Abstract base class for skeleton models. A skeleton model is a collection of related keypoints,
    segments, linkages, and chains that together define the structure of a skeleton.
    """
    keypoints: list[KeypointConstraint] = []
    segments: list[SegmentConstraint] = []
    linkages: list[LinkageConstraint] = []
    chains: list[ChainConstraint] = []

    @property
    def root(self) -> KeypointConstraint:
        return self.keypoints[0]

    @model_validator(mode="after")
    def validate(self) -> "SkeletonConstraint":
        if len(self.keypoints) == 0:
            raise ValueError("Skeleton model must have at least one keypoint.")

        # Bidirectional validation for keypoints <-> segments
        self._validate_no_orphaned_keypoints()
        self._validate_no_secret_keypoints()

        # Bidirectional validation for segments <-> linkages
        self._validate_no_orphaned_segments()
        self._validate_no_secret_segments()

        # Bidirectional validation for linkages <-> chains
        self._validate_no_secret_linkages()
        return self

    def _validate_no_orphaned_keypoints(self) -> None:
        """Ensure every keypoint in skeleton.keypoints is used in at least one segment."""
        keypoints_in_segments: list[KeypointConstraint] = []

        for segment in self.segments:
            keypoints_in_segments.append(segment.parent)
            keypoints_in_segments.extend(segment.children)

        orphaned: list[KeypointConstraint] = [kp for kp in self.keypoints if kp not in keypoints_in_segments]
        if orphaned:
            names = [kp.name for kp in orphaned]
            raise ValueError(
                f"Free-floating keypoints detected (not in any segment): {names}"
            )

    def _validate_no_secret_keypoints(self) -> None:
        """Ensure segments only reference keypoints from skeleton.keypoints."""
        for segment in self.segments:
            # Check root keypoint
            if segment.parent not in self.keypoints:
                raise ValueError(
                    f"SegmentConstraint '{segment.name}' has root keypoint '{segment.parent.name}' "
                    f"which is not in skeleton.keypoints (secret keypoint!)"
                )

            # Check all keypoints in segment
            for keypoint in segment.children:
                if keypoint not in self.keypoints:
                    raise ValueError(
                        f"SegmentConstraint '{segment.name}' references keypoint '{keypoint.name}' "
                        f"which is not in skeleton.keypoints (secret keypoint!)"
                    )

    def _validate_no_orphaned_segments(self) -> None:
        """Ensure every segment in skeleton.segments is used in at least one linkage."""
        segments_in_linkages: list[SegmentConstraint] = []

        for linkage in self.linkages:
            segments_in_linkages.append(linkage.parent)
            segments_in_linkages.extend(linkage.children)

        orphaned: list[SegmentConstraint] = [seg for seg in self.segments if seg not in segments_in_linkages]
        if orphaned:
            names = [seg.name for seg in orphaned]
            raise ValueError(
                f"Orphaned segments detected (not in any linkage): {names}"
            )

    def _validate_no_secret_segments(self) -> None:
        """Ensure linkages only reference segments from skeleton.segments."""
        for linkage in self.linkages:
            # Check parent segment
            if linkage.parent not in self.segments:
                raise ValueError(
                    f"LinkageConstraint '{linkage.name}' has parent segment '{linkage.parent.name}' "
                    f"which is not in skeleton.segments (secret segment!)"
                )

            # Check all child segments
            for segment in linkage.children:
                if segment not in self.segments:
                    raise ValueError(
                        f"LinkageConstraint '{linkage.name}' references child segment '{segment.name}' "
                        f"which is not in skeleton.segments (secret segment!)"
                    )

    def _validate_no_secret_linkages(self) -> None:
        """Ensure chains only reference linkages from skeleton.linkages."""
        for chain in self.chains:
            # Check parent linkage
            if chain.parent not in self.linkages:
                raise ValueError(
                    f"ChainConstraint '{chain.name}' has parent linkage '{chain.parent.name}' "
                    f"which is not in skeleton.linkages (secret linkage!)"
                )

            # Check all child linkages
            for linkage in chain.children:
                if linkage not in self.linkages:
                    raise ValueError(
                        f"ChainConstraint '{chain.name}' references child linkage '{linkage.name}' "
                        f"which is not in skeleton.linkages (secret linkage!)"
                    )