from pydantic import model_validator

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.data.primitive_objects.chain_model import Chain
from skellysolver.data.primitive_objects.keypoint_model import Keypoint
from skellysolver.data.primitive_objects.linkage_model import Linkage
from skellysolver.data.primitive_objects.segment_model import Segment


class Skeleton(ABaseModel):
    """
    Abstract base class for skeleton models. A skeleton model is a collection of related keypoints,
    segments, linkages, and chains that together define the structure of a skeleton.
    """
    name: str
    keypoints: list[Keypoint] = []
    segments: list[Segment] = []
    linkages: list[Linkage] = []
    chains: list[Chain] = []
    tracked_to_keypoint_mapping: dict[str, Keypoint] = {}

    @property
    def root(self) -> Keypoint:
        return self.keypoints[0]

    @model_validator(mode="after")
    def validate(self) -> "Skeleton":
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

        # Each keypoint has a tracked name mapping
        self._validate_tracked_keypoint_mapping()


        return self

    def _validate_tracked_keypoint_mapping(self) -> None:
        for keypoint in self.keypoints:
            if not keypoint in list(self.tracked_to_keypoint_mapping.values()):
                raise ValueError(
                    f"Keypoint '{keypoint.name}' has no tracked name mapping in "
                    f"skeleton.tracked_to_keypoint_mapping"
                )
    def _validate_no_orphaned_keypoints(self) -> None:
        """Ensure every keypoint in skeleton.keypoints is used in at least one segment."""
        keypoints_in_segments: list[Keypoint] = []

        for segment in self.segments:
            keypoints_in_segments.append(segment.root)
            keypoints_in_segments.extend(segment.keypoints)

        orphaned: list[Keypoint] = [kp for kp in self.keypoints if kp not in keypoints_in_segments]
        if orphaned:
            names = [kp.name for kp in orphaned]
            raise ValueError(
                f"Free-floating keypoints detected (not in any segment): {names}"
            )

    def _validate_no_secret_keypoints(self) -> None:
        """Ensure segments only reference keypoints from skeleton.keypoints."""
        for segment in self.segments:
            # Check root keypoint
            if segment.root not in self.keypoints:
                raise ValueError(
                    f"Segment '{segment.name}' has root keypoint '{segment.root.name}' "
                    f"which is not in skeleton.keypoints (secret keypoint!)"
                )

            # Check all keypoints in segment
            for keypoint in segment.keypoints:
                if keypoint not in self.keypoints:
                    raise ValueError(
                        f"Segment '{segment.name}' references keypoint '{keypoint.name}' "
                        f"which is not in skeleton.keypoints (secret keypoint!)"
                    )

    def _validate_no_orphaned_segments(self) -> None:
        """Ensure every segment in skeleton.segments is used in at least one linkage."""
        segments_in_linkages: list[Segment] = []

        for linkage in self.linkages:
            segments_in_linkages.append(linkage.parent)
            segments_in_linkages.extend(linkage.children)

        orphaned: list[Segment] = [seg for seg in self.segments if seg not in segments_in_linkages]
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
                    f"Linkage '{linkage.name}' has parent segment '{linkage.parent.name}' "
                    f"which is not in skeleton.segments (secret segment!)"
                )

            # Check all child segments
            for segment in linkage.children:
                if segment not in self.segments:
                    raise ValueError(
                        f"Linkage '{linkage.name}' references child segment '{segment.name}' "
                        f"which is not in skeleton.segments (secret segment!)"
                    )

    def _validate_no_secret_linkages(self) -> None:
        """Ensure chains only reference linkages from skeleton.linkages."""
        for chain in self.chains:
            # Check parent linkage
            if chain.parent not in self.linkages:
                raise ValueError(
                    f"Chain '{chain.name}' has parent linkage '{chain.parent.name}' "
                    f"which is not in skeleton.linkages (secret linkage!)"
                )

            # Check all child linkages
            for linkage in chain.children:
                if linkage not in self.linkages:
                    raise ValueError(
                        f"Chain '{chain.name}' references child linkage '{linkage.name}' "
                        f"which is not in skeleton.linkages (secret linkage!)"
                    )