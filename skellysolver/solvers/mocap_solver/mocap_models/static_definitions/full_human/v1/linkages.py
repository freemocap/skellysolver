"""
Refactored linkages - using actual LinkageABC instances
"""
from typing import ClassVar

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.linkage_abc import LinkageABC
from .keypoints import HumanKeypoints as kp
from .segments import HumanSegments as seg


class HumanLinkages(ABaseModel):
    """
    Container for all human skeleton linkages (joints).
    Each linkage connects segments via a shared keypoint.

    Note: There's a bug in linkage_abc.py - it defines linked_keypoint as list[KeypointABC]
    but should be a single KeypointABC. Working around this for now.
    """

    # Skull linkage - connects cervical spine to skull
    SKULL_C1: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.SPINE_CERVICAL,
        children=[
            seg.SKULL_NOSE,
            seg.SKULL_RIGHT_EYE_INNER,
            seg.SKULL_RIGHT_EYE_CENTER,
            seg.SKULL_RIGHT_EYE_OUTER,
            seg.SKULL_RIGHT_EAR,
            seg.SKULL_RIGHT_MOUTH,
            seg.SKULL_LEFT_EYE_INNER,
            seg.SKULL_LEFT_EYE_CENTER,
            seg.SKULL_LEFT_EYE_OUTER,
            seg.SKULL_LEFT_EAR,
            seg.SKULL_LEFT_MOUTH,
        ],
        linked_keypoint=[kp.SKULL_ORIGIN_FORAMEN_MAGNUM]  # Bug: should be single, not list
    )

    # Neck linkage - connects thoracic spine to cervical spine
    NECK_C7: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.SPINE_THORACIC,
        children=[
            seg.SPINE_CERVICAL,
            seg.RIGHT_CLAVICLE,
            seg.LEFT_CLAVICLE
        ],
        linked_keypoint=[kp.SPINE_CERVICAL_ORIGIN_C7]
    )

    # Chest linkage - connects lumbar spine to thoracic spine
    CHEST_T12: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.SPINE_SACRUM_LUMBAR,
        children=[seg.SPINE_THORACIC],
        linked_keypoint=[kp.SPINE_THORACIC_ORIGIN_T12]
    )

    # Right shoulder linkage
    RIGHT_SHOULDER: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.RIGHT_CLAVICLE,
        children=[seg.RIGHT_ARM_PROXIMAL],
        linked_keypoint=[kp.RIGHT_SHOULDER]
    )

    # Right elbow linkage
    RIGHT_ELBOW: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.RIGHT_ARM_PROXIMAL,
        children=[seg.RIGHT_ARM_DISTAL],
        linked_keypoint=[kp.RIGHT_ELBOW]
    )

    # Right wrist linkage
    RIGHT_WRIST: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.RIGHT_ARM_DISTAL,
        children=[
            seg.RIGHT_PALM_THUMB,
            seg.RIGHT_PALM_PINKY,
            seg.RIGHT_PALM_INDEX
        ],
        linked_keypoint=[kp.RIGHT_WRIST]
    )

    # Right hip linkage  
    # NOTE: Bug in original - used PELVIS_LEFT_HIP_ACETABULUM for right hip!
    RIGHT_HIP: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.PELVIS_RIGHT,
        children=[seg.RIGHT_LEG_THIGH],
        linked_keypoint=[kp.PELVIS_RIGHT_HIP_ACETABULUM]  # Fixed!
    )

    # Right knee linkage
    RIGHT_KNEE: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.RIGHT_LEG_THIGH,
        children=[seg.RIGHT_LEG_CALF],
        linked_keypoint=[kp.RIGHT_KNEE]
    )

    # Right ankle linkage
    RIGHT_ANKLE: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.RIGHT_LEG_CALF,
        children=[
            seg.RIGHT_FOOT_HEEL,
            seg.RIGHT_FOOT_FRONT
        ],
        linked_keypoint=[kp.RIGHT_ANKLE]
    )

    # Left shoulder linkage
    LEFT_SHOULDER: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.LEFT_CLAVICLE,
        children=[seg.LEFT_ARM_PROXIMAL],
        linked_keypoint=[kp.LEFT_SHOULDER]
    )

    # Left elbow linkage
    LEFT_ELBOW: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.LEFT_ARM_PROXIMAL,
        children=[seg.LEFT_ARM_DISTAL],
        linked_keypoint=[kp.LEFT_ELBOW]
    )

    # Left wrist linkage
    LEFT_WRIST: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.LEFT_ARM_DISTAL,
        children=[
            seg.LEFT_PALM_THUMB,
            seg.LEFT_PALM_PINKY,
            seg.LEFT_PALM_INDEX
        ],
        linked_keypoint=[kp.LEFT_WRIST]
    )

    # Left hip linkage
    LEFT_HIP: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.PELVIS_LEFT,
        children=[seg.LEFT_LEG_THIGH],
        linked_keypoint=[kp.PELVIS_LEFT_HIP_ACETABULUM]
    )

    # Left knee linkage
    LEFT_KNEE: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.LEFT_LEG_THIGH,
        children=[seg.LEFT_LEG_CALF],
        linked_keypoint=[kp.LEFT_KNEE]
    )

    # Left ankle linkage
    LEFT_ANKLE: ClassVar[LinkageABC] = LinkageABC(
        parent=seg.LEFT_LEG_CALF,
        children=[
            seg.LEFT_FOOT_HEEL,
            seg.LEFT_FOOT_FRONT
        ],
        linked_keypoint=[kp.LEFT_ANKLE]
    )

    @classmethod
    def get_all_linkages(cls) -> dict[str, LinkageABC]:
        """Get all linkages as a dictionary."""
        return {
            name: getattr(cls, name)
            for name in dir(cls)
            if not name.startswith("_") and isinstance(getattr(cls, name), LinkageABC)
        }