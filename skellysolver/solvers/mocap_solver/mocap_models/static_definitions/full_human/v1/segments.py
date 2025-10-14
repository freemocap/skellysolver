"""
Refactored segments - using actual SegmentABC instances
"""
from typing import ClassVar

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.segments_abc import (
    SimpleSegmentABC,
    CompoundSegmentABC
)
from .keypoints import HumanKeypoints as kp


class HumanSegments(ABaseModel):
    """
    Container for all human skeleton segments.
    Each segment is an instance of SimpleSegmentABC or CompoundSegmentABC.
    """

    # Skull segments (all radiate from skull origin)
    SKULL_NOSE: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.NOSE_TIP
    )

    SKULL_TOP: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.SKULL_TOP_BREGMA
    )

    SKULL_RIGHT_EYE_INNER: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.RIGHT_EYE_INNER
    )

    SKULL_RIGHT_EYE_CENTER: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.RIGHT_EYE_CENTER
    )

    SKULL_RIGHT_EYE_OUTER: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.RIGHT_EYE_OUTER
    )

    SKULL_RIGHT_EAR: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.RIGHT_ACOUSTIC_MEATUS
    )

    SKULL_RIGHT_MOUTH: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.RIGHT_CANINE_TOOTH_TIP
    )

    SKULL_LEFT_EYE_INNER: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.LEFT_EYE_INNER
    )

    SKULL_LEFT_EYE_CENTER: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.LEFT_EYE_CENTER
    )

    SKULL_LEFT_EYE_OUTER: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.LEFT_EYE_OUTER
    )

    SKULL_LEFT_EAR: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.LEFT_ACOUSTIC_MEATUS
    )

    SKULL_LEFT_MOUTH: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        z_axis_reference=kp.LEFT_CANINE_TOOTH_TIP
    )

    # Spine segments
    SPINE_CERVICAL: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SPINE_THORACIC_TOP_T1,
        z_axis_reference=kp.SPINE_CERVICAL_TOP_C1_AXIS
    )

    SPINE_THORACIC: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SPINE_LUMBAR_L1,
        z_axis_reference=kp.SPINE_THORACIC_TOP_T1
    )

    SPINE_SACRUM_LUMBAR: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.PELVIS_SPINE_SACRUM_ORIGIN,
        z_axis_reference=kp.SPINE_LUMBAR_L1
    )

    # Right arm segments
    RIGHT_CLAVICLE: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SPINE_THORACIC_TOP_T1,
        z_axis_reference=kp.RIGHT_SHOULDER
    )

    RIGHT_ARM_PROXIMAL: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.RIGHT_SHOULDER,
        z_axis_reference=kp.RIGHT_ELBOW
    )

    RIGHT_ARM_DISTAL: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.RIGHT_ELBOW,
        z_axis_reference=kp.RIGHT_WRIST
    )

    # Right hand segments
    RIGHT_PALM_INDEX: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.RIGHT_WRIST,
        z_axis_reference=kp.RIGHT_INDEX_KNUCKLE
    )

    RIGHT_PALM_PINKY: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.RIGHT_WRIST,
        z_axis_reference=kp.RIGHT_PINKY_KNUCKLE
    )

    RIGHT_PALM_THUMB: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.RIGHT_WRIST,
        z_axis_reference=kp.RIGHT_THUMB_KNUCKLE
    )

    # Right leg segments
    PELVIS_RIGHT: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.PELVIS_SPINE_SACRUM_ORIGIN,
        z_axis_reference=kp.PELVIS_RIGHT_HIP_ACETABULUM
    )

    RIGHT_LEG_THIGH: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.PELVIS_RIGHT_HIP_ACETABULUM,
        z_axis_reference=kp.RIGHT_KNEE
    )

    RIGHT_LEG_CALF: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.RIGHT_KNEE,
        z_axis_reference=kp.RIGHT_ANKLE
    )

    RIGHT_FOOT_FRONT: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.RIGHT_ANKLE,
        z_axis_reference=kp.RIGHT_HALLUX_TIP
    )

    RIGHT_FOOT_HEEL: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.RIGHT_ANKLE,
        z_axis_reference=kp.RIGHT_HEEL
    )

    # Left arm segments
    LEFT_CLAVICLE: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.SPINE_THORACIC_TOP_T1,
        z_axis_reference=kp.LEFT_SHOULDER
    )

    LEFT_ARM_PROXIMAL: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.LEFT_SHOULDER,
        z_axis_reference=kp.LEFT_ELBOW
    )

    LEFT_ARM_DISTAL: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.LEFT_ELBOW,
        z_axis_reference=kp.LEFT_WRIST
    )

    # Left hand segments
    LEFT_PALM_INDEX: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.LEFT_WRIST,
        z_axis_reference=kp.LEFT_INDEX_KNUCKLE
    )

    LEFT_PALM_PINKY: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.LEFT_WRIST,
        z_axis_reference=kp.LEFT_PINKY_KNUCKLE
    )

    LEFT_PALM_THUMB: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.LEFT_WRIST,
        z_axis_reference=kp.LEFT_THUMB_KNUCKLE
    )

    # Left leg segments
    PELVIS_LEFT: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.PELVIS_SPINE_SACRUM_ORIGIN,
        z_axis_reference=kp.PELVIS_LEFT_HIP_ACETABULUM
    )

    LEFT_LEG_THIGH: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.PELVIS_LEFT_HIP_ACETABULUM,
        z_axis_reference=kp.LEFT_KNEE
    )

    LEFT_LEG_CALF: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.LEFT_KNEE,
        z_axis_reference=kp.LEFT_ANKLE
    )

    LEFT_FOOT_FRONT: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.LEFT_ANKLE,
        z_axis_reference=kp.LEFT_HALLUX_TIP
    )

    LEFT_FOOT_HEEL: ClassVar[SimpleSegmentABC] = SimpleSegmentABC(
        origin=kp.LEFT_ANKLE,
        z_axis_reference=kp.LEFT_HEEL
    )

    @classmethod
    def get_all_segments(cls) -> dict[str, SimpleSegmentABC]:
        """Get all segments as a dictionary."""
        return {
            name: getattr(cls, name)
            for name in dir(cls)
            if not name.startswith("_")
               and isinstance(getattr(cls, name), (SimpleSegmentABC, CompoundSegmentABC))
        }


class HumanCompoundSegments:
    """
    Compound segments that represent multiple keypoints as a rigid body.
    """

    # Note: These need to be fixed - the segments attribute should reference
    # segment instances, not keypoint names
    SKULL: ClassVar[CompoundSegmentABC] = CompoundSegmentABC(
        segments=[  # This should be segment instances, not keypoint names
            HumanSegments.SKULL_NOSE,
            HumanSegments.SKULL_TOP,
            HumanSegments.SKULL_RIGHT_EYE_INNER,
            HumanSegments.SKULL_RIGHT_EYE_CENTER,
            HumanSegments.SKULL_RIGHT_EYE_OUTER,
            HumanSegments.SKULL_RIGHT_EAR,
            HumanSegments.SKULL_RIGHT_MOUTH,
            HumanSegments.SKULL_LEFT_EYE_INNER,
            HumanSegments.SKULL_LEFT_EYE_CENTER,
            HumanSegments.SKULL_LEFT_EYE_OUTER,
            HumanSegments.SKULL_LEFT_EAR,
            HumanSegments.SKULL_LEFT_MOUTH
        ],
        origin=kp.SKULL_ORIGIN_FORAMEN_MAGNUM,
        x_axis_reference=kp.NOSE_TIP,
        y_axis_reference=kp.LEFT_ACOUSTIC_MEATUS,
        z_axis_reference=None  # Will be calculated from x and y
    )

    SPINE_PELVIS_LUMBAR: ClassVar[CompoundSegmentABC] = CompoundSegmentABC(
        segments=[
            HumanSegments.SPINE_SACRUM_LUMBAR,
            HumanSegments.PELVIS_RIGHT,
            HumanSegments.PELVIS_LEFT
        ],
        origin=kp.PELVIS_SPINE_SACRUM_ORIGIN,
        z_axis_reference=kp.SPINE_LUMBAR_L1,
        x_axis_reference=kp.PELVIS_RIGHT_HIP_ACETABULUM,
        y_axis_reference=None  # Will be calculated
    )