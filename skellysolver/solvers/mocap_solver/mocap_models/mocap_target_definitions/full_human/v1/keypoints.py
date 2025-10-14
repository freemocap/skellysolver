"""
Refactored keypoints - using actual KeypointABC instances instead of Enums
"""
from typing import ClassVar

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.data.primitive_objects.keypoint_model import Keypoint


class HumanKeypoints(ABaseModel):
    """
    Container for all human skeleton keypoints.
    Each keypoint is an instance of KeypointABC, not an Enum.
    """

    # Skull
    SKULL_ORIGIN_FORAMEN_MAGNUM: ClassVar[Keypoint] = Keypoint(
        name="skull_origin_foramen_magnum",
        definition="Geometric center of the foramen magnum, the hole at the base of the skull where the spinal cord enters"
    )

    SKULL_TOP_BREGMA: ClassVar[Keypoint] = Keypoint(
        name="skull_top_bregma",
        definition="Tippy top of the head, intersection of coronal and sagittal sutures"
    )

    # Face
    NOSE_TIP: ClassVar[Keypoint] = Keypoint(
        name="nose_tip",
        definition="Tip of the nose"
    )

    # Right face
    RIGHT_EYE_INNER: ClassVar[Keypoint] = Keypoint(
        name="right_eye_inner",
        definition="Inner corner of the right eye socket, in the Lacrimal fossa"
    )

    RIGHT_EYE_CENTER: ClassVar[Keypoint] = Keypoint(
        name="right_eye_center",
        definition="Geometric center of the inner and outer keypoints of the right eye"
    )

    RIGHT_EYE_OUTER: ClassVar[Keypoint] = Keypoint(
        name="right_eye_outer",
        definition="Outer corner of the right eye"
    )

    RIGHT_ACOUSTIC_MEATUS: ClassVar[Keypoint] = Keypoint(
        name="right_acoustic_meatus",
        definition="Entrance to the right ear canal"
    )

    RIGHT_CANINE_TOOTH_TIP: ClassVar[Keypoint] = Keypoint(
        name="right_canine_tooth_tip",
        definition="Tip of the right canine tooth"
    )

    # Left face (mirror right)
    LEFT_EYE_INNER: ClassVar[Keypoint] = Keypoint(
        name="left_eye_inner",
        definition="Inner corner of the left eye socket"
    )

    LEFT_EYE_CENTER: ClassVar[Keypoint] = Keypoint(
        name="left_eye_center",
        definition="Geometric center of the inner and outer keypoints of the left eye"
    )

    LEFT_EYE_OUTER: ClassVar[Keypoint] = Keypoint(
        name="left_eye_outer",
        definition="Outer corner of the left eye"
    )

    LEFT_ACOUSTIC_MEATUS: ClassVar[Keypoint] = Keypoint(
        name="left_acoustic_meatus",
        definition="Entrance to the left ear canal"
    )

    LEFT_CANINE_TOOTH_TIP: ClassVar[Keypoint] = Keypoint(
        name="left_canine_tooth_tip",
        definition="Tip of the left canine tooth"
    )

    # Spine - Cervical
    SPINE_CERVICAL_TOP_C1_AXIS: ClassVar[Keypoint] = Keypoint(
        name="spine_cervical_top_c1_axis",
        definition="Top of the neck segment, geometric center of C2 (Axis)"
    )

    SPINE_CERVICAL_ORIGIN_C7: ClassVar[Keypoint] = Keypoint(
        name="spine_cervical_origin_c7",
        definition="Base of the neck, geometric center of C7"
    )

    # Spine - Thoracic
    SPINE_THORACIC_TOP_T1: ClassVar[Keypoint] = Keypoint(
        name="spine_thoracic_top_t1",
        definition="Geometric center of the top surface of T1"
    )

    SPINE_THORACIC_ORIGIN_T12: ClassVar[Keypoint] = Keypoint(
        name="spine_thoracic_origin_t12",
        definition="Geometric center of the bottom surface of T12"
    )

    # Sternum
    STERNUM_TOP_SUPRASTERNAL_NOTCH: ClassVar[Keypoint] = Keypoint(
        name="sternum_top_suprasternal_notch",
        definition="Geometric center of the suprasternal notch"
    )

    STERNUM_ORIGIN_XIPHOID_PROCESS: ClassVar[Keypoint] = Keypoint(
        name="sternum_origin_xiphoid_process",
        definition="Geometric center of the xiphoid process"
    )

    # Spine - Lumbar and Pelvis
    SPINE_LUMBAR_L1: ClassVar[Keypoint] = Keypoint(
        name="spine_lumbar_l1",
        definition="Geometric center of the top surface of L1"
    )

    PELVIS_SPINE_SACRUM_ORIGIN: ClassVar[Keypoint] = Keypoint(
        name="pelvis_spine_sacrum_origin",
        definition="Geometric center of the hip sockets, anterior to the Sacrum"
    )

    PELVIS_RIGHT_HIP_ACETABULUM: ClassVar[Keypoint] = Keypoint(
        name="pelvis_right_hip_acetabulum",
        definition="Geometric center of the right hip socket/acetabulum"
    )

    PELVIS_LEFT_HIP_ACETABULUM: ClassVar[Keypoint] = Keypoint(
        name="pelvis_left_hip_acetabulum",
        definition="Geometric center of the left hip socket/acetabulum"
    )

    # Right arm
    RIGHT_STERNOCLAVICULAR: ClassVar[Keypoint] = Keypoint(
        name="right_sternoclavicular",
        definition="Center of the right sternoclavicular joint"
    )

    RIGHT_SHOULDER: ClassVar[Keypoint] = Keypoint(
        name="right_shoulder",
        definition="Center of the right glenohumeral joint"
    )

    RIGHT_ELBOW: ClassVar[Keypoint] = Keypoint(
        name="right_elbow",
        definition="Center of the right elbow joint"
    )

    RIGHT_WRIST: ClassVar[Keypoint] = Keypoint(
        name="right_wrist",
        definition="Center of the right radiocarpal joint"
    )

    # Right hand
    RIGHT_THUMB_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="right_thumb_knuckle",
        definition="Center of the right thumb MCP joint"
    )

    RIGHT_INDEX_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="right_index_knuckle",
        definition="Center of the right index finger MCP joint"
    )

    RIGHT_MIDDLE_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="right_middle_knuckle",
        definition="Center of the right middle finger MCP joint"
    )

    RIGHT_RING_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="right_ring_knuckle",
        definition="Center of the right ring finger MCP joint"
    )

    RIGHT_PINKY_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="right_pinky_knuckle",
        definition="Center of the right pinky MCP joint"
    )

    # Right leg
    RIGHT_KNEE: ClassVar[Keypoint] = Keypoint(
        name="right_knee",
        definition="Center of the right knee joint"
    )

    RIGHT_ANKLE: ClassVar[Keypoint] = Keypoint(
        name="right_ankle",
        definition="Center of the right ankle joint"
    )

    RIGHT_HEEL: ClassVar[Keypoint] = Keypoint(
        name="right_heel",
        definition="Contact surface of the right heel"
    )

    RIGHT_HALLUX_TIP: ClassVar[Keypoint] = Keypoint(
        name="right_hallux_tip",
        definition="Tip of the right big toe"
    )

    # Left arm (mirror right)
    LEFT_STERNOCLAVICULAR: ClassVar[Keypoint] = Keypoint(
        name="left_sternoclavicular",
        definition="Center of the left sternoclavicular joint"
    )

    LEFT_SHOULDER: ClassVar[Keypoint] = Keypoint(
        name="left_shoulder",
        definition="Center of the left glenohumeral joint"
    )

    LEFT_ELBOW: ClassVar[Keypoint] = Keypoint(
        name="left_elbow",
        definition="Center of the left elbow joint"
    )

    LEFT_WRIST: ClassVar[Keypoint] = Keypoint(
        name="left_wrist",
        definition="Center of the left radiocarpal joint"
    )

    # Left hand
    LEFT_THUMB_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="left_thumb_knuckle",
        definition="Center of the left thumb MCP joint"
    )

    LEFT_INDEX_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="left_index_knuckle",
        definition="Center of the left index finger MCP joint"
    )

    LEFT_MIDDLE_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="left_middle_knuckle",
        definition="Center of the left middle finger MCP joint"
    )

    LEFT_RING_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="left_ring_knuckle",
        definition="Center of the left ring finger MCP joint"
    )

    LEFT_PINKY_KNUCKLE: ClassVar[Keypoint] = Keypoint(
        name="left_pinky_knuckle",
        definition="Center of the left pinky MCP joint"
    )

    # Left leg
    LEFT_KNEE: ClassVar[Keypoint] = Keypoint(
        name="left_knee",
        definition="Center of the left knee joint"
    )

    LEFT_ANKLE: ClassVar[Keypoint] = Keypoint(
        name="left_ankle",
        definition="Center of the left ankle joint"
    )

    LEFT_HEEL: ClassVar[Keypoint] = Keypoint(
        name="left_heel",
        definition="Contact surface of the left heel"
    )

    LEFT_HALLUX_TIP: ClassVar[Keypoint] = Keypoint(
        name="left_hallux_tip",
        definition="Tip of the left big toe"
    )

    @classmethod
    def get_all_keypoints(cls) -> dict[str, Keypoint]:
        """Get all keypoints as a dictionary."""
        return {
            name: getattr(cls, name)
            for name in dir(cls)
            if not name.startswith("_") and isinstance(getattr(cls, name), Keypoint)
        }

    @classmethod
    def get_keypoint_by_name(cls, name: str) -> Keypoint | None:
        """Get a keypoint by its name attribute (not class attribute name)."""
        for kp in cls.get_all_keypoints().values():
            if kp.name == name:
                return kp
        return None