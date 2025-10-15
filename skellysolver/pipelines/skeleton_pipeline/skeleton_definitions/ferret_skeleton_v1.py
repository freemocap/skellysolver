"""
Ferret skeleton definition with module-level constants.
Reference: BSAVA Manual of Rodents and Ferrets
"""
from skellysolver.solvers.constraints.chain_constraint import ChainConstraint
from skellysolver.solvers.constraints.keypoint_constraint import KeypointConstraint
from skellysolver.solvers.constraints.linkage_constraint import LinkageConstraint
from skellysolver.solvers.constraints.segment_constraint import SegmentConstraint
from skellysolver.solvers.constraints.skeleton_constraint import SkeletonConstraint

# ============================================================================
# KEYPOINTS
# ============================================================================

# Eye Cameras
RIGHT_EYE_CAMERA = KeypointConstraint(
    name="right_eye_camera",
    definition="Position of the right eye camera at the end of the eye camera mount"
)
LEFT_EYE_CAMERA = KeypointConstraint(
    name="left_eye_camera",
    definition="Position of the left eye camera at the end of the eye camera mount"
)

# Skull
CAMERA_BASE = KeypointConstraint(
    name="camera_base",
    definition="Base of the eye camera mounts on top of the skull, between the ears"
)

NOSE_TIP = KeypointConstraint(
    name="nose_tip",
    definition="Tip of the nose"
)

# Right face
RIGHT_EYE_INNER = KeypointConstraint(
    name="right_eye_inner",
    definition="Inner corner of the right eye socket, in the Lacrimal fossa (aka tear duct area)"
)

RIGHT_EYE_CENTER = KeypointConstraint(
    name="right_eye_center",
    definition="Geometric center of the inner and outer keypoints of the right eye, approximately center of eyeball socket/orbit"
)

RIGHT_EYE_OUTER = KeypointConstraint(
    name="right_eye_outer",
    definition="Outer corner of the right eye roughly between the Zygomatic Process and the Frontal Process of the Zygomatic Bone (i.e. where the eyelids meet laterally)"
)

RIGHT_ACOUSTIC_MEATUS = KeypointConstraint(
    name="right_acoustic_meatus",
    definition="Entrance to the right ear canal"
)

# Left face
LEFT_EYE_INNER = KeypointConstraint(
    name="left_eye_inner",
    definition="Inner corner of the left eye socket in the Lacrimal fossa (aka tear duct area)"
)

LEFT_EYE_CENTER = KeypointConstraint(
    name="left_eye_center",
    definition="Geometric center of the inner and outer keypoints of the left eye approximately center of eyeball socket/orbit"
)

LEFT_EYE_OUTER = KeypointConstraint(
    name="left_eye_outer",
    definition="Outer corner of the left eye roughly between the Zygomatic Process and the Frontal Process of the Zygomatic Bone, i.e. where the eyelids meet laterally"
)

LEFT_ACOUSTIC_MEATUS = KeypointConstraint(
    name="left_acoustic_meatus",
    definition="Entrance to the left ear canal"
)

# Spine
SPINE_THORACIC_TOP_T1 = KeypointConstraint(
    name="spine_thoracic_top_t1",
    definition="Geometric center of the top surface of T1"
)

PELVIS_SPINE_SACRUM_ORIGIN = KeypointConstraint(
    name="pelvis_spine_sacrum_origin",
    definition="Geometric center of the hip sockets, anterior to the Sacrum"
)

TAIL_TIP = KeypointConstraint(
    name="tail_tip",
    definition="Tip of the tail, end of the caudal vertebrae"
)

# ============================================================================
# SEGMENTS
# ============================================================================

SKULL = SegmentConstraint(
    name="skull",
    parent=CAMERA_BASE,
    children=[
        NOSE_TIP,
        RIGHT_EYE_CENTER,
        LEFT_EYE_CENTER,
        RIGHT_ACOUSTIC_MEATUS,
        LEFT_ACOUSTIC_MEATUS
    ],
    rigidity=1.0,
)

EYE_CAMERAS = SegmentConstraint(
    name="eye_cameras",
    parent=CAMERA_BASE,
    children=[
        RIGHT_EYE_CAMERA,
        LEFT_EYE_CAMERA
    ],
    rigidity=1.0,
)

CERVICAL_SPINE = SegmentConstraint(
    name="cervical_spine",
    parent=CAMERA_BASE,
    children=[SPINE_THORACIC_TOP_T1],
    rigidity=0.2,
)

THORACOLUMBAR_SPINE = SegmentConstraint(
    name="thoracolumbar_spine",
    parent=SPINE_THORACIC_TOP_T1,
    children=[PELVIS_SPINE_SACRUM_ORIGIN],
    rigidity=0.2,
)

CAUDAL_SPINE = SegmentConstraint(
    name="caudal_spine",
    parent=PELVIS_SPINE_SACRUM_ORIGIN,
    children=[TAIL_TIP],
    rigidity=0.2,
)

# ============================================================================
# LINKAGES
# ============================================================================

EYE_CAMERAS_TO_SKULL = LinkageConstraint(
    name="skull_to_eye_cameras",
    parent=EYE_CAMERAS,
    children=[SKULL],
    linked_keypoint=CAMERA_BASE,
    stiffness=0.99,
)

SKULL_TO_CERVICAL_SPINE = LinkageConstraint(
    name="skull_to_cervical_spine",
    parent=SKULL,
    children=[CERVICAL_SPINE],
    linked_keypoint=CAMERA_BASE,
    stiffness=0.1,
)

CERVICAL_TO_THORACOLUMBAR_SPINE = LinkageConstraint(
    name="cervical_to_thoracolumbar_spine",
    parent=CERVICAL_SPINE,
    children=[THORACOLUMBAR_SPINE],
    linked_keypoint=SPINE_THORACIC_TOP_T1,
    stiffness=0.1,
)

THORACIC_LUMBAR_TO_CAUDAL_SPINE = LinkageConstraint(
    name="thoracolumbar_to_caudal_spine",
    parent=THORACOLUMBAR_SPINE,
    children=[CAUDAL_SPINE],
    linked_keypoint=PELVIS_SPINE_SACRUM_ORIGIN,
    stiffness=0.1,
)

# ============================================================================
# CHAINS
# ============================================================================

SPINE_CHAIN = ChainConstraint(
    name="spine_chain",
    parent=EYE_CAMERAS_TO_SKULL,
    children=[
        SKULL_TO_CERVICAL_SPINE,
        CERVICAL_TO_THORACOLUMBAR_SPINE,
        THORACIC_LUMBAR_TO_CAUDAL_SPINE
    ]
)

# ============================================================================
# SKELETON
# ============================================================================

FERRET_SKELETON_V1 = SkeletonConstraint(
    name="ferret_skeleton_v1",
    keypoints=[
        RIGHT_EYE_CAMERA,
        LEFT_EYE_CAMERA,
        CAMERA_BASE,
        NOSE_TIP,
        RIGHT_EYE_CENTER,
        RIGHT_ACOUSTIC_MEATUS,
        LEFT_EYE_CENTER,
        LEFT_ACOUSTIC_MEATUS,
        SPINE_THORACIC_TOP_T1,
        PELVIS_SPINE_SACRUM_ORIGIN,
        TAIL_TIP,
    ],
    segments=[
        SKULL,
        EYE_CAMERAS,
        CERVICAL_SPINE,
        THORACOLUMBAR_SPINE,
        CAUDAL_SPINE,
    ],
    linkages=[
        EYE_CAMERAS_TO_SKULL,
        SKULL_TO_CERVICAL_SPINE,
        CERVICAL_TO_THORACOLUMBAR_SPINE,
        THORACIC_LUMBAR_TO_CAUDAL_SPINE,
    ],
    chains=[SPINE_CHAIN],
    keypoint_to_tracked_mapping={
         RIGHT_EYE_CAMERA:"right_cam_tip",
         LEFT_EYE_CAMERA:"left_cam_tip",
         CAMERA_BASE:"base",
         NOSE_TIP:"nose",
         RIGHT_EYE_CENTER:"right_eye",
         RIGHT_ACOUSTIC_MEATUS:"right_ear",
         LEFT_EYE_CENTER:"left_eye",
         LEFT_ACOUSTIC_MEATUS:"left_ear",
         SPINE_THORACIC_TOP_T1:"spine_t1",
         PELVIS_SPINE_SACRUM_ORIGIN:"sacrum",
         TAIL_TIP:"tail_tip",
    }
)