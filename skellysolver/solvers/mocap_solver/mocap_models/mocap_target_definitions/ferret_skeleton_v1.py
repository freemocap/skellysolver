"""
Ferret skeleton definition with module-level constants.
Reference: BSAVA Manual of Rodents and Ferrets
"""
from skellysolver.data.primitive_objects.chain_model import Chain
from skellysolver.data.primitive_objects.keypoint_model import Keypoint
from skellysolver.data.primitive_objects.linkage_model import Linkage
from skellysolver.data.primitive_objects.segment_model import Segment
from skellysolver.data.primitive_objects.skeleton_model import Skeleton

# ============================================================================
# KEYPOINTS
# ============================================================================

# Eye Cameras
RIGHT_EYE_CAMERA = Keypoint(
    name="right_eye_camera",
    definition="Position of the right eye camera at the end of the eye camera mount"
)
LEFT_EYE_CAMERA = Keypoint(
    name="left_eye_camera",
    definition="Position of the left eye camera at the end of the eye camera mount"
)

# Skull
CAMERA_BASE = Keypoint(
    name="camera_base",
    definition="Base of the eye camera mounts on top of the skull, between the ears"
)

NOSE_TIP = Keypoint(
    name="nose_tip",
    definition="Tip of the nose"
)

# Right face
RIGHT_EYE_INNER = Keypoint(
    name="right_eye_inner",
    definition="Inner corner of the right eye socket, in the Lacrimal fossa (aka tear duct area)"
)

RIGHT_EYE_CENTER = Keypoint(
    name="right_eye_center",
    definition="Geometric center of the inner and outer keypoints of the right eye, approximately center of eyeball socket/orbit"
)

RIGHT_EYE_OUTER = Keypoint(
    name="right_eye_outer",
    definition="Outer corner of the right eye roughly between the Zygomatic Process and the Frontal Process of the Zygomatic Bone (i.e. where the eyelids meet laterally)"
)

RIGHT_ACOUSTIC_MEATUS = Keypoint(
    name="right_acoustic_meatus",
    definition="Entrance to the right ear canal"
)

# Left face
LEFT_EYE_INNER = Keypoint(
    name="left_eye_inner",
    definition="Inner corner of the left eye socket in the Lacrimal fossa (aka tear duct area)"
)

LEFT_EYE_CENTER = Keypoint(
    name="left_eye_center",
    definition="Geometric center of the inner and outer keypoints of the left eye approximately center of eyeball socket/orbit"
)

LEFT_EYE_OUTER = Keypoint(
    name="left_eye_outer",
    definition="Outer corner of the left eye roughly between the Zygomatic Process and the Frontal Process of the Zygomatic Bone, i.e. where the eyelids meet laterally"
)

LEFT_ACOUSTIC_MEATUS = Keypoint(
    name="left_acoustic_meatus",
    definition="Entrance to the left ear canal"
)

# Spine
SPINE_THORACIC_TOP_T1 = Keypoint(
    name="spine_thoracic_top_t1",
    definition="Geometric center of the top surface of T1"
)

PELVIS_SPINE_SACRUM_ORIGIN = Keypoint(
    name="pelvis_spine_sacrum_origin",
    definition="Geometric center of the hip sockets, anterior to the Sacrum"
)

TAIL_TIP = Keypoint(
    name="tail_tip",
    definition="Tip of the tail, end of the caudal vertebrae"
)

# ============================================================================
# SEGMENTS
# ============================================================================

SKULL = Segment(
    name="skull",
    root=CAMERA_BASE,
    keypoints=[
        NOSE_TIP,
        RIGHT_EYE_CENTER,
        LEFT_EYE_CENTER,
        RIGHT_ACOUSTIC_MEATUS,
        LEFT_ACOUSTIC_MEATUS
    ],
    rigidity=1.0,
)

EYE_CAMERAS = Segment(
    name="eye_cameras",
    root=CAMERA_BASE,
    keypoints=[
        RIGHT_EYE_CAMERA,
        LEFT_EYE_CAMERA
    ],
    rigidity=1.0,
)

CERVICAL_SPINE = Segment(
    name="cervical_spine",
    root=CAMERA_BASE,
    keypoints=[SPINE_THORACIC_TOP_T1],
    rigidity=0.2,
)

THORACOLUMBAR_SPINE = Segment(
    name="thoracolumbar_spine",
    root=SPINE_THORACIC_TOP_T1,
    keypoints=[PELVIS_SPINE_SACRUM_ORIGIN],
    rigidity=0.2,
)

CAUDAL_SPINE = Segment(
    name="caudal_spine",
    root=PELVIS_SPINE_SACRUM_ORIGIN,
    keypoints=[TAIL_TIP],
    rigidity=0.2,
)

# ============================================================================
# LINKAGES
# ============================================================================

EYE_CAMERAS_TO_SKULL = Linkage(
    name="skull_to_eye_cameras",
    parent=EYE_CAMERAS,
    children=[SKULL],
    linked_keypoint=CAMERA_BASE,
    stiffness=0.99,
)

SKULL_TO_CERVICAL_SPINE = Linkage(
    name="skull_to_cervical_spine",
    parent=SKULL,
    children=[CERVICAL_SPINE],
    linked_keypoint=CAMERA_BASE,
    stiffness=0.1,
)

CERVICAL_TO_THORACOLUMBAR_SPINE = Linkage(
    name="cervical_to_thoracolumbar_spine",
    parent=CERVICAL_SPINE,
    children=[THORACOLUMBAR_SPINE],
    linked_keypoint=SPINE_THORACIC_TOP_T1,
    stiffness=0.1,
)

THORACIC_LUMBAR_TO_CAUDAL_SPINE = Linkage(
    name="thoracolumbar_to_caudal_spine",
    parent=THORACOLUMBAR_SPINE,
    children=[CAUDAL_SPINE],
    linked_keypoint=PELVIS_SPINE_SACRUM_ORIGIN,
    stiffness=0.1,
)

# ============================================================================
# CHAINS
# ============================================================================

SPINE_CHAIN = Chain(
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

FERRET_SKELETON_V1 = Skeleton(
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
)