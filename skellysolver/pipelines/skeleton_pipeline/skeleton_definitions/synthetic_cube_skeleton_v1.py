"""Synthetic cube skeleton with wiggling tail chain.

A simple skeleton for testing:
- Rigid cube (8 vertices)
- Flexible tail chain (3 segments)
"""
from skellysolver.solvers.constraints.chain_constraint import ChainConstraint
from skellysolver.solvers.constraints.keypoint_constraint import KeypointConstraint
from skellysolver.solvers.constraints.linkage_constraint import LinkageConstraint
from skellysolver.solvers.constraints.segment_constraint import SegmentConstraint
from skellysolver.solvers.constraints.skeleton_constraint import SkeletonConstraint

# ============================================================================
# KEYPOINTS - Cube vertices
# ============================================================================

CUBE_V0 = KeypointConstraint(
    name="cube_v0",
    definition="Cube vertex 0 (front-bottom-left)"
)

CUBE_V1 = KeypointConstraint(
    name="cube_v1",
    definition="Cube vertex 1 (front-bottom-right)"
)

CUBE_V2 = KeypointConstraint(
    name="cube_v2",
    definition="Cube vertex 2 (front-top-right)"
)

CUBE_V3 = KeypointConstraint(
    name="cube_v3",
    definition="Cube vertex 3 (front-top-left)"
)

CUBE_V4 = KeypointConstraint(
    name="cube_v4",
    definition="Cube vertex 4 (back-bottom-left)"
)

CUBE_V5 = KeypointConstraint(
    name="cube_v5",
    definition="Cube vertex 5 (back-bottom-right)"
)

CUBE_V6 = KeypointConstraint(
    name="cube_v6",
    definition="Cube vertex 6 (back-top-right)"
)

CUBE_V7 = KeypointConstraint(
    name="cube_v7",
    definition="Cube vertex 7 (back-top-left)"
)

CUBE_ASYMMETRIC_V8 = KeypointConstraint(
    name="cube_assymetric_v8",
    definition="Asymmetric cube vertex 8 (extra vertex to break symmetry)"
)

# ============================================================================
# KEYPOINTS - Tail chain segments
# ============================================================================

TAIL_BASE = KeypointConstraint(
    name="tail_base",
    definition="Base of tail, attached to cube back face"
)

TAIL_MID1 = KeypointConstraint(
    name="tail_mid1",
    definition="First tail segment endpoint"
)

TAIL_MID2 = KeypointConstraint(
    name="tail_mid2",
    definition="Second tail segment endpoint"
)

TAIL_TIP = KeypointConstraint(
    name="tail_tip",
    definition="Tip of the tail"
)

# ============================================================================
# SEGMENTS
# ============================================================================
CUBE_BODY = SegmentConstraint(
    name="cube_body",
    parent=CUBE_V0,
    children=[
        CUBE_V1,
        CUBE_V2,
        CUBE_V3,
        CUBE_V4,
        CUBE_V5,
        CUBE_V6,
        CUBE_V7,
        CUBE_ASYMMETRIC_V8
    ],
    rigidity=1.0,  # Fully rigid cube
)

TAIL_SEGMENT_0 = SegmentConstraint(
    name="tail_segment_0",
    parent=TAIL_BASE,
    children=[TAIL_MID1],
    rigidity=0.5,  # Somewhat flexible
)

TAIL_SEGMENT_1 = SegmentConstraint(
    name="tail_segment_1",
    parent=TAIL_MID1,
    children=[TAIL_MID2],
    rigidity=0.3,  # More flexible
)

TAIL_SEGMENT_2 = SegmentConstraint(
    name="tail_segment_2",
    parent=TAIL_MID2,
    children=[TAIL_TIP],
    rigidity=0.2,  # Very flexible
)

# ============================================================================
# LINKAGES
# ============================================================================

CUBE_TO_TAIL = LinkageConstraint(
    name="cube_to_tail",
    parent=CUBE_BODY,
    children=[TAIL_SEGMENT_0],
    linked_keypoint=TAIL_BASE,
    stiffness=0.5,  # Moderate connection
)

TAIL_0_TO_1 = LinkageConstraint(
    name="tail_0_to_1",
    parent=TAIL_SEGMENT_0,
    children=[TAIL_SEGMENT_1],
    linked_keypoint=TAIL_MID1,
    stiffness=0.3,
)

TAIL_1_TO_2 = LinkageConstraint(
    name="tail_1_to_2",
    parent=TAIL_SEGMENT_1,
    children=[TAIL_SEGMENT_2],
    linked_keypoint=TAIL_MID2,
    stiffness=0.2,
)

# ============================================================================
# CHAIN
# ============================================================================

TAIL_CHAIN = ChainConstraint(
    name="tail_chain",
    parent=CUBE_TO_TAIL,
    children=[
        TAIL_0_TO_1,
        TAIL_1_TO_2
    ]
)

# ============================================================================
# SKELETON
# ============================================================================

SYNTHETIC_CUBE_SKELETON = SkeletonConstraint(
    name="synthetic_cube_skeleton",
    keypoints=[
        CUBE_V0,
        CUBE_V1,
        CUBE_V2,
        CUBE_V3,
        CUBE_V4,
        CUBE_V5,
        CUBE_V6,
        CUBE_V7,
        TAIL_BASE,
        TAIL_MID1,
        TAIL_MID2,
        TAIL_TIP,
    ],
    segments=[
        CUBE_BODY,
        TAIL_SEGMENT_0,
        TAIL_SEGMENT_1,
        TAIL_SEGMENT_2,
    ],
    linkages=[
        CUBE_TO_TAIL,
        TAIL_0_TO_1,
        TAIL_1_TO_2,
    ],
    chains=[TAIL_CHAIN],
    keypoint_to_tracked_mapping={
        CUBE_V0: "cube_v0",
        CUBE_V1: "cube_v1",
        CUBE_V2: "cube_v2",
        CUBE_V3: "cube_v3",
        CUBE_V4: "cube_v4",
        CUBE_V5: "cube_v5",
        CUBE_V6: "cube_v6",
        CUBE_V7: "cube_v7",
        CUBE_ASYMMETRIC_V8: "cube_asymmetric_v8",
        TAIL_BASE: "tail_base",
        TAIL_MID1: "tail_mid1",
        TAIL_MID2: "tail_mid2",
        TAIL_TIP: "tail_tip",
    }
)
