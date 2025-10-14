"""Eye tracking visualization for Blender.

This module creates animated 3D eyeballs with gaze tracking visualization,
including proper orientation constraints relative to skull markers.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Euler


@dataclass
class EyeTrackingConfig:
    """Configuration for eye tracking visualization."""

    csv_path: Path
    """Path to eye_tracking_results.csv (tidy format)"""

    parent_marker_name: str
    """Name of skull marker to attach eyeball to (e.g., 'right_eye')"""

    track_to_marker_name: str
    """Marker to point AWAY from (e.g., opposite eye or head center)"""

    up_marker_name: str
    """Marker to use for up direction (e.g., 'base' or top of head)"""

    eyeball_radius: float = 0.012
    """Eyeball radius in meters (default: 12mm)"""

    pupil_radius: float = 0.002
    """Pupil radius in meters (default: 2mm)"""

    iris_radius: float = 0.005
    """Iris radius in meters (default: 5mm)"""

    local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset from parent marker in local space (meters)"""

    eyeball_color: tuple[float, float, float] = (0.95, 0.95, 0.95)
    """RGB color for eyeball (white)"""

    iris_color: tuple[float, float, float] = (0.3, 0.5, 0.8)
    """RGB color for iris (blue)"""

    pupil_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """RGB color for pupil (black)"""

    show_gaze_arrow: bool = True
    """Show gaze direction arrow"""

    gaze_arrow_length: float = 0.05
    """Length of gaze arrow in meters"""

    gaze_arrow_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    """RGBA color for gaze arrow (red)"""


# ============================================================================
# DATA LOADING
# ============================================================================


def load_eye_tracking_data(
    *,
    csv_path: Path,
    data_scale: float = 1.0
) -> list[dict[str, float]]:
    """
    Load eye tracking results from CSV.

    Expected columns:
    - frame
    - gaze_azimuth_rad
    - gaze_elevation_rad
    - gaze_x, gaze_y, gaze_z
    - eyeball_x_mm, eyeball_y_mm, eyeball_z_mm
    - reprojection_error_px

    Args:
        csv_path: Path to eye tracking CSV
        data_scale: Scale factor for eyeball position (e.g., 0.001 for mm->m)

    Returns:
        List of frames with eye tracking data
    """
    frames: list[dict[str, float]] = []

    with open(file=csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f=f)

        for row in reader:
            frame_data = {
                'frame': int(row['frame']),
                'gaze_azimuth_rad': float(row['gaze_azimuth_rad']),
                'gaze_elevation_rad': float(row['gaze_elevation_rad']),
                'gaze_x': float(row['gaze_x']),
                'gaze_y': float(row['gaze_y']),
                'gaze_z': float(row['gaze_z']),
                'eyeball_x_mm': float(row['eyeball_x_mm']) * data_scale,
                'eyeball_y_mm': float(row['eyeball_y_mm']) * data_scale,
                'eyeball_z_mm': float(row['eyeball_z_mm']) * data_scale,
                'reprojection_error_px': float(row['reprojection_error_px'])
            }
            frames.append(frame_data)

    print(f"✓ Loaded {len(frames)} frames of eye tracking data")
    return frames


# ============================================================================
# MATERIALS
# ============================================================================


def create_eye_material(
    *,
    name: str,
    color: tuple[float, float, float],
    metallic: float = 0.3,
    roughness: float = 0.4
) -> bpy.types.Material:
    """Create a material for eye components."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    nodes.clear()

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_mix = nodes.new(type='ShaderNodeMixShader')

    node_bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    node_bsdf.inputs['Metallic'].default_value = metallic
    node_bsdf.inputs['Roughness'].default_value = roughness

    node_emission.inputs['Color'].default_value = (*color, 1.0)
    node_emission.inputs['Strength'].default_value = 0.5

    links = mat.node_tree.links
    links.new(input=node_bsdf.outputs['BSDF'], output=node_mix.inputs[1])
    links.new(input=node_emission.outputs['Emission'], output=node_mix.inputs[2])
    links.new(input=node_mix.outputs['Shader'], output=node_output.inputs['Surface'])

    node_mix.inputs['Fac'].default_value = 0.2

    return mat


# ============================================================================
# GEOMETRY CREATION
# ============================================================================


def create_eyeball_with_pupil(
    *,
    name: str,
    eyeball_radius: float,
    pupil_radius: float,
    iris_radius: float,
    location: tuple[float, float, float],
    eyeball_material: bpy.types.Material,
    iris_material: bpy.types.Material,
    pupil_material: bpy.types.Material,
    parent: bpy.types.Object
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    """
    Create a 3D eyeball with iris and pupil.

    Args:
        name: Base name for eyeball object
        eyeball_radius: Radius of eyeball sphere
        pupil_radius: Radius of pupil circle
        iris_radius: Radius of iris circle
        location: Initial location
        eyeball_material: Material for eyeball
        iris_material: Material for iris
        pupil_material: Material for pupil
        parent: Parent object

    Returns:
        Tuple of (eyeball_sphere, iris, pupil) objects
    """
    # Create eyeball sphere
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=eyeball_radius,
        location=location,
        segments=32,
        ring_count=16
    )

    eyeball = bpy.context.active_object
    eyeball.name = name
    eyeball.parent = parent

    if eyeball.data.materials:
        eyeball.data.materials[0] = eyeball_material
    else:
        eyeball.data.materials.append(eyeball_material)

    bpy.ops.object.shade_smooth()

    # Create iris (ring on eyeball surface)
    bpy.ops.mesh.primitive_circle_add(
        vertices=32,
        radius=iris_radius,
        location=(0, 0, eyeball_radius * 0.98),
        fill_type='NGON'
    )

    iris = bpy.context.active_object
    iris.name = f"{name}_Iris"
    iris.parent = eyeball
    iris.parent_type = 'OBJECT'

    if iris.data.materials:
        iris.data.materials[0] = iris_material
    else:
        iris.data.materials.append(iris_material)

    bpy.ops.object.shade_smooth()

    # Create pupil (circle on iris)
    bpy.ops.mesh.primitive_circle_add(
        vertices=16,
        radius=pupil_radius,
        location=(0, 0, eyeball_radius * 0.99),
        fill_type='NGON'
    )

    pupil = bpy.context.active_object
    pupil.name = f"{name}_Pupil"
    pupil.parent = eyeball
    pupil.parent_type = 'OBJECT'

    if pupil.data.materials:
        pupil.data.materials[0] = pupil_material
    else:
        pupil.data.materials.append(pupil_material)

    bpy.ops.object.shade_smooth()

    return eyeball, iris, pupil


def create_gaze_arrow(
    *,
    name: str,
    length: float,
    color: tuple[float, float, float, float],
    parent: bpy.types.Object
) -> bpy.types.Object:
    """
    Create a gaze direction arrow.

    Args:
        name: Name for arrow object
        length: Length of arrow
        color: RGBA color
        parent: Parent object

    Returns:
        Arrow curve object
    """
    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.001
    curve_data.bevel_resolution = 4
    curve_data.fill_mode = 'FULL'

    spline = curve_data.splines.new(type='NURBS')
    spline.points.add(count=1)

    spline.points[0].co = (0, 0, 0, 1.0)
    spline.points[1].co = (0, 0, length, 1.0)

    spline.order_u = 2
    spline.use_endpoint_u = True

    arrow = bpy.data.objects.new(name=name, object_data=curve_data)
    bpy.context.collection.objects.link(object=arrow)
    arrow.parent = parent

    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    nodes.clear()

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_emission = nodes.new(type='ShaderNodeEmission')

    node_emission.inputs['Color'].default_value = color
    node_emission.inputs['Strength'].default_value = 2.0

    mat.node_tree.links.new(
        input=node_emission.outputs['Emission'],
        output=node_output.inputs['Surface']
    )

    if arrow.data.materials:
        arrow.data.materials[0] = mat
    else:
        arrow.data.materials.append(mat)

    return arrow


# ============================================================================
# ANIMATION
# ============================================================================


def animate_eyeball_gaze(
    *,
    eyeball: bpy.types.Object,
    eye_frames: list[dict[str, float]],
    frame_start: int,
    keyframe_step: int = 1
) -> None:
    """
    Animate eyeball rotation based on gaze angles.

    The gaze angles are in camera space, so we convert them to
    Euler rotations applied to the eyeball.

    Args:
        eyeball: Eyeball object to animate
        eye_frames: List of eye tracking frame data
        frame_start: Starting frame number
        keyframe_step: Keyframe interval
    """
    for frame_idx in range(0, len(eye_frames), keyframe_step):
        frame = frame_start + frame_idx
        eye_data = eye_frames[frame_idx]

        azimuth_rad = eye_data['gaze_azimuth_rad']
        elevation_rad = eye_data['gaze_elevation_rad']

        # Convert to Euler angles (XYZ order)
        euler = Euler((elevation_rad, azimuth_rad, 0.0), 'XYZ')
        eyeball.rotation_euler = euler

        eyeball.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Set interpolation to linear
    if eyeball.animation_data and eyeball.animation_data.action:
        for fcurve in eyeball.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


# ============================================================================
# MAIN EYE VISUALIZATION
# ============================================================================


def create_eye_visualization(
    *,
    config: EyeTrackingConfig,
    parent_marker: bpy.types.Object,
    track_to_marker: bpy.types.Object,
    up_marker: bpy.types.Object,
    rigid_body_parent: bpy.types.Object,
    frame_start: int,
    keyframe_step: int,
    data_scale: float
) -> bpy.types.Object:
    """
    Create and animate eye tracking visualization with proper orientation constraints.

    Strategy:
    1. Create base empty at eye marker location (follows head movement)
    2. Use constraints to orient it away from opposite eye, up towards head top
    3. Create eyeball as child of base empty
    4. Apply gaze rotations to eyeball (relative to base orientation)

    This ensures the eye naturally points outward and rotates with the head,
    plus the gaze tracking rotations are added on top.

    Args:
        config: Eye tracking configuration
        parent_marker: Marker object to parent eye to
        track_to_marker: Marker to point away from
        up_marker: Marker defining up direction
        rigid_body_parent: Parent object for entire rigid body system
        frame_start: Starting frame number
        keyframe_step: Keyframe interval
        data_scale: Scale factor for data

    Returns:
        The eyeball object
    """
    print(f"\nCreating eye visualization from {config.csv_path.name}...")

    # Load eye tracking data
    eye_frames = load_eye_tracking_data(
        csv_path=config.csv_path,
        data_scale=data_scale
    )

    # Create materials
    eyeball_mat = create_eye_material(
        name=f"Eye_{config.parent_marker_name}_Eyeball",
        color=config.eyeball_color,
        metallic=0.1,
        roughness=0.3
    )

    iris_mat = create_eye_material(
        name=f"Eye_{config.parent_marker_name}_Iris",
        color=config.iris_color,
        metallic=0.2,
        roughness=0.4
    )

    pupil_mat = create_eye_material(
        name=f"Eye_{config.parent_marker_name}_Pupil",
        color=config.pupil_color,
        metallic=0.0,
        roughness=0.8
    )

    # Create base orientation empty
    bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
    eye_base_empty = bpy.context.active_object
    eye_base_empty.name = f"EyeBase_{config.parent_marker_name}"
    eye_base_empty.empty_display_size = config.eyeball_radius * 0.5

    eye_base_empty.parent = parent_marker
    eye_base_empty.parent_type = 'OBJECT'
    eye_base_empty.location = config.local_offset

    # Add constraints for proper orientation
    # Damped Track: Point AWAY from opposite eye marker
    constraint_track = eye_base_empty.constraints.new(type='DAMPED_TRACK')
    constraint_track.target = track_to_marker
    constraint_track.track_axis = 'TRACK_NEGATIVE_Z'
    constraint_track.name = "TrackAwayFromOppositeEye"

    print(f"  ✓ Added constraint: back of eye points towards {config.track_to_marker_name}")

    # Locked Track: Keep up direction pointing towards head top
    constraint_up = eye_base_empty.constraints.new(type='LOCKED_TRACK')
    constraint_up.target = up_marker
    constraint_up.track_axis = 'TRACK_Y'
    constraint_up.lock_axis = 'LOCK_Z'
    constraint_up.name = "UpTowardsHeadTop"

    print(f"  ✓ Added constraint: up direction towards {config.up_marker_name}")

    # Create rotation empty for gaze animation
    bpy.ops.object.empty_add(type='SPHERE', location=(0, 0, 0))
    eye_rotation_empty = bpy.context.active_object
    eye_rotation_empty.name = f"EyeRotation_{config.parent_marker_name}"
    eye_rotation_empty.empty_display_size = config.eyeball_radius * 0.3
    eye_rotation_empty.parent = eye_base_empty
    eye_rotation_empty.parent_type = 'OBJECT'

    # Create eyeball geometry
    eyeball, iris, pupil = create_eyeball_with_pupil(
        name=f"Eyeball_{config.parent_marker_name}",
        eyeball_radius=config.eyeball_radius,
        pupil_radius=config.pupil_radius,
        iris_radius=config.iris_radius,
        location=(0, 0, 0),
        eyeball_material=eyeball_mat,
        iris_material=iris_mat,
        pupil_material=pupil_mat,
        parent=eye_rotation_empty
    )

    # Create gaze arrow if requested
    if config.show_gaze_arrow:
        gaze_arrow = create_gaze_arrow(
            name=f"GazeArrow_{config.parent_marker_name}",
            length=config.gaze_arrow_length,
            color=config.gaze_arrow_color,
            parent=eyeball
        )
        print(f"  ✓ Created gaze arrow")

    # Animate eyeball rotation
    print(f"  Animating eyeball rotation...")
    animate_eyeball_gaze(
        eyeball=eye_rotation_empty,
        eye_frames=eye_frames,
        frame_start=frame_start,
        keyframe_step=keyframe_step
    )

    print(f"  ✓ Eye visualization complete for {config.parent_marker_name}")
    print(f"    - Base orientation constrained to skull markers")
    print(f"    - Gaze rotations applied relative to base")

    return eyeball
