"""Load rigid body tracking + eye tracking into Blender with 3D eyeball visualization.

NEW: Refactored to use ONLY tidy CSV format for cleaner data loading!
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector


@dataclass
class TidyTrajectoryData:
    """Ergonomic trajectory data for Blender visualization."""

    frames: list[dict[str, tuple[float, float, float]]]
    """List of frames, each frame maps marker_name -> (x, y, z)"""

    marker_names: list[str]
    """Ordered list of all marker names"""

    frame_indices: list[int]
    """Frame numbers (may not start at 0)"""

    n_frames: int
    """Total number of frames"""

    n_markers: int
    """Total number of markers"""


@dataclass
class EyeTrackingConfig:
    """Configuration for eye tracking visualization."""

    csv_path: Path
    """Path to eye_tracking_results.csv (tidy format)"""

    parent_marker_name: str
    """Name of skull marker to attach eyeball to (e.g., 'M8' for right eye)"""

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


@dataclass
class RigidBodyVisualizationConfig:
    """Configuration for visualization."""

    csv_path: Path
    """Path to trajectory_data.csv (tidy format)"""

    topology_path: Path
    """Path to topology.json"""

    data_scale: float = 1.0
    """Scale factor for data (e.g. 0.001 for mm to meters)"""

    sphere_radius: float = 0.01
    """Radius of marker spheres (meters)"""

    tube_radius: float = 0.003
    """Radius of connecting tubes (meters)"""

    rigid_edge_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)
    """RGBA color for rigid edges (cyan)"""

    soft_edge_color: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 1.0)
    """RGBA color for soft edges (magenta)"""

    display_edge_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    """RGBA color for display edges (white)"""

    show_rigid_edges: bool = False
    """Show rigid constraint edges"""

    show_soft_edges: bool = True
    """Show soft constraint edges"""

    show_display_edges: bool = True
    """Show display edges"""

    frame_start: int = 0
    """Start frame for animation"""

    keyframe_step: int = 1
    """Keyframe every N frames (1=all frames, 2=every other, etc.)"""

    toy_csv_path: Path | None = None
    """Optional path to toy trajectory CSV (tidy format)"""

    toy_sphere_radius: float = 0.015
    """Radius of toy marker spheres (meters)"""

    toy_color: tuple[float, float, float] = (1.0, 0.8, 0.0)
    """RGB color for toy markers (gold/yellow)"""

    eye_tracking_configs: list[EyeTrackingConfig] | None = None
    """List of eye tracking configurations (one per eye)"""


# ============================================================================
# TIDY CSV LOADING - CLEAN AND ERGONOMIC
# ============================================================================


def load_tidy_csv(
    *,
    filepath: Path,
    data_scale: float = 1.0
) -> TidyTrajectoryData:
    """
    Load trajectory data from tidy CSV format.

    Tidy format expected columns:
    - frame: int
    - keypoint: str (marker name)
    - x: float
    - y: float
    - z: float (optional, defaults to 0.0)

    Args:
        filepath: Path to tidy CSV file
        data_scale: Scale factor to apply to coordinates (e.g., 0.001 for mm->m)

    Returns:
        TidyTrajectoryData with frames organized for easy Blender access
    """
    print(f"Loading tidy CSV: {filepath.name}")

    # First pass: collect all data
    data_dict: dict[str, dict[int, np.ndarray]] = {}
    frame_set: set[int] = set()

    with open(file=filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f=f)

        for row in reader:
            frame = int(row['frame'])
            keypoint = row['keypoint']
            x = float(row['x']) * data_scale
            y = float(row['y']) * data_scale
            z = (float(row['z']) if 'z' in row and row['z'] else 0.0) * data_scale

            frame_set.add(frame)

            if keypoint not in data_dict:
                data_dict[keypoint] = {}

            data_dict[keypoint][frame] = np.array([x, y, z])

    # Get sorted frames and markers
    frame_indices = sorted(frame_set)
    marker_names = sorted(data_dict.keys())
    n_frames = len(frame_indices)
    n_markers = len(marker_names)

    print(f"  Found {n_markers} markers × {n_frames} frames")

    # Convert to ergonomic format: list of dicts (one dict per frame)
    frames: list[dict[str, tuple[float, float, float]]] = []

    for frame_idx in frame_indices:
        frame_data: dict[str, tuple[float, float, float]] = {}

        for marker_name in marker_names:
            if frame_idx in data_dict[marker_name]:
                pos = data_dict[marker_name][frame_idx]
                frame_data[marker_name] = (float(pos[0]), float(pos[1]), float(pos[2]))
            else:
                # Missing data - use NaN or previous frame
                frame_data[marker_name] = (float('nan'), float('nan'), float('nan'))

        frames.append(frame_data)

    print(f"✓ Loaded trajectory data in tidy format")

    return TidyTrajectoryData(
        frames=frames,
        marker_names=marker_names,
        frame_indices=frame_indices,
        n_frames=n_frames,
        n_markers=n_markers
    )


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
# TOPOLOGY LOADING
# ============================================================================


def load_topology(*, filepath: Path) -> dict[str, object]:
    """Load topology from JSON file."""
    with open(file=filepath, mode='r', encoding='utf-8') as f:
        data = json.load(fp=f)
    return data['topology']


# ============================================================================
# BLENDER OBJECT CREATION
# ============================================================================


def clear_scene() -> None:
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def create_marker_material(
    *,
    name: str,
    color: tuple[float, float, float],
    metallic: float = 0.3,
    roughness: float = 0.4
) -> bpy.types.Material:
    """Create a material for markers."""
    mat = bpy.raw_trajectories.materials.new(name=name)
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


def create_edge_material(
    *,
    name: str,
    color: tuple[float, float, float, float],
    metallic: float = 0.8,
    roughness: float = 0.2
) -> bpy.types.Material:
    """Create a material for edges."""
    mat = bpy.raw_trajectories.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    nodes.clear()

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    node_bsdf.inputs['Base Color'].default_value = color
    node_bsdf.inputs['Metallic'].default_value = metallic
    node_bsdf.inputs['Roughness'].default_value = roughness
    node_bsdf.inputs['Alpha'].default_value = color[3]

    links = mat.node_tree.links
    links.new(input=node_bsdf.outputs['BSDF'], output=node_output.inputs['Surface'])

    mat.blend_method = 'BLEND'

    return mat


def create_sphere_marker(
    *,
    name: str,
    radius: float,
    location: tuple[float, float, float],
    material: bpy.types.Material,
    parent: bpy.types.Object
) -> bpy.types.Object:
    """Create a UV sphere marker."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=location,
        segments=32,
        ring_count=16
    )

    sphere = bpy.context.active_object
    sphere.name = name
    sphere.parent = parent

    if sphere.raw_trajectories.materials:
        sphere.raw_trajectories.materials[0] = material
    else:
        sphere.raw_trajectories.materials.append(material)

    bpy.ops.object.shade_smooth()

    return sphere


def create_edge_curves(
    *,
    name: str,
    edges: list[tuple[int, int]],
    marker_names: list[str],
    markers: dict[str, bpy.types.Object],
    initial_positions: dict[str, tuple[float, float, float]],
    radius: float,
    material: bpy.types.Material,
    parent: bpy.types.Object
) -> bpy.types.Object | None:
    """Create edge curves with drivers."""
    if not edges:
        return None

    curve_data = bpy.raw_trajectories.curves.new(name=f"{name}_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4
    curve_data.resolution_u = 2
    curve_data.fill_mode = 'FULL'
    curve_data.use_fill_caps = True

    curve_obj = bpy.raw_trajectories.objects.new(name=name, object_data=curve_data)
    bpy.context.collection.objects.link(object=curve_obj)
    curve_obj.parent = parent

    curve_obj.show_wire = False
    curve_obj.show_all_edges = False
    curve_obj.display_type = 'TEXTURED'

    if curve_obj.raw_trajectories.materials:
        curve_obj.raw_trajectories.materials[0] = material
    else:
        curve_obj.raw_trajectories.materials.append(material)

    for edge_idx, (i, j) in enumerate(edges):
        marker_i_name = marker_names[i]
        marker_j_name = marker_names[j]

        start_pos = Vector(initial_positions[marker_i_name])
        end_pos = Vector(initial_positions[marker_j_name])

        spline = curve_data.splines.new(type='NURBS')
        spline.points.add(count=1)

        spline.points[0].co = (*start_pos, 1.0)
        spline.points[1].co = (*end_pos, 1.0)

        spline.order_u = 2
        spline.use_endpoint_u = True

        marker_i = markers[marker_i_name]
        marker_j = markers[marker_j_name]

        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            fcurve = curve_data.driver_add(
                path=f'splines[{edge_idx}].points[0].co',
                index=axis_idx
            )
            driver = fcurve.driver
            driver.type = 'AVERAGE'

            var = driver.variables.new()
            var.name = 'loc'
            var.type = 'TRANSFORMS'

            target = var.targets[0]
            target.id = marker_i
            target.transform_type = f'LOC_{axis_name.upper()}'
            target.transform_space = 'WORLD_SPACE'

        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            fcurve = curve_data.driver_add(
                path=f'splines[{edge_idx}].points[1].co',
                index=axis_idx
            )
            driver = fcurve.driver
            driver.type = 'AVERAGE'

            var = driver.variables.new()
            var.name = 'loc'
            var.type = 'TRANSFORMS'

            target = var.targets[0]
            target.id = marker_j
            target.transform_type = f'LOC_{axis_name.upper()}'
            target.transform_space = 'WORLD_SPACE'

    return curve_obj


def animate_marker(
    *,
    marker: bpy.types.Object,
    trajectory: list[tuple[float, float, float]],
    frame_start: int,
    keyframe_step: int = 1
) -> None:
    """Animate a marker along its trajectory."""
    for frame_idx in range(0, len(trajectory), keyframe_step):
        frame = frame_start + frame_idx
        x, y, z = trajectory[frame_idx]
        marker.location = (x, y, z)
        marker.keyframe_insert(data_path="location", frame=frame)

    if marker.animation_data and marker.animation_data.action:
        for fcurve in marker.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


# ============================================================================
# EYE TRACKING VISUALIZATION
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

    if eyeball.raw_trajectories.materials:
        eyeball.raw_trajectories.materials[0] = eyeball_material
    else:
        eyeball.raw_trajectories.materials.append(eyeball_material)

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

    if iris.raw_trajectories.materials:
        iris.raw_trajectories.materials[0] = iris_material
    else:
        iris.raw_trajectories.materials.append(iris_material)

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

    if pupil.raw_trajectories.materials:
        pupil.raw_trajectories.materials[0] = pupil_material
    else:
        pupil.raw_trajectories.materials.append(pupil_material)

    bpy.ops.object.shade_smooth()

    return eyeball, iris, pupil


def create_gaze_arrow(
    *,
    name: str,
    length: float,
    color: tuple[float, float, float, float],
    parent: bpy.types.Object
) -> bpy.types.Object:
    """Create a gaze direction arrow."""
    curve_data = bpy.raw_trajectories.curves.new(name=f"{name}_Curve", type='CURVE')
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

    arrow = bpy.raw_trajectories.objects.new(name=name, object_data=curve_data)
    bpy.context.collection.objects.link(object=arrow)
    arrow.parent = parent

    mat = bpy.raw_trajectories.materials.new(name=f"{name}_Material")
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

    if arrow.raw_trajectories.materials:
        arrow.raw_trajectories.materials[0] = mat
    else:
        arrow.raw_trajectories.materials.append(mat)

    return arrow


def animate_eyeball_gaze(
    *,
    eyeball: bpy.types.Object,
    eye_frames: list[dict[str, float]],
    frame_start: int,
    keyframe_step: int = 1
) -> None:
    """Animate eyeball rotation based on gaze angles."""
    for frame_idx in range(0, len(eye_frames), keyframe_step):
        frame = frame_start + frame_idx
        eye_data = eye_frames[frame_idx]

        azimuth_rad = eye_data['gaze_azimuth_rad']
        elevation_rad = eye_data['gaze_elevation_rad']

        euler = Euler((elevation_rad, azimuth_rad, 0.0), 'XYZ')
        eyeball.rotation_euler = euler

        eyeball.keyframe_insert(data_path="rotation_euler", frame=frame)

    if eyeball.animation_data and eyeball.animation_data.action:
        for fcurve in eyeball.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


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
    """Create and animate eye tracking visualization with proper orientation constraints."""
    print(f"\nCreating eye visualization from {config.csv_path.name}...")

    eye_frames = load_eye_tracking_data(
        csv_path=config.csv_path,
        data_scale=data_scale
    )

    eyeball_mat = create_marker_material(
        name=f"Eye_{config.parent_marker_name}_Eyeball",
        color=config.eyeball_color,
        metallic=0.1,
        roughness=0.3
    )

    iris_mat = create_marker_material(
        name=f"Eye_{config.parent_marker_name}_Iris",
        color=config.iris_color,
        metallic=0.2,
        roughness=0.4
    )

    pupil_mat = create_marker_material(
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
    constraint_track = eye_base_empty.constraints.new(type='DAMPED_TRACK')
    constraint_track.target = track_to_marker
    constraint_track.track_axis = 'TRACK_NEGATIVE_Z'
    constraint_track.name = "TrackAwayFromOppositeEye"

    constraint_up = eye_base_empty.constraints.new(type='LOCKED_TRACK')
    constraint_up.target = up_marker
    constraint_up.track_axis = 'TRACK_Y'
    constraint_up.lock_axis = 'LOCK_Z'
    constraint_up.name = "UpTowardsHeadTop"

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

    if config.show_gaze_arrow:
        gaze_arrow = create_gaze_arrow(
            name=f"GazeArrow_{config.parent_marker_name}",
            length=config.gaze_arrow_length,
            color=config.gaze_arrow_color,
            parent=eyeball
        )
        print(f"  ✓ Created gaze arrow")

    animate_eyeball_gaze(
        eyeball=eye_rotation_empty,
        eye_frames=eye_frames,
        frame_start=frame_start,
        keyframe_step=keyframe_step
    )

    print(f"  ✓ Eye visualization complete for {config.parent_marker_name}")

    return eyeball


# ============================================================================
# MAIN VISUALIZATION
# ============================================================================


def create_rigid_body_visualization(*, config: RigidBodyVisualizationConfig) -> None:
    """Create complete rigid body visualization with eye tracking."""

    print("=" * 80)
    print("RIGID BODY + EYE TRACKING VISUALIZATION (TIDY CSV)")
    print("=" * 80)

    # Load topology
    print("\nLoading topology...")
    topology = load_topology(filepath=config.topology_path)

    # Load trajectory data in tidy format
    print("\nLoading trajectory data...")
    traj_data = load_tidy_csv(
        filepath=config.csv_path,
        data_scale=config.data_scale
    )

    print(f"\nTopology: {topology['name']}")
    print(f"Markers: {traj_data.n_markers}")
    print(f"Frames: {traj_data.n_frames}")

    # Create parent empty
    bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
    parent = bpy.context.active_object
    parent.name = f"RigidBody_{topology['name']}"
    parent.empty_display_size = 0.1

    # Create materials
    print("\nCreating materials...")

    marker_colors = [
        (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (0.5, 1.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 1.0, 0.5), (0.0, 1.0, 1.0), (0.0, 0.5, 1.0),
        (0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.5),
    ]

    marker_materials = {}
    for idx, marker_name in enumerate(traj_data.marker_names):
        color = marker_colors[idx % len(marker_colors)]
        mat = create_marker_material(name=f"Marker_{marker_name}", color=color)
        marker_materials[marker_name] = mat

    rigid_edge_mat = create_edge_material(name="RigidEdge", color=config.rigid_edge_color)
    soft_edge_mat = create_edge_material(name="SoftEdge", color=config.soft_edge_color)
    display_edge_mat = create_edge_material(name="DisplayEdge", color=config.display_edge_color)

    # Create markers
    print("\nCreating marker spheres...")
    markers = {}

    for marker_name in traj_data.marker_names:
        initial_pos = traj_data.frames[0][marker_name]
        sphere = create_sphere_marker(
            name=f"Marker_{marker_name}",
            radius=config.sphere_radius,
            location=initial_pos,
            material=marker_materials[marker_name],
            parent=parent
        )
        markers[marker_name] = sphere

    # Create edges
    print("\nCreating edge curves...")

    if config.show_rigid_edges and 'rigid_edges' in topology:
        print(f"  Creating rigid edges ({len(topology['rigid_edges'])} edges)...")
        create_edge_curves(
            name="RigidEdges",
            edges=topology['rigid_edges'],
            marker_names=traj_data.marker_names,
            markers=markers,
            initial_positions=traj_data.frames[0],
            radius=config.tube_radius,
            material=rigid_edge_mat,
            parent=parent
        )

    if config.show_soft_edges and 'soft_edges' in topology:
        print(f"  Creating soft edges ({len(topology['soft_edges'])} edges)...")
        create_edge_curves(
            name="SoftEdges",
            edges=topology['soft_edges'],
            marker_names=traj_data.marker_names,
            markers=markers,
            initial_positions=traj_data.frames[0],
            radius=config.tube_radius,
            material=soft_edge_mat,
            parent=parent
        )

    if config.show_display_edges and 'display_edges' in topology:
        print(f"  Creating display edges ({len(topology['display_edges'])} edges)...")
        create_edge_curves(
            name="DisplayEdges",
            edges=topology['display_edges'],
            marker_names=traj_data.marker_names,
            markers=markers,
            initial_positions=traj_data.frames[0],
            radius=config.tube_radius,
            material=display_edge_mat,
            parent=parent
        )

    # Animate markers
    print(f"\nAnimating markers...")
    for marker_name, sphere in markers.items():
        trajectory = [traj_data.frames[f][marker_name] for f in range(traj_data.n_frames)]
        animate_marker(
            marker=sphere,
            trajectory=trajectory,
            frame_start=config.frame_start,
            keyframe_step=config.keyframe_step
        )

    # Create eye visualizations
    if config.eye_tracking_configs is not None:
        print("\n" + "=" * 80)
        print("ADDING EYE TRACKING VISUALIZATIONS")
        print("=" * 80)

        for eye_config in config.eye_tracking_configs:
            if eye_config.parent_marker_name not in markers:
                print(f"⚠ Warning: Parent marker '{eye_config.parent_marker_name}' not found")
                continue

            if eye_config.track_to_marker_name not in markers:
                print(f"⚠ Warning: Track-to marker '{eye_config.track_to_marker_name}' not found")
                continue

            if eye_config.up_marker_name not in markers:
                print(f"⚠ Warning: Up marker '{eye_config.up_marker_name}' not found")
                continue

            create_eye_visualization(
                config=eye_config,
                parent_marker=markers[eye_config.parent_marker_name],
                track_to_marker=markers[eye_config.track_to_marker_name],
                up_marker=markers[eye_config.up_marker_name],
                rigid_body_parent=parent,
                frame_start=config.frame_start,
                keyframe_step=config.keyframe_step,
                data_scale=config.data_scale
            )

    # Set up scene
    bpy.context.scene.frame_start = config.frame_start
    bpy.context.scene.frame_end = config.frame_start + traj_data.n_frames - 1
    bpy.context.scene.frame_current = config.frame_start

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    print("\n" + "=" * 80)
    print("✓ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Timeline: frames {config.frame_start} to {config.frame_start + traj_data.n_frames - 1}")
    print(f"Markers: {len(markers)} spheres × {traj_data.n_frames // config.keyframe_step} keyframes")
    if config.eye_tracking_configs:
        print(f"Eyes: {len(config.eye_tracking_configs)} eyeball(s) with animated gaze")
    print("\nPress SPACE to play animation! 🎬")



config = RigidBodyVisualizationConfig(
    csv_path=Path(
        r"/examples/output/ferret_skull_tracking_EO5/trajectory_data.csv"
    ),
    topology_path=Path(
        r"/examples/output/ferret_skull_tracking_EO5/topology.json"
    ),
    data_scale=0.001,  # mm to meters
    sphere_radius=0.005,
    tube_radius=0.001,
    show_rigid_edges=False,
    show_soft_edges=True,
    show_display_edges=True,
    frame_start=0,
    keyframe_step=3,
    eye_tracking_configs=[
        # EyeTrackingConfig(
        #     csv_path=Path(r"C:\path\to\right_eye_tracking_results.csv"),
        #     parent_marker_name="right_eye",
        #     track_to_marker_name="left_eye",
        #     up_marker_name="base",
        #     eyeball_radius=0.0035,  # 3.5mm ferret eyeball
        #     pupil_radius=0.001,
        #     iris_radius=0.0025,
        #     local_offset=(0.0, 0.0, 0.0),
        #     show_gaze_arrow=True,
        #     gaze_arrow_length=0.03,
        #     gaze_arrow_color=(1.0, 0.0, 0.0, 1.0)
        # ),
    ]
)

# Run the visualization
clear_scene()
create_rigid_body_visualization(config=config)