"""Rigid body marker visualization for Blender.

This module handles loading and visualizing rigid body marker trajectories
from tidy CSV format, including marker spheres and connecting edges.
Includes raw data visualization alongside processed data.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


@dataclass
class TrajectoryData:
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
class RigidBodyConfig:
    """Configuration for rigid body visualization."""
    base_path: Path
    """Base path for data files, use to construct other paths if not provided"""

    trajectories_csv_path: Path | None = None
    """Path to trajectory.csv (tidy format)"""

    topology_path: Path | None = None
    """Path to topology.json"""

    raw_trajectories_csv_path: Path  | None = None
    """Path to raw_trajectory.csv """

    data_scale: float = 1.0
    """Scale factor for data (e.g. 0.001 for mm to meters)"""

    sphere_radius: float = 0.01
    """Radius of marker spheres (meters)"""

    tube_radius: float = 0.003
    """Radius of connecting tubes (meters)"""

    raw_sphere_radius: float = 0.003
    """Radius of raw data marker spheres (meters)"""

    raw_tube_radius: float = 0.0008
    """Radius of raw data connecting tubes (meters)"""

    rigid_edge_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)
    """RGBA color for rigid edges (cyan)"""

    soft_edge_color: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 1.0)
    """RGBA color for soft edges (magenta)"""

    display_edge_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    """RGBA color for display edges (white)"""

    raw_marker_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """RGB color for raw data markers (red)"""

    raw_edge_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.5)
    """RGBA color for raw data trajectory connections (semi-transparent red)"""

    show_rigid_edges: bool = True
    """Show rigid constraint edges"""

    show_soft_edges: bool = True
    """Show soft constraint edges"""

    show_display_edges: bool = True
    """Show display edges"""

    show_raw_data: bool = True
    """Show raw data markers"""

    show_trajecotories: bool = False
    """Show connections between consecutive raw data points"""

    frame_start: int = 0
    """Start frame for animation"""

    keyframe_step: int = 1
    """Keyframe every N frames (1=all frames, 2=every other, etc.)"""

    def __post_init__(self):
        if not self.trajectories_csv_path:
            self.trajectories_csv_path = self.base_path / "trajectories.csv"
        if not self.topology_path:
            self.topology_path = self.base_path / "topology.json"
        if not self.raw_trajectories_csv_path:
            self.raw_trajectories_csv_path = self.base_path / "raw_trajectories.csv"

# ============================================================================
# DATA LOADING
# ============================================================================


def load_tidy_csv(
    *,
    filepath: Path,
    data_scale: float = 1.0
) -> TrajectoryData:
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
            z =(float(row['z']) if 'z' in row and row['z'] else 0.0) * data_scale

            frame_set.add(frame)

            if keypoint not in data_dict:
                data_dict[keypoint] = {}

            data_dict[keypoint][frame] = np.array([x, y, z])

    # Get sorted frames and markers
    frame_indices = sorted(frame_set)
    marker_names = list(data_dict.keys())
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
                # Missing data - use NaN
                frame_data[marker_name] = (float('nan'), float('nan'), float('nan'))

        frames.append(frame_data)

    print(f"✓ Loaded trajectory data in tidy format")

    return TrajectoryData(
        frames=frames,
        marker_names=marker_names,
        frame_indices=frame_indices,
        n_frames=n_frames,
        n_markers=n_markers
    )


def load_topology(*, filepath: Path) -> dict[str, object]:
    """Load topology from JSON file."""
    with open(file=filepath, mode='r', encoding='utf-8') as f:
        data = json.load(fp=f)
    return data['topology']


# ============================================================================
# MATERIALS
# ============================================================================


def create_marker_material(
    *,
    name: str,
    color: tuple[float, float, float],
    metallic: float = 0.3,
    roughness: float = 0.4,
    emission_strength: float = 0.5
) -> bpy.types.Material:
    """Create a material for markers."""
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
    node_emission.inputs['Strength'].default_value = emission_strength

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
    mat = bpy.data.materials.new(name=name)
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


# ============================================================================
# GEOMETRY CREATION
# ============================================================================


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
        segments=16,
        ring_count=8
    )

    sphere = bpy.context.active_object
    sphere.name = name
    sphere.parent = parent

    if sphere.data.materials:
        sphere.data.materials[0] = material
    else:
        sphere.data.materials.append(material)

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

    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4
    curve_data.resolution_u = 2
    curve_data.fill_mode = 'FULL'
    curve_data.use_fill_caps = True

    curve_obj = bpy.data.objects.new(name=name, object_data=curve_data)
    bpy.context.collection.objects.link(object=curve_obj)
    curve_obj.parent = parent

    curve_obj.show_wire = False
    curve_obj.show_all_edges = False
    curve_obj.display_type = 'TEXTURED'

    if curve_obj.data.materials:
        curve_obj.data.materials[0] = material
    else:
        curve_obj.data.materials.append(material)

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
                f'splines[{edge_idx}].points[0].co',
                axis_idx
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
                f'splines[{edge_idx}].points[1].co',
                axis_idx
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


def create_trajectory_curves(
    *,
    name: str,
    marker_name: str,
    trajectory: list[tuple[float, float, float]],
    radius: float,
    material: bpy.types.Material,
    parent: bpy.types.Object,
    frame_start: int,
    keyframe_step: int = 1
) -> bpy.types.Object | None:
    """Create a curve showing the trajectory path for a single marker."""
    # Filter out NaN positions
    valid_positions = [
        (i, pos) for i, pos in enumerate(trajectory)
        if not (np.isnan(pos[0]) or np.isnan(pos[1]) or np.isnan(pos[2]))
    ]

    if len(valid_positions) < 2:
        return None

    curve_data = bpy.data.curves.new(name=f"{name}_{marker_name}_Trail", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 3
    curve_data.resolution_u = 4
    curve_data.fill_mode = 'FULL'
    curve_data.use_fill_caps = True

    curve_obj = bpy.data.objects.new(name=f"{name}_{marker_name}", object_data=curve_data)
    bpy.context.collection.objects.link(object=curve_obj)
    curve_obj.parent = parent

    if curve_obj.data.materials:
        curve_obj.data.materials[0] = material
    else:
        curve_obj.data.materials.append(material)

    # Create spline with all valid positions
    spline = curve_data.splines.new(type='POLY')
    spline.points.add(count=len(valid_positions) - 1)

    for point_idx, (_, pos) in enumerate(valid_positions):
        spline.points[point_idx].co = (*pos, 1.0)

    return curve_obj


# ============================================================================
# ANIMATION
# ============================================================================


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
# MAIN RIGID BODY VISUALIZATION
# ============================================================================


def create_rigid_body_visualization(
    *,
    config: RigidBodyConfig,
    parent_name: str = "RigidBody"
) -> tuple[bpy.types.Object, dict[str, bpy.types.Object], TrajectoryData, dict[str, object]]:
    """
    Create rigid body marker visualization.

    Args:
        config: Rigid body configuration
        parent_name: Name for parent empty object

    Returns:
        Tuple of (parent_object, markers_dict, trajectory_data, topology)
    """
    print("=" * 80)
    print("RIGID BODY VISUALIZATION")
    print("=" * 80)

    # Load data
    print("\nLoading topology...")
    topology = load_topology(filepath=config.topology_path)

    print("\nLoading trajectory data...")
    traj_data = load_tidy_csv(
        filepath=config.trajectories_csv_path,
        data_scale=config.data_scale
    )

    # Load raw data if provided
    raw_traj_data: TrajectoryData | None = None
    if config.raw_trajectories_csv_path and config.show_raw_data:
        print("\nLoading raw trajectory data...")
        raw_traj_data = load_tidy_csv(
            filepath=config.raw_trajectories_csv_path,
            data_scale=config.data_scale
        )

    print(f"\nTopology: {topology['name']}")
    print(f"Markers: {traj_data.n_markers}")
    print(f"Frames: {traj_data.n_frames}")
    if raw_traj_data:
        print(f"Raw data frames: {raw_traj_data.n_frames}")

    # Create parent empty
    bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
    parent = bpy.context.active_object
    parent.name = f"{parent_name}_{topology['name']}"
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

    # Raw data material (red with slight emission)
    raw_marker_mat = create_marker_material(
        name="RawMarker",
        color=config.raw_marker_color,
        metallic=0.5,
        roughness=0.6,
        emission_strength=0.3
    )

    rigid_edge_mat = create_edge_material(name="RigidEdge", color=config.rigid_edge_color)
    soft_edge_mat = create_edge_material(name="SoftEdge", color=config.soft_edge_color)
    display_edge_mat = create_edge_material(name="DisplayEdge", color=config.display_edge_color)
    raw_edge_mat = create_edge_material(name="RawEdge", color=config.raw_edge_color)

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

    # Create raw data markers
    raw_markers = {}
    if raw_traj_data and config.show_raw_data:
        print(f"\nCreating raw data markers ({raw_traj_data.n_markers} markers)...")
        for marker_name in raw_traj_data.marker_names:
            initial_pos = raw_traj_data.frames[0][marker_name]
            sphere = create_sphere_marker(
                name=f"RawMarker_{marker_name}",
                radius=config.raw_sphere_radius,
                location=initial_pos,
                material=raw_marker_mat,
                parent=parent
            )
            raw_markers[marker_name] = sphere

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
            radius=config.tube_radius/2,
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

    # Create raw data trajectory connections
    # if raw_traj_data and config.show_raw_data and config.show_trajecotories:
    #     print(f"\nCreating raw data trajectory connections...")
    #     for marker_name in raw_traj_data.marker_names:
    #         trajectory = [raw_traj_data.frames[f][marker_name] for f in range(raw_traj_data.n_frames)]
    #         create_trajectory_curves(
    #             name="RawTrail",
    #             marker_name=marker_name,
    #             trajectory=trajectory,
    #             radius=config.raw_tube_radius,
    #             material=raw_edge_mat,
    #             parent=parent,
    #             frame_start=config.frame_start,
    #             keyframe_step=config.keyframe_step
    #         )

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

    # Animate raw markers
    if raw_traj_data and raw_markers:
        print(f"\nAnimating raw data markers...")
        for marker_name, sphere in raw_markers.items():
            trajectory = [raw_traj_data.frames[f][marker_name] for f in range(raw_traj_data.n_frames)]
            animate_marker(
                marker=sphere,
                trajectory=trajectory,
                frame_start=config.frame_start,
                keyframe_step=config.keyframe_step
            )

    print("\n✓ Rigid body visualization complete")
    print(f"  Processed markers: {len(markers)} spheres")
    if raw_markers:
        print(f"  Raw data markers: {len(raw_markers)} spheres")
    print(f"  Frames: {config.frame_start} to {config.frame_start + traj_data.n_frames - 1}")

    return parent, markers, traj_data, topology


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# Example usage
config = RigidBodyConfig(
    base_path =Path(
        r"/old/old_broken_rigid_body_tracker\examples\output\2025-07-11_ferret_757_EyeCameras_P43_E15__1_0m_37s-1m_37s"),
    data_scale=0.001,  # mm to meters
    sphere_radius=0.003,
    tube_radius=0.001,
    raw_sphere_radius=0.002,
    raw_tube_radius=0.0008,
    show_rigid_edges=False,
    show_soft_edges=True,
    show_display_edges=True,
    show_raw_data=True,
    frame_start=0,
    keyframe_step=3,
)

create_rigid_body_visualization(config=config)