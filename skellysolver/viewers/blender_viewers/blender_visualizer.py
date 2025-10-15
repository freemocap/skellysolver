"""Main visualization coordinator for Blender rigid body + eye tracking.

This module brings together rigid body marker tracking and eye tracking
visualizations into a single coordinated animation.
"""

from dataclasses import dataclass
from pathlib import Path

import bpy

from blender_eye_tracking import EyeTrackingConfig, create_eye_visualization
from blender_rigid_body import RigidBodyConfig, create_rigid_body_visualization


@dataclass
class VisualizationConfig:
    """Complete configuration for rigid body + eye tracking visualization."""

    # Rigid body configuration
    rigid_body: RigidBodyConfig
    """Configuration for rigid body marker visualization"""

    # Eye tracking configurations (optional)
    eye_tracking_configs: list[EyeTrackingConfig] | None = None
    """List of eye tracking configurations (one per eye)"""


# ============================================================================
# UTILITIES
# ============================================================================


def clear_scene() -> None:
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_viewport() -> None:
    """Set up viewport for visualization."""
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'


# ============================================================================
# MAIN COORDINATOR
# ============================================================================


def create_complete_visualization(*, config: VisualizationConfig) -> None:
    """
    Create complete rigid body + eye tracking visualization.

    This is the main entry point that coordinates creation of:
    1. Rigid body marker visualization
    2. Eye tracking visualizations (if configured)
    3. Scene setup

    Args:
        config: Complete visualization configuration
    """
    print("=" * 80)
    print("COMPLETE VISUALIZATION: RIGID BODY + EYE TRACKING")
    print("=" * 80)

    # Create rigid body visualization
    parent, markers, traj_data, topology = create_rigid_body_visualization(
        config=config.rigid_body,
        parent_name="RigidBody"
    )

    # Create eye tracking visualizations
    if config.eye_tracking_configs is not None:
        print("\n" + "=" * 80)
        print("ADDING EYE TRACKING VISUALIZATIONS")
        print("=" * 80)

        for eye_config in config.eye_tracking_configs:
            # Validate markers exist
            if eye_config.parent_marker_name not in markers:
                print(f"⚠ Warning: Parent marker '{eye_config.parent_marker_name}' not found")
                continue

            if eye_config.track_to_marker_name not in markers:
                print(f"⚠ Warning: Track-to marker '{eye_config.track_to_marker_name}' not found")
                continue

            if eye_config.up_marker_name not in markers:
                print(f"⚠ Warning: Up marker '{eye_config.up_marker_name}' not found")
                continue

            # Create eye visualization
            create_eye_visualization(
                config=eye_config,
                parent_marker=markers[eye_config.parent_marker_name],
                track_to_marker=markers[eye_config.track_to_marker_name],
                up_marker=markers[eye_config.up_marker_name],
                rigid_body_parent=parent,
                frame_start=config.rigid_body.frame_start,
                keyframe_step=config.rigid_body.keyframe_step,
                data_scale=config.rigid_body.data_scale
            )

    # Set up scene timeline
    bpy.context.scene.frame_start = config.rigid_body.frame_start
    bpy.context.scene.frame_end = config.rigid_body.frame_start + traj_data.n_frames - 1
    bpy.context.scene.frame_current = config.rigid_body.frame_start

    # Set up viewport
    setup_viewport()

    # Final summary
    print("\n" + "=" * 80)
    print("✓ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Timeline: frames {config.rigid_body.frame_start} to {config.rigid_body.frame_start + traj_data.n_frames - 1}")
    print(f"Markers: {len(markers)} spheres × {traj_data.n_frames // config.rigid_body.keyframe_step} keyframes")
    if config.eye_tracking_configs:
        print(f"Eyes: {len(config.eye_tracking_configs)} eyeball(s) with animated gaze")
    print("\nPress SPACE to play animation! 🎬")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage: Ferret eye tracking visualization
    
    FERRET EYE DIMENSIONS:
    - Axial length: ~7mm (front-to-back diameter of eyeball)
    - Eyeball radius: 3.5mm (half of axial length)
    - Pupil: ~1-1.5mm radius (dilated in low light)
    - Iris: ~2.5-3mm radius
    
    HOW THE EYE ORIENTATION SYSTEM WORKS:
    
    The eye visualization uses a constraint-based hierarchy:
    
    1. EYE MARKER (skull) 
       ↓ (parent)
    2. EYE BASE EMPTY (constrained orientation)
       - Damped Track: Back of eye (-Z) points towards opposite eye marker
       - Locked Track: Up direction (+Y) points towards head top marker
       - Result: Eye naturally faces outward from skull
       ↓ (parent)
    3. EYE ROTATION EMPTY (gaze animation)
       - Animated with azimuth/elevation from eye tracking
       - Rotations are relative to the base orientation
       ↓ (parent)
    4. EYEBALL MESH (visual geometry)
       - Sphere with iris and pupil
       - Gaze arrow shows final viewing direction
    
    This ensures:
    - Eye moves with the skull (parented to marker)
    - Eye points outward by default (constraints on base empty)
    - Eye rotates with head movements (constraints follow parent)
    - Gaze tracking adds movements on top (rotation empty animation)
    - Final gaze = head orientation × base orientation × eye rotation ✓
    
    KEY MARKERS TO IDENTIFY:
    - parent_marker_name: Marker at this eye's location (e.g., "right_eye")
    - track_to_marker_name: Marker to point AWAY from (opposite eye or head center)
    - up_marker_name: Marker defining "up" direction (head top, "base", etc.)
    """
    
    # Example configuration
    config = VisualizationConfig(
        rigid_body=RigidBodyConfig(
            csv_path=Path(
                r"C:\Users\jonma\github_repos\jonmatthis\bs\python_code\rigid_body_tracker\examples\output\2025-07-11_ferret_757_EyeCameras_P43_E15__1_0m_37s-1m_37s\trajectory_data.csv"
            ),
            topology_path=Path(
                r"C:\Users\jonma\github_repos\jonmatthis\bs\python_code\rigid_body_tracker\examples\output\2025-07-11_ferret_757_EyeCameras_P43_E15__1_0m_37s-1m_37s\topology.json"
            ),
            data_scale=0.001,  # mm to meters
            sphere_radius=0.005,
            tube_radius=0.001,
            show_rigid_edges=False,
            show_soft_edges=True,
            show_display_edges=True,
            frame_start=0,
            keyframe_step=3,
        ),
        eye_tracking_configs=[
            # Right eye configuration
            EyeTrackingConfig(
                csv_path=Path(
                    r"C:\Users\jonma\github_repos\jonmatthis\bs\python_code\eye_tracking\output\eye_tracking_demo\eye_tracking_results.csv"
                ),
                parent_marker_name="right_eye",     # Right eye marker
                track_to_marker_name="left_eye",    # Left eye marker (point away from this)
                up_marker_name="base",              # Head top/base marker for up direction
                eyeball_radius=0.0035,              # 3.5mm - half of 7mm axial length
                pupil_radius=0.001,                 # 1mm - ferret pupil
                iris_radius=0.0025,                 # 2.5mm - ferret iris
                local_offset=(0.0, 0.0, 0.0),       # Adjust if needed
                show_gaze_arrow=True,
                gaze_arrow_length=0.03,             # 30mm arrow
                gaze_arrow_color=(1.0, 0.0, 0.0, 1.0)  # Red arrow
            ),
            # Left eye configuration (uncomment if you have left eye data)
            # EyeTrackingConfig(
            #     csv_path=Path(
            #         r"C:\path\to\left_eye_tracking_results.csv"
            #     ),
            #     parent_marker_name="left_eye",      # Left eye marker
            #     track_to_marker_name="right_eye",   # Right eye marker (point away)
            #     up_marker_name="base",              # Same head top marker
            #     eyeball_radius=0.0035,
            #     pupil_radius=0.001,
            #     iris_radius=0.0025,
            #     local_offset=(0.0, 0.0, 0.0),
            #     show_gaze_arrow=True,
            #     gaze_arrow_length=0.03,
            #     gaze_arrow_color=(0.0, 1.0, 0.0, 1.0)  # Green arrow
            # ),
        ]
    )

    # Run the visualization
    clear_scene()
    create_complete_visualization(config=config)
