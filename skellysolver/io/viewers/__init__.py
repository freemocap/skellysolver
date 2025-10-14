"""Viewers module for SkellySolver IO.

Provides viewer generators for interactive HTML visualizations:
- RigidBodyViewerGenerator: Rigid body tracking viewer
- EyeTrackingViewerGenerator: Eye tracking viewer
- Convenience functions for quick viewer generation

Usage:
    from skellysolver.io.viewers import generate_rigid_body_viewer
    
    viewer_path = generate_rigid_body_viewer(
        output_dir=Path("output/"),
        data_csv_path=Path("output/trajectory_data.csv"),
        topology_json_path=Path("output/topology.json"),
    )
    
    print(f"Open {viewer_path} in a browser!")
"""

from skellysolver.io.viewers.html_viewers.base_viewer import (
    BaseViewerGenerator,
    HTMLViewerGenerator,
)
from skellysolver.io.viewers.html_viewers.eye_viewer import generate_eye_tracking_viewer, generate_rigid_body_viewer
from skellysolver.io.viewers.html_viewers.rigid_viewer import RigidBodyViewerGenerator


__all__ = [
    # Base viewers
    "BaseViewerGenerator",
    "HTMLViewerGenerator",
    # Specialized viewers
    "RigidBodyViewerGenerator",
    # Convenience functions
    "generate_rigid_body_viewer",
    "generate_eye_tracking_viewer",
]


