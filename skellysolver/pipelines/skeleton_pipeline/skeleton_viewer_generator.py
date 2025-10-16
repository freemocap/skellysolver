"""Generate interactive HTML viewer for skeleton trajectories.

Creates a self-contained HTML file with embedded trajectory data and
skeleton topology for 3D visualization.
"""

import json
import logging
from pathlib import Path

from skellysolver.data.trajectory_dataset import TrajectoryDataset
from skellysolver.pipelines.skeleton_pipeline.skeleton_topology import extract_skeleton_topology, \
    SkeletonTopology
from skellysolver.solvers.constraints.skeleton_constraint import SkeletonConstraint
from skellysolver.viewers.html_viewers.templates.template_helpers import get_template_path

logger = logging.getLogger(__name__)




class SkeletonViewerData:
    """Container for viewer data preparation."""
    
    def __init__(
        self,
        *,
        topology: SkeletonTopology,
        raw_data: TrajectoryDataset,
        optimized_data: TrajectoryDataset | None = None,
        ground_truth_data: TrajectoryDataset | None = None
    ) -> None:
        self.topology = topology
        self.raw_data = raw_data
        self.optimized_data = optimized_data
        self.ground_truth_data = ground_truth_data
    
    def to_json_dict(self) -> dict[str, object]:
        """Convert to JSON-serializable dictionary."""
        n_frames = self.raw_data.n_frames
        
        # Build frame data
        frames_data: list[dict[str, float]] = []
        
        for frame_idx in range(n_frames):
            frame_dict: dict[str, float] = {"frame": frame_idx}
            
            # Add raw data
            for kp_name in self.topology.keypoint_names:
                if kp_name in self.raw_data.data:
                    pos = self.raw_data.data[kp_name].data[frame_idx]
                    frame_dict[f"raw_{kp_name}_x"] = float(pos[0])
                    frame_dict[f"raw_{kp_name}_y"] = float(pos[1])
                    frame_dict[f"raw_{kp_name}_z"] = float(pos[2])
            
            # Add optimized data if available
            if self.optimized_data is not None:
                for kp_name in self.topology.keypoint_names:
                    if kp_name in self.optimized_data.data:
                        pos = self.optimized_data.data[kp_name].data[frame_idx]
                        frame_dict[f"optimized_{kp_name}_x"] = float(pos[0])
                        frame_dict[f"optimized_{kp_name}_y"] = float(pos[1])
                        frame_dict[f"optimized_{kp_name}_z"] = float(pos[2])
            
            # Add ground truth if available
            if self.ground_truth_data is not None:
                for kp_name in self.topology.keypoint_names:
                    if kp_name in self.ground_truth_data.data:
                        pos = self.ground_truth_data.data[kp_name].data[frame_idx]
                        frame_dict[f"gt_{kp_name}_x"] = float(pos[0])
                        frame_dict[f"gt_{kp_name}_y"] = float(pos[1])
                        frame_dict[f"gt_{kp_name}_z"] = float(pos[2])
            
            frames_data.append(frame_dict)
        
        return {
            "topology": self.topology.model_dump(),
            "frames": frames_data,
            "n_frames": n_frames,
            "has_optimized": self.optimized_data is not None,
            "has_ground_truth": self.ground_truth_data is not None
        }


def generate_skeleton_viewer(
    *,
    skeleton: SkeletonConstraint,
    raw_data: TrajectoryDataset,
    optimized_data: TrajectoryDataset | None = None,
    ground_truth_data: TrajectoryDataset | None = None,
    output_path: Path,
    rigidity_threshold: float = 0.8
) -> None:
    """Generate interactive HTML viewer for skeleton trajectories.
    
    Args:
        skeleton: Skeleton constraint definition
        raw_data: Raw/noisy trajectory data
        optimized_data: Optimized trajectory data (optional)
        ground_truth_data: Ground truth data (optional)
        output_path: Where to save the HTML file
        rigidity_threshold: Threshold for rigid vs flexible edges
    """
    logger.info("Generating skeleton viewer...")
    
    # Extract topology
    topology = extract_skeleton_topology(
        skeleton=skeleton,
        rigidity_threshold=rigidity_threshold
    )
    
    # Prepare viewer data
    viewer_data = SkeletonViewerData(
        topology=topology,
        raw_data=raw_data,
        optimized_data=optimized_data,
        ground_truth_data=ground_truth_data
    )
    
    # Convert to JSON
    logger.info("Preparing data for viewer...")
    data_dict = viewer_data.to_json_dict()
    data_json = json.dumps(obj=data_dict, indent=2)
    
    # Load template
    template_path = get_template_path(viewer_type="skeleton")
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    logger.info(f"Loading template from: {template_path}")
    with open(file=template_path, mode='r', encoding='utf-8') as f:
        template_html = f.read()
    
    # Replace placeholder with actual data
    html_content = template_html.replace("__VIEWER_DATA_JSON__", data_json)
    
    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing viewer to: {output_path}")
    with open(file=output_path, mode='w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Calculate file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Viewer size: {size_mb:.2f} MB")
    logger.info("✓ Skeleton viewer generated successfully")
