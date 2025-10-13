"""Eye tracking viewer generator.

Generates interactive HTML viewer for eye tracking results.
"""

import pandas as pd
from pathlib import Path
from typing import Any

from .base_viewer import BaseViewerGenerator


class EyeTrackingViewerGenerator(BaseViewerGenerator):
    """Generate interactive HTML viewer for eye tracking.
    
    Creates viewer that displays:
    - Video with overlaid pupil tracking
    - Gaze direction visualization
    - Pupil dilation graph
    - Reprojection error metrics
    """
    
    def __init__(self, *, template_path: Path | None = None) -> None:
        """Initialize eye tracking viewer generator.
        
        Args:
            template_path: Optional custom template path
        """
        super().__init__()
        
        if template_path is None:
            # Use default template
            template_path = Path(__file__).parent / "templates" / "eye_tracking_viewer.html"
        
        self.template_path = template_path
    
    def _get_template_path(self) -> Path:
        """Get template path.
        
        Returns:
            Path to eye tracking viewer template
        """
        return self.template_path
    
    def generate(
        self,
        *,
        output_dir: Path,
        data_csv_path: Path,
        video_path: Path | None = None
    ) -> Path:
        """Generate eye tracking viewer.
        
        Args:
            output_dir: Output directory
            data_csv_path: Path to eye_tracking_results.csv
            video_path: Optional eye camera video
            
        Returns:
            Path to generated eye_tracking_viewer.html
        """
        self._ensure_output_dir(output_dir=output_dir)
        
        # Output path
        output_path = output_dir / "eye_tracking_viewer.html"
        
        # Check template exists
        template = self._get_template_path()
        if not template.exists():
            print(f"  ⚠ Template not found: {template}")
            print(f"  → Using simple viewer fallback")
            return self._generate_simple_viewer(
                output_path=output_path,
                data_csv_path=data_csv_path
            )
        
        # Copy template
        self._copy_template(
            template_path=template,
            output_path=output_path
        )
        
        # Read data
        data_json = self._read_csv_as_json(csv_path=data_csv_path)
        
        # Handle video
        video_filename = ""
        video_loaded = "false"
        if video_path is not None and video_path.exists():
            video_filename = self._copy_video(
                video_path=video_path,
                output_dir=output_dir
            )
            video_loaded = "true"
        
        # Get frame count
        df = pd.read_csv(filepath_or_buffer=data_csv_path)
        n_frames = len(df)
        
        # Embed data
        replacements = {
            "__N_FRAMES__": str(n_frames),
            "__VIDEO_SRC__": video_filename,
            "__VIDEO_LOADED__": video_loaded,
            "__FRAME_SLIDER_MAX__": str(max(n_frames - 1, 0)),
        }
        
        self._embed_data_in_html(
            html_path=output_path,
            data_json=data_json,
            replacements=replacements
        )
        
        self.last_generated_path = output_path
        
        return output_path
    
    def _generate_simple_viewer(
        self,
        *,
        output_path: Path,
        data_csv_path: Path
    ) -> Path:
        """Generate simple fallback viewer.
        
        Args:
            output_path: Output HTML path
            data_csv_path: Path to data CSV
            
        Returns:
            Path to generated HTML
        """
        # Read data
        df = pd.read_csv(filepath_or_buffer=data_csv_path)
        
        # Compute statistics
        if "gaze_azimuth_deg" in df.columns:
            azimuth_range = df["gaze_azimuth_deg"].max() - df["gaze_azimuth_deg"].min()
            elevation_range = df["gaze_elevation_deg"].max() - df["gaze_elevation_deg"].min()
        else:
            azimuth_range = 0.0
            elevation_range = 0.0
        
        if "pupil_scale" in df.columns:
            pupil_mean = df["pupil_scale"].mean()
            pupil_std = df["pupil_scale"].std()
        else:
            pupil_mean = 0.0
            pupil_std = 0.0
        
        # Generate simple HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Eye Tracking Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Eye Tracking Results</h1>
    
    <div class="info">
        <h2>Dataset Info</h2>
        <p><strong>Frames:</strong> {len(df)}</p>
    </div>
    
    <div class="info">
        <h2>Gaze Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Azimuth Range</td>
                <td>{azimuth_range:.1f}°</td>
            </tr>
            <tr>
                <td>Elevation Range</td>
                <td>{elevation_range:.1f}°</td>
            </tr>
            <tr>
                <td>Pupil Scale Mean</td>
                <td>{pupil_mean:.3f}</td>
            </tr>
            <tr>
                <td>Pupil Scale Std</td>
                <td>{pupil_std:.3f}</td>
            </tr>
        </table>
    </div>
    
    <div class="info">
        <h2>Data Preview</h2>
        <p>First 10 frames:</p>
        {df.head(10).to_html()}
    </div>
    
    <div class="info">
        <p><em>Note: Full interactive viewer template not found. This is a basic fallback.</em></p>
    </div>
</body>
</html>"""
        
        output_path.write_text(data=html, encoding='utf-8')
        
        return output_path


def generate_rigid_body_viewer(
    *,
    output_dir: Path,
    data_csv_path: Path,
    topology_json_path: Path,
    video_path: Path | None = None,
    template_path: Path | None = None
) -> Path:
    """Convenience function to generate rigid body viewer.
    
    Args:
        output_dir: Output directory
        data_csv_path: Path to trajectory_data.csv
        topology_json_path: Path to topology.json
        video_path: Optional video path
        template_path: Optional custom template
        
    Returns:
        Path to generated HTML
    """
    from .rigid_body_viewer import RigidBodyViewerGenerator
    
    generator = RigidBodyViewerGenerator(template_path=template_path)
    
    return generator.generate(
        output_dir=output_dir,
        data_csv_path=data_csv_path,
        topology_json_path=topology_json_path,
        video_path=video_path
    )


def generate_eye_tracking_viewer(
    *,
    output_dir: Path,
    data_csv_path: Path,
    video_path: Path | None = None,
    template_path: Path | None = None
) -> Path:
    """Convenience function to generate eye tracking viewer.
    
    Args:
        output_dir: Output directory
        data_csv_path: Path to eye_tracking_results.csv
        video_path: Optional eye camera video
        template_path: Optional custom template
        
    Returns:
        Path to generated HTML
    """
    from .eye_tracking_viewer import EyeTrackingViewerGenerator
    
    generator = EyeTrackingViewerGenerator(template_path=template_path)
    
    return generator.generate(
        output_dir=output_dir,
        data_csv_path=data_csv_path,
        video_path=video_path
    )
