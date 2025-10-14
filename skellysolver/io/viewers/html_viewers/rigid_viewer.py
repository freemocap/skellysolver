"""Rigid body viewer generator.

Generates interactive HTML viewer for rigid body tracking results.
"""

import pandas as pd
from pathlib import Path
from typing import Any

from .base_viewer import BaseViewerGenerator


class RigidBodyViewerGenerator(BaseViewerGenerator):
    """Generate interactive HTML viewer for rigid body tracking.

    Creates viewer that displays:
    - 3D visualization of markers and edges
    - Noisy vs optimized trajectories
    - Playback controls
    - Metric displays
    """

    template_path: Path | None = Path(__file__).parent / "templates" / "rigid_body_viewer.html"

    def _get_template_path(self) -> Path:
        """Get template path.

        Returns:
            Path to rigid body viewer template
        """
        return self.template_path

    def generate(
        self,
        *,
        output_dir: Path,
        data_csv_path: Path,
            raw_csv_path: Path,
        topology_json_path: Path,
        video_path: Path | None = None
    ) -> Path:
        """Generate rigid body viewer.

        Args:
            output_dir: Output directory
            data_csv_path: Path to trajectory_data.csv
            topology_json_path: Path to topology.json
            video_path: Optional video path

        Returns:
            Path to generated rigid_body_viewer.html
        """
        self._ensure_output_dir(output_dir=output_dir)

        # Output path
        output_path = output_dir / "rigid_body_viewer.html"

        # Check template exists
        template = self._get_template_path()
        if not template.exists():
            print(f"  âš  Template not found: {template}")
            print(f"  â†’ Using simple viewer fallback")
            return self._generate_simple_viewer(
                output_path=output_path,
                data_csv_path=data_csv_path,
                topology_json_path=topology_json_path
            )

        # Copy template
        self._copy_template(
            template_path=template,
            output_path=output_path
        )

        # Read data
        data_json = self._read_csv_as_json(csv_path=data_csv_path)

        # Read topology
        import json
        with open(topology_json_path, mode='r') as f:
            topology_data = json.load(fp=f)
        topology_json = json.dumps(topology_data, indent=2)

        # Handle video
        video_filename = ""
        if video_path is not None and video_path.exists():
            video_filename = self._copy_video(
                video_path=video_path,
                output_dir=output_dir
            )

        # Get frame count
        df = pd.read_csv(filepath_or_buffer=data_csv_path)
        n_frames = len(df)

        # Embed data
        replacements = {
            "__TOPOLOGY_JSON__": topology_json,
            "__VIDEO_SRC__": video_filename,
            "__N_FRAMES__": str(n_frames),
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
        data_csv_path: Path,
        topology_json_path: Path
    ) -> Path:
        """Generate simple fallback viewer.

        Used when template is not found.

        Args:
            output_path: Output HTML path
            data_csv_path: Path to data CSV
            topology_json_path: Path to topology JSON

        Returns:
            Path to generated HTML
        """
        # Read data
        df = pd.read_csv(filepath_or_buffer=data_csv_path)

        import json
        with open(topology_json_path, mode='r') as f:
            topology = json.load(fp=f)

        # Generate simple HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Rigid Body Tracking Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Rigid Body Tracking Results</h1>
    
    <div class="info">
        <h2>Dataset Info</h2>
        <p><strong>Frames:</strong> {len(df)}</p>
        <p><strong>Markers:</strong> {len(topology['topology']['marker_names'])}</p>
        <p><strong>Marker Names:</strong> {', '.join(topology['topology']['marker_names'])}</p>
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