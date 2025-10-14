"""Eye tracking viewer generator with full 3D + 2D visualization.

Generates interactive HTML viewer for eye tracking results showing:
- 3D eye model (eyeball, pupil, tear duct, gaze ray)
- 2D image projection with observed and reprojected points
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

from .base_viewer import BaseViewerGenerator


class EyeTrackingViewerGenerator(BaseViewerGenerator):
    """Generate interactive HTML viewer for eye tracking.

    Creates dual-view visualization:
    - Left: 3D eye model with animated gaze and pupil
    - Right: 2D image plane with observed vs reprojected pupil points
    """

    def _get_template_path(self) -> Path:
        """Get template path.

        Returns:
            Path to eye tracking viewer template
        """
        return Path(__file__).parent / "templates" / "eye_tracking_viewer.html"

    def generate(
        self,
        *,
        output_dir: Path,
        data_csv_path: Path,
        eye_model_json_path: Path | None = None,
        video_path: Path | None = None
    ) -> Path:
        """Generate eye tracking viewer.

        Args:
            output_dir: Output directory
            data_csv_path: Path to eye_tracking_results.csv
            eye_model_json_path: Path to eye_model.json (optional)
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
        df = pd.read_csv(filepath_or_buffer=data_csv_path)
        n_frames = len(df)

        # Prepare visualization data
        viz_data = self._prepare_visualization_data(
            df=df,
            eye_model_json_path=eye_model_json_path
        )

        data_json = json.dumps(viz_data, indent=2)

        # Load eye model if available
        eye_model_json = "{}"
        if eye_model_json_path is not None and eye_model_json_path.exists():
            with open(eye_model_json_path, mode='r') as f:
                eye_model_data = json.load(fp=f)
            eye_model_json = json.dumps(eye_model_data, indent=2)

        # Handle video
        video_filename = ""
        video_loaded = "false"
        if video_path is not None and video_path.exists():
            video_filename = self._copy_video(
                video_path=video_path,
                output_dir=output_dir
            )
            video_loaded = "true"

        # Embed data
        replacements = {
            "__N_FRAMES__": str(n_frames),
            "__EYE_MODEL_JSON__": eye_model_json,
            "__VIDEO_SRC__": video_filename,
            "__VIDEO_LOADED__": video_loaded,
        }

        self._embed_data_in_html(
            html_path=output_path,
            data_json=data_json,
            replacements=replacements
        )

        self.last_generated_path = output_path

        return output_path

    def _prepare_visualization_data(
        self,
        *,
        df: pd.DataFrame,
        eye_model_json_path: Path | None
    ) -> list[dict]:
        """Prepare data for visualization.

        Structures data for both 3D and 2D views.

        Args:
            df: DataFrame with eye tracking results
            eye_model_json_path: Optional path to eye model JSON

        Returns:
            List of frame data dictionaries
        """
        frames_data = []

        for idx, row in df.iterrows():
            frame_data = {
                "frame": int(row["frame"]),
                "gaze_x": float(row["gaze_x"]),
                "gaze_y": float(row["gaze_y"]),
                "gaze_z": float(row["gaze_z"]),
                "gaze_azimuth_deg": float(row["gaze_azimuth_deg"]),
                "gaze_elevation_deg": float(row["gaze_elevation_deg"]),
                "pupil_scale": float(row["pupil_scale"]),
                "pupil_error_px": float(row.get("pupil_error_px", 0)),
                "tear_duct_error_px": float(row.get("tear_duct_error_px", 0)),
            }

            # Add pupil center projections if available
            if "pupil_center_projected_u" in df.columns:
                frame_data["pupil_center_projected"] = {
                    "u": float(row["pupil_center_projected_u"]),
                    "v": float(row["pupil_center_projected_v"])
                }

            # Add tear duct projections if available
            if "tear_duct_projected_u" in df.columns:
                frame_data["tear_duct_projected"] = {
                    "u": float(row["tear_duct_projected_u"]),
                    "v": float(row["tear_duct_projected_v"])
                }

            # Add observed pupil points (p1-p8)
            observed_pupil = []
            for i in range(1, 9):
                u_col = f"pupil_p{i}_observed_u"
                v_col = f"pupil_p{i}_observed_v"
                if u_col in df.columns and v_col in df.columns:
                    u_val = row[u_col]
                    v_val = row[v_col]
                    # Only add if not NaN
                    if pd.notna(u_val) and pd.notna(v_val):
                        observed_pupil.append({
                            "u": float(u_val),
                            "v": float(v_val)
                        })

            if observed_pupil:
                frame_data["observed_pupil_points"] = observed_pupil

            # Add observed tear duct
            if "tear_duct_observed_u" in df.columns:
                td_u = row["tear_duct_observed_u"]
                td_v = row["tear_duct_observed_v"]
                if pd.notna(td_u) and pd.notna(td_v):
                    frame_data["tear_duct_observed"] = {
                        "u": float(td_u),
                        "v": float(td_v)
                    }

            frames_data.append(frame_data)

        return frames_data

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

        if "pupil_error_px" in df.columns:
            error_mean = df["pupil_error_px"].mean()
            error_std = df["pupil_error_px"].std()
        else:
            error_mean = 0.0
            error_std = 0.0

        # Generate simple HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Eye Tracking Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #0a0a0a; color: white; }}
        h1 {{ color: #60a5fa; }}
        .info {{ background: #1a1a1a; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #333; padding: 10px; text-align: left; }}
        th {{ background-color: #2563eb; color: white; }}
        .metric-value {{ color: #60a5fa; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Eye Tracking Results</h1>
    
    <div class="info">
        <h2>Dataset Info</h2>
        <p><strong>Frames:</strong> <span class="metric-value">{len(df)}</span></p>
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
                <td class="metric-value">{azimuth_range:.1f}°</td>
            </tr>
            <tr>
                <td>Elevation Range</td>
                <td class="metric-value">{elevation_range:.1f}°</td>
            </tr>
            <tr>
                <td>Pupil Scale Mean</td>
                <td class="metric-value">{pupil_mean:.3f}</td>
            </tr>
            <tr>
                <td>Pupil Scale Std</td>
                <td class="metric-value">{pupil_std:.3f}</td>
            </tr>
            <tr>
                <td>Reprojection Error</td>
                <td class="metric-value">{error_mean:.2f} ± {error_std:.2f} px</td>
            </tr>
        </table>
    </div>
    
    <div class="info">
        <h2>Data Preview</h2>
        <p>First 10 frames:</p>
        {df.head(10).to_html(classes='table', border=0)}
    </div>
    
    <div class="info">
        <p><em>Note: Full interactive 3D viewer template not found. This is a basic fallback showing summary statistics.</em></p>
        <p><em>Place eye_tracking_viewer.html template in the templates directory for full visualization.</em></p>
    </div>
</body>
</html>"""

        output_path.write_text(data=html, encoding='utf-8')

        return output_path


def generate_eye_tracking_viewer(
    *,
    output_dir: Path,
    data_csv_path: Path,
    eye_model_json_path: Path | None = None,
    video_path: Path | None = None
) -> Path:
    """Convenience function to generate eye tracking viewer.

    Args:
        output_dir: Output directory
        data_csv_path: Path to eye_tracking_results.csv
        eye_model_json_path: Path to eye_model.json
        video_path: Optional eye camera video

    Returns:
        Path to generated HTML
    """
    generator = EyeTrackingViewerGenerator()

    return generator.generate(
        output_dir=output_dir,
        data_csv_path=data_csv_path,
        eye_model_json_path=eye_model_json_path,
        video_path=video_path
    )