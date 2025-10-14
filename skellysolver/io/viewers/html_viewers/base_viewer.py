"""Base viewer generator for interactive HTML visualizations.

Provides common functionality for all viewer generators.
"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from skellysolver.data.arbitrary_types_model import ABaseModel


class BaseViewerGenerator(ABaseModel, ABC):
    """Abstract base class for viewer generators.
    
    All viewers inherit from this and implement:
    - generate(): Create HTML viewer
    - _get_template_path(): Return path to HTML template
    - _prepare_data(): Prepare data for embedding
    """
    
    last_generated_path: Path | None = None
    
    @abstractmethod
    def generate(
        self,
        *,
        output_dir: Path,
        data_csv_path: Path,
        video_path: Path | None = None
    ) -> Path:
        """Generate HTML viewer.
        
        Must be implemented by subclasses.
        
        Args:
            output_dir: Directory for output viewer
            data_csv_path: Path to data CSV
            video_path: Optional path to video file
            
        Returns:
            Path to generated HTML file
        """
        pass
    
    @abstractmethod
    def _get_template_path(self) -> Path:
        """Get path to HTML template.
        
        Must be implemented by subclasses.
        
        Returns:
            Path to HTML template file
        """
        pass
    
    def _ensure_output_dir(self, *, output_dir: Path) -> None:
        """Ensure output directory exists.
        
        Args:
            output_dir: Directory to create
        """
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def _copy_template(
        self,
        *,
        template_path: Path,
        output_path: Path
    ) -> None:
        """Copy HTML template to output location.
        
        Args:
            template_path: Path to template file
            output_path: Path to output file
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        shutil.copy(src=template_path, dst=output_path)
    
    def _embed_data_in_html(
        self,
        *,
        html_path: Path,
        data_json: str,
        replacements: dict[str, str] | None = None
    ) -> None:
        """Embed data into HTML file using placeholders.
        
        Args:
            html_path: Path to HTML file
            data_json: JSON string of data to embed
            replacements: Optional additional string replacements
        """
        # Read HTML
        html = html_path.read_text(encoding='utf-8')
        
        # Replace data placeholder
        html = html.replace("__DATA_JSON__", data_json)
        
        # Additional replacements
        if replacements is not None:
            for placeholder, value in replacements.items():
                html = html.replace(placeholder, value)
        
        # Write back
        html_path.write_text(data=html, encoding='utf-8')
    
    def _read_csv_as_json(self, *, csv_path: Path) -> str:
        """Read CSV and convert to JSON string.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            JSON string of CSV data
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        df = pd.read_csv(filepath_or_buffer=csv_path)
        return df.to_json(orient='records')
    
    def _copy_video(
        self,
        *,
        video_path: Path,
        output_dir: Path
    ) -> str:
        """Copy video to output directory.
        
        Args:
            video_path: Path to video file
            output_dir: Destination directory
            
        Returns:
            Video filename (relative to HTML)
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        video_dest = output_dir / video_path.name
        
        # Only copy if doesn't exist
        if not video_dest.exists():
            shutil.copy2(src=video_path, dst=video_dest)
        
        return video_path.name


class HTMLViewerGenerator(BaseViewerGenerator):
    """Generic HTML viewer generator.
    
    Can generate viewers from templates with placeholder replacement.
    """
    template_path: Path

    def _get_template_path(self) -> Path:
        """Get template path.
        
        Returns:
            Path to HTML template
        """
        return self.template_path
    
    def generate(
        self,
        *,
        output_dir: Path,
        data_csv_path: Path,
        video_path: Path | None = None,
        placeholders: dict[str, str] | None = None
    ) -> Path:
        """Generate HTML viewer.
        
        Args:
            output_dir: Output directory
            data_csv_path: Path to data CSV
            video_path: Optional video path
            placeholders: Optional placeholder replacements
            
        Returns:
            Path to generated HTML
        """
        self._ensure_output_dir(output_dir=output_dir)
        
        # Output path
        output_path = output_dir / "viewer.html"
        
        # Copy template
        self._copy_template(
            template_path=self.template_path,
            output_path=output_path
        )
        
        # Read data as JSON
        data_json = self._read_csv_as_json(csv_path=data_csv_path)
        
        # Handle video
        video_filename = ""
        if video_path is not None:
            video_filename = self._copy_video(
                video_path=video_path,
                output_dir=output_dir
            )
        
        # Prepare replacements
        replacements = {
            "video_src": video_filename,
            "n_frames": str(pd.read_csv(data_csv_path).shape[0]),
        }
        
        if placeholders is not None:
            replacements.update(placeholders)
        
        # Embed data
        self._embed_data_in_html(
            html_path=output_path,
            data_json=data_json,
            replacements=replacements
        )
        
        self.last_generated_path = output_path
        
        return output_path
