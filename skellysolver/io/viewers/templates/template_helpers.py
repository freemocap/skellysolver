"""Constants and utilities for viewer generation.

Provides standard paths, filenames, and helper functions for viewer generators.
"""

from pathlib import Path
from typing import Literal

# Template filenames
RIGID_BODY_TEMPLATE_NAME = "rigid_body_viewer.html"
EYE_TRACKING_TEMPLATE_NAME = "eye_tracking_viewer.html"

# Output filenames
RIGID_BODY_OUTPUT_NAME = "rigid_body_viewer.html"
EYE_TRACKING_OUTPUT_NAME = "eye_tracking_viewer.html"


# Template directory (relative to this file)
def get_templates_dir() -> Path:
    """Get templates directory path.

    Returns:
        Path to templates directory
    """
    return Path(__file__).parent


def get_template_path(
        *,
        viewer_type: Literal["rigid_body", "eye_tracking"]
) -> Path:
    """Get path to viewer template.

    Args:
        viewer_type: Type of viewer ("rigid_body" or "eye_tracking")

    Returns:
        Path to template file

    Raises:
        ValueError: If viewer_type is invalid
    """
    templates_dir = get_templates_dir()

    if viewer_type == "rigid_body":
        return templates_dir / RIGID_BODY_TEMPLATE_NAME
    elif viewer_type == "eye_tracking":
        return templates_dir / EYE_TRACKING_TEMPLATE_NAME
    else:
        raise ValueError(f"Invalid viewer_type: {viewer_type}")


def get_output_path(
        *,
        output_dir: Path,
        viewer_type: Literal["rigid_body", "eye_tracking"]
) -> Path:
    """Get output path for viewer.

    Args:
        output_dir: Output directory
        viewer_type: Type of viewer ("rigid_body" or "eye_tracking")

    Returns:
        Path to output HTML file

    Raises:
        ValueError: If viewer_type is invalid
    """
    if viewer_type == "rigid_body":
        return output_dir / RIGID_BODY_OUTPUT_NAME
    elif viewer_type == "eye_tracking":
        return output_dir / EYE_TRACKING_OUTPUT_NAME
    else:
        raise ValueError(f"Invalid viewer_type: {viewer_type}")


def template_exists(
        *,
        viewer_type: Literal["rigid_body", "eye_tracking"]
) -> bool:
    """Check if template exists.

    Args:
        viewer_type: Type of viewer

    Returns:
        True if template file exists
    """
    template_path = get_template_path(viewer_type=viewer_type)
    return template_path.exists()

if not get_template_path(viewer_type="rigid_body").exists():
    raise FileNotFoundError(f"Template not found: {get_template_path(viewer_type ='rigid_body')}")
if not get_template_path(viewer_type="eye_tracking").exists():
    pass
    # raise FileNotFoundError(f"Template not found: {get_template_path(viewer_type ='eye_tracking')}")