"""CSV format detection for SkellySolver.

This module automatically detects which CSV format is being used:
- tidy: Long format with columns [frame, keypoint, x, y, z]
- wide: Wide format with columns [frame, {marker}_x, {marker}_y, {marker}_z]
- dlc: DeepLabCut 3-row header format

Consolidates format detection from:
- loaders.py::detect_csv_format
- load_trajectories.py::detect_csv_format
"""

import csv
from pathlib import Path
from typing import Literal

CSVFormat = Literal["tidy", "wide", "dlc"]


def detect_csv_format(*, filepath: Path) -> CSVFormat:
    """Automatically detect CSV format.
    
    Checks file structure to determine format type.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Format type: "tidy", "wide", or "dlc"
        
    Raises:
        ValueError: If format cannot be determined
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    # Read first few lines
    with open(filepath, mode='r', encoding='utf-8') as f:
        lines = [f.readline().strip() for _ in range(min(3, sum(1 for _ in f) + 1))]
        f.seek(0)
        all_lines = f.readlines()
    
    if len(lines) < 1:
        raise ValueError(f"CSV file is empty: {filepath}")
    
    # Check for DeepLabCut format (3-row header)
    if len(lines) >= 3:
        if _is_dlc_format(lines=lines):
            return "dlc"
    
    # Check for tidy vs wide format
    # Need to read the actual header
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
    
    if headers is None:
        raise ValueError(f"CSV has no headers: {filepath}")
    
    # Tidy format: has 'keypoint' column and separate x, y columns
    if _is_tidy_format(headers=headers):
        return "tidy"
    
    # Wide format: has {marker}_x, {marker}_y pattern
    if _is_wide_format(headers=headers):
        return "wide"
    
    # Could not determine
    raise ValueError(
        f"Unknown CSV format: {filepath}\n"
        f"Expected one of:\n"
        f"  - Tidy: columns ['frame', 'keypoint', 'x', 'y', 'z']\n"
        f"  - Wide: columns ['frame', '{{marker}}_x', '{{marker}}_y', ...]\n"
        f"  - DLC: 3-row header (scorer/bodyparts/coords)\n"
        f"Got headers: {headers}"
    )


def _is_dlc_format(*, lines: list[str]) -> bool:
    """Check if lines match DeepLabCut format.
    
    DLC format has 3-row header:
    - Row 0: scorer names
    - Row 1: bodypart names
    - Row 2: coordinate types (x, y, likelihood)
    
    Args:
        lines: First 3 lines of file
        
    Returns:
        True if DLC format
    """
    if len(lines) < 3:
        return False
    
    # Check row 3 for coordinate type indicators
    row3_values = lines[2].split(',')
    
    # DLC has 'x', 'y', 'likelihood' in row 3
    dlc_indicators = ['x', 'y', 'likelihood', 'coords']
    has_dlc_indicators = any(
        val.strip().lower() in dlc_indicators
        for val in row3_values
    )
    
    return has_dlc_indicators


def _is_tidy_format(*, headers: list[str]) -> bool:
    """Check if headers match tidy format.
    
    Tidy format has columns: frame, keypoint, x, y, z
    
    Args:
        headers: CSV column headers
        
    Returns:
        True if tidy format
    """
    headers_lower = [h.lower().strip() for h in headers]
    
    # Must have 'keypoint' column
    has_keypoint = 'keypoint' in headers_lower
    
    # Must have x and y columns
    has_x = 'x' in headers_lower
    has_y = 'y' in headers_lower
    
    return has_keypoint and has_x and has_y


def _is_wide_format(*, headers: list[str]) -> bool:
    """Check if headers match wide format.
    
    Wide format has columns: frame, {marker}_x, {marker}_y, ...
    
    Args:
        headers: CSV column headers
        
    Returns:
        True if wide format
    """
    headers_lower = [h.lower().strip() for h in headers]
    
    # Look for _x and _y suffixes
    has_x_suffix = any(h.endswith('_x') for h in headers_lower)
    has_y_suffix = any(h.endswith('_y') for h in headers_lower)
    
    return has_x_suffix and has_y_suffix


def validate_tidy_format(*, filepath: Path) -> None:
    """Validate that file is properly formatted tidy CSV.
    
    Args:
        filepath: Path to CSV file
        
    Raises:
        ValueError: If format is invalid
    """
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        if headers is None:
            raise ValueError("CSV has no headers")
        
        # Check required columns
        required = ['frame', 'keypoint', 'x', 'y']
        headers_lower = [h.lower().strip() for h in headers]
        missing = [col for col in required if col not in headers_lower]
        
        if missing:
            raise ValueError(f"Tidy format missing required columns: {missing}")


def validate_wide_format(*, filepath: Path) -> dict[str, list[str]]:
    """Validate wide format and extract marker names.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Dictionary with marker names and their columns
        
    Raises:
        ValueError: If format is invalid
    """
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        if headers is None:
            raise ValueError("CSV has no headers")
        
        # Extract marker names from _x suffixes
        marker_names = set()
        for header in headers:
            if header.endswith('_x'):
                marker_name = header[:-2]
                marker_names.add(marker_name)
        
        if not marker_names:
            raise ValueError("No markers found (expected columns ending in '_x')")
        
        # Check each marker has required columns
        marker_columns = {}
        for marker in marker_names:
            cols = []
            for suffix in ['_x', '_y', '_z']:
                col_name = f"{marker}{suffix}"
                if col_name in headers:
                    cols.append(col_name)
            
            if len(cols) < 2:
                raise ValueError(f"Marker '{marker}' missing required columns")
            
            marker_columns[marker] = cols
        
        return marker_columns


def validate_dlc_format(*, filepath: Path) -> dict[str, list[str]]:
    """Validate DLC format and extract bodypart names.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Dictionary mapping bodypart names to column indices
        
    Raises:
        ValueError: If format is invalid
    """
    with open(filepath, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 4:
        raise ValueError("DLC CSV must have at least 4 rows (3 header + 1 data)")
    
    # Parse 3-row header
    scorer_row = lines[0].strip().split(',')
    bodypart_row = lines[1].strip().split(',')
    coords_row = lines[2].strip().split(',')
    
    # Build column mapping
    column_map = {}
    for col_idx, (bodypart, coord_type) in enumerate(zip(bodypart_row, coords_row)):
        bodypart = bodypart.strip()
        coord_type = coord_type.strip()
        
        if not bodypart or bodypart.lower() == 'scorer':
            continue
        
        if bodypart not in column_map:
            column_map[bodypart] = {}
        
        column_map[bodypart][coord_type] = col_idx
    
    # Validate each bodypart has x and y
    valid_bodyparts = {}
    for bodypart, coords in column_map.items():
        if 'x' in coords and 'y' in coords:
            valid_bodyparts[bodypart] = list(coords.keys())
        else:
            print(f"Warning: Bodypart '{bodypart}' missing x or y coordinate")
    
    if not valid_bodyparts:
        raise ValueError("No valid bodyparts found with both x and y coordinates")
    
    return valid_bodyparts


def get_format_info(*, filepath: Path) -> dict[str, any]:
    """Get detailed information about CSV format.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Dictionary with format information
    """
    format_type = detect_csv_format(filepath=filepath)
    
    info = {
        "format": format_type,
        "filepath": str(filepath),
    }
    
    if format_type == "tidy":
        validate_tidy_format(filepath=filepath)
        info["description"] = "Tidy format (long)"
        info["columns"] = ["frame", "keypoint", "x", "y", "z"]
    
    elif format_type == "wide":
        marker_columns = validate_wide_format(filepath=filepath)
        info["description"] = "Wide format"
        info["markers"] = list(marker_columns.keys())
        info["n_markers"] = len(marker_columns)
    
    elif format_type == "dlc":
        bodyparts = validate_dlc_format(filepath=filepath)
        info["description"] = "DeepLabCut format"
        info["bodyparts"] = list(bodyparts.keys())
        info["n_bodyparts"] = len(bodyparts)
    
    return info


def is_3d_data(*, filepath: Path) -> bool:
    """Check if CSV contains 3D data (has z column).
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        True if data is 3D
    """
    format_type = detect_csv_format(filepath=filepath)
    
    if format_type == "tidy":
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers is None:
                return False
            headers_lower = [h.lower().strip() for h in headers]
            return 'z' in headers_lower
    
    elif format_type == "wide":
        marker_columns = validate_wide_format(filepath=filepath)
        # Check if any marker has _z column
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers is None:
                return False
            return any(h.endswith('_z') for h in headers)
    
    elif format_type == "dlc":
        # DLC is typically 2D
        bodyparts = validate_dlc_format(filepath=filepath)
        # Check if any bodypart has z coordinate
        return any('z' in coords for coords in bodyparts.values())
    
    return False
