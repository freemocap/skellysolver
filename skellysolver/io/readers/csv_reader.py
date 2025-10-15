"""CSV readers for different formats.

Provides specialized readers for:
- Tidy format (long format)
- Wide format (wide format)
- DeepLabCut format (3-row header)
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np

from skellysolver.io.readers.reader_base import BaseReader

import logging
logger = logging.getLogger(__name__)

class CSVReader(BaseReader):
    """Base class for CSV readers.

    Provides common CSV reading functionality.
    Subclasses implement format-specific parsing.
    """
    encoding: str = 'utf-8'

    def can_read(self, *, filepath: Path) -> bool:
        """Check if file is CSV.

        Args:
            filepath: Path to file

        Returns:
            True if file has .csv extension
        """
        return filepath.suffix.lower() == '.csv' and filepath.exists()

    def read_lines(self, *, filepath: Path, max_lines: int | None = None) -> list[str]:
        """Read lines from CSV file.

        Args:
            filepath: Path to CSV file
            max_lines: Maximum number of lines to read (None = all)

        Returns:
            List of lines (without newline characters)
        """
        self.validate_file(filepath=filepath)

        with open(filepath, mode='r', encoding=self.encoding) as f:
            if max_lines is None:
                lines = f.readlines()
            else:
                lines = [f.readline() for _ in range(max_lines)]

        return [line.strip() for line in lines]

    def read_header(self, *, filepath: Path) -> list[str]:
        """Read CSV header row.

        Args:
            filepath: Path to CSV file

        Returns:
            List of column names
        """

        self.validate_file(filepath=filepath)

        with open(filepath, mode='r', encoding=self.encoding) as f:
            reader = csv.reader(f)
            header = next(reader)

        return [col.strip() for col in header]


class TidyCSVReader(CSVReader):
    """Reader for tidy/long-format CSV files.

    Expected format:
        frame, keypoint, x, y, z
        0, marker1, 1.0, 2.0, 3.0
        0, marker2, 4.0, 5.0, 6.0
        1, marker1, 1.1, 2.1, 3.1
        ...
    """

    def read(self, *, filepath: Path) -> dict[str, Any]:
        """Read tidy CSV.

        Args:
            filepath: Path to CSV file

        Returns:
            Dictionary with:
                - trajectories: dict mapping keypoint -> (n_frames, 3) array
                - frame_indices: array of frame numbers
                - format: "tidy"
        """
        self.validate_file(filepath=filepath)

        # Read all rows
        data_dict = {}
        frame_set = set()

        with open(filepath, mode='r', encoding=self.encoding) as f:
            reader = csv.DictReader(f)

            for row in reader:
                frame = int(row['frame'])
                keypoint = row['keypoint']
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z']) if 'z' in row and row['z'] else 0.0

                frame_set.add(frame)

                if keypoint not in data_dict:
                    data_dict[keypoint] = {}

                data_dict[keypoint][frame] = np.array([x, y, z])

        # Convert to arrays
        frame_indices = np.array(sorted(frame_set))
        n_frames = len(frame_indices)

        trajectories = {}
        for keypoint, frame_data in data_dict.items():
            positions = np.zeros((n_frames, 3))
            positions[:] = np.nan

            for i, frame_idx in enumerate(frame_indices):
                if frame_idx in frame_data:
                    positions[i] = frame_data[frame_idx]

            trajectories[keypoint] = positions

        result = {
            "trajectories": trajectories,
            "frame_indices": frame_indices,
            "format": "tidy",
            "n_frames": n_frames,
            "n_markers": len(trajectories),
        }

        self.last_read_path = filepath
        self.last_read_data = result

        return result


class WideCSVReader(CSVReader):
    """Reader for wide-format CSV files.

    Expected format:
        frame, marker1_x, marker1_y, marker1_z, marker2_x, marker2_y, marker2_z, ...
        0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ...
        1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, ...
        ...
    """
    encoding: str = 'utf-8'
    default_z: float = 0.0

    def read(self, *, filepath: Path) -> dict[str, Any]:
        """Read wide CSV.

        Args:
            filepath: Path to CSV file

        Returns:
            Dictionary with:
                - trajectories: dict mapping marker -> (n_frames, 3) array
                - frame_indices: array of frame numbers
                - format: "wide"
        """
        self.validate_file(filepath=filepath)

        # Read CSV
        with open(filepath, mode='r', encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            if headers is None:
                raise ValueError("CSV has no headers")

            # Find marker names
            marker_names = set()
            for header in headers:
                if header.endswith('_x'):
                    marker_names.add(header[:-2])

            if not marker_names:
                raise ValueError("No markers found (expected columns ending in '_x')")

            # Read all rows
            rows = list(reader)

        n_frames = len(rows)
        frame_indices = np.arange(n_frames)

        # Extract data for each marker
        trajectories = {}
        for marker_name in marker_names:
            positions = np.zeros((n_frames, 3))

            x_col = f"{marker_name}_x"
            y_col = f"{marker_name}_y"
            z_col = f"{marker_name}_z"

            has_z = z_col in headers

            for i, row in enumerate(rows):
                try:
                    x_str = row[x_col].strip()
                    y_str = row[y_col].strip()

                    x = float(x_str) if x_str else np.nan
                    y = float(y_str) if y_str else np.nan

                    if has_z:
                        z_str = row[z_col].strip()
                        z = float(z_str) if z_str else self.default_z
                    else:
                        z = self.default_z

                    positions[i] = np.array([x, y, z])

                except (ValueError, KeyError):
                    positions[i] = np.array([np.nan, np.nan, self.default_z])

            trajectories[marker_name] = positions

        result = {
            "trajectories": trajectories,
            "frame_indices": frame_indices,
            "format": "wide",
            "n_frames": n_frames,
            "n_markers": len(trajectories),
        }

        self.last_read_path = filepath
        self.last_read_data = result

        return result


class DLCCSVReader(CSVReader):
    """Reader for DeepLabCut CSV files.

    Expected format (3-row header):
        scorer, scorer, scorer, ...           (row 0)
        bodypart1, bodypart1, bodypart2, ...  (row 1)
        x, y, likelihood, x, y, likelihood    (row 2)
        1.0, 2.0, 0.95, 3.0, 4.0, 0.92, ...   (row 3+)
    """

    encoding: str = 'utf-8'
    default_z: float = 0.0
    min_likelihood: float | None = None

    def read(self, *, filepath: Path) -> dict[str, Any]:
        """Read DeepLabCut CSV.

        Args:
            filepath: Path to CSV file

        Returns:
            Dictionary with:
                - trajectories: dict mapping bodypart -> (n_frames, 2 or 3) array
                - confidence: dict mapping bodypart -> (n_frames,) likelihood array
                - frame_indices: array of frame numbers
                - format: "dlc"
        """
        self.validate_file(filepath=filepath)

        # Read all lines
        with open(filepath, mode='r', encoding=self.encoding) as f:
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

        # Filter to valid bodyparts
        valid_bodyparts = [
            bp for bp, coords in column_map.items()
            if 'x' in coords and 'y' in coords
        ]

        if not valid_bodyparts:
            raise ValueError("No valid bodyparts with x and y coordinates")

        # Check if any bodypart has z
        has_z = any('z' in column_map[bp] for bp in valid_bodyparts)
        n_dims = 3 if has_z else 2

        # Parse data rows
        n_frames = len(lines) - 3
        frame_indices = np.arange(n_frames)

        trajectories = {}
        confidence_data = {}

        for bodypart in valid_bodyparts:
            coords = column_map[bodypart]
            has_likelihood = 'likelihood' in coords

            positions = np.zeros((n_frames, n_dims))
            confidence = np.zeros(n_frames) if has_likelihood else None

            for i, line in enumerate(lines[3:]):
                values = line.strip().split(',')

                try:
                    x_str = values[coords['x']].strip()
                    y_str = values[coords['y']].strip()

                    x = float(x_str) if x_str else np.nan
                    y = float(y_str) if y_str else np.nan

                    # Check likelihood
                    if has_likelihood:
                        likelihood_str = values[coords['likelihood']].strip()
                        likelihood = float(likelihood_str) if likelihood_str else 0.0
                        confidence[i] = likelihood

                        # Filter by threshold
                        if self.min_likelihood is not None and likelihood < self.min_likelihood:
                            x = np.nan
                            y = np.nan

                    # Get z if available
                    if has_z and 'z' in coords:
                        z_str = values[coords['z']].strip()
                        z = float(z_str) if z_str else self.default_z
                        positions[i] = np.array([x, y, z])
                    else:
                        positions[i] = np.array([x, y])

                except (ValueError, IndexError):
                    if has_z:
                        positions[i] = np.array([np.nan, np.nan, self.default_z])
                    else:
                        positions[i] = np.array([np.nan, np.nan])

            trajectories[bodypart] = positions
            if confidence is not None:
                confidence_data[bodypart] = confidence

        result = {
            "trajectories": trajectories,
            "confidence": confidence_data if confidence_data else None,
            "frame_indices": frame_indices,
            "format": "dlc",
            "n_frames": n_frames,
            "n_markers": len(trajectories),
            "is_3d": has_z,
        }

        self.last_read_path = filepath
        self.last_read_data = result

        return result


class AutoCSVReader(CSVReader):
    """Automatic CSV reader that detects format and uses appropriate parser.

    Tries readers in order:
        1. DeepLabCut (3-row header format)
        2. Tidy/Long format (frame, keypoint, x, y, z columns)
        3. Wide format (marker_x, marker_y, marker_z columns)

    Usage:
        reader = AutoCSVReader()
        data = reader.read(filepath=Path("data.csv"))
    """

    min_likelihood: float | None = None
    default_z: float = 0.0

    def __init__(
            self,
            *,
            min_likelihood: float | None = None,
            default_z: float = 0.0,
            encoding: str = 'utf-8'
    ) -> None:
        """Initialize auto CSV reader.

        Args:
            min_likelihood: Minimum likelihood threshold for DLC data
            default_z: Default z value for 2D data
            encoding: File encoding
        """
        super().__init__()
        self.min_likelihood = min_likelihood
        self.default_z = default_z
        self.encoding = encoding

    def detect_format(self, *, filepath: Path) -> str | None:
        """Detect CSV format by inspecting header.

        Args:
            filepath: Path to CSV file

        Returns:
            Format name ("dlc", "tidy", "wide") or None if unknown
        """
        try:
            # Read first few lines
            lines = self.read_lines(filepath=filepath, max_lines=5)
            if len(lines) < 2:
                return None

            header = self.read_header(filepath=filepath)

            # Check for DLC format (3+ rows, repeating pattern)
            if len(lines) >= 3:
                # DLC has bodypart names in row 1 and x,y,likelihood in row 2
                row2 = lines[1].split(',')
                row3 = lines[2].split(',')

                # DLC typically has repeated bodypart names and x,y,likelihood pattern
                if any(coord.strip().lower() in ['x', 'y', 'likelihood'] for coord in row3):
                    logger.debug("Detected DLC format (3-row header)")
                    return "dlc"

            # Check for Tidy format (has 'frame', 'keypoint', 'x', 'y' columns)
            header_lower = [col.lower() for col in header]
            if ('frame' in header_lower and
                    'keypoint' in header_lower and
                    'x' in header_lower and
                    'y' in header_lower):
                logger.debug("Detected Tidy format (long format)")
                return "tidy"

            # Check for Wide format (has columns ending in _x, _y, _z)
            if any(col.endswith('_x') for col in header):
                logger.debug("Detected Wide format (wide format)")
                return "wide"

            return None

        except Exception as e:
            logger.debug(f"Format detection failed: {e}")
            return None

    def read(self, *, filepath: Path) -> dict[str, Any]:
        """Read CSV using automatic format detection.

        Args:
            filepath: Path to CSV file

        Returns:
            Dictionary with:
                - trajectories: dict mapping marker -> (n_frames, 3) array
                - frame_indices: array of frame numbers
                - format: detected format name
                - (optional) confidence: dict mapping marker -> likelihood array

        Raises:
            ValueError: If no reader could parse the file
        """
        self.validate_file(filepath=filepath)

        # Try to detect format first
        detected_format = self.detect_format(filepath=filepath)

        # Create readers in priority order
        readers: list[tuple[str, CSVReader]] = []

        if detected_format == "dlc":
            # Try DLC first if detected
            readers.append(("DLC", DLCCSVReader(
                min_likelihood=self.min_likelihood,
                default_z=self.default_z
            )))
            readers.append(("Tidy", TidyCSVReader()))
            readers.append(("Wide", WideCSVReader(default_z=self.default_z)))
        elif detected_format == "tidy":
            # Try Tidy first if detected
            readers.append(("Tidy", TidyCSVReader()))
            readers.append(("Wide", WideCSVReader(default_z=self.default_z)))
            readers.append(("DLC", DLCCSVReader(
                min_likelihood=self.min_likelihood,
                default_z=self.default_z
            )))
        else:
            # Default order: Wide, Tidy, DLC
            readers.append(("Wide", WideCSVReader(default_z=self.default_z)))
            readers.append(("Tidy", TidyCSVReader()))
            readers.append(("DLC", DLCCSVReader(
                min_likelihood=self.min_likelihood,
                default_z=self.default_z
            )))

        # Try each reader
        errors: dict[str, str] = {}

        for reader_name, reader in readers:
            try:
                logger.debug(f"Attempting to read with {reader_name} reader...")
                result = reader.read(filepath=filepath)

                # Success!
                logger.info(
                    f"Successfully parsed CSV with {reader_name} reader "
                    f"({result['n_markers']} markers, {result['n_frames']} frames)"
                )

                self.last_read_path = filepath
                self.last_read_data = result

                return result

            except Exception as e:
                error_msg = str(e)
                errors[reader_name] = error_msg
                logger.debug(f"{reader_name} reader failed: {error_msg}")
                continue

        # All readers failed
        error_details = "\n".join(f"  - {name}: {err}" for name, err in errors.items())
        raise ValueError(
            f"Could not parse CSV file with any reader.\n"
            f"Tried formats: {', '.join(errors.keys())}\n"
            f"Errors:\n{error_details}\n\n"
            f"Supported formats:\n"
            f"  - Wide: marker_x, marker_y, marker_z columns\n"
            f"  - Tidy: frame, keypoint, x, y, z columns\n"
            f"  - DLC: 3-row header with bodyparts and x,y,likelihood"
        )