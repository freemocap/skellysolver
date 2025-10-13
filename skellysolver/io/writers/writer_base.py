"""Abstract base classes for data writers.

Provides consistent interface for writing different file formats.
All writers inherit from BaseWriter and implement write() method.
"""

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from skellysolver.data.arbitrary_types_model import ArbitraryTypesModel
from pydantic import Field

class BaseWriter(ArbitraryTypesModel,ABC):
    """Abstract base class for all data writers.
    
    Provides standard interface for writing files.
    Subclasses implement format-specific writing logic.
    
    Usage:
        class MyWriter(BaseWriter):
            def write(
                self,
                *,
                filepath: Path,
                data: dict[str, Any]
            ) -> None:
                # Implementation
                pass
        
        writer = MyWriter()
        writer.write(filepath=Path("output.txt"), data={"key": "value"})
    """
    
    last_write_path: Path | None = None
    
    @abstractmethod
    def write(
        self,
        *,
        filepath: Path,
        data: dict[str, Any]
    ) -> None:
        """Write data to file.
        
        Must be implemented by subclasses.
        
        Args:
            filepath: Path to output file
            data: Data to write
        """
        pass
    
    def ensure_directory(self, *, filepath: Path) -> None:
        """Ensure output directory exists.
        
        Args:
            filepath: Path to file (directory will be created)
        """
        directory = filepath.parent
        directory.mkdir(parents=True, exist_ok=True)
    
    def validate_data(self, *, data: dict[str, Any], required_keys: list[str]) -> None:
        """Validate that data contains required keys.
        
        Args:
            data: Data dictionary
            required_keys: List of required keys
            
        Raises:
            ValueError: If any required keys are missing
        """
        missing = set(required_keys) - set(data.keys())
        if missing:
            raise ValueError(f"Data missing required keys: {missing}")


class CSVWriter(BaseWriter):
    """Base class for CSV writers.
    
    Provides common CSV writing functionality.
    Subclasses implement format-specific structure.
    """
    encoding: str = 'utf-8'

    def write_rows(
        self,
        *,
        filepath: Path,
        rows: list[dict[str, Any]],
        fieldnames: list[str] | None = None
    ) -> None:
        """Write rows to CSV file.
        
        Args:
            filepath: Path to output CSV
            rows: List of dictionaries (one per row)
            fieldnames: Optional field names (None = infer from first row)
        """
        import csv
        
        self.ensure_directory(filepath=filepath)
        
        if not rows:
            raise ValueError("No rows to write")
        
        if fieldnames is None:
            fieldnames = list(rows[0].keys())
        
        with open(filepath, mode='w', encoding=self.encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        self.last_write_path = filepath


class JSONWriter(BaseWriter):
    """Writer for JSON files."""
    indent: int = 2

    def write(
        self,
        *,
        filepath: Path,
        data: dict[str, Any]
    ) -> None:
        """Write data to JSON file.
        
        Args:
            filepath: Path to output JSON
            data: Data to write (must be JSON-serializable)
        """
        import json
        
        self.ensure_directory(filepath=filepath)
        
        # Convert numpy types to Python types
        data_serializable = self._convert_to_serializable(data=data)
        
        with open(filepath, mode='w', encoding='utf-8') as f:
            json.dump(obj=data_serializable, fp=f, indent=self.indent)
        
        self.last_write_path = filepath
    
    def _convert_to_serializable(self, *, data: Any) -> Any:
        """Convert numpy types to JSON-serializable types.
        
        Args:
            data: Data to convert
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, dict):
            return {key: self._convert_to_serializable(data=value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(data=item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)


class NPYWriter(BaseWriter):
    """Writer for NumPy .npy files."""
    
    def write(
        self,
        *,
        filepath: Path,
        data: dict[str, Any]
    ) -> None:
        """Write numpy array to .npy file.
        
        Args:
            filepath: Path to output .npy file
            data: Dictionary with 'array' key containing numpy array
        """
        self.ensure_directory(filepath=filepath)
        
        if "array" not in data:
            raise ValueError("Data must contain 'array' key")
        
        array = data["array"]
        
        if not isinstance(array, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(array)}")
        
        np.save(file=filepath, arr=array)
        
        self.last_write_path = filepath
    
    def write_array(
        self,
        *,
        filepath: Path,
        array: np.ndarray
    ) -> None:
        """Write numpy array directly.
        
        Convenience method.
        
        Args:
            filepath: Path to output .npy file
            array: Numpy array to write
        """
        self.write(filepath=filepath, data={"array": array})


class MultiFormatWriter(BaseWriter):
    """Writer that automatically selects format based on file extension.
    
    Supports: .csv, .json, .npy
    """
    


    csv_writer :CSVWriter = Field(default_factory=CSVWriter)
    json_writer:JSONWriter = Field(default_factory=JSONWriter)
    npy_writer :NPYWriter = Field(default_factory=NPYWriter)
    
    def write(
        self,
        *,
        filepath: Path,
        data: dict[str, Any]
    ) -> None:
        """Write data to file (auto-detect format).
        
        Args:
            filepath: Path to output file
            data: Data to write
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix == '.json':
            self.json_writer.write(filepath=filepath, data=data)
        elif suffix == '.npy':
            self.npy_writer.write(filepath=filepath, data=data)
        elif suffix == '.csv':
            # Assume it's trajectory data
            if "rows" in data:
                self.csv_writer.write_rows(
                    filepath=filepath,
                    rows=data["rows"],
                    fieldnames=data.get("fieldnames")
                )
            else:
                raise ValueError("CSV writing requires 'rows' key in data")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
