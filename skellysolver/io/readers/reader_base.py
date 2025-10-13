"""Abstract base classes for data readers.

Provides consistent interface for reading different file formats.
All readers inherit from BaseReader and implement read() method.
"""

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseReader(ABC):
    """Abstract base class for all data readers.
    
    Provides standard interface for reading files.
    Subclasses implement format-specific reading logic.
    
    Usage:
        class MyReader(BaseReader):
            def read(self, *, filepath: Path) -> dict[str, Any]:
                # Implementation
                pass
        
        reader = MyReader()
        data = reader.read(filepath=Path("data.txt"))
    """
    
    def __init__(self) -> None:
        """Initialize reader."""
        self.last_read_path: Path | None = None
        self.last_read_data: dict[str, Any] | None = None
    
    @abstractmethod
    def read(self, *, filepath: Path) -> dict[str, Any]:
        """Read data from file.
        
        Must be implemented by subclasses.
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary with loaded data
        """
        pass
    
    def can_read(self, *, filepath: Path) -> bool:
        """Check if this reader can read the file.
        
        Default implementation checks file extension.
        Subclasses can override for more sophisticated checks.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if this reader can handle the file
        """
        return filepath.exists()
    
    def validate_file(self, *, filepath: Path) -> None:
        """Validate that file exists and is readable.
        
        Args:
            filepath: Path to file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not filepath.is_file():
            raise ValueError(f"Not a file: {filepath}")
        
        if filepath.stat().st_size == 0:
            raise ValueError(f"File is empty: {filepath}")


class CSVReader(BaseReader):
    """Base class for CSV readers.
    
    Provides common CSV reading functionality.
    Subclasses implement format-specific parsing.
    """
    
    def __init__(self, *, encoding: str = 'utf-8') -> None:
        """Initialize CSV reader.
        
        Args:
            encoding: Text encoding for CSV file
        """
        super().__init__()
        self.encoding = encoding
    
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
        import csv
        
        self.validate_file(filepath=filepath)
        
        with open(filepath, mode='r', encoding=self.encoding) as f:
            reader = csv.reader(f)
            header = next(reader)
        
        return [col.strip() for col in header]


class BinaryReader(BaseReader):
    """Base class for binary file readers.
    
    Provides common binary reading functionality.
    """
    
    def read_bytes(self, *, filepath: Path) -> bytes:
        """Read entire file as bytes.
        
        Args:
            filepath: Path to file
            
        Returns:
            File contents as bytes
        """
        self.validate_file(filepath=filepath)
        
        with open(filepath, mode='rb') as f:
            data = f.read()
        
        return data


class JSONReader(BaseReader):
    """Reader for JSON files."""
    
    def can_read(self, *, filepath: Path) -> bool:
        """Check if file is JSON.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file has .json extension
        """
        return filepath.suffix.lower() == '.json' and filepath.exists()
    
    def read(self, *, filepath: Path) -> dict[str, Any]:
        """Read JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        import json
        
        self.validate_file(filepath=filepath)
        
        with open(filepath, mode='r', encoding='utf-8') as f:
            data = json.load(fp=f)
        
        self.last_read_path = filepath
        self.last_read_data = data
        
        return data


class NPYReader(BaseReader):
    """Reader for NumPy .npy files."""
    
    def can_read(self, *, filepath: Path) -> bool:
        """Check if file is NPY.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file has .npy extension
        """
        return filepath.suffix.lower() == '.npy' and filepath.exists()
    
    def read(self, *, filepath: Path) -> dict[str, np.ndarray]:
        """Read NPY file.
        
        Args:
            filepath: Path to NPY file
            
        Returns:
            Dictionary with 'array' key containing loaded array
        """
        self.validate_file(filepath=filepath)
        
        array = np.load(file=filepath)
        
        data = {
            "array": array,
            "shape": array.shape,
            "dtype": str(array.dtype),
        }
        
        self.last_read_path = filepath
        self.last_read_data = data
        
        return data
