"""Readers module for SkellySolver IO.

Provides readers for different file formats:
- CSV readers (tidy, wide, DeepLabCut)
- JSON readers
- NPY readers
- Base reader abstractions

Usage:
    from skellysolver.io.readers import TidyCSVReader
    
    reader = TidyCSVReader()
    data = reader.read(filepath=Path("data.csv"))
"""

from .reader_base import (
    BaseReader,
    BinaryReader,
    JSONReader,
    NPYReader,
)

from .csv_reader import (
    CSVReader,
    TidyCSVReader,
    WideCSVReader,
    DLCCSVReader,
)

__all__ = [
    # Base readers
    "BaseReader",
    "CSVReader",
    "BinaryReader",
    "JSONReader",
    "NPYReader",
    # CSV readers
    "TidyCSVReader",
    "WideCSVReader",
    "DLCCSVReader",
]
