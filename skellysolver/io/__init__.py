"""IO module for SkellySolver.

Provides organized input/output functionality:
- Readers: Read various file formats (CSV, JSON, NPY)
- Writers: Write results to various formats
- Viewers: Generate interactive HTML visualizations

Consolidates and organizes:
- io/loaders.py
- io/savers.py
- io/eye_savers.py
- io/blender_rigid_body_viz.py (moved to visualization/)

Structure:
    io/
    ├── readers/     - Read files (CSV, JSON, NPY)
    ├── writers/     - Write files (CSV, JSON, NPY)
    └── viewers/     - Generate HTML viewers

Usage:
    # Reading
    from skellysolver.io.readers import TidyCSVReader
    reader = TidyCSVReader()
    data = reader.read(filepath=Path("data.csv"))
    
    # Writing
    from skellysolver.io.writers import ResultsWriter
    writer = ResultsWriter(output_dir=Path("output/"))
    writer.save_rigid_body_results(...)
    
    # Viewing
    from skellysolver.io.viewers import generate_rigid_body_viewer
    viewer_path = generate_rigid_body_viewer(...)
"""

# Readers
from .readers import (
    BaseReader,
    CSVReader,
    BinaryReader,
    JSONReader,
    NPYReader,
    TidyCSVReader,
    WideCSVReader,
    DLCCSVReader,
)

# Writers
from .writers import (
    BaseWriter,
    CSVWriter,
    JSONWriter,
    NPYWriter,
    MultiFormatWriter,
    TrajectoryCSVWriter,
    SimpleTrajectoryCSVWriter,
    EyeTrackingCSVWriter,
    TidyCSVWriter,
    ResultsWriter,
)

# Viewers
from .viewers import (
    BaseViewerGenerator,
    HTMLViewerGenerator,
    RigidBodyViewerGenerator,
    generate_rigid_body_viewer,
    generate_eye_tracking_viewer,
)

__all__ = [
    # Readers
    "BaseReader",
    "CSVReader",
    "BinaryReader",
    "JSONReader",
    "NPYReader",
    "TidyCSVReader",
    "WideCSVReader",
    "DLCCSVReader",
    # Writers
    "BaseWriter",
    "CSVWriter",
    "JSONWriter",
    "NPYWriter",
    "MultiFormatWriter",
    "TrajectoryCSVWriter",
    "SimpleTrajectoryCSVWriter",
    "EyeTrackingCSVWriter",
    "TidyCSVWriter",
    "ResultsWriter",
    # Viewers
    "BaseViewerGenerator",
    "HTMLViewerGenerator",
    "RigidBodyViewerGenerator",
    "generate_rigid_body_viewer",
    "generate_eye_tracking_viewer",
]

