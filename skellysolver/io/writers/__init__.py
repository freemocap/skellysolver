"""Writers module for SkellySolver IO.

Provides writers for different file formats:
- CSV writers (trajectory, eye tracking, tidy)
- JSON writers
- NPY writers
- Unified results writer

Usage:
    from skellysolver.io.writers import ResultsWriter
    
    writer = ResultsWriter(output_dir=Path("output/"))
    writer.save_rigid_body_results(
        result=result,
        raw_data=raw_data,
        marker_names=marker_names,
        topology_dict=topology_dict,
        metrics=metrics
    )
"""

from .writer_base import (
    BaseWriter,
    CSVWriter,
    JSONWriter,
    NPYWriter,
    MultiFormatWriter,
)

from .csv_writer import (
    TrajectoryCSVWriter,
    SimpleTrajectoryCSVWriter,
    EyeTrackingCSVWriter,
    TidyCSVWriter,
)

from .results_writer import (
    ResultsWriter,
)

__all__ = [
    # Base writers
    "BaseWriter",
    "CSVWriter",
    "JSONWriter",
    "NPYWriter",
    "MultiFormatWriter",
    # CSV writers
    "TrajectoryCSVWriter",
    "SimpleTrajectoryCSVWriter",
    "EyeTrackingCSVWriter",
    "TidyCSVWriter",
    # Results writer
    "ResultsWriter",
]
