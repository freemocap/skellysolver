"""Tests for IO readers and writers (Phase 4).

Tests reading and writing various file formats.
"""

import numpy as np
import pytest
from pathlib import Path
import json

from skellysolver.io import TidyCSVReader, WideCSVReader, DLCCSVReader
from skellysolver.io.readers.reader_base import JSONReader, NPYReader
from skellysolver.io.writers.csv_writer import TrajectoryCSVWriter, SimpleTrajectoryCSVWriter
from skellysolver.io.writers.results_writer import ResultsWriter
from skellysolver.io.writers.writer_base import JSONWriter, NPYWriter


class TestCSVReaders:
    """Test CSV readers."""
    
    def test_tidy_reader(self, create_tidy_csv: Path) -> None:
        """Should read tidy CSV."""
        reader = TidyCSVReader()
        data = reader.read(filepath=create_tidy_csv)
        
        assert data["format"] == "tidy"
        assert data["n_markers"] == 3
        assert data["n_frames"] == 2
        assert "marker1" in data["trajectories"]
    
    def test_wide_reader(self, create_wide_csv: Path) -> None:
        """Should read wide CSV."""
        reader = WideCSVReader()
        data = reader.read(filepath=create_wide_csv)
        
        assert data["format"] == "wide"
        assert data["n_markers"] == 2
        assert data["n_frames"] == 3
    
    def test_dlc_reader(self, create_dlc_csv: Path) -> None:
        """Should read DLC CSV."""
        reader = DLCCSVReader()
        data = reader.read(filepath=create_dlc_csv)
        
        assert data["format"] == "dlc"
        assert data["n_markers"] == 2
        assert data["n_frames"] == 3
        assert "confidence" in data
    
    def test_reader_can_read(self, create_tidy_csv: Path) -> None:
        """Should check if file is readable."""
        reader = TidyCSVReader()
        
        assert reader.can_read(filepath=create_tidy_csv) is True
        assert reader.can_read(filepath=Path("nonexistent.txt")) is False


class TestJSONReader:
    """Test JSON reader."""
    
    def test_read_json(self, temp_dir: Path) -> None:
        """Should read JSON file."""
        json_path = temp_dir / "test.json"
        
        data_to_write = {"key": "value", "number": 42}
        
        with open(json_path, mode='w') as f:
            json.dump(obj=data_to_write, fp=f)
        
        reader = JSONReader()
        data = reader.read(filepath=json_path)
        
        assert data["key"] == "value"
        assert data["number"] == 42


class TestNPYReader:
    """Test NPY reader."""
    
    def test_read_npy(self, temp_dir: Path) -> None:
        """Should read NPY file."""
        npy_path = temp_dir / "test.npy"
        
        array = np.random.randn(10, 3)
        np.save(file=npy_path, arr=array)
        
        reader = NPYReader()
        data = reader.read(filepath=npy_path)
        
        assert "array" in data
        assert np.allclose(data["array"], array)


class TestCSVWriters:
    """Test CSV writers."""
    
    def test_trajectory_writer(self, temp_dir: Path) -> None:
        """Should write trajectory CSV."""
        noisy = np.random.randn(10, 3, 3)
        optimized = np.random.randn(10, 3, 3)
        marker_names = ["m1", "m2", "m3"]
        
        writer = TrajectoryCSVWriter()
        output_path = temp_dir / "output.csv"
        
        writer.write(
            filepath=output_path,
            data={
                "noisy_data": noisy,
                "optimized_data": optimized,
                "marker_names": marker_names,
            }
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_simple_trajectory_writer(self, temp_dir: Path) -> None:
        """Should write simple trajectory CSV."""
        positions = np.random.randn(10, 3, 3)
        marker_names = ["m1", "m2", "m3"]
        
        writer = SimpleTrajectoryCSVWriter()
        output_path = temp_dir / "simple.csv"
        
        writer.write(
            filepath=output_path,
            data={
                "positions": positions,
                "marker_names": marker_names,
            }
        )
        
        assert output_path.exists()


class TestJSONWriter:
    """Test JSON writer."""
    
    def test_write_json(self, temp_dir: Path) -> None:
        """Should write JSON file."""
        writer = JSONWriter(indent=2)
        output_path = temp_dir / "test.json"
        
        data = {"key": "value", "number": 42}
        writer.write(filepath=output_path, data=data)
        
        assert output_path.exists()
        
        # Verify content
        with open(output_path, mode='r') as f:
            loaded = json.load(fp=f)
        
        assert loaded["key"] == "value"
        assert loaded["number"] == 42
    
    def test_write_numpy_types(self, temp_dir: Path) -> None:
        """Should convert numpy types to JSON-serializable."""
        writer = JSONWriter()
        output_path = temp_dir / "test.json"
        
        data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
        }
        
        # Should not raise
        writer.write(filepath=output_path, data=data)
        
        assert output_path.exists()


class TestNPYWriter:
    """Test NPY writer."""
    
    def test_write_npy(self, temp_dir: Path) -> None:
        """Should write NPY file."""
        writer = NPYWriter()
        output_path = temp_dir / "test.npy"
        
        array = np.random.randn(10, 3)
        
        writer.write_array(filepath=output_path, array=array)
        
        assert output_path.exists()
        
        # Verify content
        loaded = np.load(file=output_path)
        assert np.allclose(loaded, array)


class TestResultsWriter:
    """Test unified results writer."""
    
    def test_create_results_writer(self, temp_dir: Path) -> None:
        """Should create results writer."""
        writer = ResultsWriter(output_dir=temp_dir)
        
        assert writer.output_dir == temp_dir
        assert temp_dir.exists()
    
    def test_directory_created_on_save(self, temp_dir: Path) -> None:
        """Should create output directory when saving results."""
        output_dir = temp_dir / "new_dir"

        # Directory shouldn't exist yet
        assert not output_dir.exists()

        writer = ResultsWriter(output_dir=output_dir)

        # Directory still shouldn't exist after instantiation
        assert not output_dir.exists()

        # Create minimal result data for testing
        from skellysolver.core import OptimizationResult

        result = OptimizationResult(
            success=True,
            num_iterations=10,
            initial_cost=1.0,
            final_cost=0.1,
            solve_time_seconds=0.5,
            reconstructed=None,
            rotations=None,
            translations=None,
        )

        metrics = {"mean_error": 0.5}

        # Save results - this should create the directory
        writer.save_generic_results(
            result=result,
            metrics=metrics
        )

        # Now the directory should exist
        assert output_dir.exists()
        assert (output_dir / "metrics.json").exists()