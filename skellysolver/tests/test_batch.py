"""Tests for batch processing (Phase 5).

Tests batch configuration, processor, and utilities.
"""

from pathlib import Path

from skellysolver.batch.batch_config import BatchConfig, BatchJobConfig, ParameterSweepConfig
from skellysolver.batch.batch_processor import BatchJobResult, BatchResult
from skellysolver.batch.batch_utils import create_parameter_sweep, estimate_batch_time, create_batch_from_files
from skellysolver.core import OptimizationConfig
from skellysolver.pipelines import RigidBodyConfig
from skellysolver.pipelines.rigid_body_pipeline.rigid_body_topology import RigidBodyTopology


class TestBatchConfig:
    """Test batch configuration."""
    
    def test_create_batch_config(self, temp_dir: Path) -> None:
        """Should create batch config."""
        topology = RigidBodyTopology(
            marker_names=["m1", "m2"],
            rigid_edges=[(0, 1)],
        )
        
        job_1 = BatchJobConfig(
            job_id="job_001",
            job_name="Test Job 1",
            pipeline_config=RigidBodyConfig(
                input_path=temp_dir / "data1.csv",
                output_dir=temp_dir / "output1",
                topology=topology,
                optimization=OptimizationConfig(max_iterations=10),
            )
        )
        
        batch_config = BatchConfig(
            batch_name="test_batch",
            jobs=[job_1],
            output_root=temp_dir
        )
        
        assert batch_config.n_jobs == 1
        assert batch_config.batch_name == "test_batch"
    
    def test_get_sorted_jobs(self, temp_dir: Path) -> None:
        """Should sort jobs by priority."""
        topology = RigidBodyTopology(
            marker_names=["m1", "m2"],
            rigid_edges=[(0, 1)],
        )
        
        jobs = [
            BatchJobConfig(
                job_id="low",
                job_name="Low Priority",
                pipeline_config=RigidBodyConfig(
                    input_path=temp_dir / "test.csv",
                    output_dir=temp_dir / "output",
                    topology=topology,
                    optimization=OptimizationConfig(),
                ),
                priority=0
            ),
            BatchJobConfig(
                job_id="high",
                job_name="High Priority",
                pipeline_config=RigidBodyConfig(
                    input_path=temp_dir / "test.csv",
                    output_dir=temp_dir / "output",
                    topology=topology,
                    optimization=OptimizationConfig(),
                ),
                priority=10
            ),
        ]
        
        batch_config = BatchConfig(
            batch_name="test",
            jobs=jobs,
            output_root=temp_dir
        )
        
        sorted_jobs = batch_config.get_sorted_jobs()
        
        # High priority should be first
        assert sorted_jobs[0].job_id == "high"
        assert sorted_jobs[1].job_id == "low"
    
    def test_should_use_parallel(self, temp_dir: Path) -> None:
        """Should determine parallel mode."""
        topology = RigidBodyTopology(
            marker_names=["m1", "m2"],
            rigid_edges=[(0, 1)],
        )
        
        job = BatchJobConfig(
            job_id="job_001",
            job_name="Job",
            pipeline_config=RigidBodyConfig(
                input_path=temp_dir / "test.csv",
                output_dir=temp_dir / "output",
                topology=topology,
                optimization=OptimizationConfig(),
            )
        )
        
        # Single job - sequential
        batch_config = BatchConfig(
            batch_name="test",
            jobs=[job],
            output_root=temp_dir,
            parallel_mode="auto"
        )
        
        assert batch_config.should_use_parallel() is False
        
        # Multiple jobs - parallel
        batch_config = BatchConfig(
            batch_name="test",
            jobs=[job, job],
            output_root=temp_dir,
            parallel_mode="auto"
        )
        
        assert batch_config.should_use_parallel() is True


class TestParameterSweep:
    """Test parameter sweep configuration."""
    
    def test_create_parameter_sweep(self, temp_dir: Path) -> None:
        """Should create parameter sweep."""
        topology = RigidBodyTopology(
            marker_names=["m1", "m2"],
            rigid_edges=[(0, 1)],
        )
        
        base_config = RigidBodyConfig(
            input_path=temp_dir / "test.csv",
            output_dir=temp_dir / "output",
            topology=topology,
            optimization=OptimizationConfig(max_iterations=100),
        )
        
        parameter_grid = {
            "optimization.max_iterations": [50, 100, 150],
        }
        
        sweep_config = ParameterSweepConfig(
            base_config=base_config,
            parameter_grid=parameter_grid,
            output_root=temp_dir,
            sweep_name="test_sweep"
        )
        
        batch_config = sweep_config.generate_batch_config()
        
        # Should create 3 jobs
        assert batch_config.n_jobs == 3
    
    def test_parameter_sweep_combinations(self, temp_dir: Path) -> None:
        """Should generate all parameter combinations."""
        topology = RigidBodyTopology(
            marker_names=["m1", "m2"],
            rigid_edges=[(0, 1)],
        )
        
        base_config = RigidBodyConfig(
            input_path=temp_dir / "test.csv",
            output_dir=temp_dir / "output",
            topology=topology,
            optimization=OptimizationConfig(),
        )
        
        parameter_grid = {
            "optimization.max_iterations": [100, 200],
            "weights.lambda_rigid": [100.0, 500.0],
        }
        
        batch_config = create_parameter_sweep(
            base_config=base_config,
            parameter_grid=parameter_grid,
            output_root=temp_dir
        )
        
        # Should create 2 × 2 = 4 jobs
        assert batch_config.n_jobs == 4


class TestBatchUtils:
    """Test batch utility functions."""
    
    def test_create_batch_from_files(self, temp_dir: Path) -> None:
        """Should create batch from file list."""
        # Create test files
        files = [
            temp_dir / "file1.csv",
            temp_dir / "file2.csv",
        ]
        
        for f in files:
            f.write_text("frame,m1_x,m1_y,m1_z\n0,1,2,3\n")
        
        topology = RigidBodyTopology(
            marker_names=["m1"],
            rigid_edges=[],
        )
        
        def make_config(filepath: Path) -> RigidBodyConfig:
            return RigidBodyConfig(
                input_path=filepath,
                output_dir=temp_dir / filepath.stem,
                topology=topology,
                optimization=OptimizationConfig(),
            )
        
        batch_config = create_batch_from_files(
            file_paths=files,
            config_factory=make_config,
            output_root=temp_dir
        )
        
        assert batch_config.n_jobs == 2
    
    def test_estimate_batch_time(self, temp_dir: Path) -> None:
        """Should estimate batch processing time."""
        topology = RigidBodyTopology(
            marker_names=["m1", "m2"],
            rigid_edges=[(0, 1)],
        )
        
        jobs = [
            BatchJobConfig(
                job_id=f"job_{i:03d}",
                job_name=f"Job {i}",
                pipeline_config=RigidBodyConfig(
                    input_path=temp_dir / f"data{i}.csv",
                    output_dir=temp_dir / f"output{i}",
                    topology=topology,
                    optimization=OptimizationConfig(),
                )
            )
            for i in range(10)
        ]
        
        batch_config = BatchConfig(
            batch_name="test",
            jobs=jobs,
            output_root=temp_dir,
            parallel_mode="parallel",
            max_workers=5
        )
        
        estimate = estimate_batch_time(
            batch_config=batch_config,
            time_per_job_seconds=60.0
        )
        
        assert estimate["n_jobs"] == 10
        assert estimate["sequential_time_seconds"] == 600.0
        assert estimate["speedup"] > 1.0


class TestBatchResult:
    """Test batch result."""
    
    def test_batch_result_creation(self) -> None:
        """Should create batch result."""

        job_results = [
            BatchJobResult(
                job_id="job_001",
                job_name="Job 1",
                success=True,
                duration_seconds=10.0
            ),
            BatchJobResult(
                job_id="job_002",
                job_name="Job 2",
                success=False,
                error="Test error",
                duration_seconds=5.0
            ),
        ]
        
        batch_result = BatchResult(
            batch_name="test_batch",
            job_results=job_results,
            total_duration_seconds=15.0,
            n_jobs_total=2,
            n_jobs_successful=1,
            n_jobs_failed=1
        )
        
        assert batch_result.success_rate == 0.5
        assert batch_result.n_jobs_total == 2
    
    def test_batch_result_summary(self) -> None:
        """Should generate summary."""

        job_result = BatchJobResult(
            job_id="job_001",
            job_name="Job 1",
            success=True,
            duration_seconds=10.0
        )
        
        batch_result = BatchResult(
            batch_name="test",
            job_results=[job_result],
            total_duration_seconds=10.0,
            n_jobs_total=1,
            n_jobs_successful=1,
            n_jobs_failed=0
        )
        
        summary = batch_result.summary()
        
        assert "BATCH RESULT" in summary
        assert "test" in summary
