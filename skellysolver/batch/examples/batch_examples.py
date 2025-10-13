"""Basic batch processing examples.

Shows how to use SkellySolver batch processing for common tasks.
"""

import numpy as np
from pathlib import Path

from ...batch.config import BatchConfig, BatchJobConfig
from ...batch.processor import BatchProcessor
from ...batch.report import BatchReportGenerator
from ...batch.utils import (
    create_batch_from_files,
    create_parameter_sweep,
    create_batch_from_directory,
)
from ...pipelines.rigid_body import RigidBodyConfig, RigidBodyPipeline
from ...core import OptimizationConfig, RigidBodyWeightConfig
from ...core.topology import RigidBodyTopology


def example_1_process_multiple_files() -> None:
    """Example 1: Process multiple files with same configuration."""
    
    print("="*80)
    print("EXAMPLE 1: Process Multiple Files")
    print("="*80)
    
    # Define topology (same for all files)
    topology = RigidBodyTopology(
        marker_names=["nose", "left_eye", "right_eye"],
        rigid_edges=[(0, 1), (1, 2), (2, 0)],
        name="face_triangle"
    )
    
    # List of files to process
    file_paths = [
        Path("data/recording_001.csv"),
        Path("data/recording_002.csv"),
        Path("data/recording_003.csv"),
    ]
    
    # Create config factory
    def make_config(filepath: Path) -> RigidBodyConfig:
        return RigidBodyConfig(
            input_path=filepath,
            output_dir=Path("batch_output") / filepath.stem,
            topology=topology,
            optimization=OptimizationConfig(max_iterations=300),
        )
    
    # Create batch
    batch_config = create_batch_from_files(
        file_paths=file_paths,
        config_factory=make_config,
        output_root=Path("batch_output/"),
        batch_name="multi_file_batch"
    )
    
    # Run batch
    processor = BatchProcessor(config=batch_config)
    result = processor.run()
    
    # Print summary
    print(result.summary())


def example_2_parameter_sweep() -> None:
    """Example 2: Parameter sweep to find best weights."""
    
    print("="*80)
    print("EXAMPLE 2: Parameter Sweep")
    print("="*80)
    
    # Base configuration
    topology = RigidBodyTopology(
        marker_names=["nose", "left_eye", "right_eye"],
        rigid_edges=[(0, 1), (1, 2), (2, 0)],
    )
    
    base_config = RigidBodyConfig(
        input_path=Path("data/test_data.csv"),
        output_dir=Path("sweep_output/"),  # Will be overridden per job
        topology=topology,
        optimization=OptimizationConfig(max_iterations=200),
        weights=RigidBodyWeightConfig(),
    )
    
    # Define parameter grid
    parameter_grid = {
        "weights.lambda_rigid": [100.0, 500.0, 1000.0],
        "weights.lambda_rot_smooth": [50.0, 100.0, 200.0],
        "optimization.max_iterations": [100, 200, 300],
    }
    
    # This creates 3 × 3 × 3 = 27 jobs!
    batch_config = create_parameter_sweep(
        base_config=base_config,
        parameter_grid=parameter_grid,
        output_root=Path("sweep_output/"),
        sweep_name="weight_sweep"
    )
    
    print(f"Created parameter sweep with {batch_config.n_jobs} jobs")
    
    # Run batch
    processor = BatchProcessor(config=batch_config)
    result = processor.run()
    
    # Find best parameters
    from ...batch.report import find_best_parameters
    
    best_params = find_best_parameters(batch_result=result)
    print(f"\nBest parameters: {best_params}")


def example_3_process_directory() -> None:
    """Example 3: Process all files in a directory."""
    
    print("="*80)
    print("EXAMPLE 3: Process Directory")
    print("="*80)
    
    # Define topology
    topology = RigidBodyTopology(
        marker_names=["m1", "m2", "m3"],
        rigid_edges=[(0, 1), (1, 2)],
    )
    
    # Config factory
    def make_config(filepath: Path) -> RigidBodyConfig:
        return RigidBodyConfig(
            input_path=filepath,
            output_dir=Path("directory_output") / filepath.stem,
            topology=topology,
            optimization=OptimizationConfig(max_iterations=300),
        )
    
    # Create batch from all CSVs in directory
    batch_config = create_batch_from_directory(
        directory=Path("data/"),
        pattern="*.csv",
        config_factory=make_config,
        output_root=Path("directory_output/"),
        recursive=False
    )
    
    # Run batch
    processor = BatchProcessor(config=batch_config)
    result = processor.run()
    
    # Generate reports
    report_gen = BatchReportGenerator(batch_result=result)
    report_gen.save_all_reports(output_dir=Path("directory_output/reports/"))


def example_4_manual_batch_creation() -> None:
    """Example 4: Manually create batch with custom jobs."""
    
    print("="*80)
    print("EXAMPLE 4: Manual Batch Creation")
    print("="*80)
    
    # Create topology
    topology = RigidBodyTopology(
        marker_names=["nose", "left_eye", "right_eye"],
        rigid_edges=[(0, 1), (1, 2), (2, 0)],
    )
    
    # Create jobs manually
    jobs = []
    
    # Job 1: High rigidity
    config1 = RigidBodyConfig(
        input_path=Path("data/test.csv"),
        output_dir=Path("manual_batch/job_1_high_rigid/"),
        topology=topology,
        optimization=OptimizationConfig(max_iterations=300),
        weights=RigidBodyWeightConfig(lambda_rigid=1000.0),
    )
    
    jobs.append(BatchJobConfig(
        job_id="job_001",
        job_name="High Rigidity",
        pipeline_config=config1,
        priority=1,
        metadata={"rigidity": "high"}
    ))
    
    # Job 2: Medium rigidity
    config2 = RigidBodyConfig(
        input_path=Path("data/test.csv"),
        output_dir=Path("manual_batch/job_2_medium_rigid/"),
        topology=topology,
        optimization=OptimizationConfig(max_iterations=300),
        weights=RigidBodyWeightConfig(lambda_rigid=500.0),
    )
    
    jobs.append(BatchJobConfig(
        job_id="job_002",
        job_name="Medium Rigidity",
        pipeline_config=config2,
        priority=0,
        metadata={"rigidity": "medium"}
    ))
    
    # Job 3: Low rigidity
    config3 = RigidBodyConfig(
        input_path=Path("data/test.csv"),
        output_dir=Path("manual_batch/job_3_low_rigid/"),
        topology=topology,
        optimization=OptimizationConfig(max_iterations=300),
        weights=RigidBodyWeightConfig(lambda_rigid=100.0),
    )
    
    jobs.append(BatchJobConfig(
        job_id="job_003",
        job_name="Low Rigidity",
        pipeline_config=config3,
        priority=-1,
        metadata={"rigidity": "low"}
    ))
    
    # Create batch config
    batch_config = BatchConfig(
        batch_name="manual_batch",
        jobs=jobs,
        output_root=Path("manual_batch/"),
        parallel_mode="parallel",
        max_workers=3
    )
    
    # Run batch
    processor = BatchProcessor(config=batch_config)
    result = processor.run()
    
    print(result.summary())


def example_5_analyze_sweep_results() -> None:
    """Example 5: Analyze parameter sweep results."""
    
    print("="*80)
    print("EXAMPLE 5: Analyze Sweep Results")
    print("="*80)
    
    # First run a parameter sweep
    topology = RigidBodyTopology(
        marker_names=["m1", "m2", "m3"],
        rigid_edges=[(0, 1), (1, 2)],
    )
    
    base_config = RigidBodyConfig(
        input_path=Path("data/test.csv"),
        output_dir=Path("analysis_sweep/"),
        topology=topology,
        optimization=OptimizationConfig(max_iterations=100),
        weights=RigidBodyWeightConfig(),
    )
    
    parameter_grid = {
        "weights.lambda_rigid": [100.0, 300.0, 500.0, 700.0, 1000.0],
    }
    
    batch_config = create_parameter_sweep(
        base_config=base_config,
        parameter_grid=parameter_grid,
        output_root=Path("analysis_sweep/"),
        sweep_name="rigidity_analysis"
    )
    
    # Run
    processor = BatchProcessor(config=batch_config)
    result = processor.run()
    
    # Analyze results
    from ...batch.report import (
        compare_parameter_sweep_results,
        find_best_parameters,
    )
    
    # Compare across parameter
    comparison = compare_parameter_sweep_results(
        batch_result=result,
        parameter_name="weights.lambda_rigid"
    )
    
    print("\nParameter Sweep Comparison:")
    print(comparison)
    
    # Find best
    best_params = find_best_parameters(batch_result=result)
    print(f"\nBest parameters: {best_params}")
    
    # Generate report
    report_gen = BatchReportGenerator(batch_result=result)
    report_gen.save_all_reports(output_dir=Path("analysis_sweep/reports/"))


if __name__ == "__main__":
    # Run examples
    print("Choose an example to run:")
    print("1. Process multiple files")
    print("2. Parameter sweep")
    print("3. Process directory")
    print("4. Manual batch creation")
    print("5. Analyze sweep results")
    
    choice = input("\nEnter choice (1-5): ")
    
    if choice == "1":
        example_1_process_multiple_files()
    elif choice == "2":
        example_2_parameter_sweep()
    elif choice == "3":
        example_3_process_directory()
    elif choice == "4":
        example_4_manual_batch_creation()
    elif choice == "5":
        example_5_analyze_sweep_results()
    else:
        print("Invalid choice")
