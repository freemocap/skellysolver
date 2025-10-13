"""Batch processor for running multiple optimization jobs.

Handles:
- Sequential or parallel job execution
- Error handling and recovery
- Progress tracking
- Result aggregation
"""

import numpy as np
import time
import logging
from pathlib import Path
from typing import Any
import multiprocessing as mp

from pydantic import Field

from skellysolver.batch.batch_config import BatchConfig, BatchJobConfig
from skellysolver.core import OptimizationResult
from skellysolver.data.arbitrary_types_model import ArbitraryTypesModel
from skellysolver.pipelines import PipelineConfig, BasePipeline

logger = logging.getLogger(__name__)


class BatchJobResult(ArbitraryTypesModel):
    """Result from a single batch job.
    
    Attributes:
        job_id: Job identifier
        job_name: Job name
        success: Whether job completed successfully
        optimization_result: Result from optimization (if successful)
        error: Error message (if failed)
        start_time: Job start timestamp
        end_time: Job end timestamp
        duration_seconds: Job duration
        metadata: Additional result metadata
    """
    
    job_id: str
    job_name: str
    success: bool
    optimization_result: OptimizationResult | None = None
    error: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def cost_reduction_percent(self) -> float:
        """Get cost reduction percentage.
        
        Returns:
            Cost reduction percent (0 if failed)
        """
        if self.optimization_result is None:
            return 0.0
        return self.optimization_result.cost_reduction_percent



class BatchResult(ArbitraryTypesModel):
    """Results from batch processing.
    
    Attributes:
        batch_name: Name of batch
        job_results: List of individual job results
        total_duration_seconds: Total batch duration
        n_jobs_total: Total number of jobs
        n_jobs_successful: Number of successful jobs
        n_jobs_failed: Number of failed jobs
        metadata: Additional batch metadata
    """
    
    batch_name: str
    job_results: list[BatchJobResult]
    total_duration_seconds: float
    n_jobs_total: int
    n_jobs_successful: int
    n_jobs_failed: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Compute success rate.
        
        Returns:
            Fraction of jobs that succeeded [0-1]
        """
        if self.n_jobs_total == 0:
            return 0.0
        return self.n_jobs_successful / self.n_jobs_total
    
    @property
    def average_duration(self) -> float:
        """Compute average job duration.
        
        Returns:
            Average duration in seconds
        """
        if not self.job_results:
            return 0.0
        return np.mean([r.duration_seconds for r in self.job_results])
    
    def get_best_result(self) -> BatchJobResult | None:
        """Get job with lowest final cost.
        
        Returns:
            Best job result (None if no successful jobs)
        """
        successful = [r for r in self.job_results if r.success and r.optimization_result is not None]
        
        if not successful:
            return None
        
        return min(successful, key=lambda r: r.optimization_result.final_cost)
    
    def get_worst_result(self) -> BatchJobResult | None:
        """Get job with highest final cost.
        
        Returns:
            Worst job result (None if no successful jobs)
        """
        successful = [r for r in self.job_results if r.success and r.optimization_result is not None]
        
        if not successful:
            return None
        
        return max(successful, key=lambda r: r.optimization_result.final_cost)
    
    def summary(self) -> str:
        """Generate summary string.
        
        Returns:
            Multi-line summary
        """
        lines = [
            "="*80,
            f"BATCH RESULT: {self.batch_name}",
            "="*80,
            f"Total jobs:     {self.n_jobs_total}",
            f"Successful:     {self.n_jobs_successful} ({self.success_rate*100:.1f}%)",
            f"Failed:         {self.n_jobs_failed}",
            f"Total time:     {self.total_duration_seconds:.2f}s ({self.total_duration_seconds/60:.1f}m)",
            f"Average time:   {self.average_duration:.2f}s per job",
            "="*80,
        ]
        
        # Add best/worst info
        best = self.get_best_result()
        if best is not None:
            lines.append(f"\nBest job:  {best.job_name}")
            lines.append(f"  Final cost: {best.optimization_result.final_cost:.6f}")
            lines.append(f"  Reduction:  {best.cost_reduction_percent:.1f}%")
        
        worst = self.get_worst_result()
        if worst is not None:
            lines.append(f"\nWorst job: {worst.job_name}")
            lines.append(f"  Final cost: {worst.optimization_result.final_cost:.6f}")
            lines.append(f"  Reduction:  {worst.cost_reduction_percent:.1f}%")
        
        return "\n".join(lines)


class BatchProcessor(ArbitraryTypesModel):
    """Process multiple optimization jobs in batch.
    
    Handles:
    - Job queue management
    - Sequential or parallel execution
    - Error handling and recovery
    - Progress tracking
    - Result aggregation
    
    Usage:
        config = BatchConfig(...)
        processor = BatchProcessor(config=config)
        result = processor.run()
    """
    
    config: BatchConfig
    job_results: list[BatchJobResult] = []
    
    def run(self) -> BatchResult:
        """Run all jobs in batch.
        
        Returns:
            BatchResult with all job results
        """
        logger.info("="*80)
        logger.info(f"BATCH PROCESSOR: {self.config.batch_name}")
        logger.info("="*80)
        logger.info(f"Jobs:         {self.config.n_jobs}")
        logger.info(f"Mode:         {self.config.parallel_mode}")
        logger.info(f"Output root:  {self.config.output_root}")
        
        start_time = time.time()
        
        # Run jobs
        if self.config.should_use_parallel():
            logger.info(f"Workers:      {self.config.get_num_workers()}")
            self.job_results = self._run_parallel()
        else:
            self.job_results = self._run_sequential()
        
        total_duration = time.time() - start_time
        
        # Count successes/failures
        n_successful = sum(1 for r in self.job_results if r.success)
        n_failed = len(self.job_results) - n_successful
        
        # Create batch result
        batch_result = BatchResult(
            batch_name=self.config.batch_name,
            job_results=self.job_results,
            total_duration_seconds=total_duration,
            n_jobs_total=self.config.n_jobs,
            n_jobs_successful=n_successful,
            n_jobs_failed=n_failed,
            metadata=self.config.metadata
        )
        
        # Generate summary report
        if self.config.generate_summary_report:
            self._generate_summary_report(batch_result=batch_result)
        
        logger.info("\n" + "="*80)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Successful: {n_successful}/{self.config.n_jobs}")
        logger.info(f"Total time: {total_duration:.2f}s ({total_duration/60:.1f}m)")
        
        return batch_result
    
    def _run_sequential(self) -> list[BatchJobResult]:
        """Run jobs sequentially.
        
        Returns:
            List of job results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING JOBS SEQUENTIALLY")
        logger.info("="*80)
        
        results = []
        jobs = self.config.get_sorted_jobs()
        
        for i, job in enumerate(jobs, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"JOB {i}/{len(jobs)}: {job.job_name}")
            logger.info(f"{'='*80}")
            
            result = self._run_single_job(job=job)
            results.append(result)
            
            if not result.success and not self.config.continue_on_error:
                logger.error(f"Job failed: {result.error}")
                logger.error("Stopping batch (continue_on_error=False)")
                break
        
        return results
    
    def _run_parallel(self) -> list[BatchJobResult]:
        """Run jobs in parallel.
        
        Returns:
            List of job results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING JOBS IN PARALLEL")
        logger.info("="*80)
        
        jobs = self.config.get_sorted_jobs()
        n_workers = self.config.get_num_workers()
        
        logger.info(f"Processing {len(jobs)} jobs with {n_workers} workers...")
        
        # Create worker pool
        with mp.Pool(processes=n_workers) as pool:
            # Map jobs to workers
            results = pool.map(self._run_single_job, jobs)
        
        return results
    
    def _run_single_job(self, *, job: BatchJobConfig) -> BatchJobResult:
        """Run a single job.
        
        Args:
            job: Job configuration
            
        Returns:
            BatchJobResult
        """
        job_start = time.time()
        
        try:
            logger.info(f"Starting job: {job.job_name}")
            
            # Import pipeline class
            pipeline = self._create_pipeline(config=job.pipeline_config)
            
            # Run pipeline
            result = pipeline.run()
            
            job_end = time.time()
            duration = job_end - job_start
            
            logger.info(f"✓ Job completed: {job.job_name} ({duration:.2f}s)")
            
            return BatchJobResult(
                job_id=job.job_id,
                job_name=job.job_name,
                success=True,
                optimization_result=result,
                error=None,
                start_time=job_start,
                end_time=job_end,
                duration_seconds=duration,
                metadata=job.metadata
            )
            
        except Exception as e:
            job_end = time.time()
            duration = job_end - job_start
            
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"✗ Job failed: {job.job_name}")
            logger.error(f"  Error: {error_msg}")
            
            return BatchJobResult(
                job_id=job.job_id,
                job_name=job.job_name,
                success=False,
                optimization_result=None,
                error=error_msg,
                start_time=job_start,
                end_time=job_end,
                duration_seconds=duration,
                metadata=job.metadata
            )
    
    def _create_pipeline(self, *, config: PipelineConfig) -> BasePipeline:
        """Create pipeline instance from config.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Pipeline instance
        """
        # Determine pipeline type from config class
        config_class_name = config.__class__.__name__
        
        if "RigidBody" in config_class_name:
            from ..pipelines.rigid_body_pipeline import RigidBodyPipeline
            return RigidBodyPipeline(config=config)
        elif "EyeTracking" in config_class_name:
            from ..pipelines.eye_pipeline  import EyeTrackingPipeline
            return EyeTrackingPipeline(config=config)
        else:
            raise ValueError(f"Unknown pipeline config type: {config_class_name}")
    
    def _generate_summary_report(self, *, batch_result: BatchResult) -> None:
        """Generate summary report for batch.
        
        Args:
            batch_result: Batch result to summarize
        """
        logger.info("\nGenerating batch summary report...")
        
        report_path = self.config.output_root / "batch_summary.json"
        
        # Build report
        import json
        
        report = {
            "batch_name": batch_result.batch_name,
            "n_jobs_total": batch_result.n_jobs_total,
            "n_jobs_successful": batch_result.n_jobs_successful,
            "n_jobs_failed": batch_result.n_jobs_failed,
            "success_rate": batch_result.success_rate,
            "total_duration_seconds": batch_result.total_duration_seconds,
            "average_duration_seconds": batch_result.average_duration,
            "jobs": []
        }
        
        # Add job summaries
        for job_result in batch_result.job_results:
            job_summary = {
                "job_id": job_result.job_id,
                "job_name": job_result.job_name,
                "success": job_result.success,
                "duration_seconds": job_result.duration_seconds,
                "error": job_result.error,
                "metadata": job_result.metadata,
            }
            
            if job_result.optimization_result is not None:
                opt = job_result.optimization_result
                job_summary["optimization"] = {
                    "iterations": opt.num_iterations,
                    "initial_cost": opt.initial_cost,
                    "final_cost": opt.final_cost,
                    "cost_reduction_percent": opt.cost_reduction_percent,
                }
            
            report["jobs"].append(job_summary)
        
        # Save report
        with open(report_path, mode='w') as f:
            json.dump(obj=report, fp=f, indent=2)
        
        logger.info(f"  ✓ Saved batch summary: {report_path}")


class ProgressTracker:
    """Track progress of batch processing.
    
    Provides real-time updates on job completion.
    """
    
    def __init__(self, *, total_jobs: int) -> None:
        """Initialize progress tracker.
        
        Args:
            total_jobs: Total number of jobs
        """
        self.total_jobs = total_jobs
        self.completed_jobs = 0
        self.start_time = time.time()
    
    def update(self, *, job_name: str, success: bool) -> None:
        """Update progress with completed job.
        
        Args:
            job_name: Name of completed job
            success: Whether job succeeded
        """
        self.completed_jobs += 1
        
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.completed_jobs
        remaining_jobs = self.total_jobs - self.completed_jobs
        eta = remaining_jobs * avg_time
        
        status = "✓" if success else "✗"
        logger.info(
            f"{status} [{self.completed_jobs}/{self.total_jobs}] {job_name} - "
            f"ETA: {eta/60:.1f}m"
        )
    
    @property
    def percent_complete(self) -> float:
        """Get completion percentage.
        
        Returns:
            Completion percent [0-100]
        """
        if self.total_jobs == 0:
            return 100.0
        return (self.completed_jobs / self.total_jobs) * 100.0
