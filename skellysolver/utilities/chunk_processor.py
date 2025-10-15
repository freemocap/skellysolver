"""Minimal chunking manager for pipeline execution.

Provides a simple interface for chunked optimization with automatic parallelization.
Each chunk gets its own solver instance for independent optimization.
"""

import logging
import multiprocessing as mp
import time
from typing import Callable

from typing_extensions import Self

from skellysolver.utilities.arbitrary_types_model import ABaseModel
from skellysolver.data.trajectory_dataset import TrajectoryDataset
from skellysolver.solvers.base_solver import SolverResult

logger = logging.getLogger(__name__)


class ChunkingConfig(ABaseModel):
    """Configuration for chunked optimization.

    Attributes:
        enabled: Whether to use chunked processing
        chunk_size: Number of frames per chunk
        overlap_size: Number of overlapping frames between chunks
        blend_window: Size of blending window in overlap region
        min_chunk_size: Minimum frames to process as separate chunk
        num_workers: Number of parallel workers (None = auto-detect)
    """
    enabled: bool = True
    chunk_size: int = 500
    overlap_size: int = 50
    blend_window: int = 25
    min_chunk_size: int = 100
    num_workers: int | None = None

    def __str__(self) -> str:
        """Human-readable chunking config."""
        status = "Enabled" if self.enabled else "Disabled"
        workers = self.num_workers if self.num_workers is not None else "auto"
        lines = [
            "Chunking Configuration:",
            f"  Status:        {status}",
            f"  Chunk size:    {self.chunk_size} frames",
            f"  Overlap:       {self.overlap_size} frames",
            f"  Blend window:  {self.blend_window} frames",
            f"  Min chunk:     {self.min_chunk_size} frames",
            f"  Workers:       {workers}"
        ]
        return "\n".join(lines)


class ChunkProcessor(ABaseModel):
    """Minimal manager for chunked pipeline execution.

    Handles splitting data into chunks, parallel/sequential processing,
    and stitching results back together. Each chunk gets its own solver instance.
    """
    config: ChunkingConfig

    @classmethod
    def from_config(cls, *, config: ChunkingConfig) -> Self:
        """Create ChunkProcessor from configuration.

        Args:
            config: Chunking configuration

        Returns:
            ChunkProcessor instance
        """
        return cls(config=config)

    def chunk_run_pipeline(
        self,
        *,
        input_data: TrajectoryDataset,
        setup_and_solve_fn: Callable[[TrajectoryDataset], SolverResult],
    ) -> SolverResult:
        """Run pipeline with chunking and optional parallelization.

        Each chunk will create its own solver by calling setup_and_solve_fn.

        Args:
            input_data: Full trajectory dataset
            setup_and_solve_fn: Function that creates solver, sets up costs, and solves
                               for a chunk of data. Should return SolverResult.

        Returns:
            Merged SolverResult from all chunks
        """
        n_frames = input_data.n_frames

        # Check if we should use chunking
        if not self._should_chunk(n_frames=n_frames):
            logger.info("Processing without chunking")
            return setup_and_solve_fn(input_data)

        # Split into chunks
        chunks = self._split_into_chunks(n_frames=n_frames)
        use_parallel = self._should_use_parallel(n_chunks=len(chunks))

        logger.info("=" * 80)
        logger.info(f"{'PARALLEL' if use_parallel else 'SEQUENTIAL'} CHUNKED OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Total frames: {n_frames}")
        logger.info(f"Chunk size: {self.config.chunk_size}")
        logger.info(f"Overlap: {self.config.overlap_size}")
        logger.info(f"Blend window: {self.config.blend_window}")
        logger.info(f"\nSplit into {len(chunks)} chunks:")
        for i, (start, end) in enumerate(chunks):
            logger.info(f"  Chunk {i}: frames {start}-{end} ({end-start} frames)")

        # Process chunks
        start_time = time.time()

        if use_parallel:
            chunk_results = self._process_parallel(
                input_data=input_data,
                chunks=chunks,
                setup_and_solve_fn=setup_and_solve_fn
            )
        else:
            chunk_results = self._process_sequential(
                input_data=input_data,
                chunks=chunks,
                setup_and_solve_fn=setup_and_solve_fn
            )

        total_time = time.time() - start_time

        logger.info(f"\n{'=' * 80}")
        logger.info(f"{'PARALLEL' if use_parallel else 'SEQUENTIAL'} OPTIMIZATION COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Average per chunk: {total_time/len(chunks):.1f}s")

        # Stitch results together
        return self._stitch_results(chunk_results=chunk_results)

    def _should_chunk(self, *, n_frames: int) -> bool:
        """Determine if chunking should be used.

        Args:
            n_frames: Total number of frames

        Returns:
            True if should use chunking
        """
        if not self.config.enabled:
            return False

        # Only chunk if we have enough frames
        return n_frames > (self.config.chunk_size + self.config.min_chunk_size)

    def _should_use_parallel(self, *, n_chunks: int) -> bool:
        """Determine if parallel processing should be used.

        Args:
            n_chunks: Number of chunks

        Returns:
            True if should use parallel processing
        """
        if n_chunks <= 1:
            return False

        # Use parallel if we have multiple workers available
        num_workers = self._get_num_workers()
        return num_workers > 1

    def _get_num_workers(self) -> int:
        """Get number of workers to use.

        Returns:
            Number of workers
        """
        if self.config.num_workers is not None:
            return max(self.config.num_workers, 1)

        cpu_count = mp.cpu_count()
        return max((cpu_count or 1) - 1, 1)

    def _split_into_chunks(self, *, n_frames: int) -> list[tuple[int, int]]:
        """Split frame range into overlapping chunks.

        Args:
            n_frames: Total number of frames

        Returns:
            List of (start, end) tuples for each chunk
        """
        if self.config.overlap_size >= self.config.chunk_size:
            raise ValueError(
                f"overlap_size ({self.config.overlap_size}) must be < "
                f"chunk_size ({self.config.chunk_size})"
            )

        chunks = []
        stride = self.config.chunk_size - self.config.overlap_size
        start = 0

        while start < n_frames:
            end = min(start + self.config.chunk_size, n_frames)

            # Check if this would be a tiny final chunk
            if end < n_frames and (n_frames - end) < self.config.min_chunk_size:
                # Extend this chunk to include the remainder
                end = n_frames

            chunks.append((start, end))

            if end == n_frames:
                break

            start += stride

        return chunks

    def _process_sequential(
        self,
        *,
        input_data: TrajectoryDataset,
        chunks: list[tuple[int, int]],
        setup_and_solve_fn: Callable[[TrajectoryDataset], SolverResult]
    ) -> list[tuple[int, int, SolverResult]]:
        """Process chunks sequentially, creating a new solver for each.

        Args:
            input_data: Full input dataset
            chunks: List of (start, end) chunk ranges
            setup_and_solve_fn: Function to setup solver and solve for chunk

        Returns:
            List of (start, end, result) tuples
        """
        results = []

        for chunk_idx, (start, end) in enumerate(chunks):
            logger.info(f"  Optimizing chunk {chunk_idx}: frames {start}-{end}")

            # Extract chunk data
            chunk_data = input_data.slice_frames(start_frame=start, end_frame=end)

            # Setup new solver and optimize
            result = setup_and_solve_fn(chunk_data)

            results.append((start, end, result))
            logger.info(f"  Chunk {chunk_idx} complete ({result.solve_time_seconds:.1f}s)")

        return results

    def _process_parallel(
        self,
        *,
        input_data: TrajectoryDataset,
        chunks: list[tuple[int, int]],
        setup_and_solve_fn: Callable[[TrajectoryDataset], SolverResult]
    ) -> list[tuple[int, int, SolverResult]]:
        """Process chunks in parallel, each with its own solver.

        Args:
            input_data: Full input dataset
            chunks: List of (start, end) chunk ranges
            setup_and_solve_fn: Function to setup solver and solve for chunk

        Returns:
            List of (start, end, result) tuples
        """
        num_workers = self._get_num_workers()
        logger.info(f"Workers: {num_workers}")

        # Prepare tasks
        tasks = []
        for start, end in chunks:
            chunk_data = input_data.slice_frames(start_frame=start, end_frame=end)
            tasks.append((start, end, chunk_data, setup_and_solve_fn))

        # Process in parallel
        try:
            with mp.Pool(processes=num_workers) as pool:
                raw_results = pool.starmap(_setup_and_solve_chunk_worker, tasks)
        except Exception as e:
            error_msg = (
                f"❌ PARALLEL OPTIMIZATION CRASHED!\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error message: {e}\n"
                f"\n"
                f"One or more chunks failed during parallel processing.\n"
                f"Check the full traceback above for the exact failure point.\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        return raw_results

    def _stitch_results(
        self,
        *,
        chunk_results: list[tuple[int, int, SolverResult]]
    ) -> SolverResult:
        """Stitch chunk results together with blending.

        Args:
            chunk_results: List of (start, end, result) tuples

        Returns:
            Complete stitched SolverResult
        """
        logger.info(f"\n{'=' * 80}")
        logger.info("STITCHING CHUNKS WITH BLENDING")
        logger.info(f"{'=' * 80}")

        # Extract results and metadata
        results = [result for _, _, result in chunk_results]
        chunk_ranges = [(start, end) for start, end, _ in chunk_results]

        # Use TrajectoryDataset's stitch method
        if results[0].optimized_data is None:
            raise ValueError("Cannot stitch results without optimized_data")

        stitched_data = TrajectoryDataset.stitch_with_blending(
            datasets=[r.optimized_data for r in results],
            chunk_ranges=chunk_ranges,
            overlap_size=self.config.overlap_size,
            blend_window=self.config.blend_window
        )

        # Create combined result
        total_time = sum(r.solve_time_seconds for r in results)
        total_iterations = sum(r.num_iterations for r in results)

        # Use first chunk's result as template
        first_result = results[0]

        return type(first_result)(
            success=all(r.success for r in results),
            num_iterations=total_iterations,
            initial_cost=results[0].initial_cost,
            final_cost=results[-1].final_cost,
            solve_time_seconds=total_time,
            raw_data=chunk_results[0][2].raw_data,
            optimized_data=stitched_data
        )


def _setup_and_solve_chunk_worker(
    start: int,
    end: int,
    chunk_data: TrajectoryDataset,
    setup_and_solve_fn: Callable[[TrajectoryDataset], SolverResult]
) -> tuple[int, int, SolverResult]:
    """Worker function for parallel chunk optimization.

    Creates a new solver for this chunk and solves it independently.

    Args:
        start: Start frame in global data
        end: End frame in global data
        chunk_data: Data for this chunk
        setup_and_solve_fn: Function to create solver, setup, and solve

    Returns:
        Tuple of (start, end, result)
    """
    result = setup_and_solve_fn(chunk_data)
    return (start, end, result)