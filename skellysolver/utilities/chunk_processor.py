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

class ChunkResults(ABaseModel):
    result: SolverResult
    start_frame: int
    end_frame: int
class ChunkingConfig(ABaseModel):
    enabled: bool = True
    chunk_size: int = 500
    overlap_size: int = 50
    blend_window: int = 25
    min_chunk_ratio: float = 0.1  # Minimum chunk size as ratio of chunk_size
    num_workers: int | None = None

    @property
    def min_chunk_size(self) -> int:
        return int(self.chunk_size * self.min_chunk_ratio)

    def __str__(self) -> str:
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
    config: ChunkingConfig

    def chunk_run_pipeline(
        self,
        *,
        input_data: TrajectoryDataset,
        setup_and_solve_fn: Callable[[TrajectoryDataset], SolverResult],
    ) -> SolverResult:
        chunks = self._split_into_chunks(n_frames=input_data.n_frames)
        logger.info("=" * 80)
        logger.info("=" * 80)
        logger.info(f"Total frames: {input_data.n_frames}")
        logger.info(f"Chunk size: {self.config.chunk_size}")
        logger.info(f"Overlap: {self.config.overlap_size}")
        logger.info(f"Blend window: {self.config.blend_window}")
        logger.info(f"\nSplit into {len(chunks)} chunks:")
        for i, (start, end) in enumerate(chunks):
            logger.info(f"  Chunk {i}: frames {start}-{end} ({end-start} frames)")
        start_time = time.time()

        chunk_results = self._process_parallel(
            input_data=input_data,
            chunks=chunks,
            setup_and_solve_fn=setup_and_solve_fn
            )

        total_time = time.time() - start_time
        logger.info(f"\n{'=' * 80}")
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Average per chunk: {total_time/len(chunks):.1f}s")

        return self._stitch_results(chunk_results=chunk_results)

    def _get_num_workers(self) -> int:
        if self.config.num_workers is not None:
            return max(self.config.num_workers, 1)
        cpu_count = mp.cpu_count()
        return max((cpu_count or 1) - 1, 1)

    def _split_into_chunks(self, *, n_frames: int) -> list[tuple[int, int]]:
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


    def _process_parallel(
        self,
        *,
        input_data: TrajectoryDataset,
        chunks: list[tuple[int, int]],
        setup_and_solve_fn: Callable[[TrajectoryDataset], SolverResult]
    ) -> list[tuple[int, int, SolverResult]]:
        logger.info(f"Workers: {self._get_num_workers()}")
        tasks = []
        for start, end in chunks:
            chunk_data = input_data.slice_frames(start_frame=start, end_frame=end)
            tasks.append((start, end, chunk_data, setup_and_solve_fn))
        try:

            with mp.Pool(processes=self._get_num_workers()) as pool:
                chunk_results = pool.starmap(_setup_and_solve_chunk_worker, tasks)

        except Exception as e:
            error_msg = (
                f" PARALLEL OPTIMIZATION CRASHED!\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error message: {e}\n"
                f"\n"
                f"One or more chunks failed during parallel processing.\n"
                f"Check the full traceback above for the exact failure point.\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        return chunk_results

    def _stitch_results(
        self,
        *,
        chunk_results: list[tuple[int, int, SolverResult]]
    ) -> TrajectoryDataset:
        logger.info(f"\n{'=' * 80}")
        logger.info("STITCHING CHUNKS WITH BLENDING")
        logger.info(f"{'=' * 80}")
        return  TrajectoryDataset.stitch_with_blending(
            datasets=chunk_datasets,
            chunk_result=chunk_results,
            overlap_size=self.config.overlap_size,
            blend_window=self.config.blend_window
        )

def _setup_and_solve_chunk_worker(
    start: int,
    end: int,
    chunk_data: TrajectoryDataset,
    setup_and_solve_fn: Callable[[TrajectoryDataset], SolverResult]
) -> ChunkResults:

    result = setup_and_solve_fn(chunk_data)

    return ChunkResults(
        result=result,
        start_frame=start,
        end_frame=end
    )