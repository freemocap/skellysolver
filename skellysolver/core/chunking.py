"""Chunked optimization for long recordings with smooth blending.

This module provides functionality to split long motion capture recordings
into overlapping chunks, optimize each chunk independently (potentially in parallel),
and blend the results smoothly using SLERP for rotations.
"""

import logging
import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from skellysolver.data.arbitrary_types_model import ABaseModel

logger = logging.getLogger(__name__)



class ChunkInfo(ABaseModel):
    """Information about a single chunk.

    Attributes:
        chunk_id: Index of this chunk
        global_start: Start frame in original data
        global_end: End frame in original data (exclusive)
        local_blend_start: Local index where blending starts
        local_blend_end: Local index where blending ends
    """
    chunk_id: int
    global_start: int
    global_end: int
    local_blend_start: int
    local_blend_end: int

    @property
    def n_frames(self) -> int:
        """Number of frames in this chunk."""
        return self.global_end - self.global_start



class ChunkResult(ABaseModel, ABC):
    """Result from optimizing a single chunk, abc class - need instance for each Solver.

    Attributes:
        chunk_info: Information about the chunk
        rotations: (n_frames, 3, 3) rotation matrices
        translations: (n_frames, 3) translation vectors
        reconstructed: (n_frames, n_markers, 3) reconstructed positions
        reference_geometry: (n_markers, 3) optimized reference shape
        computation_time: Time spent optimizing this chunk
        success: Whether optimization succeeded
    """
    chunk_info: ChunkInfo
    # add chunk data appropriate for each Solver subclass
    computation_time: float
    success: bool

    @abstractmethod
    @classmethod
    def from_solver_result(cls, chunk_info: ChunkInfo, result: Any, computation_time: float) -> "ChunkResult":
        """Create ChunkResult from solver-specific result object."""
        pass


def split_into_chunks(
        *,
        n_frames: int,
        chunk_size: int,
        overlap_size: int,
        min_chunk_size: int = 100
) -> list[ChunkInfo]:
    """Split frame range into overlapping chunks.

    Args:
        n_frames: Total number of frames
        chunk_size: Target frames per chunk
        overlap_size: Number of overlapping frames between chunks
        min_chunk_size: Minimum frames to create a separate chunk

    Returns:
        List of ChunkInfo objects
    """
    if overlap_size >= chunk_size:
        raise ValueError(f"overlap_size ({overlap_size}) must be < chunk_size ({chunk_size})")

    chunks = []
    stride = chunk_size - overlap_size
    chunk_id = 0

    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)

        # Check if this would be a tiny final chunk
        if end < n_frames and (n_frames - end) < min_chunk_size:
            # Extend this chunk to include the remainder
            end = n_frames

        # Determine blend regions
        if chunk_id == 0:
            # First chunk: no blending at start
            local_blend_start = 0
        else:
            # Blend region is at the start
            local_blend_start = 0

        if end == n_frames:
            # Last chunk: no blending at end
            local_blend_end = end - start
        else:
            # Blend region is at the end
            local_blend_end = overlap_size

        chunks.append(ChunkInfo(
            chunk_id=chunk_id,
            global_start=start,
            global_end=end,
            local_blend_start=local_blend_start,
            local_blend_end=local_blend_end
        ))

        if end == n_frames:
            break

        start += stride
        chunk_id += 1

    return chunks


def create_blend_weights(
        *,
        n_frames: int,
        blend_type: str = "cosine"
) -> np.ndarray:
    """Create smooth blending weights from 0 to 1.

    Args:
        n_frames: Length of blend region
        blend_type: "linear" or "cosine" (cosine is smoother)

    Returns:
        (n_frames,) weights transitioning from 0 to 1
    """
    if blend_type == "linear":
        return np.linspace(start=0.0, stop=1.0, num=n_frames)
    elif blend_type == "cosine":
        # Cosine interpolation for smoother blending
        t = np.linspace(start=0.0, stop=1.0, num=n_frames)
        return (1.0 - np.cos(t * np.pi)) / 2.0
    else:
        raise ValueError(f"Unknown blend_type: {blend_type}")


def blend_rotations(
        *,
        R1: np.ndarray,
        R2: np.ndarray,
        weights: np.ndarray
) -> np.ndarray:
    """Blend rotation matrices using spherical linear interpolation (SLERP).

    Args:
        R1: (n_frames, 3, 3) rotation matrices from chunk 1
        R2: (n_frames, 3, 3) rotation matrices from chunk 2
        weights: (n_frames,) blend weights (0=R1, 1=R2)

    Returns:
        (n_frames, 3, 3) blended rotations
    """
    n_frames = len(weights)
    blended = np.zeros((n_frames, 3, 3))

    for i in range(n_frames):
        if weights[i] <= 0.0:
            blended[i] = R1[i]
        elif weights[i] >= 1.0:
            blended[i] = R2[i]
        else:
            # Convert to quaternions for SLERP
            q1 = Rotation.from_matrix(matrix=R1[i]).as_quat()
            q2 = Rotation.from_matrix(matrix=R2[i]).as_quat()

            # Ensure shortest path (same hemisphere)
            if np.dot(q1, q2) < 0:
                q2 = -q2

            # Spherical linear interpolation
            rot_interp = Slerp(
                times=np.array([0.0, 1.0]),
                rotations=Rotation.from_quat(quat=[q1, q2])
            )
            blended[i] = rot_interp(times=weights[i]).as_matrix()

    return blended


def blend_translations(
        *,
        T1: np.ndarray,
        T2: np.ndarray,
        weights: np.ndarray
) -> np.ndarray:
    """Blend translations using linear interpolation.

    Args:
        T1: (n_frames, 3) translations from chunk 1
        T2: (n_frames, 3) translations from chunk 2
        weights: (n_frames,) blend weights (0=T1, 1=T2)

    Returns:
        (n_frames, 3) blended translations
    """
    weights_expanded = weights[:, np.newaxis]
    return (1.0 - weights_expanded) * T1 + weights_expanded * T2


def optimize_chunk_sequential(
        *,
        chunk_info: ChunkInfo,
        chunk_result_model: type[ChunkResult],
        raw_data: np.ndarray,
        optimize_fn: Callable,
        **optimize_kwargs: Any
) -> ChunkResult:
    """Optimize a single chunk (for sequential processing).

    Args:
        chunk_info: Information about this chunk
        chunk_result_model: ChunkResult subclass to use for results
        raw_data: (n_total_frames, n_markers, 3) full dataset
        optimize_fn: Optimization function to call
        **optimize_kwargs: Additional arguments for optimize_fn

    Returns:
        ChunkResult with optimization results

    Raises:
        Exception: Any error during optimization propagates immediately with full traceback
    """
    start_time = time.time()

    # Extract chunk data
    chunk_data = raw_data[chunk_info.global_start:chunk_info.global_end]

    logger.info(f"  Optimizing chunk {chunk_info.chunk_id}: frames {chunk_info.global_start}-{chunk_info.global_end}")

    # NO try-except - let errors crash immediately with full traceback!
    result = optimize_fn(data=chunk_data, **optimize_kwargs)

    computation_time = time.time() - start_time

    return chunk_result_model.from_solver_result(
        chunk_info=chunk_info,
        result=result,
        computation_time=computation_time
    )


def optimize_chunk_parallel_worker(
        chunk_info: ChunkInfo,
        chunk_result_model: type[ChunkResult],
        chunk_data: np.ndarray,
        optimize_fn: Callable,
        optimize_kwargs: dict[str, Any]
) -> ChunkResult:
    """Worker function for parallel chunk optimization.

    This is a separate function to work with multiprocessing.

    IMPORTANT: No '*,' in signature - must accept positional args for pool.starmap!

    Args:
        chunk_info: Information about this chunk
        chunk_result_model: ChunkResult subclass to use for results
        chunk_data: (n_chunk_frames, n_markers, 3) data for this chunk
        optimize_fn: Optimization function
        optimize_kwargs: Arguments for optimize_fn

    Returns:
        ChunkResult

    Raises:
        Exception: Any error during optimization propagates immediately.
                  Multiprocessing will capture and re-raise in main process.
    """
    # Suppress verbose logging in worker processes
    logging.getLogger('skellysolver').setLevel(logging.WARNING)

    start_time = time.time()

    # NO try-except - let errors crash immediately with full traceback!
    # Multiprocessing will capture the exception and re-raise it in the main process
    result = optimize_fn(data=chunk_data, **optimize_kwargs)

    return chunk_result_model.from_solver_result(
        chunk_info=chunk_info,
        result=result,
        computation_time=time.time() - start_time
    )


def optimize_chunked_parallel(
        *,
        raw_data: np.ndarray,
        chunk_result_model: type[ChunkResult],
        chunk_size: int,
        overlap_size: int,
        blend_window: int,
        min_chunk_size: int,
        n_workers: int | None,
        optimize_fn: Callable,
        **optimize_kwargs: Any
) -> ChunkResult:
    """Optimize using parallel chunked processing.

    Args:
        raw_data: (n_frames, n_markers, 3) full dataset
        chunk_result_model: ChunkResult subclass to use for results
        chunk_size: Frames per chunk
        overlap_size: Overlapping frames between chunks
        blend_window: Size of blending window
        min_chunk_size: Minimum chunk size
        n_workers: Number of parallel workers (None = auto)
        optimize_fn: Optimization function
        **optimize_kwargs: Additional arguments for optimize_fn

    Returns:
        ChunkResult with full optimized data

    Raises:
        RuntimeError: If any chunk fails to optimize (with detailed context)
    """
    n_frames, n_markers, _ = raw_data.shape

    if n_workers is None:
        n_workers = max(mp.cpu_count() - 1, 1)

    logger.info("=" * 80)
    logger.info("PARALLEL CHUNKED OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Total frames: {n_frames}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Overlap: {overlap_size}")
    logger.info(f"Blend window: {blend_window}")
    logger.info(f"Workers: {n_workers}")

    # Split into chunks
    chunks = split_into_chunks(
        n_frames=n_frames,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        min_chunk_size=min_chunk_size
    )

    logger.info(f"\nSplit into {len(chunks)} chunks:")
    for chunk in chunks:
        logger.info(
            f"  Chunk {chunk.chunk_id}: frames {chunk.global_start}-{chunk.global_end} ({chunk.n_frames} frames)")

    # Prepare tasks for parallel processing
    tasks = []
    for chunk in chunks:
        chunk_data = raw_data[chunk.global_start:chunk.global_end]
        tasks.append((
            chunk,
            chunk_data,
            optimize_fn,
            optimize_kwargs
        ))

    # Process in parallel
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING {len(tasks)} CHUNKS IN PARALLEL")
    logger.info(f"{'=' * 80}\n")

    start_time = time.time()

    # Catch multiprocessing errors and provide context
    try:
        with mp.Pool(processes=n_workers) as pool:
            chunk_results_unsorted = pool.starmap(
                optimize_chunk_parallel_worker,
                tasks
            )
    except Exception as e:
        # If ANY chunk fails, multiprocessing will raise an exception
        # Re-raise with more context to help debugging
        error_msg = (
            f"❌ PARALLEL OPTIMIZATION CRASHED!\n"
            f"  Error type: {type(e).__name__}\n"
            f"  Error message: {e}\n"
            f"\n"
            f"One or more chunks failed during parallel processing.\n"
            f"Check the full traceback above for the exact failure point.\n"
            f"\n"
            f"Common causes:\n"
            f"  1. Bad/missing input data in a specific time range\n"
            f"  2. Numerical instability in the optimizer\n"
            f"  3. Invalid configuration parameters (weights, tolerances)\n"
            f"  4. Out of memory (try reducing chunk_size or n_workers)\n"
            f"  5. Bug in optimization function\n"
            f"\n"
            f"Debug steps:\n"
            f"  1. Check the full error traceback above\n"
            f"  2. Try sequential mode (ParallelConfig(enabled=False))\n"
            f"  3. Reduce chunk_size to isolate problematic time range\n"
            f"  4. Inspect input data for NaN/Inf values"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    total_time = time.time() - start_time

    # Sort by chunk_id
    chunk_results = [chunk_result_model.from_solver_result(
        chunk_info=result.chunk_info,
        result=result,
        computation_time=result.computation_time
    ) for result in  sorted(chunk_results_unsorted, key=lambda x: x.chunk_info.chunk_id)]

    # NO check for success=False - errors crash immediately now!
    # If we reached here, all chunks succeeded (or would have crashed)

    logger.info(f"\n{'=' * 80}")
    logger.info("PARALLEL OPTIMIZATION COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Average per chunk: {total_time / len(tasks):.1f}s")

    # Stitch results
    return stitch_chunk_results(
        chunk_results=chunk_results,
        n_frames=n_frames,
        n_markers=n_markers,
        blend_window=blend_window,
        overlap_size=overlap_size
    )


def optimize_chunked_sequential(
        *,
        raw_data: np.ndarray,
        chunk_size: int,
        overlap_size: int,
        blend_window: int,
        min_chunk_size: int,
        optimize_fn: Callable,
        **optimize_kwargs: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize using sequential chunked processing.

    Args:
        raw_data: (n_frames, n_markers, 3) full dataset
        chunk_size: Frames per chunk
        overlap_size: Overlapping frames between chunks
        blend_window: Size of blending window
        min_chunk_size: Minimum chunk size
        optimize_fn: Optimization function
        **optimize_kwargs: Additional arguments for optimize_fn

    Returns:
        Tuple of (rotations, translations, reconstructed)

    Raises:
        Exception: Any optimization error propagates immediately with full traceback
    """
    n_frames, n_markers, _ = raw_data.shape

    logger.info("=" * 80)
    logger.info("SEQUENTIAL CHUNKED OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Total frames: {n_frames}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Overlap: {overlap_size}")
    logger.info(f"Blend window: {blend_window}")

    # Split into chunks
    chunks = split_into_chunks(
        n_frames=n_frames,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        min_chunk_size=min_chunk_size
    )

    logger.info(f"\nSplit into {len(chunks)} chunks:")
    for chunk in chunks:
        logger.info(
            f"  Chunk {chunk.chunk_id}: frames {chunk.global_start}-{chunk.global_end} ({chunk.n_frames} frames)")

    # Process sequentially
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING {len(chunks)} CHUNKS SEQUENTIALLY")
    logger.info(f"{'=' * 80}\n")

    chunk_results = []
    for chunk in chunks:
        result = optimize_chunk_sequential(
            chunk_info=chunk,
            raw_data=raw_data,
            optimize_fn=optimize_fn,
            **optimize_kwargs
        )
        chunk_results.append(result)
        logger.info(f"Chunk {chunk.chunk_id} complete ({result.computation_time:.1f}s)")

    # Stitch results
    return stitch_chunk_results(
        chunk_results=chunk_results,
        n_frames=n_frames,
        n_markers=n_markers,
        blend_window=blend_window,
        overlap_size=overlap_size
    )


def stitch_chunk_results(
        *,
        chunk_results: list[ChunkResult],
        n_frames: int,
        n_markers: int,
        blend_window: int,
        overlap_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch chunk results with smooth blending.

    Args:
        chunk_results: List of ChunkResult objects (sorted by chunk_id)
        n_frames: Total number of frames
        n_markers: Number of markers
        blend_window: Size of blending window
        overlap_size: Overlap size between chunks

    Returns:
        Tuple of (rotations, translations, reconstructed)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("STITCHING CHUNKS WITH BLENDING")
    logger.info(f"{'=' * 80}")

    # Allocate output arrays
    all_rotations = np.zeros((n_frames, 3, 3))
    all_translations = np.zeros((n_frames, 3))
    all_reconstructed = np.zeros((n_frames, n_markers, 3))

    # Use first chunk's reference geometry for reconstruction
    reference_geometry = chunk_results[0].reference_geometry

    for chunk_idx, chunk_result in enumerate(chunk_results):
        chunk_info = chunk_result.chunk_info
        global_start = chunk_info.global_start
        global_end = chunk_info.global_end

        if chunk_idx == 0:
            # First chunk: copy directly (no previous chunk to blend with)
            blend_end = global_end - overlap_size if chunk_idx < len(chunk_results) - 1 else global_end

            all_rotations[global_start:blend_end] = chunk_result.rotations[:blend_end - global_start]
            all_translations[global_start:blend_end] = chunk_result.translations[:blend_end - global_start]
            all_reconstructed[global_start:blend_end] = chunk_result.reconstructed[:blend_end - global_start]

            logger.info(f"Chunk 0: Copied frames {global_start}-{blend_end}")

        else:
            # Subsequent chunks: blend overlap region with previous chunk
            prev_result = chunk_results[chunk_idx - 1]
            overlap_start = global_start
            overlap_end = min(global_start + overlap_size, global_end)
            blend_size = min(blend_window, overlap_end - overlap_start)

            # Blend region
            blend_global_start = overlap_start
            blend_global_end = overlap_start + blend_size

            # Extract data from both chunks
            prev_local_start = blend_global_start - prev_result.chunk_info.global_start
            prev_local_end = blend_global_end - prev_result.chunk_info.global_start

            curr_local_start = blend_global_start - global_start
            curr_local_end = blend_global_end - global_start

            R_prev = prev_result.rotations[prev_local_start:prev_local_end]
            T_prev = prev_result.translations[prev_local_start:prev_local_end]

            R_curr = chunk_result.rotations[curr_local_start:curr_local_end]
            T_curr = chunk_result.translations[curr_local_start:curr_local_end]

            # Create blend weights (cosine for smoothness)
            weights = create_blend_weights(n_frames=blend_size, blend_type="cosine")

            # Blend using SLERP for rotations, linear for translations
            R_blended = blend_rotations(R1=R_prev, R2=R_curr, weights=weights)
            T_blended = blend_translations(T1=T_prev, T2=T_curr, weights=weights)

            # Reconstruct from blended poses
            recon_blended = np.zeros((blend_size, n_markers, 3))
            for i in range(blend_size):
                recon_blended[i] = (R_blended[i] @ reference_geometry.T).T + T_blended[i]

            # Store blended region
            all_rotations[blend_global_start:blend_global_end] = R_blended
            all_translations[blend_global_start:blend_global_end] = T_blended
            all_reconstructed[blend_global_start:blend_global_end] = recon_blended

            logger.info(f"Chunk {chunk_idx}: Blended frames {blend_global_start}-{blend_global_end}")

            # Copy non-overlapping region from current chunk
            copy_start = blend_global_end
            copy_end = global_end - (overlap_size if chunk_idx < len(chunk_results) - 1 else 0)

            if copy_start < copy_end:
                local_copy_start = copy_start - global_start
                local_copy_end = copy_end - global_start

                all_rotations[copy_start:copy_end] = chunk_result.rotations[local_copy_start:local_copy_end]
                all_translations[copy_start:copy_end] = chunk_result.translations[local_copy_start:local_copy_end]
                all_reconstructed[copy_start:copy_end] = chunk_result.reconstructed[local_copy_start:local_copy_end]

                logger.info(f"Chunk {chunk_idx}: Copied frames {copy_start}-{copy_end}")

    logger.info(f"\nStitching complete: {n_frames} frames")

    return all_rotations, all_translations, all_reconstructed


class ChunkingConfig(ABaseModel):
    """Configuration for parallel chunked optimization.

    Used when processing long recordings that need to be split
    into chunks and processed in parallel.

    Attributes:
        enabled: Whether to use parallel processing
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

    def get_num_workers(self) -> int:
        """Get number of workers.

        Returns:
            Number of workers to use
        """
        if self.num_workers is None:
            cpu_count = os.cpu_count()
            return cpu_count if cpu_count else 1
        return self.num_workers

    def should_use_parallel(self, *, n_frames: int) -> bool:
        """Determine if parallel processing should be used.

        Args:
            n_frames: Total number of frames

        Returns:
            True if should use parallel processing
        """
        if not self.enabled:
            return False

        # Only use parallel if we have enough frames
        return n_frames > (self.chunk_size + self.min_chunk_size)
