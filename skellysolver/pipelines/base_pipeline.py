import logging
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import model_validator
from typing_extensions import Self

from skellysolver.utilities.arbitrary_types_model import ABaseModel
from skellysolver.data.dataset_manager import load_trajectory_csv, save_solver_result
from skellysolver.data.trajectory_dataset import TrajectoryDataset
from skellysolver.solvers.base_solver import SolverResult, SolverOptimizationReport, PyceresSolver, SolverConfig
from skellysolver.utilities.chunk_processor import ChunkingConfig, ChunkProcessor

logger = logging.getLogger(__name__)


class PipelineSummary(ABaseModel):
    pipeline: str
    status: str
    optimization: SolverOptimizationReport

    def __str__(self) -> str:
        lines = [
            "=" * 80,
            f"Pipeline: {self.pipeline}",
            f"Status: {self.status}",
            str(self.optimization)
        ]
        return "\n".join(lines)


class PipelineConfig(ABaseModel):
    input_path: Path
    output_dir: Path
    solver_config: SolverConfig
    parallel: ChunkingConfig
    input_data_confidence_threshold: float | None = None

    @model_validator(mode='after')
    def ensure_paths_and_create_output_dir(self) -> Self:
        """Ensure paths are Path objects and output dir exists."""
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __str__(self) -> str:
        lines = [
            "Pipeline Configuration:",
            f"  Input:  {self.input_path}",
            f"  Output: {self.output_dir}",
            "",
            str(self.solver_config),
            "",
            str(self.parallel)
        ]
        return "\n".join(lines)


class BasePipeline(ABaseModel, ABC):
    """Base class for optimization pipelines.

    Subclasses must implement setup_and_solve() to define how to:
    1. Create a solver for a chunk of data
    2. Add costs/constraints to the solver
    3. Solve and return results
    """
    config: PipelineConfig
    input_data: TrajectoryDataset
    solver_result: SolverResult | None = None

    @classmethod
    def from_config(cls, *, config: PipelineConfig) -> Self:
        return cls(
            config=config,
            input_data=load_trajectory_csv(
                filepath=config.input_path,
                min_confidence=config.input_data_confidence_threshold
            )
        )

    @abstractmethod
    def setup_and_solve(self, chunk_data: TrajectoryDataset) -> SolverResult:
        """Setup solver and solve for a chunk of data.

        This method is called for each chunk (or once for non-chunked data).
        It should:
        1. Create a new PyceresSolver instance
        2. Add all necessary costs/constraints for the chunk
        3. Call solver.solve()
        4. Return the SolverResult

        Args:
            chunk_data: Trajectory data for this chunk

        Returns:
            SolverResult from optimization
        """
        pass

    def run(self) -> None:
        """Run the pipeline with optional chunking and parallelization."""
        logger.info(f"Data shape: {self.input_data.get_summary()}")
        logger.info(f"Pipeline config: {self.config}")

        # Run chunked optimization with parallel processing
        chunk_processor = ChunkProcessor(config=self.config.parallel)
        self.solver_result = chunk_processor.chunk_run_pipeline(
            input_data=self.input_data,
            setup_and_solve_fn=self.setup_and_solve,
        )

        save_solver_result(
            result=self.solver_result,
            save_directory=self.config.output_dir
        )

        logger.info(f"\n{self.solver_result.summary()}")

    def generate_viewer(self) -> None:
        raise NotImplementedError