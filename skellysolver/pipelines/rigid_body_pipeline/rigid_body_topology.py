"""Rigid body topology - defines structure of rigid bodies for tracking.

This is the constraint definition that describes which markers form rigid bodies
and how they're connected.
"""

import json
from pathlib import Path

import numpy as np
from pydantic import model_validator
from typing_extensions import Self

from skellysolver.solvers.constraints.base_constraint import BaseConstraint


class RigidBodyTopology(BaseConstraint):
    """Defines which markers form a rigid body and their connectivity.
    
    Attributes:
        marker_names: Names of markers belonging to this rigid body
        rigid_edges: Pairs of marker indices that must maintain fixed distance
        soft_edges: Optional pairs that prefer but don't require fixed distance
        display_edges: Optional edges for visualization (defaults to rigid_edges)
        name: Descriptive name for this rigid body configuration
    """
    
    marker_names: list[str]
    rigid_edges: list[tuple[int, int]]
    soft_edges: list[tuple[int, int]] | None = None
    display_edges: list[tuple[int, int]] | None = None
    name: str = "rigid_body"
    
    @model_validator(mode='after')
    def validate_and_set_defaults(self) -> Self:
        """Validate indices and set default display edges."""
        # Set display edges to rigid edges if not provided
        if self.display_edges is None:
            self.display_edges = self.rigid_edges.copy()
        
        # Validate edge indices
        n_markers = len(self.marker_names)
        all_edges = self.rigid_edges + (self.soft_edges or [])
        
        for i, j in all_edges:
            if i < 0 or i >= n_markers or j < 0 or j >= n_markers:
                raise ValueError(
                    f"Invalid edge ({i}, {j}): indices must be in range [0, {n_markers})"
                )
            if i == j:
                raise ValueError(f"Invalid edge ({i}, {j}): cannot connect marker to itself")
        
        return self
    
    @property
    def n_markers(self) -> int:
        """Number of markers in this rigid body."""
        return len(self.marker_names)
    
    @property
    def n_rigid_edges(self) -> int:
        """Number of rigid edges."""
        return len(self.rigid_edges)
    
    @property
    def n_soft_edges(self) -> int:
        """Number of soft edges."""
        return len(self.soft_edges) if self.soft_edges else 0
    
    def compute_reference_distances(
        self,
        *,
        reference_geometry: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise distances for all edges.
        
        Args:
            reference_geometry: (n_markers, 3) reference positions
            
        Returns:
            (n_markers, n_markers) distance matrix (0 for non-edge pairs)
        """
        n = self.n_markers
        distances = np.zeros((n, n))
        
        # Compute distances for rigid edges
        for i, j in self.rigid_edges:
            dist = np.linalg.norm(reference_geometry[i] - reference_geometry[j])
            distances[i, j] = dist
            distances[j, i] = dist
        
        # Compute distances for soft edges
        if self.soft_edges:
            for i, j in self.soft_edges:
                dist = np.linalg.norm(reference_geometry[i] - reference_geometry[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def validate_trajectory_data(
        self,
        *,
        trajectory_dict: dict[str, np.ndarray]
    ) -> None:
        """Validate that trajectory data contains all required markers.
        
        Args:
            trajectory_dict: Dictionary mapping marker names to trajectories
            
        Raises:
            ValueError: If any markers are missing
        """
        missing = set(self.marker_names) - set(trajectory_dict.keys())
        if missing:
            raise ValueError(
                f"Missing {len(missing)} markers in data: {sorted(missing)}"
            )
    
    def extract_marker_array(
        self,
        *,
        trajectory_dict: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Extract and order trajectories according to topology.
        
        Args:
            trajectory_dict: Maps marker names to (n_frames, 3) arrays
            
        Returns:
            (n_frames, n_markers, 3) ordered trajectory array
        """
        self.validate_trajectory_data(trajectory_dict=trajectory_dict)
        
        trajectories = [trajectory_dict[name] for name in self.marker_names]
        return np.stack(trajectories, axis=1)
    
    def to_dict(self) -> dict[str, object]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "marker_names": self.marker_names,
            "rigid_edges": self.rigid_edges,
            "soft_edges": self.soft_edges,
            "display_edges": self.display_edges,
        }
    
    @classmethod
    def from_dict(cls, *, data: dict[str, object]) -> Self:
        """Create topology from dictionary."""
        return cls(
            name=str(data["name"]),
            marker_names=list(data["marker_names"]),
            rigid_edges=list(data["rigid_edges"]),
            soft_edges=list(data.get("soft_edges")) if data.get("soft_edges") else None,
            display_edges=list(data.get("display_edges")) if data.get("display_edges") else None,
        )
    
    def save_json(self, *, filepath: Path) -> None:
        """Save topology to JSON file."""
        with open(filepath, "w") as f:
            json.dump(obj=self.to_dict(), fp=f, indent=2)
    
    @classmethod
    def load_json(cls, *, filepath: Path) -> Self:
        """Load topology from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(fp=f)
        return cls.from_dict(data=data)
    
    @classmethod
    def from_marker_names(
        cls,
        *,
        marker_names: list[str],
        name: str = "auto_generated",
        edge_strategy: str = "full",
        soft_edge_strategy: str | None = None
    ) -> Self:
        """Create topology automatically from marker names.
        
        Args:
            marker_names: List of marker names
            name: Name for this topology
            edge_strategy: Strategy for rigid edges:
                - "full": Connect all pairs (n*(n-1)/2 edges)
                - "minimal": Create minimal spanning tree (star from first marker)
                - "skeleton": Connect adjacent markers in sequence
            soft_edge_strategy: Optional strategy for soft edges (same options)
            
        Returns:
            RigidBodyTopology instance
        """
        n = len(marker_names)
        
        # Generate rigid edges
        if edge_strategy == "full":
            rigid_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        elif edge_strategy == "minimal":
            rigid_edges = [(0, i) for i in range(1, n)]
        elif edge_strategy == "skeleton":
            rigid_edges = [(i, i + 1) for i in range(n - 1)]
        else:
            raise ValueError(f"Unknown edge_strategy: {edge_strategy}")
        
        # Generate soft edges if requested
        soft_edges = None
        if soft_edge_strategy is not None:
            if soft_edge_strategy == "full":
                soft_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            elif soft_edge_strategy == "minimal":
                soft_edges = [(0, i) for i in range(1, n)]
            elif soft_edge_strategy == "skeleton":
                soft_edges = [(i, i + 1) for i in range(n - 1)]
            else:
                raise ValueError(f"Unknown soft_edge_strategy: {soft_edge_strategy}")
        
        return cls(
            marker_names=marker_names,
            rigid_edges=rigid_edges,
            soft_edges=soft_edges,
            name=name
        )
    
    def __str__(self) -> str:
        """Human-readable summary."""
        soft_str = f", soft={self.n_soft_edges}" if self.n_soft_edges > 0 else ""
        return (
            f"RigidBodyTopology(name='{self.name}', "
            f"markers={self.n_markers}, "
            f"rigid={self.n_rigid_edges}{soft_str})"
        )
