"""Pytest configuration and fixtures for SkellySolver tests.

Provides reusable fixtures for:
- Test data (trajectories, CSVs)
- Configurations
- Mock pipelines
- Temporary directories
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for tests.
    
    Yields:
        Path to temporary directory (auto-cleaned up)
    """
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def sample_3d_trajectory() -> np.ndarray:
    """Create sample 3D trajectory data.
    
    Returns:
        (n_frames, n_markers, 3) array with synthetic trajectory
    """
    n_frames = 100
    n_markers = 5
    
    # Generate smooth trajectory
    t = np.linspace(0, 2 * np.pi, n_frames)
    
    positions = np.zeros((n_frames, n_markers, 3))
    
    for i in range(n_markers):
        # Circular motion with offset per marker
        offset = i * 0.1
        positions[:, i, 0] = np.cos(t) + offset
        positions[:, i, 1] = np.sin(t) + offset
        positions[:, i, 2] = 0.0
    
    return positions


@pytest.fixture
def sample_2d_observations() -> np.ndarray:
    """Create sample 2D observation data.
    
    Returns:
        (n_frames, n_points, 2) array with synthetic observations
    """
    n_frames = 100
    n_points = 8
    
    # Generate ellipse points
    observations = np.zeros((n_frames, n_points, 2))
    
    for frame_idx in range(n_frames):
        # Pupil points on ellipse
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        
        # Vary size over time
        scale = 1.0 + 0.2 * np.sin(frame_idx * 0.1)
        
        observations[frame_idx, :, 0] = 200 + 50 * scale * np.cos(angles)
        observations[frame_idx, :, 1] = 200 + 30 * scale * np.sin(angles)
    
    return observations


@pytest.fixture
def sample_marker_names() -> list[str]:
    """Sample marker names.
    
    Returns:
        List of marker names
    """
    return ["marker1", "marker2", "marker3", "marker4", "marker5"]


@pytest.fixture
def sample_topology_dict() -> dict[str, Any]:
    """Sample topology dictionary.
    
    Returns:
        Topology dictionary
    """
    return {
        "name": "test_topology",
        "marker_names": ["m1", "m2", "m3"],
        "rigid_edges": [(0, 1), (1, 2), (2, 0)],
        "display_edges": [(0, 1), (1, 2), (2, 0)],
    }


@pytest.fixture
def tidy_csv_content() -> str:
    """Content for tidy format CSV.
    
    Returns:
        CSV content string
    """
    return """frame,keypoint,x,y,z
0,marker1,1.0,2.0,3.0
0,marker2,4.0,5.0,6.0
0,marker3,7.0,8.0,9.0
1,marker1,1.1,2.1,3.1
1,marker2,4.1,5.1,6.1
1,marker3,7.1,8.1,9.1
"""


@pytest.fixture
def wide_csv_content() -> str:
    """Content for wide format CSV.
    
    Returns:
        CSV content string
    """
    return """frame,marker1_x,marker1_y,marker1_z,marker2_x,marker2_y,marker2_z
0,1.0,2.0,3.0,4.0,5.0,6.0
1,1.1,2.1,3.1,4.1,5.1,6.1
2,1.2,2.2,3.2,4.2,5.2,6.2
"""


@pytest.fixture
def dlc_csv_content() -> str:
    """Content for DeepLabCut format CSV.
    
    Returns:
        CSV content string
    """
    return """scorer,scorer,scorer,scorer,scorer,scorer
bodypart1,bodypart1,bodypart1,bodypart2,bodypart2,bodypart2
x,y,likelihood,x,y,likelihood
1.0,2.0,0.95,4.0,5.0,0.92
1.1,2.1,0.93,4.1,5.1,0.94
1.2,2.2,0.96,4.2,5.2,0.91
"""


@pytest.fixture
def create_tidy_csv(temp_dir: Path, tidy_csv_content: str) -> Path:
    """Create tidy CSV file.
    
    Args:
        temp_dir: Temporary directory
        tidy_csv_content: CSV content
        
    Returns:
        Path to created CSV
    """
    csv_path = temp_dir / "tidy.csv"
    csv_path.write_text(data=tidy_csv_content)
    return csv_path


@pytest.fixture
def create_wide_csv(temp_dir: Path, wide_csv_content: str) -> Path:
    """Create wide CSV file.
    
    Args:
        temp_dir: Temporary directory
        wide_csv_content: CSV content
        
    Returns:
        Path to created CSV
    """
    csv_path = temp_dir / "wide.csv"
    csv_path.write_text(data=wide_csv_content)
    return csv_path


@pytest.fixture
def create_dlc_csv(temp_dir: Path, dlc_csv_content: str) -> Path:
    """Create DLC CSV file.
    
    Args:
        temp_dir: Temporary directory
        dlc_csv_content: CSV content
        
    Returns:
        Path to created CSV
    """
    csv_path = temp_dir / "dlc.csv"
    csv_path.write_text(data=dlc_csv_content)
    return csv_path


@pytest.fixture
def sample_quaternion() -> np.ndarray:
    """Sample unit quaternion.
    
    Returns:
        (4,) quaternion [w, x, y, z]
    """
    return np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def sample_translation() -> np.ndarray:
    """Sample translation vector.
    
    Returns:
        (3,) translation
    """
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def noisy_trajectory_data() -> np.ndarray:
    """Generate noisy trajectory data for testing.
    
    Returns:
        (n_frames, n_markers, 3) noisy positions
    """
    from scipy.spatial.transform import Rotation
    
    n_frames = 50
    n_markers = 4
    
    # Generate ground truth
    reference = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5],
    ])
    
    ground_truth = np.zeros((n_frames, n_markers, 3))
    
    for i in range(n_frames):
        # Smooth rotation
        angle = i / n_frames * 2 * np.pi
        R = Rotation.from_euler(seq='z', angles=angle).as_matrix()
        
        # Translation
        translation = np.array([np.sin(angle), np.cos(angle), 0.0])
        
        # Transform
        ground_truth[i] = (R @ reference.T).T + translation
    
    # Add noise
    noise = np.random.normal(loc=0.0, scale=0.05, size=ground_truth.shape)
    noisy = ground_truth + noise
    
    return noisy
