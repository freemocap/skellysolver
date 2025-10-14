"""
Refactored chains - using actual ChainABC instances
"""
from typing import ClassVar

from skellysolver.data.arbitrary_types_model import ABaseModel
from skellysolver.solvers.mocap_solver.mocap_models.abstract_base_classes.chain_abc import ChainABC
from .linkages import HumanLinkages as link


class HumanChains(ABaseModel):
    """
    Container for all human skeleton chains.
    A chain is a series of connected linkages.
    """

    AXIAL: ClassVar[ChainABC] = ChainABC(
        parent=link.CHEST_T12,
        children=[
            link.NECK_C7,
            link.SKULL_C1
        ],
        shared_segments=[]
    )

    RIGHT_ARM: ClassVar[ChainABC] = ChainABC(
        parent=link.RIGHT_SHOULDER,
        children=[
            link.RIGHT_ELBOW,
            link.RIGHT_WRIST,
        ],
        shared_segments=[]
    )

    RIGHT_LEG: ClassVar[ChainABC] = ChainABC(
        parent=link.RIGHT_HIP,
        children=[
            link.RIGHT_KNEE,
            link.RIGHT_ANKLE,
        ],
        shared_segments=[]
    )

    LEFT_ARM: ClassVar[ChainABC] = ChainABC(
        parent=link.LEFT_SHOULDER,
        children=[
            link.LEFT_ELBOW,
            link.LEFT_WRIST,
        ],
        shared_segments=[]
    )

    LEFT_LEG: ClassVar[ChainABC] = ChainABC(
        parent=link.LEFT_HIP,
        children=[
            link.LEFT_KNEE,
            link.LEFT_ANKLE,
        ],
        shared_segments=[]
    )

    @classmethod
    def get_all_chains(cls) -> dict[str, ChainABC]:
        """Get all chains as a dictionary."""
        return {
            name: getattr(cls, name)
            for name in dir(cls)
            if not name.startswith("_") and isinstance(getattr(cls, name), ChainABC)
        }