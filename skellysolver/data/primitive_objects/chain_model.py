from pydantic import model_validator, BaseModel

from skellysolver.data.primitive_objects.keypoint_model import Keypoint
from skellysolver.data.primitive_objects.linkage_model import Linkage


class Chain(BaseModel):
    """
    A Chain is a set of linkages that are connected via shared Segments.
    The parent linkage is the root of the chain, and the children linkages are the
    linkages that are connected to the parent linkage via shared segments (in order according to their list index).
    the distance between the origin keypoint and the end keypoint of the chain is the sum of the lengths of the segments in the chain.
    """
    name:str
    parent: Linkage
    children: list[Linkage]


    @property
    def root(self) -> Keypoint:
        # Chain -> Linkage -> Segment -> Keypoint
        return self.parent.root


    @model_validator(mode="after")
    def check_linkages(self) -> "Chain":
        """
        Ensure that the linkages are connected via shared segments.
        """
        if len(self.children) == 0:
            raise ValueError("Chain must have at least one child linkage.")
        if not self.children[0].parent in self.parent.children:
            raise ValueError(f"First child linkage must be connected to parent linkage. {self.children[0]} not in {self.parent.children}")
        for i in range(1, len(self.children)):
            if not self.children[i].parent in self.children[i - 1].children:
                raise ValueError(f"Child linkage {self.children[i]} not connected to previous linkage {self.children[i - 1]}.")

        return self

    @property
    def linkages(self) -> list[Linkage]:
        return [self.parent] + self.children


class RingChainABC(Chain):
    """
    A RingChain is a Chain where the last child linkage is connected to the parent linkage via a shared segment.
    """

    @model_validator(mode="after")
    def check_ring(self) -> "RingChainABC":
        """
        Ensure that the last child linkage is connected to the parent linkage via a shared segment.
        """
        if not self.parent.parent in self.children[-1].children:
            raise ValueError(f"Last child linkage must be connected to parent linkage. {self.parent} not in {self.children[-1].children}")
        return self