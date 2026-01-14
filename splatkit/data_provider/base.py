from typing import Any, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import torch

from ..splat import SplatModel
from ..modules import SplatRenderPayloadT, SplatBaseModule

@dataclass(frozen=True)
class SplatDataItem():
    """
    Data set item type.
    """
    id: int
    image_name: str
    camera_model: str

    K: torch.Tensor                # (B, 3, 3)
    cam_to_world: torch.Tensor     # (B, 4, 4)
    image: torch.Tensor            # (B, H, W, 3), float32
    
    mask: torch.Tensor | None = None    # (B, H, W), bool
    points: torch.Tensor | None = None      # (B, M, 2)
    depths: torch.Tensor | None = None      # (B, M,)

    
    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)
    
    def to(self, device: torch.device | str) -> "SplatDataItem":
        """
        Move the data item to the given device and return a new instance.
        """
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if torch.is_tensor(v):
                values[f.name] = v.to(torch.device(device))
            else:
                values[f.name] = v
        return type(self)(**values)

    
    @classmethod
    def from_batch(cls, batch: list[dict[str, Any]]) -> "SplatDataItem":
        if not batch:
            raise ValueError("Empty batch")

        values: dict[str, Any] = {}

        for f in fields(cls):
            name = f.name
            elems = [b[name] for b in batch]

            first = elems[0]

            # Tensor → stack
            if torch.is_tensor(first):
                values[name] = torch.stack(elems, dim=0)

            # Optional tensor
            elif first is None:
                values[name] = None

            # Metadata → list
            else:
                values[name] = elems

        return cls(**values)

SplatDataItemT = TypeVar("SplatDataItemT", bound="SplatDataItem")

class SplatDataProvider(
    SplatBaseModule[SplatRenderPayloadT], 
    Generic[SplatRenderPayloadT, SplatDataItemT], 
    ABC
):
    """
    Abstract base class for data providers.
    """

    @abstractmethod
    def load_data(self) -> float:
        """
        Load the data and return the scene scale.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def next_train_data(self, step: int) -> SplatDataItemT:
        """
        Return the next training data item.

        Args:
            step: The current step.
            world_rank: The current world rank.
            world_size: The current world size.

        Returns:
            The next training data item.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_train_data_size(
        self,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> int:
        """
        Return the length of the training data.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def next_test_data(
        self,
        step: int,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> SplatDataItemT:
        """
        Return the next test data item.

        Args:
            step: The current step.
            world_rank: The current world rank.
            world_size: The current world size.

        Returns:
            The next test data item.
        """
        raise NotImplementedError("Subclasses must implement this method")
    

    @abstractmethod
    def get_test_data_size(
        self,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> int:
        """
        Return the size of the test data.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def init_splat_model(
        self,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        world_rank: int = 0,
        world_size: int = 1,

        leader_rank: int = 0,
    ) -> SplatModel | None:
        """
        Initialize a SplatModel.
            
        Args:
            sh_degree: Maximum SH degree
            init_opacity: Initial opacity
            init_scale: Scale multiplier
            world_rank: The current world rank.
            world_size: The current world size.
        Returns:
            SplatModel initialized from data provider
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def scene_scale(self) -> float:
        """
        Return the scene scale.
        """
        raise NotImplementedError("Subclasses must implement this method")