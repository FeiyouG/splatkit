from typing import Any, KeysView, ItemsView, ValuesView, TypeVar, Generic
from typing_extensions import NotRequired
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ..splat import SplatModel
from ..modules import SplatBaseFrameT, SplatBaseModule

@dataclass(frozen=True)
class SplatDataItem():
    """
    Data set item type.
    """
    id: int
    image_name: str
    camera_model: str

    K: torch.Tensor                # (3, 3)
    cam_to_world: torch.Tensor     # (4, 4)
    image: torch.Tensor            # (H, W, 3), float32
    
    mask: NotRequired[torch.Tensor]        # (H, W), bool
    points: NotRequired[torch.Tensor]      # (M, 2)
    depths: NotRequired[torch.Tensor]      # (M,)

    
    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    def values(self) -> ValuesView[Any]:
        return self.__dict__.values()

    def items(self) -> ItemsView[str, Any]:
        return self.__dict__.items()

    def to_dict(self) -> dict:
        return dict(self.__dict__)

SplatDataItemT = TypeVar("SplatDataItemType", bound="SplatDataItem")

class SplatDataProvider(
    SplatBaseModule[SplatBaseFrameT], 
    Generic[SplatBaseFrameT, SplatDataItemT], 
    ABC
):
    """
    Abstract base class for data providers.
    """

    @abstractmethod
    def next_train_data(
        self,
        step: int,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> SplatDataItemT:
        """
        Return the next training data item.

        Args:
            step: The current step.
            world_rank: The current world rank.
            world_size: The current world size.

        Returns:
            The next training data item.
        """
        pass

    @abstractmethod
    def get_train_data_size(
        self,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> int:
        """
        Return the length of the training data.
        """
        pass

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
        pass
    

    @abstractmethod
    def get_test_data_size(
        self,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> int:
        """
        Return the size of the test data.
        """
        pass

    @abstractmethod
    def init_splat_model(
        self,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
    ) -> SplatModel:
        """
        Initialize a SplatModel.
            
        Args:
            sh_degree: Maximum SH degree
            init_opacity: Initial opacity
            init_scale: Scale multiplier
            
        Returns:
            SplatModel initialized from data provider
        """
        pass