from typing import Any, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import torch

from ..splat import SplatModel
from ..modules import SplatRenderPayloadT, SplatBaseModule
from ..logger import SplatLogger

@dataclass(frozen=True)
class SplatDataItem():
    """
    Single training data item containing an image and camera parameters.
    
    This is returned by data providers for each training step. Contains all
    information needed to render and compare against ground truth.

    Args:
        id: Unique identifier for this data item
        image_name: Original image filename (for logging/debugging)
        camera_model: Camera type ("pinhole", "ortho", "fisheye", or "ftheta")
        K: Camera intrinsics matrix, shape (B, 3, 3). Contains focal lengths (fx, fy) and principal point (cx, cy)
        cam_to_world: Camera-to-world transformation, shape (B, 4, 4). Converts camera space to world space
        image: RGB image in [0, 1] range, shape (B, H, W, 3), dtype float32
        mask: Optional binary mask, shape (B, H, W), dtype bool. True = valid pixel, False = ignore (e.g., sky, dynamic objects)
        points: Optional 2D feature points, shape (B, M, 2). For depth supervision or sparse matching
        depths: Optional depth values at points, shape (B, M). Corresponding depths for the 2D points
    """
    
    id: int
    image_name: str
    camera_model: str
    K: torch.Tensor
    cam_to_world: torch.Tensor
    image: torch.Tensor
    mask: torch.Tensor | None = None
    points: torch.Tensor | None = None
    depths: torch.Tensor | None = None

    
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
    Base class for loading and serving training data.
    
    Data providers handle loading images, cameras, and optional metadata
    from various formats (COLMAP, NeRF, etc.). They sample training data
    and provide initialization points for Gaussians.
    
    Subclasses must implement:
        - load_data(logger: "SplatLogger") -> float: Load all data from disk
        - next_train_data(): Sample next training item(s)
        - get_init_point_cloud(): Get initial 3D points for Gaussians
    
    Available data providers:
        - SplatColmapDataProvider: Loads COLMAP sparse reconstruction
    
    Example:
        >>> from splatkit.data_provider import SplatColmapDataProvider, SplatColmapDataProviderConfig
        >>> config = SplatColmapDataProviderConfig(
        ...     colmap_dir="data/sparse/0",
        ...     images_dir="data/images",
        ... )
        >>> provider = SplatColmapDataProvider(config)
        >>> scene_scale = provider.load_data(logger)
        >>> data_item = provider.next_train_data(step=0)
    
    Gotchas:
        - Call load_data() before next_train_data()
        - Images returned in [0, 1] range as float32
        - Scene scale affects densification thresholds
    """

    @abstractmethod
    def load_data(self, logger: SplatLogger) -> float:
        """
        Load all data from disk and return scene scale.
        
        This is called once at the start of training. Loads images, cameras,
        and any metadata needed for training. Computes scene scale based on
        camera positions.
        
        Returns:
            scene_scale: Radius of scene's bounding sphere
                        Used to normalize densification thresholds
        
        Gotchas:
            - Must be called before next_train_data()
            - Scene scale affects Gaussian splitting/cloning
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def next_train_data(self, step: int) -> SplatDataItemT:
        """
        Sample and return the next training data item(s).

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