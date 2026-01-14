import os
from typing import Iterator, Sequence

from typing_extensions import override
import numpy as np
from torch.utils.data import DataLoader

from .config import SplatColmapDataProviderConfig
from .dataset import SplatColmapDataset
from ..base import SplatDataProvider
from .item import ColmapDataItem
from ...splat import SplatModel
from ...modules import SplatRenderPayload, SplatRenderPayloadT
from ...modules.base import SplatBaseModule


class SplatColmapDataProvider(
    SplatDataProvider[SplatRenderPayload, ColmapDataItem],
):
    """
    Data provider for COLMAP datasets.
    """
    _config: SplatColmapDataProviderConfig

    _colmap_dataset: SplatColmapDataset
    _train_dataset: SplatColmapDataset
    _test_dataset: SplatColmapDataset

    _train_data_loader: DataLoader[ColmapDataItem] | None = None
    _test_data_loader: DataLoader[ColmapDataItem] | None = None

    _train_data_iter: Iterator[ColmapDataItem] | None = None
    _test_data_iter: Iterator[ColmapDataItem] | None = None

    def __init__(self, config: SplatColmapDataProviderConfig):
        self._config = config
        self.__post_init__()
    
    def __post_init__(self):
        """Post initialization validation."""
        if not os.path.exists(self._config.colmap_dir):
            raise FileNotFoundError(f"COLMAP directory not found: {self._config.colmap_dir}")
        
        if not os.path.exists(self._config.images_dir):
            raise FileNotFoundError(f"Image directory not found: {self._config.images_dir}")
        
        if self._config.masks_dir is not None:
            if not os.path.exists(self._config.masks_dir):
                raise FileNotFoundError(f"Mask directory not found: {self._config.masks_dir}")
            
    @override
    def load_data(self): 
        self._colmap_dataset = SplatColmapDataset(self._config)

        stride = int(self._config.train_test_ratio * 10)
        self._train_dataset = self._colmap_dataset.split(lambda i, _: i % stride != 0)
        self._test_dataset = self._colmap_dataset.split(lambda i, _: i % stride == 0)

        return self._colmap_dataset.scene_scale

    @override
    def on_setup(
        self,
        render_payload_T: type,
        data_item_T: type,
        modules: Sequence[SplatBaseModule[SplatRenderPayloadT]], 
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ): 
       

        self._train_data_loader = DataLoader[ColmapDataItem](
            dataset=self._train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=self._config.num_workers,
            collate_fn=ColmapDataItem.from_batch,
        )

        self._test_data_loader = DataLoader[ColmapDataItem](
            dataset=self._test_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=ColmapDataItem.from_batch,
        )
    
    def get_test_data_size(self, world_rank: int = 0, world_size: int = 1) -> int:
        return len(self._test_dataset)
    
    def get_train_data_size(self, world_rank: int = 0, world_size: int = 1) -> int:
        return len(self._train_dataset)

    def next_train_data(self, step: int, world_rank: int = 0, world_size: int = 1) -> ColmapDataItem:
        batch, self._train_iter = self._next_from_iter(
            self._train_data_loader,
            self._train_data_iter,
        )

        return batch

    @override
    def next_test_data(self, step: int, world_rank: int = 0, world_size: int = 1) -> ColmapDataItem:
        batch, self._test_data_iter = self._next_from_iter(
            self._test_data_loader, 
            self._test_data_iter
        )
        return batch

    @override
    def init_splat_model(
        self,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        world_rank: int = 0,
        world_size: int = 1,

        leader_rank: int = 0,
    ) -> SplatModel | None:

        if world_rank != leader_rank:
            return None

        points = self._colmap_dataset.points.astype(np.float32)
        colors = (self._colmap_dataset.points_rgb / 255.0).astype(np.float32)

        return SplatModel.from_points(
            points,
            colors,
            sh_degree,
            init_opacity,
            init_scale,
        )
    
    @property
    @override
    def scene_scale(self) -> float:
        return self._colmap_dataset.scene_scale

    def __getstate__(self):
        return {
            "config": self._config,
        }
    
    def __setstate__(self, state):
        self._config = state["config"]
        self.__post_init__()

    def _next_from_iter(
        self, 
        loader: DataLoader[ColmapDataItem] | None,
        it: Iterator[ColmapDataItem] | None = None,
    ) -> tuple[ColmapDataItem, Iterator[ColmapDataItem]]:
        """
        Get the next batch from the loader and iterator.
        """
        if loader is None:
            raise ValueError("Loader is not set")
        
        if it is None:
            it = iter(loader)
        
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        return batch, it