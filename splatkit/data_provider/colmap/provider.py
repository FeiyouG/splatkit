import os
from typing import Generic

import numpy as np
import torch

from .config import SplatColmapDataProviderConfig
from .dataset import SplatColmapDataset
from ..base import SplatDataProvider, SplatDataItem
from ...splat import SplatModel
from ...modules import SplatBaseFrame


class SplatColmapDataProvider(
    SplatDataProvider[SplatBaseFrame, SplatDataItem],
):
    """
    Data provider for COLMAP datasets.
    """

    _config: SplatColmapDataProviderConfig

    _colmap_dataset: SplatColmapDataset
    _train_dataset: SplatColmapDataset
    _test_dataset: SplatColmapDataset

    _train_data_loader: torch.utils.data.DataLoader
    _test_data_loader: torch.utils.data.DataLoader

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

    def on_setup(
        self,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        self._colmap_dataset = SplatColmapDataset(self._config)

        stride = int(1 / self._config.train_test_ratio)
        self._train_dataset = self._colmap_dataset.split(lambda i, _: i % stride != 0)
        self._test_dataset = self._colmap_dataset.split(lambda i, _: i % stride == 0)

        self._train_data_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=self._config.num_workers,
        )

        self._test_data_loader = torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
        )
    
    def get_test_data_size(self, world_rank: int = 0, world_size: int = 1) -> int:
        return len(self._test_dataset)
    
    def get_train_data_size(self, world_rank: int = 0, world_size: int = 1) -> int:
        return len(self._train_dataset)

    def next_train_data(self, step: int, world_rank: int = 0, world_size: int = 1) -> SplatDataItem:
        batch, self._train_iter = self._next_from_iter(
            self._train_data_loader,
            self._train_data_iter,
        )

        return batch

    def next_test_data(self, step: int, world_rank: int = 0, world_size: int = 1) -> SplatDataItem:
        batch, self._test_data_iter = self._next_from_iter(
            self._test_data_loader,
            self._test_data_iter,
        )
        return batch, self._test_data_iter

    def init_splat_model(
        self,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
    ) -> SplatModel:

        points = self._colmap_dataset.points.astype(np.float32)
        colors = (self._colmap_dataset.points_rgb / 255.0).astype(np.float32)

        return SplatModel.from_points(
            points,
            colors,
            sh_degree,
            init_opacity,
            init_scale,
        )

    def __getstate__(self):
        return {
            "config": self._config,
        }
    
    def __setstate__(self, state):
        self._config = state["config"]
        self.__post_init__()

    def _next_from_iter(self, loader, it):
        """
        Get the next batch from the loader and iterator.
        """
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        return batch, it
