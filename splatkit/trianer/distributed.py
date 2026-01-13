from typing import Generic

import torch
from gsplat.distributed import cli

from .trainer import SplatTrainer
from .config import SplatTrainerConfig
from ..renderer import SplatRenderer
from ..loss_fn import SplatLossFn
from ..data_provider import SplatDataProvider, SplatDataItemT
from ..modules import SplatBaseModule, SplatBaseFrameT

class SplatDistributedTrainer(Generic[SplatBaseFrameT, SplatDataItemT]):
    """
    Responsible only for distribution and process orchestration.
    """

    def __init__(
        self,
        renderer: SplatRenderer[SplatBaseFrameT],
        loss_fn: SplatLossFn[SplatBaseFrameT],
        train_data_provider: SplatDataProvider[SplatBaseFrameT, SplatDataItemT],
        test_data_provider: SplatDataProvider[SplatBaseFrameT, SplatDataItemT] | None = None,
        modules: list[SplatBaseModule[SplatBaseFrameT]] = [],
        config: SplatTrainerConfig = SplatTrainerConfig(),
    ):
        self._config = config
        self._renderer = renderer
        self._loss_fn = loss_fn
        self._train_data_provider = train_data_provider
        self._test_data_provider = test_data_provider
        self._modules = modules

    def run(self):
        """
        Entry point users call.
        """
        self._spawn_workers()

    def _should_spawn(self) -> bool:
        return (
            torch.distributed.is_available()
            and not torch.distributed.is_initialized()
            and torch.cuda.device_count() > 1
        )

    def _run_single(self):
        trainer = SplatTrainer(
            config=self._config,
            renderer=self._renderer,
            loss_fn=self._loss_fn,
            train_data_provider=self._train_data_provider,
            test_data_provider=self._test_data_provider,
            modules=self._modules,
            world_rank=0,
            world_size=1,
            local_rank=torch.cuda.current_device()
            if torch.cuda.is_available()
            else None,
        )
        trainer.train()

    def _spawn_workers(self):
        args = (
            self._config,
            self._renderer,
            self._loss_fn,
            self._train_data_provider,
            self._test_data_provider,
            self._modules,
        )
        cli(_splat_worker_entry, args=args, verbose=True)


def _splat_worker_entry(
    local_rank: int,
    world_rank: int,
    world_size: int,
    args: tuple,
):
    (
        config,
        renderer,
        loss_fn,
        train_data_provider,
        test_data_provider,
        modules,
    ) = args

    trainer = SplatTrainer(
        config=config,
        renderer=renderer,
        loss_fn=loss_fn,
        train_data_provider=train_data_provider,
        test_data_provider=test_data_provider,
        modules=modules,
        world_rank=world_rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    trainer.run()
