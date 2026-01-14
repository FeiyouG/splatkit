from typing import Generic, get_args

import torch
from torch import distributed as dist
from gsplat.distributed import cli

from .trainer import SplatTrainer
from .config import SplatTrainerConfig
from ..renderer import SplatRenderer
from ..loss_fn import SplatLossFn
from ..data_provider import SplatDataProvider, SplatDataItemT
from ..modules import SplatBaseModule, SplatRenderPayloadT
from ..densification import SplatDensification
from ..utils.generics import extrace_instance_generics

class SplatDistributedTrainer(Generic[SplatDataItemT, SplatRenderPayloadT]):
    """
    Responsible only for distribution and process orchestration.
    """

    def __init__(
        self,
        renderer: SplatRenderer[SplatRenderPayloadT],
        loss_fn: SplatLossFn[SplatRenderPayloadT],
        data_provider: SplatDataProvider[SplatRenderPayloadT, SplatDataItemT],
        densification: SplatDensification[SplatRenderPayloadT],
        modules: list[SplatBaseModule[SplatRenderPayloadT]] = [],
        config: SplatTrainerConfig = SplatTrainerConfig(),
    ):
        self._config = config
        self._renderer = renderer
        self._loss_fn = loss_fn
        self._data_provider = data_provider
        self._densification = densification
        self._modules = modules

    def run(self):
        """
        Entry point users call.
        """
        self._spawn_workers()

    def _should_spawn(self) -> bool:
        return (
            dist.is_available()
            and not dist.is_initialized()
            and torch.cuda.device_count() > 1
        )

    def _run_single(self):
        trainer = SplatTrainer[SplatDataItemT, SplatRenderPayloadT](
            config=self._config,
            renderer=self._renderer,
            loss_fn=self._loss_fn,
            data_provider=self._data_provider,
            densification=self._densification,
            modules=self._modules,
            world_rank=0,
            world_size=1,
            local_rank=0
        )
        trainer.run()

    def _spawn_workers(self):
        data_t, payload_t = extrace_instance_generics(self)

        args = (
            data_t,
            payload_t,
            self._config,
            self._renderer,
            self._loss_fn,
            self._data_provider,
            self._densification,
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
        data_t,
        payload_t,
        config,
        renderer,
        loss_fn,
        data_provider,
        densification,
        modules,
    ) = args

    trainer = SplatTrainer[data_t, payload_t](
        config=config,
        renderer=renderer,
        loss_fn=loss_fn,
        data_provider=data_provider,
        densification=densification,
        modules=modules,
        world_rank=world_rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    trainer.run()