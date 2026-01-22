from typing import TYPE_CHECKING, Any, Generic, Sequence
from typing_extensions import override

import torch

from .base import SplatBaseModule
from .frame import SplatRenderPayloadT
from ..splat.training_state import SplatTrainingState

if TYPE_CHECKING:
    from ..logger import SplatLogger
    from ..renderer.base import SplatRenderer
    from ..data_provider.base import SplatDataProvider
    from ..loss_fn.base import SplatLossFn
    from ..densification.base import SplatDensification


class SplatModuleComposite(SplatBaseModule[SplatRenderPayloadT], Generic[SplatRenderPayloadT]):
    """
    A composite Splat module that forwards hook calls to a list of modules
    in order.
    """

    def __init__(self, *modules: SplatBaseModule[SplatRenderPayloadT]):
        super().__init__()
        self._modules = list(modules)

    @override
    def on_setup(
        self,
        logger: "SplatLogger",
        renderer: "SplatRenderer[SplatRenderPayloadT]",
        data_provider: "SplatDataProvider[SplatRenderPayloadT, Any]",
        loss_fn: "SplatLossFn[SplatRenderPayloadT]",
        densification: "SplatDensification[SplatRenderPayloadT]",
        modules: Sequence["SplatBaseModule[SplatRenderPayloadT]"], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        for m in self._modules:
            m.on_setup(
                logger=logger,
                renderer=renderer,
                data_provider=data_provider,
                loss_fn=loss_fn,
                densification=densification,
                modules=modules,
                max_steps=max_steps,
                world_rank=world_rank,
                world_size=world_size,
                scene_scale=scene_scale,
            )

    @override
    def pre_step(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        target_frames: torch.Tensor,
        training_state: SplatTrainingState,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        for m in self._modules:
            m.pre_step(
                logger=logger,
                step=step,
                max_steps=max_steps,
                target_frames=target_frames,
                training_state=training_state,
                masks=masks,
                world_rank=world_rank,
                world_size=world_size,
            )

    @override
    def pre_compute_loss(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor,
        target_frames: torch.Tensor,
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayloadT,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        for m in self._modules:
            m.pre_compute_loss(
                logger=logger,
                step=step,
                max_steps=max_steps,
                rendered_frames=rendered_frames,
                target_frames=target_frames,
                training_state=training_state,
                rend_out=rend_out,
                masks=masks,
                world_rank=world_rank,
                world_size=world_size,
            )
    
    @override
    def post_compute_loss(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        loss: torch.Tensor,
        training_state: SplatTrainingState,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        for m in self._modules:
            m.post_compute_loss(
                logger=logger,
                step=step,
                max_steps=max_steps,
                loss=loss,
                training_state=training_state,
                masks=masks,
                world_rank=world_rank,
                world_size=world_size,
            )
    
    @override
    def on_optimize(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        training_state: SplatTrainingState,
    ):
        for m in self._modules:
            m.on_optimize(
                logger=logger,
                step=step,
                max_steps=max_steps,
                training_state=training_state,
            )

    @override
    def post_step(
        self,
        logger: "SplatLogger",
        step: int, 
        max_steps: int, 
        rendered_frames: torch.Tensor,
        target_frames: torch.Tensor,
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayloadT,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        for m in self._modules:
            m.post_step(
                logger=logger,
                step=step,
                max_steps=max_steps,
                rendered_frames=rendered_frames,
                target_frames=target_frames,
                training_state=training_state,
                rend_out=rend_out,
                masks=masks,
                world_rank=world_rank,
                world_size=world_size,
            )
    
    @override
    def on_cleanup(
        self,
        logger: "SplatLogger",
        world_rank: int = 0,
        world_size: int = 1,
    ):
        for m in self._modules:
            m.on_cleanup(
                logger=logger,
                world_rank=world_rank,
                world_size=world_size,
            )
        
    def __len__(self) -> int:
        return len(self._modules)