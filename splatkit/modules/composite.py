from typing import Generic, Sequence
from typing_extensions import override

import torch

from .base import SplatBaseModule
from .frame import SplatRenderPayloadT
from ..splat.training_state import SplatTrainingState


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
        render_payload_T: type,
        data_item_T: type,
        modules: Sequence["SplatBaseModule[SplatRenderPayloadT]"], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        for m in self._modules:
            m.on_setup(
                render_payload_T=render_payload_T,
                data_item_T=data_item_T,
                modules=modules,
                max_steps=max_steps,
                world_rank=world_rank,
                world_size=world_size,
                scene_scale=scene_scale,
            )

    @override
    def pre_step(
        self,
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
        step: int,
        max_steps: int,
        training_state: SplatTrainingState,
    ):
        for m in self._modules:
            m.on_optimize(
                step=step,
                max_steps=max_steps,
                training_state=training_state,
            )

    @override
    def post_step(
        self, 
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
        world_rank: int = 0,
        world_size: int = 1,
    ):
        for m in self._modules:
            m.on_cleanup(
                world_rank=world_rank,
                world_size=world_size,
            )
        
    def __len__(self) -> int:
        return len(self._modules)