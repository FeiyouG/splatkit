import os
from typing import Literal, Sequence
from typing_extensions import override

import torch

from ..splat.training_state import SplatTrainingState

from .base import SplatBaseModule
from .frame import SplatRenderPayload, SplatRenderPayloadT

class SplatExporter(SplatBaseModule[SplatRenderPayload]):
    """
    Splat module for saving splat models and checkpoints.
    """

    _splat_dir: str | None
    _splat_format: Literal["ply"]
    _splat_save_on: list[int]

    _ckpt_dir: str | None
    _ckpt_save_on: list[int]
    
    def __init__(
        self, 

        splat_dir: str | None = None,
        splat_save_on: list[int] = [],
        splat_format: Literal["ply"] = "ply",
        ckpt_dir: str | None = None,
        ckpt_save_on: list[int] = [],
    ):
        super().__init__()
        self._splat_dir = splat_dir
        self._splat_format = splat_format
        self._splat_save_on = splat_save_on

        self._ckpt_dir = ckpt_dir
        self._ckpt_save_on = ckpt_save_on
    
    @override
    def on_setup(
        self,
        render_payload_T: type,
        data_item_T: type,
        modules: Sequence[SplatBaseModule[SplatRenderPayloadT]], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        if self._splat_dir is not None:
            os.makedirs(self._splat_dir, exist_ok=True)

        if self._ckpt_dir is not None:
            os.makedirs(self._ckpt_dir, exist_ok=True)

    @override
    def post_step(
        self, 
        step: int, 
        max_steps: int, 
        rendered_frames: torch.Tensor, # (..., H, W, 3)
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState, 
        rend_out: SplatRenderPayload,
        masks: torch.Tensor | None = None, 
        world_rank: int = 0, 
        world_size: int = 1
    ):

        if (
            self._splat_dir is not None
            and (
                step in self._splat_save_on
                or step == max_steps - 1
            )
        ):
            splat_model = training_state.to_splat_model()
            if splat_model is not None:
                if self._splat_format == "ply":
                    splat_model.save_ply(os.path.join(self._splat_dir, f"{step}.ply"))
                    print(f"Saved splat model to {os.path.join(self._splat_dir, f'{step}.ply')}")
                else: 
                    raise ValueError(f"Invalid splat format: {self._splat_format}")
            
        if (
            self._ckpt_dir is not None
            and (
                step in self._ckpt_save_on
                or step == max_steps - 1
            )
        ):
            training_state.save_ckpt(os.path.join(self._ckpt_dir, f"{step}.ckpt"))
            print(f"Saved checkpoint to {os.path.join(self._ckpt_dir, f'{step}.ckpt')}")