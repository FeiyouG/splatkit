import os
from typing import TYPE_CHECKING, Literal, Sequence
from typing_extensions import override

import torch

from ..splat.training_state import SplatTrainingState

from .base import SplatBaseModule
from .frame import SplatRenderPayload, SplatRenderPayloadT

if TYPE_CHECKING:
    from ..logger import SplatLogger

class SplatExporter(SplatBaseModule[SplatRenderPayload]):
    """
    Splat module for saving splat models and checkpoints.
    
    Example:
        exporter = SplatExporter(
            output_dir="results/output",
            save_splat=True,   # Default: True, saves to output_dir/splat/{step}.ply
            save_ckpt=False,   # Default: False, saves to output_dir/ckpt/{step}.ckpt
            export_steps=[7_000, 30_000],
        )
    """

    _output_dir: str
    _save_splat: bool
    _save_ckpt: bool
    _splat_format: Literal["ply"]
    _export_steps: list[int]
    
    def __init__(
        self, 
        output_dir: str,
        export_steps: list[int] = [],
        save_splat: bool = True,
        save_ckpt: bool = False,
        splat_format: Literal["ply"] = "ply",
    ):

        super().__init__()
        self._output_dir = output_dir
        self._save_splat = save_splat
        self._save_ckpt = save_ckpt
        self._splat_format = splat_format
        self._export_steps = export_steps
    
    @override
    def on_setup(
        self,
        logger: "SplatLogger",
        render_payload_T: type,
        data_item_T: type,
        renderer: SplatBaseModule[SplatRenderPayloadT],
        data_provider: SplatBaseModule[SplatRenderPayloadT],
        loss_fn: SplatBaseModule[SplatRenderPayloadT],
        densification: SplatBaseModule[SplatRenderPayloadT],
        modules: Sequence[SplatBaseModule[SplatRenderPayloadT]], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        if self._save_splat:
            splat_dir = os.path.join(self._output_dir, "splat")
            os.makedirs(splat_dir, exist_ok=True)
            logger.info(f"Splat output directory: {splat_dir}", module=self.module_name)

        if self._save_ckpt:
            ckpt_dir = os.path.join(self._output_dir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Checkpoint directory: {ckpt_dir}", module=self.module_name)

    @override
    def post_step(
        self,
        logger: "SplatLogger",
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

        # Save splat if enabled and step matches criteria
        if self._save_splat and step in self._export_steps:
            splat_model = training_state.to_splat_model()
            if splat_model is not None:
                if self._splat_format == "ply":
                    splat_dir = os.path.join(self._output_dir, "splat")
                    ply_path = os.path.join(splat_dir, f"{step}.ply")
                    splat_model.save_ply(ply_path)
                    logger.info(f"Saved splat model to {ply_path}", module=self.module_name)
                else: 
                    raise ValueError(f"Invalid splat format: {self._splat_format}")
            
        # Save checkpoint if enabled and step matches criteria
        if self._save_ckpt and step in self._export_steps:
            ckpt_dir = os.path.join(self._output_dir, "ckpt")
            ckpt_path = os.path.join(ckpt_dir, f"{step}.ckpt")
            training_state.save_ckpt(ckpt_path, step=step)
            logger.info(f"Saved checkpoint to {ckpt_path}", module=self.module_name)