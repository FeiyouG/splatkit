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
    Module for exporting splat models and checkpoints during training.
    
    Example:
        >>> from splatkit.modules import SplatExporter
        >>> exporter = SplatExporter(
        ...     output_dir="results/output",
        ...     export_steps=[7_000, 15_000, 30_000],
        ...     save_splat=True,  # Save .ply files
        ...     save_ckpt=True,   # Save training checkpoints
        ... )
        >>> # Add to trainer's modules list
        >>> # Models saved to:
        >>> #   results/output/splats/step_007000.ply
        >>> #   results/output/ckpts/step_007000.ckpt
    
    NOTE:
        - Only rank 0 saves files in distributed training
        - Creates output_dir automatically if it doesn't exist
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
        save_ckpt: bool = True,
        splat_format: Literal["ply"] = "ply",
    ):
        """
        Initialize the exporter module.
        
        Args:
            output_dir: Directory to save exports (required)
            export_steps: Training steps at which to save (default: [])
            save_splat: Save splat model as .ply file (default: True)
            save_ckpt: Save full checkpoint for resuming training (default: True)
            splat_format: Export format for splat models (default: "ply")
        """
        super().__init__()
        self._output_dir = output_dir
        self._splat_dir = os.path.join(self._output_dir, "splats")
        self._ckpt_dir = os.path.join(self._output_dir, "ckpts")

        self._save_splat = save_splat
        self._save_ckpt = save_ckpt
        self._splat_format = splat_format
        self._export_steps = export_steps
    
    @override
    def on_setup(
        self,
        logger: "SplatLogger",
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
            os.makedirs(self._splat_dir, exist_ok=True)
            logger.info(f"Create splat output directory: {self._splat_dir}", module=self.module_name)

        if self._save_ckpt:
            os.makedirs(self._ckpt_dir, exist_ok=True)
            logger.info(f"Create checkpoint output directory: {self._ckpt_dir}", module=self.module_name)

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
        world_size: int = 1,
    ):

        # Save splat if enabled and step matches criteria
        if self._save_splat and step in self._export_steps:
            splat_model = training_state.to_splat_model()
            if splat_model is not None:
                if self._splat_format == "ply":
                    ply_path = os.path.join(self._splat_dir, f"{step}.ply")
                    splat_model.save_ply(ply_path)
                    logger.info(f"Saved splat model to {ply_path}", module=self.module_name)
                else: 
                    raise ValueError(f"Invalid splat format: {self._splat_format}")
            
        # Save checkpoint if enabled and step matches criteria
        if self._save_ckpt and step in self._export_steps:
            ckpt_path = os.path.join(self._ckpt_dir, f"{step}.ckpt")
            training_state.save_ckpt(ckpt_path, step=step)
            logger.info(f"Saved checkpoint to {ckpt_path}", module=self.module_name)