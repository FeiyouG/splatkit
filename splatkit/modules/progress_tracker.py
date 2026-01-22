from typing import TYPE_CHECKING, Sequence
from typing_extensions import override

import torch
from tqdm import tqdm

from ..splat.training_state import SplatTrainingState

from .base import SplatBaseModule
from .frame import SplatRenderPayload, SplatRenderPayloadT

if TYPE_CHECKING:
    from ..logger import SplatLogger


class SplatProgressTracker(SplatBaseModule[SplatRenderPayload]):
    """
    Progress tracking module for training visualization.
    
    Features:
    - Real-time progress bar with tqdm
    - Loss and metrics display
    - Step timing information
    """

    _update_every: int
    _pbar: "tqdm | None"
    _world_rank: int
    
    _last_loss: float | None
    _last_metrics: dict[str, float | int | str]
    
    def __init__(self, update_every: int = 1):
        """
        Initialize progress tracker.
        
        Args:
            update_every: Update progress bar every N steps (default: 1)
        """
        super().__init__()
        self._update_every = update_every
        self._pbar = None
        self._last_loss = None
        self._last_metrics = {}
    
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
        """Setup progress tracker."""
        if world_rank != 0:
            return
        
        self._pbar = tqdm(
            total=max_steps,
            desc="Training",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]'
        )
        logger.info(f"Initialized progress tracker for {max_steps} steps", module=self.module_name)
    
    @override
    def post_step(
        self,
        logger: "SplatLogger",
        step: int, 
        max_steps: int, 
        rendered_frames: torch.Tensor,
        target_frames: torch.Tensor,
        training_state: SplatTrainingState, 
        rend_out: SplatRenderPayload,
        masks: torch.Tensor | None = None,
        world_rank: int = 0, 
        world_size: int = 1,
    ):
        """Update progress bar after each training step."""

        if world_rank != 0:
            return
        
        if self._pbar is not None and step % self._update_every == 0:
            self._pbar.update(self._update_every)

            postfix: dict[str, float | int | str] = {
                "#GS": training_state.num_gaussians,
            }

            if self._last_loss is not None:
                postfix["loss"] = f"{self._last_loss:.4f}"

            for k, v in self._last_metrics.items():
                postfix[k] = f"{v:.4f}" if isinstance(v, float) else v

            self._pbar.set_postfix(postfix, refresh=False)
    
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
        world_size: int = 1
    ):
        """Hook before loss backward - can access loss from training_state if needed."""
        if world_rank != 0:
            return
        self._last_loss = loss.item()

    @override
    def on_cleanup(self, logger: "SplatLogger", world_rank: int = 0, world_size: int = 1):  # type: ignore
        """Close progress bar after training."""
        if world_rank != 0:
            return
        
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
            logger.info("Training completed!", module=self.module_name)