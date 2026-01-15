from typing import Sequence
from typing_extensions import override

import torch
from tqdm import tqdm

from ..splat.training_state import SplatTrainingState

from .base import SplatBaseModule
from .frame import SplatRenderPayload, SplatRenderPayloadT


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
        render_payload_T: type,
        data_item_T: type,
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
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
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
        """Initialize progress bar at first step."""
        if world_rank != 0:
            return

        if self._pbar is not None:
            self._pbar.set_description("Initializing...")
    
    @override
    def post_step(
        self, 
        step: int, 
        max_steps: int, 
        rendered_frames: torch.Tensor,
        target_frames: torch.Tensor,
        training_state: SplatTrainingState, 
        rend_out: SplatRenderPayload,
        masks: torch.Tensor | None = None,
        world_rank: int = 0, 
        world_size: int = 1
    ):
        """Update progress bar after each training step."""

        if world_rank != 0:
            return
        
        if self._pbar is not None and step % self._update_every == 0:
            self._pbar.update(self._update_every)
            
            # Build description with available metrics
            desc_parts = []
            
            if self._last_loss is not None:
                desc_parts.append(f"loss={self._last_loss:.4f}")
            
            # Add number of Gaussians
            num_gs = training_state.num_gaussians
            desc_parts.append(f"#GS={num_gs:,}")
            
            # Add any custom metrics
            for key, value in self._last_metrics.items():
                if isinstance(value, float):
                    desc_parts.append(f"{key}={value:.4f}")
                else:
                    desc_parts.append(f"{key}={value}")
            
            desc = " | ".join(desc_parts)
            self._pbar.set_description(desc)
    
    @override
    def post_compute_loss(
        self, 
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
    def on_cleanup(self, world_rank: int = 0, world_size: int = 1):
        """Close progress bar after training."""
        if self._world_rank != 0:
            return
        
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
            print("\n✓ Training completed!")

