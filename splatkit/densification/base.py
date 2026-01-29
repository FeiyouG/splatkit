from typing import TYPE_CHECKING, Any, Dict, Generic
from typing_extensions import override
from abc import ABC, abstractmethod

import torch

from ..modules import SplatRenderPayloadT
from ..splat.training_state import SplatTrainingState
from ..data_provider import SplatDataItemT
from ..modules.base import SplatBaseModule

if TYPE_CHECKING:
    from ..logger import SplatLogger

class SplatDensification(
    SplatBaseModule[SplatRenderPayloadT],
    Generic[SplatRenderPayloadT],
):
    """
    Base class for densification strategies.
    
    Densification adds and removes Gaussians during training to improve
    reconstruction quality. Strategies decide when and how to:
        - Clone Gaussians (duplicate in under-reconstructed areas)
        - Split Gaussians (divide large ones into smaller ones)
        - Prune Gaussians (remove transparent or invisible ones)
    
    Subclasses must implement:
        - state: Dict[str, Any] - state of the densification strategy
        - densify(): Perform densification operations
    
    Example:
        >>> from splatkit.densification import SplatDefaultDensification
        >>> densification = SplatDefaultDensification()
    
    NOTE:
        - Called after loss.backward() but before optimizer.step()
        - Modifies training_state in-place (adds/removes Gaussians)
        - Must work correctly in distributed training (all ranks)
    """
    @property
    def state(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement state property")

    @abstractmethod
    def densify(
        self,   
        logger: "SplatLogger",
        step: int, 
        max_steps:  int, 
        rendered_frames: torch.Tensor, 
        target_frames: torch.Tensor, 
        training_state: SplatTrainingState, 
        rend_out: SplatRenderPayloadT, 
        masks: torch.Tensor | None = None, 
        world_rank: int = 0, 
        world_size: int = 1
    ) -> torch.Tensor | None:
        """
        Perform densification: clone, split, and prune Gaussians.
        
        Called during training after gradients are computed but before
        optimizer updates parameters.
        
        Args:
            logger: Logger instance for structured logging
            step: Current training step
            max_steps: Total training steps
            rendered_frames: Rendered images from this step
            target_frames: Ground truth images
            training_state: Current Gaussian parameters (modified in-place)
            rend_out: Render payload with depths, alphas, radii, etc.
            masks: Optional masks for valid pixels
            world_rank: Current process rank
            world_size: Total number of processes
            
        Returns:
            Last prune mask, if any Gaussians were ever pruned.
        
        NOTE:
            - Modifies training_state.params and optimizers in-place
            - Must synchronize across all ranks in distributed training
        """
        pass