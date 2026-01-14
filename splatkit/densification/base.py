from typing import Generic
from typing_extensions import override
from abc import ABC, abstractmethod

import torch

from ..modules import SplatRenderPayloadT
from ..splat.training_state import SplatTrainingState
from ..data_provider import SplatDataItemT
from ..modules.base import SplatBaseModule

class SplatDensification(
    SplatBaseModule[SplatRenderPayloadT],
    Generic[SplatRenderPayloadT],
):
    """
    Abstract base class for densification modules.
    """

    @abstractmethod
    def densify(
        self, 
        step: int, 
        max_steps:  int, 
        rendered_frames: torch.Tensor, 
        target_frames: torch.Tensor, 
        training_state: SplatTrainingState, 
        rend_out: SplatRenderPayloadT, 
        masks: torch.Tensor | None = None, 
        world_rank: int = 0, 
        world_size: int = 1
    ):
        """
        Densify the splat model.
        This will be invoked right after loss.backward() and before optimizer.step().
        """
        pass