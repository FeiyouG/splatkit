from typing import Generic, TypeVar

import torch

from .frame import SplatBaseFrameT
from ..splat.training_state import SplatTrainingState

class SplatBaseModule(Generic[SplatBaseFrameT]):
    """
    Abstract base class for all Splat modules.
    """

    def on_setup(
        self,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Setup hook.

        This hook will be called before training loop starts on the each rank.
        Use this to initialize the module.

        Args:
            world_rank: The current world rank.
            world_size: The current world size.
        """
        pass

    def post_render(
        self, 
        renders: torch.Tensor, # (..., H, W, 3)
        targets: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: SplatBaseFrameT,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Post render hook.
        """
        pass