from typing import Generic, Sequence, get_args, get_origin
from abc import ABC

import torch

from .frame import SplatRenderPayloadT
from ..splat.training_state import SplatTrainingState
from ..utils.generics import extract_subclass_generics, extrace_instance_generics

class SplatBaseModule(Generic[SplatRenderPayloadT], ABC):
    """
    Abstract base class for all Splat modules.
    """

    def on_setup(
        self,
        render_payload_T: type,
        data_item_T: type,
        modules: Sequence["SplatBaseModule[SplatRenderPayloadT]"], 
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
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
    
    def pre_step(
        self,
        step: int,
        max_steps: int,
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ): 
        """
        Hook invoked before a step has been executed.
        """
        pass

    def pre_compute_loss(
        self, 
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor, # (..., H, W, 3)
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayloadT,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Hook invoked before a loss has been computed.
        """
        pass

    def on_optimize(
        self,
        step: int,
        max_steps: int,
        training_state: SplatTrainingState,
    ):
        """
        Hook invoked after an optimizer has been stepped.
        """
        pass



    def post_step(
        self, 
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor, # (..., H, W, 3)
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayloadT,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Hook invoked after a step has been executed.
        """
        pass

    @property
    def render_payload_T(self) -> type:
        """
        Return the concrete SplatRenderPayload type this module operates on.

        Works for both:
        1. Subclass specialization:
              class X(SplatBaseModule[Payload])
        2. Instance specialization:
              X[Payload]()
            
        Note:
           This property can only be called after __init__ is finished, 
           otherwise __orig_class__ is not available and the type cannot be determined 
           in case of instance specialization.
        """
        if not hasattr(self, '_render_payload_T'):
            self._render_payload_T = None

        if self._render_payload_T is not None:
            return self._render_payload_T

        cls = type(self)
        cls_name = cls.__name__
        try:
            (payload_t,) = extract_subclass_generics(cls, base=SplatBaseModule)
            self._render_payload_T = payload_t
            return payload_t
        except TypeError:
            pass

        try:
            (payload_t,) = extrace_instance_generics(self)
            self._render_payload_T = payload_t
            return payload_t
        except TypeError:
            pass

        raise TypeError(
            f"Cannot determine render payload type for {cls.__name__}.\n\n"
            f"Expected one of:\n"
            f"  - class X(SplatBaseModule[Payload])\n"
            f"  - X[Payload]()"
        )