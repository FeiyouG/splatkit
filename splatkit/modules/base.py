from typing import TYPE_CHECKING, Any, Generic, Sequence, get_args, get_origin
from abc import ABC

import torch

from .frame import SplatRenderPayloadT
from ..splat.training_state import SplatTrainingState
from ..utils.generics import extract_subclass_generics, extrace_instance_generics

if TYPE_CHECKING:
    from ..logger import SplatLogger
    from ..renderer.base import SplatRenderer
    from ..data_provider.base import SplatDataProvider
    from ..loss_fn.base import SplatLossFn
    from ..densification.base import SplatDensification

class SplatBaseModule(Generic[SplatRenderPayloadT], ABC):
    """
    Abstract base class for all Splat modules.
    """
    
    @property
    def module_name(self) -> str:
        """
        Return the name of this module for logging.
        Override in subclasses to provide custom names.
        """
        return self.__class__.__name__

    def on_setup(
        self,
        logger: "SplatLogger",
        render_payload_T: type,
        data_item_T: type,
        renderer: "SplatRenderer[SplatRenderPayloadT]",
        data_provider: "SplatDataProvider[SplatRenderPayloadT, Any]",
        loss_fn: "SplatLossFn[SplatRenderPayloadT]",
        densification: "SplatDensification[SplatRenderPayloadT]",
        modules: Sequence["SplatBaseModule[SplatRenderPayloadT]"], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        """
        Setup hook.

        This hook will be called before training loop starts on the each rank.
        Use this to initialize the module.

        Args:
            logger: Logger instance for structured logging
            render_payload_T: The concrete type for render payloads
            data_item_T: The concrete type for data items
            renderer: The renderer instance
            data_provider: The data provider instance
            loss_fn: The loss function instance
            densification: The densification strategy instance
            modules: List of all modules (including this one)
            max_steps: Maximum number of training steps
            world_rank: The current world rank
            world_size: The current world size
            scene_scale: The scene scale factor
        """
        pass
    
    def pre_step(
        self,
        logger: "SplatLogger",
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
        logger: "SplatLogger",
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
    
    def post_compute_loss(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        loss: torch.Tensor,
        training_state: SplatTrainingState,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Hook invoked before a loss has been backwarded.
        """
        pass

    def on_optimize(
        self,
        logger: "SplatLogger",
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
        logger: "SplatLogger",
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
    
    def on_cleanup(
        self,
        logger: "SplatLogger",
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Hook invoked after the training loop has finished.
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