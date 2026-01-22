from typing import TYPE_CHECKING, Dict, Any, Sequence

from gsplat.strategy import DefaultStrategy
from typing_extensions import override
import torch

from ..renderer import Splat3dgsRenderPayload

from ..modules import SplatBaseModule
from ..splat.training_state import SplatTrainingState
from ..data_provider import SplatDataItemT
from .base import SplatDensification

if TYPE_CHECKING:
    from ..logger import SplatLogger

class SplatDefaultDensification(
    SplatDensification[Splat3dgsRenderPayload]
):
    """
    Default densification module.
    """

    _default_strategy: DefaultStrategy
    _state: Dict[str, Any]

    @override
    def on_setup(self,
        logger: "SplatLogger",
        renderer: SplatBaseModule[Splat3dgsRenderPayload],
        data_provider: SplatBaseModule[Splat3dgsRenderPayload],
        loss_fn: SplatBaseModule[Splat3dgsRenderPayload],
        densification: SplatBaseModule[Splat3dgsRenderPayload],
        modules: Sequence[SplatBaseModule[Splat3dgsRenderPayload]], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        """
        Setup hook.
        """
        self._default_strategy = DefaultStrategy()
        self._state = self._default_strategy.initialize_state(scene_scale)
        logger.info("Initialized default densification strategy", module=self.module_name)

    @override
    def pre_compute_loss(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor, # (..., H, W, 3)
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: Splat3dgsRenderPayload,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Hook invoked before a loss has been computed.
        """

        self._default_strategy.step_pre_backward(
            params=training_state.params,
            optimizers=training_state.optimizers,
            state=self._state,
            step=step,
            info=rend_out.to_dict(),
        )

    @override
    def densify(
        self, 
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor, # (..., H, W, 3)
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: Splat3dgsRenderPayload,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Hook invoked after a loss has been computed.
        """
        info = rend_out.to_dict()
        info["gaussian_ids"] = None # So default strategy won't crash

        self._default_strategy.step_post_backward(
            params=training_state.params,
            optimizers=training_state.optimizers,
            state=self._state,
            step=step,
            info=info
        )