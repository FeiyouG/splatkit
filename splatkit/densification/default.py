from typing import TYPE_CHECKING, Dict, Any, Sequence

from gsplat.strategy import DefaultStrategy
from typing_extensions import override
import torch

from ..modules import SplatBaseModule, SplatRenderPayload
from ..splat.training_state import SplatTrainingState
from ..data_provider import SplatDataItemT
from .base import SplatDensification

if TYPE_CHECKING:
    from ..logger import SplatLogger

class SplatDefaultDensification(
    SplatDensification[SplatRenderPayload]
):
    """
    Default densification module

    Built on top of gsplat's DefaultStrategy.
    
    Automatically detects the correct gradient key from render output:
        - "gradient_2dgs" for 2DGS
        - "means2d" as fallback for 3DGS and 2DGS-inria
    """

    _default_strategy: DefaultStrategy
    _state: Dict[str, Any]
    _prune_opa: float
    _grow_grad2d: float
    _grow_scale3d: float
    _prune_scale3d: float
    _refine_start_iter: int
    _refine_stop_iter: int
    _reset_every: int
    _refine_every: int
    _absgrad: bool
    _revised_opacity: bool
    _detected_gradient_key: str | None = None

    def __init__(
        self,
        prune_opa: float = 0.005,
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        prune_scale3d: float = 0.1,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 15000,
        reset_every: int = 3000,
        refine_every: int = 100,
        absgrad: bool = False,
        revised_opacity: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize default densification strategy.
        
        Args:
            prune_opa: GSs with opacity below this value will be pruned
            grow_grad2d: GSs with image plane gradient above this value will be split/duplicated
            grow_scale3d: GSs with scale below this value will be duplicated. Above will be split
            prune_scale3d: GSs with scale above this value will be pruned
            refine_start_iter: Start refining GSs after this iteration
            refine_stop_iter: Stop refining GSs after this iteration
            reset_every: Reset opacities every this steps
            refine_every: Refine GSs every this steps
            absgrad: Use absolute gradient for pruning
            revised_opacity: Use revised opacity heuristic from arXiv:2404.06109
            verbose: Print densification logs (default: False)
        """
        self._prune_opa = prune_opa
        self._grow_grad2d = grow_grad2d
        self._grow_scale3d = grow_scale3d
        self._prune_scale3d = prune_scale3d
        self._refine_start_iter = refine_start_iter
        self._refine_stop_iter = refine_stop_iter
        self._reset_every = reset_every
        self._refine_every = refine_every
        self._absgrad = absgrad
        self._revised_opacity = revised_opacity
        self._verbose = verbose

    @override
    def on_setup(self,
        logger: "SplatLogger",
        renderer: SplatBaseModule[SplatRenderPayload],
        data_provider: SplatBaseModule[SplatRenderPayload],
        loss_fn: SplatBaseModule[SplatRenderPayload],
        densification: SplatBaseModule[SplatRenderPayload],
        modules: Sequence[SplatBaseModule[SplatRenderPayload]], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        """
        Setup hook. Gradient key will be auto-detected on first render.
        """
        # Don't set key_for_gradient yet - will be detected automatically
        self._default_strategy = DefaultStrategy(
            verbose=self._verbose,
            prune_opa=self._prune_opa,
            grow_grad2d=self._grow_grad2d,
            grow_scale3d=self._grow_scale3d,
            prune_scale3d=self._prune_scale3d,
            refine_start_iter=self._refine_start_iter,
            refine_stop_iter=self._refine_stop_iter,
            reset_every=self._reset_every,
            refine_every=self._refine_every,
            absgrad=self._absgrad,
            revised_opacity=self._revised_opacity,
        )
        self._state = self._default_strategy.initialize_state(scene_scale)
        logger.info(
            f"Initialized default densification strategy (gradient key will be auto-detected)",
            module=self.module_name
        )

    def _detect_gradient_key(self, info: Dict[str, Any], logger: "SplatLogger") -> str:
        """
        Auto-detect the gradient key from render output.
        
        Priority order:
        1. "gradient_2dgs" - for 2DGS
        2. "means2d" - fallback for 3DGS
        """
        if self._detected_gradient_key is not None:
            return self._detected_gradient_key
        
        # Try gradient_2dgs first (2DGS)
        if "gradient_2dgs" in info and info["gradient_2dgs"] is not None:
            self._detected_gradient_key = "gradient_2dgs"
            logger.info(
                f"Auto-detected gradient key: 'gradient_2dgs' (2DGS mode)",
                module=self.module_name
            )
        # Fallback to means2d (3DGS or 2DGS-inria)
        elif "means2d" in info:
            self._detected_gradient_key = "means2d"
            logger.info(
                f"Auto-detected gradient key: 'means2d' (3DGS/2DGS-inria mode)",
                module=self.module_name
            )
        else:
            # Default to means2d if nothing found
            self._detected_gradient_key = "means2d"
            logger.warning(
                f"Could not detect gradient key, defaulting to 'means2d'",
                module=self.module_name
            )
        
        # Update the strategy with the detected key
        self._default_strategy.key_for_gradient = self._detected_gradient_key
        return self._detected_gradient_key

    @override
    def pre_compute_loss(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor, # (..., H, W, 3)
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayload,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Hook invoked before a loss has been computed.
        """
        info = rend_out.to_dict()
        
        # Auto-detect gradient key on first call
        self._detect_gradient_key(info, logger)

        self._default_strategy.step_pre_backward(
            params=training_state.params,
            optimizers=training_state.optimizers,
            state=self._state,
            step=step,
            info=info,
        )

    @override
    def densify(
        self, 
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor, # (..., H, W, 3)
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayload,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):
        info = rend_out.to_dict()

        self._default_strategy.step_post_backward(
            params=training_state.params,
            optimizers=training_state.optimizers,
            state=self._state,
            step=step,
            info=info
        )