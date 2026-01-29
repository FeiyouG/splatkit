from typing import TYPE_CHECKING, Dict, Any, Sequence

from gsplat.strategy import MCMCStrategy
from typing_extensions import override
import torch

from ..modules import SplatBaseModule, SplatRenderPayload
from ..splat.training_state import SplatTrainingState
from ..data_provider import SplatDataItemT
from .base import SplatDensification

if TYPE_CHECKING:
    from ..logger import SplatLogger

class SplatMCMCDensification(
    SplatDensification[SplatRenderPayload]
):
    """
    MCMC-based densification module.

    Built on top of gsplat's MCMCStrategy, following the paper:
    "3D Gaussian Splatting as Markov Chain Monte Carlo"
    
    The strategy will:
        - Periodically teleport GSs with low opacity to regions with high opacity
        - Periodically add new GSs sampled based on opacity distribution
        - Periodically perturb GS locations with noise
    """

    _mcmc_strategy: MCMCStrategy
    _state: Dict[str, Any]
    _cap_max: int
    _noise_lr: float
    _refine_start_iter: int
    _refine_stop_iter: int
    _refine_every: int
    _min_opacity: float

    def __init__(
        self,
        cap_max: int = 1_000_000,
        noise_lr: float = 5e5,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 25_000,
        refine_every: int = 100,
        min_opacity: float = 0.005,
        verbose: bool = False,
    ):
        """
        Initialize MCMC densification strategy.
        
        Args:
            cap_max: Maximum number of GSs
            noise_lr: MCMC sampling noise learning rate
            refine_start_iter: Start refining GSs after this iteration
            refine_stop_iter: Stop refining GSs after this iteration
            refine_every: Refine GSs every this steps
            min_opacity: GSs with opacity below this value will be relocated
            verbose: Print densification logs (default: False)
        """
        self._cap_max = cap_max
        self._noise_lr = noise_lr
        self._refine_start_iter = refine_start_iter
        self._refine_stop_iter = refine_stop_iter
        self._refine_every = refine_every
        self._min_opacity = min_opacity
        self._verbose = verbose
    
    @property
    def state(self) -> Dict[str, Any]:
        return self._state

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
        Setup hook. Initializes MCMC strategy state.
        """
        self._mcmc_strategy = MCMCStrategy(
            cap_max=self._cap_max,
            noise_lr=self._noise_lr,
            refine_start_iter=self._refine_start_iter,
            refine_stop_iter=self._refine_stop_iter,
            refine_every=self._refine_every,
            min_opacity=self._min_opacity,
            verbose=self._verbose,
        )
        self._state = self._mcmc_strategy.initialize_state()
        logger.info(
            f"Initialized MCMC densification strategy with cap_max={self._cap_max}",
            module=self.module_name
        )

    @override
    def densify(
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
        info = rend_out.to_dict()
        
        # Get learning rate for means from optimizer
        means_optimizer = training_state.optimizers.get("means")
        if means_optimizer is None:
            raise ValueError("MCMC strategy requires 'means' optimizer")
        
        # Extract learning rate from optimizer's param_groups
        lr = means_optimizer.param_groups[0]["lr"]
        
        self._mcmc_strategy.step_post_backward(
            params=training_state.params,
            optimizers=training_state.optimizers,
            state=self._state,
            step=step,
            info=info,
            lr=lr,
        )

        # In gsplat's MCMCStrategy, the gaussians were never pruned.
        return None