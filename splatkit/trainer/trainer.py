from typing import Generic, Sequence, get_args, get_origin

import torch

from ..modules import SplatModuleComposite

from ..data_provider import SplatDataItemT, SplatDataProvider
from ..densification import SplatDensification
from ..logger import SplatLogger
from ..loss_fn import SplatLossFn
from ..modules import SplatRenderPayloadT
from ..modules.base import SplatBaseModule
from ..renderer import SplatRenderer
from ..splat import SplatTrainingState
from .config import SplatTrainerConfig

class SplatTrainer(Generic[SplatDataItemT, SplatRenderPayloadT]):
    """
    Trainer for Gaussian Splatting.
    
    Coordinates rendering, loss computation, optimization, and densification.
    Supports distributed training and custom modules for logging/visualization.
    
    Example:
        >>> from splatkit.trainer import SplatTrainer, SplatTrainerConfig
        >>> from splatkit.renderer import Splat3DGSRenderer
        >>> from splatkit.loss_fn import Splat3DGSLossFn
        >>> from splatkit.densification import SplatDefaultDensification
        >>> 
        >>> config = SplatTrainerConfig(max_steps=30000)
        >>> trainer = SplatTrainer(
        ...     config=config,
        ...     renderer=Splat3DGSRenderer(),
        ...     loss_fn=Splat3DGSLossFn(),
        ...     data_provider=data_provider,
        ...     densification=SplatDefaultDensification(),
        ...     modules=[],  # Add viewer, progress bar, etc.
        ... )
        >>> trainer.run()
    """

    _config: SplatTrainerConfig
    _logger: SplatLogger

    # Distributed training variables
    _local_rank: int 
    _world_rank: int 
    _world_size: int
    _device: str

    def __init__(
        self,
        config: SplatTrainerConfig,
        renderer: SplatRenderer[SplatRenderPayloadT],
        loss_fn: SplatLossFn[SplatRenderPayloadT],
        data_provider: SplatDataProvider[SplatRenderPayloadT, SplatDataItemT],
        densification: SplatDensification[SplatRenderPayloadT],
        modules: Sequence[SplatBaseModule[SplatRenderPayloadT]] = [],
        logger: SplatLogger | None = None,
        ckpt_path: str | None = None,
        local_rank: int = 0,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize the trainer with all required components.
        
        Args:
            config: Training configuration (see SplatTrainerConfig)
            renderer: Renders Gaussians to images (e.g., Splat3DGSRenderer)
            loss_fn: Computes training loss (e.g., Splat3DGSLossFn)
            data_provider: Loads training images and cameras (e.g., SplatColmapDataProvider)
            densification: Strategy for adding/removing Gaussians (e.g., SplatDefaultDensification)
            modules: Optional list of modules for logging, visualization, etc.
                     Examples: SplatViewer, SplatProgressTracker, SplatTensorboard
            logger: Custom logger (creates default INFO-level logger if None)
            ckpt_path: Path to checkpoint file to resume training (optional)
            local_rank: GPU index on current machine (for multi-GPU, default: 0)
            world_rank: Global process rank across all machines (default: 0)
            world_size: Total number of processes in distributed training (default: 1)
        
        NOTE:
            - Components must be compatible (same render payload type)
            - For single GPU training, keep local_rank=0, world_rank=0, world_size=1
            - Modules' hooks are called in the same order as they appear in the list
        """
        self._config = config
        self._renderer = renderer
        self._loss_fn = loss_fn
        self._data_provider = data_provider
        self._densification = densification
        self._modules = modules
        self._logger = logger if logger else SplatLogger(level="INFO")
        self._ckpt_path = ckpt_path

        self._local_rank = local_rank
        self._world_rank = world_rank
        self._world_size = world_size
        self._device = f"cuda:{local_rank}"

        self.__post_init__()
        
    def __post_init__(self):
        """Post-initialization setup."""
        self._validate()
        # self._setup_distributed()
    
    def __class_getitem__(cls, generic_type):        
        """
        Class getitem hook to store the concrete generic types.
        """
        data_t, payload_t = generic_type
        cls._data_item_T = data_t
        cls._render_payload_T = payload_t

        new_cls = type(cls.__name__, cls.__bases__, dict(cls.__dict__))
        return new_cls
    
    def _validate(self):
        """Validate configuration."""
        if self._config is None:
            raise ValueError("Config is required")
        if self._data_provider is None:
            raise ValueError("Trainset is required")
        if self._renderer is None:
            raise ValueError("Renderer is required")
        if self._loss_fn is None:
            raise ValueError("Loss function is required")

        if self._world_rank == 0:
            if self._world_size == 1:
                self._logger.info("Starting single GPU training", module="SplatTrainer")
            elif self._world_size > 1:
                self._logger.info(f"Starting multi-GPU training: world_size={self._world_size}", module="SplatTrainer")
            
    @property
    def module_name(self) -> str:
        return "SplatTrainer"

    def run(self, leader_rank: int = 0):
        """
        Core training loop implementation.
        
        NOTE:
            - For distributed training, use SplatDistributedTrainer or manage with CLI manually
        """
        scene_scale = self._data_provider.load_data(logger=self._logger)

        all_modules = SplatModuleComposite[SplatRenderPayloadT](
            self._logger,
            self._renderer,
            self._loss_fn,
            self._data_provider,
            self._densification,
            *self._modules
        )

        # Setup modules
        all_modules.on_setup(
            logger=self._logger,
            renderer=self._renderer,
            data_provider=self._data_provider,
            loss_fn=self._loss_fn,
            densification=self._densification,
            modules=self._modules,
            max_steps=self._config.max_steps,
            world_rank=self._world_rank,
            world_size=self._world_size,
            scene_scale=scene_scale,
        )
        self._logger.info(f"Successfully set up all modules", module=self.module_name)

        # Load checkpoint or initialize from scratch
        if self._ckpt_path:
            splat_training_state, loaded_step = SplatTrainingState.from_ckpt(
                path=self._ckpt_path,
                world_rank=self._world_rank,
                world_size=self._world_size,
                leader_rank=leader_rank,
                scene_scale=scene_scale,
            )
            start_step = loaded_step + 1  # Resume from next step
            self._logger.info(f"Loaded checkpoint from step {loaded_step}, resuming from step {start_step}", module="SplatTrainer")
        else:
        
            # Initialize splat traininig state
            splat_model = self._data_provider.init_splat_model(
                leader_rank=leader_rank,
                sh_degree=self._config.sh_degree,
                init_opacity=self._config.init_opacity,
                init_scale=self._config.init_scale,
                world_rank=self._world_rank,
                world_size=self._world_size,
            )
            
            splat_training_state = SplatTrainingState.from_splat_model(
                model=splat_model,
                world_rank=self._world_rank,
                world_size=self._world_size,
                leader_rank=leader_rank,
                scene_scale=scene_scale,
            )
            del splat_model
            start_step = 1
            self._logger.info(f"Successfully initialized splat training state from data provider", module=self.module_name)

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                splat_training_state.optimizers['means'],
                gamma=0.01 ** (1.0 / self._config.max_steps)
            )
        ]
        
        # NOTE: step is 1-indexed
        self._logger.info(f"Starting training loop from step {start_step} to {self._config.max_steps}", module=self.module_name)
        for step in range(start_step, self._config.max_steps + 1):

            # STEP 1: Getting training data
            data = self._data_provider.next_train_data(step).to(self._device)

            target_frames = data.image
            height, width = target_frames.shape[1:3]
            masks = data.mask

            all_modules.pre_step(
                logger=self._logger,
                step=step,
                max_steps=self._config.max_steps,
                target_frames=target_frames,
                training_state=splat_training_state,
                masks=masks,
                world_rank=self._world_rank,
                world_size=self._world_size,
            )

            # STEP 2: Get rendered frames
            
            sh_degree_to_train = min(step // self._config.sh_degree_interval, self._config.sh_degree)
            renders, rend_out = self._renderer.render(
                splat_state=splat_training_state,
                cam_to_worlds=data["cam_to_world"],
                Ks=data["K"],
                width=width,
                height=height,
                sh_degree=sh_degree_to_train,
            )

            # STEP 3: Computing Loss
            all_modules.pre_compute_loss(
                logger=self._logger,
                step=step,
                max_steps=self._config.max_steps,
                rendered_frames=renders,
                target_frames=target_frames,
                training_state=splat_training_state,
                rend_out=rend_out,
            )

            # STEP 4: Backward propagate loss
            loss = self._loss_fn.compute_loss(
                logger=self._logger,
                renders=renders,
                targets=target_frames,
                training_state=splat_training_state,
                rend_out=rend_out,
                masks=masks,
            )

            all_modules.post_compute_loss(
                logger=self._logger,
                step=step,
                max_steps=self._config.max_steps,
                loss=loss,
                training_state=splat_training_state,
                masks=masks,
                world_rank=self._world_rank,
                world_size=self._world_size,
            )

            loss.backward()

            # Step 5: Step Optimizers
            for optimizer in splat_training_state.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()

            all_modules.on_optimize(
                logger=self._logger,
                step=step,
                max_steps=self._config.max_steps,
                training_state=splat_training_state,
            )

            # Step 6: Densification
            self._densification.densify(
                step=step,
                max_steps=self._config.max_steps,
                rendered_frames=renders,
                target_frames=target_frames,
                training_state=splat_training_state,
                rend_out=rend_out,
            )

            all_modules.post_step(
                logger=self._logger,
                step=step,
                max_steps=self._config.max_steps,
                rendered_frames=renders,
                target_frames=target_frames,
                training_state=splat_training_state,
                rend_out=rend_out,
            )
        
        self._logger.info(f"Training loop completed", module=self.module_name)
        # Cleanup resources
        all_modules.on_cleanup(
            logger=self._logger,
            world_rank=self._world_rank,
            world_size=self._world_size,
        )
        self._logger.info(f"Successfully cleaned up all modules", module=self.module_name)
