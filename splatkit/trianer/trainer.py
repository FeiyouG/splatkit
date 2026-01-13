from typing import Generic

import torch
import torch.distributed as dist
from gsplat.distributed import cli

from ..modules import SplatBaseFrameT
from ..data_provider import SplatDataProvider, SplatDataItemT
from ..modules.base import SplatBaseModule
from ..renderer import SplatRenderer
from ..loss_fn import SplatLossFn
from .config import SplatTrainerConfig


class SplatTrainer(Generic[SplatBaseFrameT, SplatDataItemT]):
    """
    Trainer for 3D Gaussian Splatting.
    
    Automatically detects whether it's running in:
    - Parent mode: Not in a distributed context → spawns workers via cli()
    - Worker mode: In a distributed context → runs training
    """

    _config: SplatTrainerConfig
    _train_data_provider: SplatDataProvider
    _test_data_provider: SplatDataProvider | None
    _modules: list[SplatBaseModule[SplatBaseFrameT]]

    # Distributed training variables
    _local_rank: int 
    _world_rank: int 
    _world_size: int
    
    def __init__(
        self,
        renderer: SplatRenderer[SplatBaseFrameT],
        loss_fn: SplatLossFn[SplatBaseFrameT],
        train_data_provider: SplatDataProvider[SplatBaseFrameT, SplatDataItemT],
        test_data_provider: SplatDataProvider[SplatBaseFrameT, SplatDataItemT] | None = None,
        modules: list[SplatBaseModule[SplatBaseFrameT]] = [],
        config: SplatTrainerConfig = SplatTrainerConfig(),
        local_rank: int = 0,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            trainset: Lazy wrapper for training dataset
            testset: Lazy wrapper for test dataset (optional)
            renderer: Lazy wrapper for renderer
            loss: Lazy wrapper for loss function
        """
        self._config = config
        self._renderer = renderer
        self._loss_fn = loss_fn
        self._train_data_provider = train_data_provider
        self._test_data_provider = test_data_provider
        self._modules = modules

        self._local_rank = local_rank
        self._world_rank = world_rank
        self._world_size = world_size

        self.__post_init__()
        
    def __post_init__(self):
        """Post-initialization setup."""
        self._validate()
        # self._setup_distributed()
    
    def _validate(self):
        """Validate configuration."""
        if self._config is None:
            raise ValueError("Config is required")
        if self._train_data_provider is None:
            raise ValueError("Trainset is required")
        if self._renderer is None:
            raise ValueError("Renderer is required")
        if self._loss_fn is None:
            raise ValueError("Loss function is required")

        if self._world_rank == 0:
            if self._world_size == 1:
                print(f"[SplatTrainer] Start Single GPU training")
            elif self._world_size > 1:
                print(f"[SplatTrainer] Start Multi-GPU training: world_size={self._world_size}")

        # vars_set = [self._world_rank is not None, self._world_size is not None]
        # if any(vars_set) and not all(vars_set):
        #     raise ValueError(f"world rank and world size must be set together: world_rank={self._world_rank}, world_size={self._world_size}")
    
    # def _setup_distributed(self):
    #     """
    #     Auto-detect if we're in a distributed context.
        
    #     Sets self._is_distributed and rank information based on detection.
        
    #     Detection logic:
    #     1. Check if torch.distributed is available
    #     2. Check if torch.distributed is initialized
    #     3. If yes, we're in a worker, get ranks
    #     4. If no, we're in parent mode
    #     """

    #    # Check if distributed is available and initialized
    #     if not dist.is_available() or not dist.is_initialized():
    #         if self._world_size is not None and self._world_size > 1:
    #             raise ValueError(f"World size is set to {self._world_size} but distributed is not available or initialized")
    #         if self._world_rank is not None and self._world_rank != 0:
    #             raise ValueError(f"World rank is set to {self._world_rank} but distributed is not available or initialized")
    #         if self._local_rank is not None and self._local_rank != 0:
    #             raise ValueError(f"Local rank is set to {self._local_rank} but distributed is not available or initialized")
    #         if self._is_distributed:
    #             raise ValueError(f"Distributed is not available or initialized but is_distributed is True")
            
    #         # No distributed support → parent mode OR single GPU mode
    #         # self._is_distributed = False
    #         # self._world_rank = 0
    #         # self._world_size = 1
            
    #         # For single GPU training, still set device properly
    #         if torch.cuda.is_available():
    #             # If we're here from worker_entry, CUDA device was set by cli()
    #             self._local_rank = torch.cuda.current_device()
    #     else:
    #         # Distributed is initialized → multi-GPU worker mode
    #         self._is_distributed = True
            
    #         # Get ranks from torch.distributed
    #         self._world_rank = dist.get_rank()
    #         self._world_size = dist.get_world_size()
            
    #         # Get local rank from current CUDA device
    #         if torch.cuda.is_available():
    #             self._local_rank = torch.cuda.current_device()
    #         else:
    #             raise RuntimeError(
    #                 "Distributed training detected but CUDA is not available"
    #             )
        
        # Log worker info (only rank 0)
        # if self._world_rank == 0:
        #     if self._world_size == 1:
        #         print(f"[Auto-detected] Single GPU training")
        #     elif self._world_size > 1:
        #         print(f"[Auto-detected] Multi-GPU training: world_size={self._world_size}")
    
    def run(self):
        """
        Core training loop implementation.
        
        Works for both single-GPU and distributed modes.
        """
        print(f"Running training on rank {self._world_rank} of {self._world_size} GPUs")
        print(self._train_data_provider)
        print(self._test_data_provider)
        print(self._renderer)
        print(self._loss_fn)

        modules = [
            self._renderer,
            self._loss_fn,
            self._train_data_provider,
            *([self._test_data_provider] if self._test_data_provider is not None else []),
            *self._modules,
        ]

        print("Number of modules: ", len(modules))

        # Setup
        for module in self._modules:
            module.on_setup(
                world_rank=self._world_rank,
                world_size=self._world_size,
            )
        
        for step in range(self._config.max_steps):
            pass

        raise NotImplementedError("Not implemented")
        
        # local_rank = self._local_rank
        # world_rank = self._world_rank
        # world_size = self._world_size
        # is_distributed = self._is_distributed
        
        # # Set device
        # device = torch.device(f"cuda:{local_rank}")
        
        # if world_rank == 0:
        #     mode = f"distributed ({world_size} GPUs)" if world_size > 1 else "single-GPU"
        #     print(f"[Running] {mode} training")
        
        # # Construct objects from lazy wrappers
        # if world_rank == 0:
        #     print("Loading datasets...")
        # trainset = self._trainset_lazy.get()
        # testset = self._testset_lazy.get() if self._testset_lazy is not None else None
        
        # if world_rank == 0:
        #     print("Creating renderer and loss...")
        # renderer = self._renderer_lazy.get()
        # loss_fn = self._loss_lazy.get()
        
        # # Get scene scale
        # scene_scale = trainset.scene_scale * 1.1 * self._config.global_scale
        
        # # Initialize Gaussians (only on rank 0)
        # if world_rank == 0:
        #     print("Initializing Gaussians...")
        #     splat_model = SplatModel.from_pcd(
        #         trainset,
        #         sh_degree=self._config.sh_degree,
        #         init_opacity=self._config.init_opacity,
        #         init_scale=self._config.init_scale,
        #     )
        # else:
        #     splat_model = None
        
        # # Create training state (handles distribution if world_size > 1)
        # splat_state = SplatTrainingState.from_splat_model(
        #     splat_model,
        #     device=device,
        #     world_rank=world_rank,
        #     world_size=world_size,
        #     lr_means=self._config.lr_means,
        #     lr_scales=self._config.lr_scales,
        #     lr_quats=self._config.lr_quats,
        #     lr_opacities=self._config.lr_opacities,
        #     lr_sh0=self._config.lr_sh0,
        #     lr_shN=self._config.lr_shN,
        #     scene_scale=scene_scale,
        #     batch_size=self._config.batch_size,
        # )
        
        # if world_rank == 0:
        #     print(f"Number of Gaussians: {len(splat_state.params['means'])}")
        
        # # Setup dataloader
        # trainloader = DataLoader(
        #     trainset,
        #     batch_size=self._config.batch_size,
        #     shuffle=True,
        #     num_workers=self._config.num_workers,
        # )
        # trainloader_iter = iter(trainloader)
        
        # # Training loop
        # pbar = range(self._config.max_steps)
        # if world_rank == 0:
        #     import tqdm
        #     pbar = tqdm.tqdm(pbar)
        
        # for step in pbar:
        #     # Get batch
        #     try:
        #         data = next(trainloader_iter)
        #     except StopIteration:
        #         trainloader_iter = iter(trainloader)
        #         data = next(trainloader_iter)
            
        #     # Move to device
        #     cam_to_worlds = data["cam_to_world"].to(device)
        #     Ks = data["K"].to(device)
        #     images = data["image"].to(device) / 255.0
        #     height, width = images.shape[1:3]
            
        #     # SH degree scheduling
        #     sh_degree = min(step // self._config.sh_degree_interval, self._config.sh_degree)
            
        #     # Render
        #     renders, render_outputs = renderer.render(
        #         splat_state=splat_state,
        #         cam_to_worlds=cam_to_worlds,
        #         Ks=Ks,
        #         width=width,
        #         height=height,
        #         sh_degree=sh_degree,
        #         world_rank=world_rank,
        #         world_size=world_size,
        #     )
            
        #     # Compute loss
        #     loss = loss_fn.compute(
        #         renders=renders,
        #         targets=images,
        #         splat_state=splat_state,
        #         rend_out=render_outputs,
        #     )
            
        #     # Backward
        #     loss.backward()
            
        #     # Optimize
        #     for optimizer in splat_state.optimizers.values():
        #         optimizer.step()
        #         optimizer.zero_grad(set_to_none=True)
            
        #     # Logging
        #     if world_rank == 0 and isinstance(pbar, tqdm.tqdm):
        #         mode = "dist" if world_size > 1 else "single"
        #         pbar.set_description(
        #             f"[{mode}] loss={loss.item():.3f} | "
        #             f"gs={len(splat_state.params['means'])} | "
        #             f"sh={sh_degree}"
        #         )


def splat_trainer_worker_entry(
    local_rank: int,
    world_rank: int,
    world_size: int,
    args: tuple,
):
    """
    Entry point for each worker process.
    
    Creates a new SplatTrainer instance.
    The trainer will auto-detect that it's in a distributed context.
    """
    cls, config, train_data_provider, test_data_provider, renderer, loss_fn, modules = args
    
    # Create worker trainer
    # In __post_init__, it will detect dist.is_initialized() == True
    # and automatically set _is_distributed=True and get ranks
    worker_trainer = cls.__new__(cls)
    worker_trainer._config = config
    worker_trainer._train_data_provider = train_data_provider
    worker_trainer._test_data_provider = test_data_provider
    worker_trainer._renderer = renderer
    worker_trainer._loss_fn = loss_fn
    worker_trainer._modules = modules
    worker_trainer._local_rank = local_rank
    worker_trainer._world_rank = world_rank
    worker_trainer._world_size = world_size
    worker_trainer.__post_init__()

    worker_trainer.train()