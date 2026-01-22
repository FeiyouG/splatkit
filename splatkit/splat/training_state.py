import torch
import torch.nn as nn
import torch.distributed as dist
import math
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .model import SplatModel
from ..utils.sh import sh_to_K
from ..utils.distributed import distribute_metadata, distribute_tensor, gather_tensor

@dataclass
class SplatTrainingState:
    """
    Mutable training state for 3D Gaussian Splatting

    ### Invariant:
    - Each rank owns params[rank::world_size]
    - Optimizer steps are synchronized
    - Only rank 0 may serialize state (save/load checkpoints, convert to/from splat model, etc.)

    ### Distributed Training Strategy:

    Parameters are distributed across ranks using a "striped" distribution pattern.
    For N Gaussians and K ranks:
    - Rank 0 owns Gaussians [0, K, 2K, 3K, ...]
    - Rank 1 owns Gaussians [1, K+1, 2K+1, 3K+1, ...]
    - Rank r owns Gaussians [r, r+K, r+2K, r+3K, ...]
    - ...

    This ensures balanced distribution and simplifies gather/scatter operations.

    Example with 10 Gaussians and 3 ranks:
    - Rank 0: [0, 3, 6, 9]
    - Rank 1: [1, 4, 7]
    - Rank 2: [2, 5, 8]

    ### Batching:
    - Support batching by specifying a batch size in factory methods
    """

    REQUIRED_PARAMS = frozenset([
        'means',
        'scales',
        'quats',
        'opacities',
        'sh0',
        'shN',
    ])
    DEFAULT_LEADER_RANK = 0  # Rank 0 is the default leader for distributed training
    
    params: nn.ParameterDict
    optimizers: Dict[str, torch.optim.Optimizer]
    sh_degree: int
    device: str
    world_rank: int = 0
    world_size: int = 1
    
    def __post_init__(self):
        """Validate state after initialization"""
        self.__validate__()
    
    def __validate__(self):
        """
        Validate consistency between parameters and optimizers.

        Copied from gsplat: 
        https://github.com/nerfstudio-project/gsplat/blob/b60e917c95afc449c5be33a634f1f457e116ff5e/gsplat/strategy/base.py#L15
        """

        if self.world_size > 1:
            if not dist.is_initialized():
                raise RuntimeError("Distributed training has not initialized for distributed training")

        # Check parameters are present
        keys = frozenset(self.params.keys())
        missing = SplatTrainingState.REQUIRED_PARAMS - keys
        extra = keys - SplatTrainingState.REQUIRED_PARAMS
        if len(missing) > 0:
            raise ValueError(f"Missing required parameters: {missing}")
        if len(extra) > 0:
            raise ValueError(f"Unknown parameters: {extra}")

        # Check trainable parameters match optimizer keys
        trainable_params = set(
            [name for name, param in self.params.items() if param.requires_grad]
        )
        optimizer_keys = set(self.optimizers.keys())

        if trainable_params != optimizer_keys:
            raise ValueError(f"Trainable parameters and optimizers must have the same keys, "
                             f"but got trainable={trainable_params}, optimizers={optimizer_keys}")
        
        # Check each optimizer has exactly one param_group
        for name, optimizer in self.optimizers.items():
            if len(optimizer.param_groups) != 1:
                raise ValueError(f"Optimizer '{name}' must have exactly one param_group, "
                                 f"but got {len(optimizer.param_groups)}")
        
        # Check parameter shapes are consistent
        N = self.params['means'].shape[0]
        if self.params['means'].shape != (N, 3):
            raise ValueError(f"Parameter 'means' shape mismatch: {self.params['means'].shape} != ({N}, 3)")
        if self.params['scales'].shape != (N, 3):
            raise ValueError(f"Parameter 'scales' shape mismatch: {self.params['scales'].shape} != ({N}, 3)")
        if self.params['quats'].shape != (N, 4):
            raise ValueError(f"Parameter 'quats' shape mismatch: {self.params['quats'].shape} != ({N}, 4)")
        if self.params['opacities'].shape != (N,):
            raise ValueError(f"Parameter 'opacities' shape mismatch: {self.params['opacities'].shape} != ({N},)")
        if self.params['sh0'].shape != (N, 1, 3):
            raise ValueError(f"Parameter 'sh0' shape mismatch: {self.params['sh0'].shape} != ({N}, 1, 3)")

        # Validate SH degree consistency    
        K = sh_to_K(self.sh_degree)
        if self.params['shN'].shape != (N, K - 1, 3):
            raise ValueError(f"shN shape mismatch: {self.params['shN'].shape} != ({N}, {K - 1}, 3)")
    
    @classmethod
    def from_splat_model(
        cls,
        model: Optional[SplatModel],  # Only rank 0 needs to provide this
        device: str = "cuda",
        world_rank: int = 0,
        world_size: int = 1,
        lr_means: float = 1.6e-4,
        lr_scales: float = 5e-3,
        lr_quats: float = 1e-3,
        lr_opacities: float = 5e-2,
        lr_sh0: float = 2.5e-3,
        lr_shN: float = 2.5e-3 / 20,
        scene_scale: float = 1.0,
        batch_size: int = 1,

        leader_rank: int = DEFAULT_LEADER_RANK,
    ) -> 'SplatTrainingState':
        """
        Create training state from SplatModel.
        
        For distributed training:
        - Rank 0: Must provide model, broadcasts to other ranks
        - Other ranks: Pass None, receive via broadcast
        
        Args:
            model: SplatModel to convert (only rank 0 needs to provide this)
            device: Device to place parameters
            world_rank: Current rank for distributed training
            world_size: Total number of ranks
            lr_means, lr_scales, lr_quats, lr_opacities, lr_sh0, lr_shN: Learning rates
            scene_scale: Scene scale (multiplies lr_means)
            batch_size: Batch size for LR scaling

            leader_rank: the rank that will be used to broadcast the model to all other ranks. Defaults to 0.
        
        Returns:
            SplatTrainingState ready for training
        """

        if world_size == 1 or world_rank == leader_rank:
            if model is None:
                raise ValueError("Model is required but missing on rank {world_rank}")
            means_full = torch.from_numpy(model.points).clone().float()
            scales_full = torch.from_numpy(model.scales).clone().float()
            quats_full = torch.from_numpy(model.quats).clone().float()
            opacities_full = torch.from_numpy(model.opacities).clone().float()
            sh0_full = torch.from_numpy(model.sh0).clone().float()
            shN_full = torch.from_numpy(model.shN).clone().float()
            sh_degree = model.sh_degree

            params = {
                'means': means_full,
                'scales': scales_full,
                'quats': quats_full,
                'opacities': opacities_full,
                'sh0': sh0_full,
                'shN': shN_full,
            }
        else:
            params = None
            sh_degree = None

        return cls._distribute(
            params, 
            None, 
            sh_degree,

            device, 
            world_rank, 
            world_size, 
            lr_means, 
            lr_scales, 
            lr_quats, 
            lr_opacities, 
            lr_sh0,
            lr_shN,
            scene_scale,
            batch_size,
            leader_rank,
        )
    

    @classmethod
    def from_ckpt(
        cls,
        path: str,
        device: str = "cuda",
        world_rank: int = 0,
        world_size: int = 1,
        load_optimizers: bool = True,
        lr_means: float = 1.6e-4,
        lr_scales: float = 5e-3,
        lr_quats: float = 1e-3,
        lr_opacities: float = 5e-2,
        lr_sh0: float = 2.5e-3,
        lr_shN: float = 2.5e-3 / 20,
        scene_scale: float = 1.0,
        batch_size: int = 1,
        
        leader_rank: int = DEFAULT_LEADER_RANK,
    ) -> Tuple['SplatTrainingState', int]:
        """
        Load checkpoint and distribute to ranks.
        
        Loads unified checkpoint and distributes parameters across ranks.
        
        Args:
            path: Checkpoint path
            device: Device to load onto
            world_rank: Current rank
            world_size: Total number of ranks
            load_optimizers: If True, load optimizer state from checkpoint
            lr_means, lr_scales, etc.: Learning rates for recreating optimizers
            scene_scale: Scene scale
            batch_size: Batch size
            
            leader_rank: the rank that will be used to broadcast the model to all other ranks. Defaults to 0.
        Returns:
            Tuple of (SplatTrainingState with distributed parameters, step)

            After reloading from ckpt, the tensor will not be ditributed 
            exactly the same way as they were before.
        """
        # Only leader rank loads from disk
        if world_size == 1 or world_rank == leader_rank:
            if not Path(path).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            ckpt = torch.load(path, map_location='cpu')

            required_keys = {'params', 'step', 'sh_degree'}
            missing = required_keys - set(ckpt.keys())
            if missing:
                raise ValueError(f"Invalid checkpoint file: missing required keys: {missing}")
            
            params_state = ckpt['params']
            sh_degree = ckpt['sh_degree']
            step_loaded: int = ckpt['step']
            optimizers_state = ckpt.get('optimizers', None) if load_optimizers else None
        else:
            params_state = None
            sh_degree = None
            step_loaded = None
            optimizers_state = None
        
        # Distribute step to all ranks
        if world_size > 1:
            data = distribute_metadata(
                {'step': step_loaded},
                world_rank,
                world_size,
                leader_rank=leader_rank,
            )
            assert data is not None
            step = data['step']
        else:
            step = step_loaded
        
        training_state = cls._distribute(
            params_state, 
            optimizers_state, 
            sh_degree, 
            device, 
            world_rank, 
            world_size,
            lr_means, lr_scales, lr_quats, lr_opacities, lr_sh0, lr_shN, scene_scale, batch_size,
            leader_rank=leader_rank,
        )
        
        return training_state, step
    
    def to_splat_model(self, leader_rank: int = DEFAULT_LEADER_RANK) -> SplatModel | None:
        """
        Convert training state to splat model.

        Args:
            leader_rank: the rank that will be used to broadcast the model to all other ranks. Defaults to 0.
        """
        params = SplatTrainingState._gather_params(self.params, self.world_rank, self.world_size, leader_rank=leader_rank)
        if self.world_rank != leader_rank:
            return None
        
        if params is None:
            raise RuntimeError(f"Failed to gather parameters on leader rank {leader_rank}")
        
        return SplatModel(
            _points=params['means'].cpu().numpy(),
            _scales=params['scales'].cpu().numpy(),
            _quats=params['quats'].cpu().numpy(),
            _opacities=params['opacities'].cpu().numpy(),
            _sh0=params['sh0'].cpu().numpy(),
            _shN=params['shN'].cpu().numpy(),
            _sh_degree=self.sh_degree,
        )
    
    def save_ckpt(
        self,
        path: str,
        step: int,
        metadata: Optional[Dict] = None,

        leader_rank: int = DEFAULT_LEADER_RANK,
    ) -> None:
        """
        Save checkpoint to disk.
        
        Only main rank writes to disk in distributed mode.
        
        Args:
            path: Checkpoint path
            step: Training step (explicit from training loop)
            metadata: Optional additional metadata

            leader_rank: the rank that will be used to broadcast the model to all other ranks. Defaults to 0.
        """

        gather_result = SplatTrainingState._gather(self, leader_rank=leader_rank)
        if self.world_rank != leader_rank:
            return
        if gather_result is None:
            raise RuntimeError(f"Failed to gather state on leader rank {leader_rank}")
        params, optimizers = gather_result

        if params is None or optimizers is None:
            raise RuntimeError(f"Failed to gather state on leader rank {leader_rank}: params={params}, optimizers={optimizers}")
        
        data = {
            'step': step,
            'sh_degree': self.sh_degree,
            'params': params,
            'optimizers': optimizers,
        }

        if metadata is not None:
            data['metadata'] = metadata
        
        torch.save(data, path)

    @staticmethod
    def _distribute_params(
        params_dict: Dict[str, torch.Tensor] | None,
        world_rank: int,
        world_size: int,
        device: str,

        leader_rank: int = DEFAULT_LEADER_RANK,
    ) -> nn.ParameterDict:
        """
        Distribute parameter tensors from rank 0 to all ranks.
        
        Args:
            params_dict: Dict of parameter tensors (only rank 0 provides)
            world_rank: Current rank
            world_size: Total ranks
            device: Target device
        
        Returns:
            ParameterDict with distributed parameters
        """
        if world_size == 1:
            if params_dict is None:
                raise ValueError("Params dict is missing but required for single GPU")
            return nn.ParameterDict({
                k: nn.Parameter(v.clone().to(device)) for k, v in params_dict.items()
            })
            
        # Gather parameter names from all ranks so each rank knows how many parameters to expect
        if world_rank == leader_rank:
            if params_dict is None:
                raise ValueError("Params dict is missing but required for leader rank")
            param_names = list(params_dict.keys())
        else:
            param_names = None # type: ignore

        data = distribute_metadata(
            {'param_names': param_names},
            world_rank,
            world_size,
            leader_rank=leader_rank,
        )
        assert data is not None
        param_names: list[str] = data['param_names']

        # Distribute each parameter tensor
        local_params = {}
        for name in param_names:
            tensor = params_dict[name] if params_dict is not None else None
            local_tensor = distribute_tensor(
                tensor, world_rank, world_size, device, leader_rank=leader_rank, striped=True
            )
            local_params[name] = nn.Parameter(local_tensor)
        
        return nn.ParameterDict(local_params)
    
    @staticmethod
    def _distribute_optimizer_state_dict(
        optimizer_states: Optional[Dict[str, Dict[str, Any]]],
        world_rank: int,
        world_size: int,
        device: str,

        leader_rank: int = DEFAULT_LEADER_RANK,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Distribute optimizer state dicts from rank 0 to all ranks.

        Note:
            Optimizer scalar metadata is globally broadcast; tensor states are distributed via striped P2P.
            All ranks must call these in identical order.
        
        Args:
            optimizer_states: Dict mapping param_name to optimizer state
                            Format: {'means': {'step': 100, 'exp_avg': tensor, ...}}
                            Only rank 0 provides this
            world_rank: Current rank
            world_size: Total ranks
            device: Target device for tensors

            leader_rank: the rank that will be used to broadcast the model to all other ranks. Defaults to 0.
        
        Returns:
            Dict of distributed optimizer states (same format as input)
            Returns None if no optimizer states to distribute
        """
        if optimizer_states is None or len(optimizer_states) == 0:
            return None
        
        # Broadcast parameter names so all ranks know what to expect
        if world_rank == leader_rank:
            param_names = list(optimizer_states.keys())
        else:
            param_names = None # type: ignore
        
        data = distribute_metadata(
            {'param_names': param_names},
            world_rank,
            world_size,
            leader_rank=leader_rank,
        )
        assert data is not None
        param_names: list[str] = data['param_names']
        
        # Multi-GPU: distribute optimizer states
        distributed_states = {}
        
        for param_name in param_names:
            # Get full state on rank 0
            # NOTE: Assuming using Adam optimizer for now
            if world_rank == leader_rank:
                full_state = optimizer_states[param_name]
                step = full_state['step']
                param_groups = full_state['param_groups']
                exp_avg_full = full_state['exp_avg']
                exp_avg_sq_full = full_state['exp_avg_sq']
            else:
                step = None # type: ignore
                param_groups = None
                exp_avg_full = None
                exp_avg_sq_full = None
            
            # Broadcast scalar metadata (step, param_groups)
            data = distribute_metadata(
                {'step': step, 'param_groups': param_groups},
                world_rank,
                world_size,
                leader_rank=leader_rank,
            )
            assert data is not None
            step: int = data['step']
            param_groups = data['param_groups']
            
            # Distribute tensor state (exp_avg, exp_avg_sq)
            exp_avg_local = distribute_tensor(
                exp_avg_full, world_rank, world_size, device, leader_rank=leader_rank, striped=True
            )
            exp_avg_sq_local = distribute_tensor(
                exp_avg_sq_full, world_rank, world_size, device, leader_rank=leader_rank, striped=True
            )
            
            # Store distributed state in same format
            distributed_states[param_name] = {
                'step': step,
                'exp_avg': exp_avg_local,
                'exp_avg_sq': exp_avg_sq_local,
                'param_groups': param_groups,
            }
        
        return distributed_states

    @classmethod
    def _distribute(
        cls,
        params_dict: Dict[str, torch.Tensor] | None,
        optimizer_states: Dict[str, Dict[str, Any]] | None,
        sh_degree: int | None, # type: ignore
        device: str,
        world_rank: int,
        world_size: int,
        lr_means: float = 1.6e-4,
        lr_scales: float = 5e-3,
        lr_quats: float = 1e-3,
        lr_opacities: float = 5e-2,
        lr_sh0: float = 2.5e-3,
        lr_shN: float = 2.5e-3 / 20,
        scene_scale: float = 1.0,
        batch_size: int = 1,

        leader_rank: int = DEFAULT_LEADER_RANK,
    ) -> 'SplatTrainingState':
        """
        Distribute parameters, optimizer states, and metadata to create training state.
        
        This is the main distribution primitive used by from_splat_model and from_ckpt.
        
        Args:
            params_dict: Dict of parameter tensors (only rank 0 provides)
            optimizer_states: Dict of optimizer states (only rank 0 provides)
            sh_degree: SH degree (only rank 0 provides)
            device: Target device
            world_rank: Current rank
            world_size: Total ranks
            lr_*: Learning rates for creating optimizers
            scene_scale: Scene scale
            batch_size: Batch size

            leader_rank: the rank that will be used to broadcast the model to all other ranks. Defaults to 0.
        Returns:
            SplatTrainingState with distributed parameters and optimizers
        """

        if world_size == 1:
            if sh_degree is None:
                raise ValueError("SH degree is missing but required for single GPU")
        else:
            data = distribute_metadata(
                {'sh_degree': sh_degree},
                world_rank,
                world_size,
                leader_rank=leader_rank,
            )
            assert data is not None
            sh_degree: int = data['sh_degree']
        
        # Distribute parameters
        params = cls._distribute_params(params_dict, world_rank, world_size, device, leader_rank=leader_rank)
        
        # Distribute optimizer states (just the state dicts, not optimizers yet)
        distributed_opt_states = cls._distribute_optimizer_state_dict(
            optimizer_states, world_rank, world_size, device,
            leader_rank=leader_rank,
        )
        
        # Create optimizers for all parameters
        optimizers = cls._create_optimizers(
            params=params,
            lr_means=lr_means,
            lr_scales=lr_scales,
            lr_quats=lr_quats,
            lr_opacities=lr_opacities,
            lr_sh0=lr_sh0,
            lr_shN=lr_shN,
            scene_scale=scene_scale,
            batch_size=batch_size,
            world_size=world_size,
        )
        
        # Load distributed states into optimizers (if available)
        if distributed_opt_states is not None:
            for param_name, opt_state in distributed_opt_states.items():
                optimizer = optimizers[param_name]
                param = params[param_name]
                param_id = id(param)
                
                # Convert our custom format to PyTorch state_dict format
                pytorch_state_dict = {
                    'state': {
                        param_id: {
                            'step': opt_state['step'],
                            'exp_avg': opt_state['exp_avg'],
                            'exp_avg_sq': opt_state['exp_avg_sq'],
                        }
                    },
                    'param_groups': [
                        {
                            'params': [param_id],
                            **opt_state['param_groups'][0]  # Merge hyperparameters
                        }
                    ]
                }
                
                optimizer.load_state_dict(pytorch_state_dict)
        
        return cls(
            params=params,
            optimizers=optimizers,
            sh_degree=sh_degree,
            device=device,
            world_rank=world_rank,
            world_size=world_size,
        )

    @staticmethod
    def _gather_params(
        params: nn.ParameterDict,
        world_rank: int,
        world_size: int,

        leader_rank: int = DEFAULT_LEADER_RANK,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Gather parameters from all ranks to leader rank in cpu.
        
        Args:
            params: Local ParameterDict from this rank
            world_rank: Current rank
            world_size: Total number of ranks

            leader_rank: the rank that will be used to broadcast the model to all other ranks. Defaults to 0.
        
        Returns:
            Dict of gathered tensors on leader rank (None for other ranks)
        """
        if world_size == 1:
            return {k: v.data.cpu() for k, v in params.items()}
        
        gathered_params = {}
        for param_name, param in params.items():
            gathered_tensor = gather_tensor(
                param.data, world_rank, world_size, leader_rank=leader_rank
            )
            
            if world_rank == leader_rank:
                if gathered_tensor is None:
                    raise RuntimeError(f"Failed to gather parameter {param_name} on rank {world_rank}")
                gathered_params[param_name] = gathered_tensor.cpu()
        
        return gathered_params if world_rank == leader_rank else None

    @staticmethod
    def _gather_optimizer_state_dict(
        optimizers: Dict[str, torch.optim.Optimizer],
        world_rank: int,
        world_size: int,

        leader_rank: int = DEFAULT_LEADER_RANK,
    ) -> Dict[str, Dict[str, Any]] | None:
        """
        Gather optimizer states from all ranks to leader rank in cpu.
        
        Args:
            optimizers: Dict of optimizers from this rank
            world_rank: Current rank
            world_size: Total number of ranks
        
            leader_rank: the rank that will be used to broadcast the model to all other ranks. Defaults to 0.
        Returns:
            Dict of gathered optimizer states on leader rank (None for other ranks)
        """
        if world_size == 1:
            # Single GPU: extract state in custom format
            result = {}
            for name, optimizer in optimizers.items():
                state_dict = optimizer.state_dict()
                param_state = list(state_dict['state'].values())[0]
                
                # NOTE: This assumes using Adam optimizer for now
                result[name] = {
                    'step': param_state.get('step', 0),
                    'exp_avg': param_state.get('exp_avg').cpu(),
                    'exp_avg_sq': param_state.get('exp_avg_sq').cpu(),
                    'param_groups': state_dict['param_groups'],
                }
            return result
        
        # Gather states from all ranks to leader rank
        gathered_states: Dict[str, Dict[str, Any]] = {}
        for param_name, optimizer in optimizers.items():
            state_dict = optimizer.state_dict()
            param_state = list(state_dict['state'].values())[0]
            
            # Extract local state
            step = param_state.get('step', 0)
            exp_avg = param_state.get('exp_avg')
            exp_avg_sq = param_state.get('exp_avg_sq')
            
            # Gather exp_avg and exp_avg_sq tensors
            exp_avg_gathered = gather_tensor(
                exp_avg, world_rank, world_size, leader_rank=leader_rank
            )
            exp_avg_sq_gathered = gather_tensor(
                exp_avg_sq, world_rank, world_size, leader_rank=leader_rank
            )
            
            if world_rank == leader_rank:
                if exp_avg_gathered is None or exp_avg_sq_gathered is None:
                    raise RuntimeError(f"Failed to gather exp_avg or exp_avg_sq for parameter {param_name} on rank {world_rank}")

                gathered_states[param_name] = {
                    'step': step,
                    'exp_avg': exp_avg_gathered.cpu(),
                    'exp_avg_sq': exp_avg_sq_gathered.cpu(),
                    'param_groups': state_dict['param_groups'],
                }

        if world_rank == leader_rank:
            return gathered_states
        else:
            return None
        

    @classmethod
    def _gather(
        cls,
        training_state: 'SplatTrainingState',

        leader_rank: int = DEFAULT_LEADER_RANK,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]]:
        """
        Gather all state from distributed ranks to rank leader rank in cpu.
        
        Args:
            training_state: Current training state
        
        Returns:
            Tuple of (params_dict, optimizer_states) on rank 0
            Returns None for other ranks

            leader_rank: the rank that will be used to broadcast the model to all other ranks. Defaults to 0.
        """
        world_rank = training_state.world_rank
        world_size = training_state.world_size
        
        # Gather parameters
        params_dict = cls._gather_params(
            training_state.params, world_rank, world_size, leader_rank=leader_rank
        )
        
        # Gather optimizer states
        optimizer_states = cls._gather_optimizer_state_dict(
            training_state.optimizers, world_rank, world_size, leader_rank=leader_rank
        )
        
        if world_rank == leader_rank:
            if params_dict is None:
                raise RuntimeError(f"Failed to gather parameters on leader rank {leader_rank}")
            if optimizer_states is None:
                raise RuntimeError(f"Failed to gather optimizer states on leader rank {leader_rank}")
            
            return params_dict, optimizer_states
        return None

    
    @staticmethod
    def _create_optimizers(
        params: nn.ParameterDict,
        lr_means: float,
        lr_scales: float,
        lr_quats: float,
        lr_opacities: float,
        lr_sh0: float,
        lr_shN: float,
        scene_scale: float,
        batch_size: int,
        world_size: int,
    ) -> Dict[str, torch.optim.Optimizer]:
        """Create Adam optimizers for all parameters"""
        # Learning rate scaling (following gsplat)
        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this would not make the training exactly equivalent, see
        # https://arxiv.org/pdf/2402.18824v1
        BS = batch_size * world_size
        lr_scale = math.sqrt(BS)
        eps_scale = 1.0 / math.sqrt(BS)
        
        param_lr_map = {
            'means': lr_means * scene_scale * lr_scale,
            'scales': lr_scales * lr_scale,
            'quats': lr_quats * lr_scale,
            'opacities': lr_opacities * lr_scale,
            'sh0': lr_sh0 * lr_scale,
            'shN': lr_shN * lr_scale,
        }
        
        optimizers = {}
        beta1 = max(0.9 ** BS, 0.5)
        beta2 = max(0.999 ** BS, 0.99)
        for name, param in params.items():
            optimizers[name] = torch.optim.Adam(
                [{'params': [param], 'lr': param_lr_map[name], 'name': name}],
                eps=1e-15 * eps_scale,
                betas=(beta1, beta2),
            )
        
        return optimizers
    
    # Properties and utilities
    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1
    
    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians on this rank"""
        return len(self.params['means'])
    
    @property
    def colors(self) -> torch.Tensor:
        """Colors of the Gaussians on this rank"""
        return torch.cat([self.params["sh0"], self.params["shN"]], dim=1)  # [N, K, 3]
    
    def zero_grad(self):
        """Zero all gradients"""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
    
    def step(self):
        """Perform optimizer step"""
        for optimizer in self.optimizers.values():
            optimizer.step()

    
    def get_metadata(self) -> Dict[str, Any]:
        """Get current training state metadata."""
        return {
            'num_gaussians': self.num_gaussians,
            'sh_degree': self.sh_degree,
            'device': self.device,
            'world_rank': self.world_rank,
            'world_size': self.world_size,
            'is_distributed': self.is_distributed,
        }
    
    def __repr__(self) -> str:
        return (
            f"SplatTrainingState(\n"
            f"  {self.get_metadata()}\n"
            f")"
        )
        
    def _debug_distributed_consistency(self, leader_rank: int = DEFAULT_LEADER_RANK) -> None:
        """Verify that distributed state is consistent across ranks (for debugging)."""
        if not self.is_distributed:
            return
        
        # Check num_gaussians matches across ranks
        local_count = torch.tensor([self.num_gaussians], dtype=torch.long, device=self.device)
        all_counts = [torch.zeros_like(local_count) for _ in range(self.world_size)]
        dist.all_gather(all_counts, local_count)
        
        if self.world_rank == leader_rank:
            total = sum(t.item() for t in all_counts)
            print(f"Total Gaussians across {self.world_size} ranks: {total}")
            for rank, count in enumerate(all_counts):
                print(f"  Rank {rank}: {count.item()} Gaussians")