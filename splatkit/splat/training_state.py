import torch
import torch.nn as nn
import torch.distributed as dist
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .model import SplatModel
from ..utils.sh_utils import sh_to_K

@dataclass
class SplatTrainingState:
    """
    SplatTrainingState: Mutable training state for 3D Gaussian Splatting

    Distributed Training Strategy:
    ------------------------------
    Parameters are distributed across ranks using a "striped" distribution pattern.
    For N Gaussians and K ranks:
    - Rank 0 owns Gaussians [0, K, 2K, 3K, ...]
    - Rank 1 owns Gaussians [1, K+1, 2K+1, 3K+1, ...]
    - Rank r owns Gaussians [r, r+K, r+2K, r+3K, ...]

    This ensures balanced distribution and simplifies gather/scatter operations.

    Example with 10 Gaussians and 3 ranks:
    - Rank 0: [0, 3, 6, 9]
    - Rank 1: [1, 4, 7]
    - Rank 2: [2, 5, 8]
    """

    REQUIRED_PARAMS = frozenset([
        'means',
        'scales',
        'quats',
        'opacities',
        'sh0',
        'shN',
    ])
    LEADER_RANK = 0  # Rank 0 is the leader for distributed training
    
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

        # Check parameters are present
        keys = self.params.keys()
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
        N = len(self.params['means'])
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
        model: SplatModel,
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
    ) -> 'SplatTrainingState':
        """
        Create training state from SplatModel.
        
        Args:
            model: SplatModel to convert
            device: Device to place parameters
            world_rank: Current rank for distributed training
            world_size: Total number of ranks
            lr_means, lr_scales, lr_quats, lr_opacities, lr_sh0, lr_shN: Learning rates
            scene_scale: Scene scale (multiplies lr_means)
            batch_size: Batch size for LR scaling
        
        Returns:
            SplatTrainingState ready for training
        """
        # Get numpy arrays from model
        points = model.points
        scales = model.scales
        quats = model.quats
        opacities = model.opacities
        sh0 = model.sh0
        shN = model.shN
        
        # Distribute across ranks (striped)
        if world_size > 1:
            points = points[world_rank::world_size]
            scales = scales[world_rank::world_size]
            quats = quats[world_rank::world_size]
            opacities = opacities[world_rank::world_size]
            sh0 = sh0[world_rank::world_size]
            shN = shN[world_rank::world_size]
        
        # Convert to torch and wrap in Parameters
        params = nn.ParameterDict({
            'means': nn.Parameter(torch.from_numpy(points).clone()),
            'scales': nn.Parameter(torch.from_numpy(scales).clone()),
            'quats': nn.Parameter(torch.from_numpy(quats).clone()),
            'opacities': nn.Parameter(torch.from_numpy(opacities).clone()),
            'sh0': nn.Parameter(torch.from_numpy(sh0).clone()),
            'shN': nn.Parameter(torch.from_numpy(shN).clone()),
        }).to(device)
        
        # Create optimizers
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
        
        return cls(
            params=params,
            optimizers=optimizers,
            sh_degree=model.sh_degree,
            device=device,
            world_rank=world_rank,
            world_size=world_size,
        )
    
    def to_splat_model(self) -> SplatModel | None:
        """
        Convert training state back to SplatModel.
        
        If distributed, gathers from all ranks first.
        
        Returns:
            SplatModel with current parameters (None for non-main ranks in distributed mode)
        """

        gathered_params = self._gather_params()
        if gathered_params is None:
            return None  # Only main rank returns model
        
        # Convert to numpy
        points = gathered_params['means'].detach().cpu().numpy()
        scales = gathered_params['scales'].detach().cpu().numpy()
        quats = gathered_params['quats'].detach().cpu().numpy()
        opacities = gathered_params['opacities'].detach().cpu().numpy()
        sh0 = gathered_params['sh0'].detach().cpu().numpy()
        shN = gathered_params['shN'].detach().cpu().numpy()
        
        return SplatModel(
            _sh_degree=self.sh_degree,
            _points=points,
            _scales=scales,
            _quats=quats,
            _opacities=opacities,
            _sh0=sh0,
            _shN=shN,
        )
    
    def save_ckpt(
        self,
        path: str,
        step: int,
        include_optimizers: bool = True,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Save checkpoint to disk.
        
        Only main rank writes to disk in distributed mode.
        
        Args:
            path: Checkpoint path
            step: Training step
            include_optimizers: If True, save optimizer state (recommended for resuming training)
            metadata: Optional additional metadata
        """

        gathered_params = self._gather_params()
        if gathered_params is None:
            if self.is_distributed and not self.is_main_rank:
                return  # Non-main ranks exit successfully
            else:
                raise RuntimeError("Failed to gather parameters")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'step': step,
            'sh_degree': self.sh_degree,
            'params': {k: v.detach().cpu() for k, v in gathered_params.items()},
        }
        
        if include_optimizers:
            gathered_optimizer_states = self._gather_optimizer_states()
            if gathered_optimizer_states is None:
                # At this point, only main rank can raise an error
                raise RuntimeError("Failed to gather optimizer states")
            data['optimizers'] = gathered_optimizer_states
        
        if metadata is not None:
            data['metadata'] = metadata
        
        torch.save(data, path)

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
    ) -> 'SplatTrainingState':
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
        
        Returns:
            SplatTrainingState with distributed parameters
        """
        # Only leader rank loads from disk
        if world_size == 1 or world_rank == SplatTrainingState.LEADER_RANK:
            if not Path(path).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            ckpt = torch.load(path, map_location='cpu')

            required_keys = {'params', 'step', 'sh_degree'}
            missing = required_keys - set(ckpt.keys())
            if missing:
                raise ValueError(f"Invalid checkpoint file: missing required keys: {missing}")
            
            params_state = ckpt['params']
            step = ckpt['step']
            sh_degree = ckpt['sh_degree']
            metadata = ckpt.get('metadata', {})
            optimizers_state = ckpt.get('optimizers', {}) if load_optimizers else {}
        else:
            params_state = None
            sh_degree = None
            step = None
            metadata = {}
            optimizers_state = {}
        
        # Broadcast metadata if distributed
        if world_size > 1:
            metadata_to_broadcast = {
                'sh_degree': sh_degree,
                'step': step,
                'metadata': metadata,
            }
            metadata_to_broadcast = cls._broadcast_metadata(metadata_to_broadcast)
            step = metadata_to_broadcast['step']
            sh_degree = metadata_to_broadcast['sh_degree']
            metadata = metadata_to_broadcast['metadata']
        
        # Scatter parameters to all ranks
        params = nn.ParameterDict()
        for param_name in SplatTrainingState.REQUIRED_PARAMS:
            param = params_state[param_name] if params_state is not None else None
            local_param = cls._scatter_tensor(param, world_rank, world_size, device)
            params[param_name] = nn.Parameter(local_param)
        
        # Create optimizers
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
        
        # Load optimizer states if available
        if load_optimizers and optimizers_state:
            for param_name in optimizers.keys():
                # Get full state on leader rank
                full_opt_state = optimizers_state.get(param_name) if world_rank == cls.LEADER_RANK else None
                
                # Check if state exists (broadcast this info)
                has_state = full_opt_state is not None if world_rank == cls.LEADER_RANK else None
                has_state_list = [has_state]
                if world_size > 1:
                    dist.broadcast_object_list(has_state_list, src=cls.LEADER_RANK) 
                has_state = has_state_list[0]
                
                if not has_state:
                    continue  # Skip if no saved state for this optimizer
                
                # Scatter optimizer state
                local_opt_state = cls._scatter_optimizer_state(
                    full_state=full_opt_state,
                    local_param=params[param_name],
                    world_rank=world_rank,
                    world_size=world_size,
                )
                
                # Load into optimizer
                optimizers[param_name].load_state_dict(local_opt_state)
        
        return cls(
            params=params,
            optimizers=optimizers,
            sh_degree=sh_degree,
            device=device,
            world_rank=world_rank,
            world_size=world_size,
        )
    
    # Helper methods
    def _gather_params(self) -> Optional[nn.ParameterDict]:
        """Gather parameters from all ranks to rank 0 (handles variable sizes)"""
        if not self.is_distributed:
            return self.params
        
        leader_rank = SplatTrainingState.LEADER_RANK
        
        # Gather to CPU using object collectives (handles variable sizes)
        gathered_params = nn.ParameterDict() if self.is_main_rank else None
        
        for param_name, param in self.params.items():
            # Convert to CPU for gathering
            local_tensor = param.data.cpu()
            
            # Gather all tensors (as objects, so variable size is OK)
            gathered_objects = [None] * self.world_size
            dist.all_gather_object(gathered_objects, local_tensor)
            
            if self.is_main_rank:
                # Calculate total size for proper interleaving
                sizes = [t.shape[0] for t in gathered_objects]
                total_size = sum(sizes)
                
                # Interleave to reconstruct striped distribution
                gathered_tensor = torch.zeros(
                    (total_size, *param.shape[1:]),
                    dtype=param.dtype,
                    device='cpu'
                )
                
                # Interleave: rank i owns indices [i, i+K, i+2K, ...]
                for rank_idx in range(self.world_size):
                    chunk = gathered_objects[rank_idx]
                    gathered_tensor[rank_idx::self.world_size][:len(chunk)] = chunk
                
                gathered_params[param_name] = nn.Parameter(gathered_tensor.to(param.device))
        
        return gathered_params

    def _gather_optimizer_states(self) -> Optional[Dict[str, Dict]]:
        """Gather optimizer states from all ranks (manual, avoids param ID issues)"""
        if not self.is_distributed:
            # Single GPU: save in simple format
            result = {}
            for name, optimizer in self.optimizers.items():
                state_dict = optimizer.state_dict()
                param_state = list(state_dict['state'].values())[0]
                
                result[name] = {
                    'step': param_state.get('step', 0),
                    'exp_avg': param_state.get('exp_avg'),
                    'exp_avg_sq': param_state.get('exp_avg_sq'),
                    'param_groups': state_dict['param_groups'],
                }
            return result
        
        # Distributed: gather each component
        gathered_states = {} if self.is_main_rank else None
        
        for param_name, optimizer in self.optimizers.items():
            state_dict = optimizer.state_dict()
            param_state = list(state_dict['state'].values())[0]
            
            # Extract state components
            step = param_state.get('step', 0)
            exp_avg = param_state.get('exp_avg')
            exp_avg_sq = param_state.get('exp_avg_sq')
            
            # Gather exp_avg using all_gather_object (handles variable sizes)
            gathered_exp_avg = [None] * self.world_size
            gathered_exp_avg_sq = [None] * self.world_size
            
            # Slow but safe, acceptable for saving checkpoints
            dist.all_gather_object(gathered_exp_avg, exp_avg.cpu() if exp_avg is not None else None)
            dist.all_gather_object(gathered_exp_avg_sq, exp_avg_sq.cpu() if exp_avg_sq is not None else None)
            
            if self.is_main_rank:
                # Interleave gathered tensors
                if gathered_exp_avg[0] is not None:
                    total_size = sum(t.shape[0] for t in gathered_exp_avg if t is not None)
                    
                    full_exp_avg = torch.zeros((total_size, *exp_avg.shape[1:]), dtype=exp_avg.dtype)
                    full_exp_avg_sq = torch.zeros((total_size, *exp_avg_sq.shape[1:]), dtype=exp_avg_sq.dtype)
                    
                    for rank_idx in range(self.world_size):
                        if gathered_exp_avg[rank_idx] is not None:
                            chunk_size = gathered_exp_avg[rank_idx].shape[0]
                            full_exp_avg[rank_idx::self.world_size][:chunk_size] = gathered_exp_avg[rank_idx]
                            full_exp_avg_sq[rank_idx::self.world_size][:chunk_size] = gathered_exp_avg_sq[rank_idx]
                else:
                    full_exp_avg = None
                    full_exp_avg_sq = None
                
                gathered_states[param_name] = {
                    'step': step,
                    'exp_avg': full_exp_avg,
                    'exp_avg_sq': full_exp_avg_sq,
                    'param_groups': state_dict['param_groups'],
                }
        
        return gathered_states
    
    @staticmethod
    def _broadcast_metadata(metadata: Optional[Dict]) -> Dict:
        """Broadcast metadata from rank 0 to all ranks"""
        object_list = [metadata]
        dist.broadcast_object_list(object_list, src=SplatTrainingState.LEADER_RANK)
        return object_list[0]
    
    @staticmethod
    def _scatter_tensor(
        full_tensor: Optional[torch.Tensor],
        world_rank: int,
        world_size: int,
        device: str,
    ) -> torch.Tensor:
        """
        Scatter tensor from rank 0 to all ranks (striped distribution).
        
        Strategy: Broadcast full tensor, then each rank takes its slice.
        While this sends more data than necessary, it's much simpler and more robust.
        For truly large models, consider using a more sophisticated approach.
        """
        if world_size == 1:
            return full_tensor.to(device)
        
        # Broadcast tensor from rank 0
        tensor_list = [full_tensor]
        dist.broadcast_object_list(tensor_list, src=SplatTrainingState.LEADER_RANK) # Slower but safe
        full_tensor = tensor_list[0]
        
        # Each rank takes its striped slice
        local_tensor = full_tensor[world_rank::world_size].clone().to(device)
        
        return local_tensor
    
   
    @staticmethod
    def _scatter_optimizer_state(
        full_state: Optional[Dict[str, Any]],
        local_param: nn.Parameter,
        world_rank: int,
        world_size: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Scatter optimizer state (manual format, not PyTorch state_dict).
        
        Args:
            full_state: Dict with keys: 'step', 'exp_avg', 'exp_avg_sq', 'param_groups'
            local_param: Local parameter
            world_rank: Current rank
            world_size: Total ranks
        
        Returns:
            PyTorch-compatible state dict for loading
        """
        if world_size == 1:
            if full_state is None:
                return None
            
            # Build PyTorch state dict
            local_param_id = id(local_param)
            return {
                'state': {
                    local_param_id: {
                        'step': full_state['step'],
                        'exp_avg': full_state['exp_avg'],
                        'exp_avg_sq': full_state['exp_avg_sq'],
                    }
                },
                'param_groups': full_state['param_groups']
            }
        
        leader_rank = SplatTrainingState.LEADER_RANK
        
        # Broadcast scalar metadata
        if world_rank == leader_rank:
            step = full_state['step']
        else:
            step = None
        step_list = [step]
        dist.broadcast_object_list(step_list, src=leader_rank)
        step = step_list[0]
        
        # Broadcast param_groups
        if world_rank == leader_rank:
            param_groups = full_state['param_groups']
        else:
            param_groups = None
        pg_list = [param_groups]
        dist.broadcast_object_list(pg_list, src=leader_rank)
        param_groups = pg_list[0]
        
        # Scatter exp_avg and exp_avg_sq
        if world_rank == leader_rank:
            exp_avg = full_state['exp_avg']
            exp_avg_sq = full_state['exp_avg_sq']
        else:
            exp_avg = None
            exp_avg_sq = None
        
        # Use broadcast + slice (simple approach)
        exp_avg_list = [exp_avg]
        exp_avg_sq_list = [exp_avg_sq]
        dist.broadcast_object_list(exp_avg_list, src=leader_rank)
        dist.broadcast_object_list(exp_avg_sq_list, src=leader_rank)
        
        exp_avg = exp_avg_list[0]
        exp_avg_sq = exp_avg_sq_list[0]
        
        if exp_avg is not None:
            local_exp_avg = exp_avg[world_rank::world_size].clone().to(local_param.device)
            local_exp_avg_sq = exp_avg_sq[world_rank::world_size].clone().to(local_param.device)
        else:
            local_exp_avg = None
            local_exp_avg_sq = None
        
        # Build PyTorch state dict
        local_param_id = id(local_param)
        local_state = {
            'step': step,
        }
        if local_exp_avg is not None:
            local_state['exp_avg'] = local_exp_avg
            local_state['exp_avg_sq'] = local_exp_avg_sq
        
        # Update param_groups to reference correct param
        param_groups_copy = [pg.copy() for pg in param_groups]
        param_groups_copy[0]['params'] = [local_param_id]
        
        return {
            'state': {local_param_id: local_state},
            'param_groups': param_groups_copy
        }
    
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
    def is_main_rank(self) -> bool:
        return self.world_rank == SplatTrainingState.LEADER_RANK
    
    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians on this rank"""
        return len(self.params['means'])
    
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
        
    def _debug_distributed_consistency(self) -> None:
        """Verify that distributed state is consistent across ranks (for debugging)."""
        if not self.is_distributed:
            return
        
        # Check num_gaussians matches across ranks
        local_count = torch.tensor([self.num_gaussians], dtype=torch.long, device=self.device)
        all_counts = [torch.zeros_like(local_count) for _ in range(self.world_size)]
        dist.all_gather(all_counts, local_count)
        
        if self.is_main_rank:
            total = sum(t.item() for t in all_counts)
            print(f"Total Gaussians across {self.world_size} ranks: {total}")
            for rank, count in enumerate(all_counts):
                print(f"  Rank {rank}: {count.item()} Gaussians")