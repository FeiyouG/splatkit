# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import math
# from typing import Dict, Any, Optional, Tuple, List
# from dataclasses import dataclass
# from pathlib import Path

# from .model import SplatModel
# from ..utils.sh_utils import sh_to_K


# @dataclass
# class SplatTrainingState:
#     """
#     SplatTrainingState: Mutable training state for 3D Gaussian Splatting

#     Distributed Training Strategy:
#     ------------------------------
#     Parameters are distributed across ranks using striped distribution.
#     Gaussians can be added/removed independently on each rank during training.
#     When gathering, all Gaussians are concatenated (order doesn't matter).
#     """

#     REQUIRED_PARAMS = frozenset([
#         'means', 'scales', 'quats', 'opacities', 'sh0', 'shN',
#     ])
#     LEADER_RANK = 0
    
#     params: nn.ParameterDict
#     optimizers: Dict[str, torch.optim.Optimizer]
#     sh_degree: int
#     device: str
#     world_rank: int = 0
#     world_size: int = 1
    
#     def __post_init__(self):
#         """Validate state after initialization"""
#         self.__validate__()
    
#     def __validate__(self):
#         """Validate consistency between parameters and optimizers."""
#         # Check parameters are present
#         keys = self.params.keys()
#         missing = SplatTrainingState.REQUIRED_PARAMS - keys
#         extra = keys - SplatTrainingState.REQUIRED_PARAMS
#         if len(missing) > 0:
#             raise ValueError(f"Missing required parameters: {missing}")
#         if len(extra) > 0:
#             raise ValueError(f"Unknown parameters: {extra}")

#         # Check trainable parameters match optimizer keys
#         trainable_params = set(
#             [name for name, param in self.params.items() if param.requires_grad]
#         )
#         optimizer_keys = set(self.optimizers.keys())

#         if trainable_params != optimizer_keys:
#             raise ValueError(f"Trainable parameters and optimizers must have the same keys, "
#                              f"but got trainable={trainable_params}, optimizers={optimizer_keys}")
        
#         # Check each optimizer has exactly one param_group
#         for name, optimizer in self.optimizers.items():
#             if len(optimizer.param_groups) != 1:
#                 raise ValueError(f"Optimizer '{name}' must have exactly one param_group, "
#                                  f"but got {len(optimizer.param_groups)}")
        
#         # Check parameter shapes are consistent
#         N = len(self.params['means'])
#         if self.params['means'].shape != (N, 3):
#             raise ValueError(f"Parameter 'means' shape mismatch: {self.params['means'].shape} != ({N}, 3)")
#         if self.params['scales'].shape != (N, 3):
#             raise ValueError(f"Parameter 'scales' shape mismatch: {self.params['scales'].shape} != ({N}, 3)")
#         if self.params['quats'].shape != (N, 4):
#             raise ValueError(f"Parameter 'quats' shape mismatch: {self.params['quats'].shape} != ({N}, 4)")
#         if self.params['opacities'].shape != (N,):
#             raise ValueError(f"Parameter 'opacities' shape mismatch: {self.params['opacities'].shape} != ({N},)")
#         if self.params['sh0'].shape != (N, 1, 3):
#             raise ValueError(f"Parameter 'sh0' shape mismatch: {self.params['sh0'].shape} != ({N}, 1, 3)")

#         # Validate SH degree consistency    
#         K = sh_to_K(self.sh_degree)
#         if self.params['shN'].shape != (N, K - 1, 3):
#             raise ValueError(f"shN shape mismatch: {self.params['shN'].shape} != ({N}, {K - 1}, 3)")

#     # =========================================================================
#     # DISTRIBUTE PRIMITIVES (Rank 0 → All Ranks)
#     # =========================================================================

#     @staticmethod
#     def _distribute_metadata(
#         metadata: Dict[str, Any] | None,
#         world_rank: int,
#         world_size: int,
#     ) -> Dict[str, Any] | None:
#         """
#         Broadcast metadata from rank 0 to all ranks.
#         """
#         if metadata is None or len(metadata) == 0:
#             return None

#         if world_size == 1:
#             return metadata
        
#         obj_list = [metadata]
#         dist.broadcast_object_list(obj_list, src=SplatTrainingState.LEADER_RANK)
#         return obj_list[0]
    
#     @staticmethod
#     def _distribute_tensor(
#         tensor: Optional[torch.Tensor],
#         world_rank: int,
#         world_size: int,
#         device: str,
#     ) -> torch.Tensor:
#         """
#         Distribute a tensor from rank 0 to all ranks using striped distribution.
        
#         Args:
#             tensor: Tensor to distribute (only rank 0 provides this)
#             world_rank: Current rank
#             world_size: Total number of ranks
#             device: Target device
        
#         Returns:
#             Local slice of tensor on target device
#         """
#         if world_size == 1:
#             if tensor is None:
#                 raise ValueError("Tenosr is missing but required for single GPU")
#             return tensor.clone().to(device)

#         if world_rank == SplatTrainingState.LEADER_RANK:
#             if tensor is None:
#                 raise ValueError(f"Tensor is missing but required for leader rank {world_rank}")
#         else:
#             if tensor is not None:
#                 raise ValueError(f"Tensor is provided but not expected for non-leader rank {world_rank}")
        
#         # Broadcast tensor from rank 0
#         tensor_list = [tensor]
#         dist.broadcast_object_list(tensor_list, src=SplatTrainingState.LEADER_RANK)
#         full_tensor = tensor_list[0]
        
#         # Each rank takes striped slice: [rank, rank+K, rank+2K, ...]
#         local_tensor = full_tensor[world_rank::world_size].clone().to(device)
        
#         return local_tensor
    
#     @staticmethod
#     def _distribute_params(
#         params_dict: Dict[str, torch.Tensor] | None,
#         world_rank: int,
#         world_size: int,
#         device: str,
#     ) -> nn.ParameterDict:
#         """
#         Distribute parameter tensors from rank 0 to all ranks.
        
#         Args:
#             params_dict: Dict of parameter tensors (only rank 0 provides)
#             world_rank: Current rank
#             world_size: Total ranks
#             device: Target device
        
#         Returns:
#             ParameterDict with distributed parameters
#         """
#         if world_size == 1:
#             if params_dict is None:
#                 raise ValueError("Params dict is missing but required for single GPU")
#             return nn.ParameterDict({
#                 k: nn.Parameter(v.clone().to(device)) for k, v in params_dict.items()
#             })
            
#         # Gather parameter names from all ranks so each rank knows how many parameters to expect
#         if world_rank == SplatTrainingState.LEADER_RANK:
#             if params_dict is None:
#                 raise ValueError("Params dict is missing but required for leader rank")
#             param_names = list(params_dict.keys())
#         else:
#             param_names = None

#         data = SplatTrainingState._distribute_metadata(
#             {'param_names': param_names},
#             world_rank,
#             world_size,
#         )
#         param_names = data['param_names']

#         # Distribute each parameter tensor
#         local_params = {}
#         for name in param_names:
#             tensor = params_dict[name] if params_dict is not None else None
#             local_tensor = SplatTrainingState._distribute_tensor(
#                 tensor, world_rank, world_size, device
#             )
#             local_params[name] = nn.Parameter(local_tensor)
        
#         return nn.ParameterDict(local_params)
    
#     @staticmethod
#     def _distribute_optimizer_state_dict(
#         optimizer_states: Optional[Dict[str, Dict[str, Any]]],
#         world_rank: int,
#         world_size: int,
#         device: str,
#     ) -> Optional[Dict[str, Dict[str, Any]]]:
#         """
#         Distribute optimizer state dicts from rank 0 to all ranks.
        
#         Args:
#             optimizer_states: Dict mapping param_name to optimizer state
#                             Format: {'means': {'step': 100, 'exp_avg': tensor, ...}}
#                             Only rank 0 provides this
#             world_rank: Current rank
#             world_size: Total ranks
#             device: Target device for tensors
        
#         Returns:
#             Dict of distributed optimizer states (same format as input)
#             Returns None if no optimizer states to distribute
#         """
#         if optimizer_states is None or len(optimizer_states) == 0:
#             return None
        
#         # Broadcast parameter names so all ranks know what to expect
#         if world_rank == SplatTrainingState.LEADER_RANK:
#             param_names = list(optimizer_states.keys())
#         else:
#             param_names = None
        
#         data = SplatTrainingState._distribute_metadata(
#             {'param_names': param_names},
#             world_rank,
#             world_size,
#         )
#         param_names = data['param_names']
        
#         # Multi-GPU: distribute optimizer states
#         distributed_states = {}
        
#         for param_name in param_names:
#             # Get full state on rank 0
#             # NOTE: Assuming using Adam optimizer for now
#             if world_rank == SplatTrainingState.LEADER_RANK:
#                 full_state = optimizer_states[param_name]
#                 step = full_state['step']
#                 param_groups = full_state['param_groups']
#                 exp_avg_full = full_state['exp_avg']
#                 exp_avg_sq_full = full_state['exp_avg_sq']
#             else:
#                 step = None
#                 param_groups = None
#                 exp_avg_full = None
#                 exp_avg_sq_full = None
            
#             # Broadcast scalar metadata (step, param_groups)
#             data = SplatTrainingState._distribute_metadata(
#                 {'step': step, 'param_groups': param_groups},
#                 world_rank,
#                 world_size,
#             )
#             step = data['step']
#             param_groups = data['param_groups']
            
#             # Distribute tensor state (exp_avg, exp_avg_sq)
#             exp_avg_local = SplatTrainingState._distribute_tensor(
#                 exp_avg_full, world_rank, world_size, device
#             )
#             exp_avg_sq_local = SplatTrainingState._distribute_tensor(
#                 exp_avg_sq_full, world_rank, world_size, device
#             )
            
#             # Store distributed state in same format
#             distributed_states[param_name] = {
#                 'step': step,
#                 'exp_avg': exp_avg_local,
#                 'exp_avg_sq': exp_avg_sq_local,
#                 'param_groups': param_groups,
#             }
        
#         return distributed_states

#     @classmethod
#     def _distribute(
#         cls,
#         params_dict: Dict[str, torch.Tensor] | None,
#         optimizer_states: Dict[str, Dict[str, Any]] | None,
#         sh_degree: int | None,
#         device: str,
#         world_rank: int,
#         world_size: int,
#         lr_means: float = 1.6e-4,
#         lr_scales: float = 5e-3,
#         lr_quats: float = 1e-3,
#         lr_opacities: float = 5e-2,
#         lr_sh0: float = 2.5e-3,
#         lr_shN: float = 2.5e-3 / 20,
#         scene_scale: float = 1.0,
#         batch_size: int = 1,
#     ) -> 'SplatTrainingState':
#         """
#         Distribute parameters, optimizer states, and metadata to create training state.
        
#         This is the main distribution primitive used by from_splat_model and from_ckpt.
        
#         Args:
#             params_dict: Dict of parameter tensors (only rank 0 provides)
#             optimizer_states: Dict of optimizer states (only rank 0 provides)
#             sh_degree: SH degree (only rank 0 provides)
#             device: Target device
#             world_rank: Current rank
#             world_size: Total ranks
#             lr_*: Learning rates for creating optimizers
#             scene_scale: Scene scale
#             batch_size: Batch size
        
#         Returns:
#             SplatTrainingState with distributed parameters and optimizers
#         """

#         if world_size == 1:
#             if sh_degree is None:
#                 raise ValueError("SH degree is missing but required for single GPU")
#         else:
#             data = cls._distribute_metadata(
#                 {'sh_degree': sh_degree},
#                 world_rank,
#                 world_size,
#             )
#             sh_degree = data['sh_degree']
        
#         # Distribute parameters
#         params = cls._distribute_params(params_dict, world_rank, world_size, device)
        
#         # Distribute optimizer states (just the state dicts, not optimizers yet)
#         distributed_opt_states = cls._distribute_optimizer_state_dict(
#             optimizer_states, world_rank, world_size, device,
#         )
        
#         # Create optimizers for all parameters
#         optimizers = cls._create_optimizers(
#             params=params,
#             lr_means=lr_means,
#             lr_scales=lr_scales,
#             lr_quats=lr_quats,
#             lr_opacities=lr_opacities,
#             lr_sh0=lr_sh0,
#             lr_shN=lr_shN,
#             scene_scale=scene_scale,
#             batch_size=batch_size,
#             world_size=world_size,
#         )
        
#         # Load distributed states into optimizers (if available)
#         if distributed_opt_states is not None:
#             for param_name, opt_state in distributed_opt_states.items():
#                 optimizer = optimizers[param_name]
#                 param = params[param_name]
#                 param_id = id(param)
                
#                 # Convert our custom format to PyTorch state_dict format
#                 pytorch_state_dict = {
#                     'state': {
#                         param_id: {
#                             'step': opt_state['step'],
#                             'exp_avg': opt_state['exp_avg'],
#                             'exp_avg_sq': opt_state['exp_avg_sq'],
#                         }
#                     },
#                     'param_groups': [
#                         {
#                             'params': [param_id],
#                             **opt_state['param_groups'][0]  # Merge hyperparameters
#                         }
#                     ]
#                 }
                
#                 optimizer.load_state_dict(pytorch_state_dict)
        
#         return cls(
#             params=params,
#             optimizers=optimizers,
#             sh_degree=sh_degree,
#             device=device,
#             world_rank=world_rank,
#             world_size=world_size,
#         )

#     @staticmethod
#     def _gather_tensor(
#         tensor: torch.Tensor,
#         world_rank: int,
#         world_size: int,
#     ) -> Optional[torch.Tensor]:
#         """
#         Gather tensors from all ranks to leader rank using concatenation.
        
#         Note: Order doesn't matter for Gaussians, so we simply concatenate.
        
#         Args:
#             tensor: Local tensor from this rank
#             world_rank: Current rank
#             world_size: Total number of ranks
        
#         Returns:
#             Concatenated tensor on leader rank (None for other ranks)
#         """
#         if world_size == 1:
#             return tensor.cpu()  # Move to CPU for saving
        
#         # Gather all tensors to leader rank
#         gathered_tensors = [None] * world_size
#         dist.gather_object(gathered_tensors, tensor.cpu(), dst=SplatTrainingState.LEADER_RANK)
#         return gathered_tensors[0]

#     @staticmethod
#     def _gather_params(
#         params: nn.ParameterDict,
#         world_rank: int,
#         world_size: int,
#     ) -> Optional[Dict[str, torch.Tensor]]:
#         """
#         Gather parameters from all ranks to leader rank.
        
#         Args:
#             params: Local ParameterDict from this rank
#             world_rank: Current rank
#             world_size: Total number of ranks
        
#         Returns:
#             Dict of gathered tensors on leader rank (None for other ranks)
#         """
#         if world_size == 1:
#             return {k: v.data.cpu() for k, v in params.items()}
        
#         gathered_params = {} if world_rank == SplatTrainingState.LEADER_RANK else None
        
#         for param_name, param in params.items():
#             gathered_tensor = SplatTrainingState._gather_tensor(
#                 param.data, world_rank, world_size
#             )
            
#             if world_rank == SplatTrainingState.LEADER_RANK:
#                 gathered_params[param_name] = gathered_tensor
        
#         return gathered_params

#     @staticmethod
#     def _gather_optimizer_state_dict(
#         optimizers: Dict[str, torch.optim.Optimizer],
#         world_rank: int,
#         world_size: int,
#     ) -> Optional[Dict[str, Dict[str, Any]]]:
#         """
#         Gather optimizer states from all ranks to leader rank.
        
#         Validates that all ranks have consistent step counts.
        
#         Args:
#             optimizers: Dict of optimizers from this rank
#             world_rank: Current rank
#             world_size: Total number of ranks
        
#         Returns:
#             Dict of gathered optimizer states on leader rank (None for other ranks)
#         """
#         if world_size == 1:
#             # Single GPU: extract state in custom format
#             result = {}
#             for name, optimizer in optimizers.items():
#                 state_dict = optimizer.state_dict()
#                 param_state = list(state_dict['state'].values())[0]
                
#                 result[name] = {
#                     'step': param_state.get('step', 0),
#                     'exp_avg': param_state.get('exp_avg'),
#                     'exp_avg_sq': param_state.get('exp_avg_sq'),
#                     'param_groups': state_dict['param_groups'],
#                 }
#             return result
        
#         # Gather states from all ranks to leader rank
#         gathered_states = {} if world_rank == SplatTrainingState.LEADER_RANK else None
        
#         for param_name, optimizer in optimizers.items():
#             state_dict = optimizer.state_dict()
#             param_state = list(state_dict['state'].values())[0]
            
#             # Extract local state
#             step = param_state.get('step', 0)
#             exp_avg = param_state.get('exp_avg')
#             exp_avg_sq = param_state.get('exp_avg_sq')
            
#             # Validate step consistency across ranks
#             step_list = [None] * world_size
#             dist.all_gather_object(step_list, step)
            
#             if world_rank == SplatTrainingState.LEADER_RANK:
#                 if not all(s == step_list[0] for s in step_list):
#                     raise ValueError(
#                         f"Inconsistent step counts for '{param_name}': {step_list}. "
#                         f"All ranks should have the same step count."
#                     )
            
#             # Gather exp_avg and exp_avg_sq tensors
#             exp_avg_gathered = SplatTrainingState._gather_tensor(
#                 exp_avg, world_rank, world_size
#             )
#             exp_avg_sq_gathered = SplatTrainingState._gather_tensor(
#                 exp_avg_sq, world_rank, world_size
#             )
            
#             if world_rank == SplatTrainingState.LEADER_RANK:
#                 gathered_states[param_name] = {
#                     'step': step,
#                     'exp_avg': exp_avg_gathered,
#                     'exp_avg_sq': exp_avg_sq_gathered,
#                     'param_groups': state_dict['param_groups'],
#                 }
        
#         return gathered_states

#     @classmethod
#     def _gather(
#         cls,
#         training_state: 'SplatTrainingState',
#     ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]], Dict[str, Any]]]:
#         """
#         Gather all state from distributed ranks to rank 0.
        
#         Args:
#             training_state: Current training state
        
#         Returns:
#             Tuple of (params_dict, optimizer_states, metadata) on rank 0
#             Returns None for other ranks
#         """
#         world_rank = training_state.world_rank
#         world_size = training_state.world_size
        
#         # Gather parameters
#         params_dict = cls._gather_params(
#             training_state.params, world_rank, world_size
#         )
        
#         # Gather optimizer states
#         optimizer_states = cls._gather_optimizer_state_dict(
#             training_state.optimizers, world_rank, world_size
#         )
        
#         # Gather metadata (already on all ranks, just return)
#         if world_rank == SplatTrainingState.LEADER_RANK:
#             metadata = {
#                 'sh_degree': training_state.sh_degree,
#                 'device': training_state.device,
#                 'world_size': world_size,
#             }
#             return params_dict, optimizer_states, metadata
#         else:
#             return None