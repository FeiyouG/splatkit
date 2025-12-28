import torch
import torch.distributed as dist
from typing import Dict, Any

def distribute_metadata(
    metadata: Dict[str, Any] | None,
    world_rank: int,
    world_size: int,
    leader_rank: int = 0,
) -> Dict[str, Any] | None:
    """
    Broadcast metadata from leader rank to all ranks.

    Args:
        metadata: Metadata to distribute (only leader rank provides this)
        world_rank: Current rank
        world_size: Total number of ranks
        leader_rank: Rank to broadcast from

    Returns:
        Broadcasted metadata
    """
    if world_size == 1:
        return metadata
    
    obj_list = [metadata]
    dist.broadcast_object_list(obj_list, src=leader_rank)
    return obj_list[0]

def distribute_tensor(
        tensor: torch.Tensor | None,
        world_rank: int,
        world_size: int,
        device: str = 'cuda',
        leader_rank: int = 0,
        striped: bool = True,
    ) -> torch.Tensor:
        """
        Distribute a tensor from leader rank to all ranks using striped distribution.
        
        Args:
            tensor: Tensor to distribute (only leader rank provides this)
            world_rank: Current rank
            world_size: Total number of ranks
            device: Target device
            leader_rank: Rank to broadcast from
            striped: Whether to distribute the tensor in a striped manner; if false, the same tensor is distributed to all ranks
        
        Returns:
            Local slice of tensor on target device
        """
        common_device = torch.device(device)

        if world_size == 1:
            if tensor is None:
                raise ValueError("Tensor is missing but required for single GPU")
            return tensor.clone().to(common_device)

        if world_rank == leader_rank:
            if tensor is None:
                raise ValueError(f"Tensor is missing but required for leader rank {world_rank}")

            full_tensor = tensor.to(common_device)
            local_tensor = full_tensor[leader_rank::world_size].contiguous() if striped else full_tensor

            for r in range(world_size):
                if r == leader_rank:
                    continue

                r_tensor = full_tensor[r::world_size].contiguous() if striped else full_tensor
                shape = torch.tensor(r_tensor.shape, dtype=torch.long, device=common_device)
                n_dim = torch.tensor([r_tensor.ndim], dtype=torch.long, device=common_device)
                dtype_code = torch.tensor([_dtype_to_code(r_tensor.dtype)], dtype=torch.long, device=common_device)

                dist.send(dtype_code, dst=r)
                dist.send(n_dim, dst=r)
                dist.send(shape, dst=r)
                dist.send(r_tensor, dst=r)
        else:
            if tensor is not None:
                raise ValueError(f"Tensor is provided but not expected for non-leader rank {world_rank}")

            dtype_code = torch.empty(1, dtype=torch.long, device=common_device)
            dist.recv(dtype_code, src=leader_rank)
            dtype = _code_to_dtype(dtype_code.item())

            n_dim = torch.empty(1, dtype=torch.long, device=common_device)
            dist.recv(n_dim, src=leader_rank)

            shape = torch.empty(n_dim.item(), dtype=torch.long, device=common_device)
            dist.recv(shape, src=leader_rank)

            local_tensor = torch.empty(tuple(shape.tolist()), dtype=dtype, device=common_device)
            dist.recv(local_tensor, src=leader_rank)

        return local_tensor.clone()

def gather_tensor(
    tensor: torch.Tensor,
    world_rank: int,
    world_size: int,
    device: str = 'cuda',
    leader_rank: int = 0,
) -> torch.Tensor | None:
    """
    Gather tensors from all ranks to leader rank using concatenation.
    Shapes may differ along dim 0; other dims must match.
    
    Args:
        tensor: Tensor to gather (only leader rank provides this)
        world_rank: Current rank
        world_size: Total number of ranks
        leader_rank: Rank to gather to

    Returns:
        Concatenated tensor on leader rank (None for other ranks)
    """
    common_device = torch.device(device)
                
    if world_size == 1:
        return tensor.clone().to(common_device)
    
    tensor = tensor.to(common_device)
    if world_rank == leader_rank:
        all_tensors = []
        for r in range(world_size):
            if r == leader_rank:
                r_tensor = tensor
            else:
                dtype_code = torch.empty(1, dtype=torch.long, device=common_device)
                dist.recv(dtype_code, src=r)
                dtype = _code_to_dtype(dtype_code.item())
                if dtype != tensor.dtype:
                    raise ValueError(f"Inconsistent dtype: {dtype} != {tensor.dtype}")

                n_dim = torch.empty(1, dtype=torch.long, device=common_device)
                dist.recv(n_dim, src=r)
                if n_dim.item() != tensor.ndim:
                    raise ValueError(f"Inconsistent n_dim: {n_dim.item()} != {tensor.ndim}")

                shape = torch.empty(n_dim.item(), dtype=torch.long, device=common_device)
                dist.recv(shape, src=r)
                r_tensor = torch.empty(tuple(shape.tolist()), dtype=dtype, device=common_device)
                dist.recv(r_tensor, src=r)

            all_tensors.append(r_tensor)

        return torch.cat(all_tensors, dim=0)
    else:
        shape = torch.tensor(tensor.shape, dtype=torch.long, device=common_device)
        dtype_code = torch.tensor([_dtype_to_code(tensor.dtype)], dtype=torch.long, device=common_device)
        n_dim = torch.tensor([tensor.ndim], dtype=torch.long, device=common_device)

        dist.send(dtype_code, dst=leader_rank)
        dist.send(n_dim, dst=leader_rank)
        dist.send(shape, dst=leader_rank)
        dist.send(tensor, dst=leader_rank)
        return None


DETYPES = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]

def _dtype_to_code(dtype: torch.dtype) -> int:
    try:
        return DETYPES.index(dtype)
    except ValueError:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes: {DETYPES}")

def _code_to_dtype(code: int) -> torch.dtype:
    return DETYPES[code]