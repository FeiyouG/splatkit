import torch
from typing import NotRequired, TypedDict

class DataSetItem(TypedDict):
    id: int
    image_name: str

    K: torch.Tensor                # (3, 3)
    cam_to_world: torch.Tensor     # (4, 4)
    image: torch.Tensor            # (H, W, 3), float32
    
    mask: NotRequired[torch.Tensor]        # (H, W), bool
    points: NotRequired[torch.Tensor]      # (M, 2)
    depths: NotRequired[torch.Tensor]      # (M,)