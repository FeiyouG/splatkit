import torch
from dataclasses import dataclass

@dataclass(slots=True)
class DataSetItem:
    """
    A item in a dataset.
    """

    id: int
    K: torch.Tensor                # (3, 3)
    cam_to_world: torch.Tensor     # (4, 4)
    image: torch.Tensor            # (H, W, 3), float32
    image_name: str

    mask: torch.Tensor | None = None        # (H, W), bool
    points: torch.Tensor | None = None      # (M, 2)
    depths: torch.Tensor | None = None      # (M,)