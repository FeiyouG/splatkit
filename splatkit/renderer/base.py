from abc import ABC, abstractmethod
from typing import Generic, Literal, Tuple

from torch import Tensor

from ..modules import SplatBaseFrameT
from ..splat.training_state import SplatTrainingState
from ..modules.base import SplatBaseModule

class SplatRenderer(
    SplatBaseModule[SplatBaseFrameT], 
    Generic[SplatBaseFrameT], 
    ABC
):
    """Abstract base class for all renderers.
    
    Encapsulates rendering configuration and provides a simple render() interface.
    All configuration parameters are private and set via constructor or setters.
    """
    
    @abstractmethod
    def render(
        self,
        splat_state: SplatTrainingState,
        cam_to_worlds: Tensor, # (..., 4, 4)
        Ks: Tensor, # (..., 3, 3)
        camera_model: str,
        width: int,
        height: int,
        backgrounds: Tensor | None = None,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        sh_degree: int | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> Tuple[Tensor, SplatBaseFrameT]:
        """
        Render splats from camera viewpoints.
        
        Args:
            splat_state: SplatTrainingState containing gaussian parameters
            cam_to_worlds: Camera-to-world matrices [B, 4, 4]
            Ks: Camera intrinsics [B, 3, 3]
            camera_model: Camera model
            width: Image width
            height: Image height
            sh_degree: SH degree to use (None = use splat_state.sh_degree)
            world_rank: World rank
            world_size: World size
        
        Returns:
            renders: Rendered images [B, H, W, C]
            render_meta: Dict containing:
                - alphas: [B, H, W, 1]
                - depths: [B, H, W, 1] (if rendered)
                - gaussian_ids: Visible gaussian indices
                - radii: 2D radii
                - means2d: Projected 2D positions
                - ... other intermediate results
        """
        pass
