from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, TypedDict, TypeVar, Generic, Literal
from torch import Tensor

from ..splat.training_state import SplatTrainingState

@dataclass(frozen=True)
class SplatRendererOutput:
    """Metadata for rendered images.
    """
    renders: Tensor  # (..., H, W, 3)
    alphas: Tensor  # (..., H, W, 1)
    radii: Tensor  # (..., H, W, 1)
    depths: Tensor # (..., H, W, 1)
    n_cameras: int
    n_batches: int
    width: int
    height: int

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def to_dict(self) -> dict:
        return dict(self.__dict__)

SplatRendererOutputType = TypeVar("SplatRendererOutputType", bound="SplatRendererOutput")

class SplatRenderer(ABC, Generic[SplatRendererOutputType]):
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
    ) -> Tuple[Tensor, SplatRendererOutputType]:
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
