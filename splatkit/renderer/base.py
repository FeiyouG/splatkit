from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Literal, Sequence, Tuple, Type

import numpy as np
from torch import Tensor

from ..modules import SplatRenderPayloadT
from ..splat.training_state import SplatTrainingState
from ..modules.base import SplatBaseModule

if TYPE_CHECKING:
    from ..logger import SplatLogger
    from nerfview import CameraState

class SplatRenderer(
    SplatBaseModule[SplatRenderPayloadT], 
    Generic[SplatRenderPayloadT], 
    ABC
):
    """Abstract base class for all renderers.
    
    Encapsulates rendering configuration and provides a simple render() interface.
    All configuration parameters are private and set via constructor or setters.
    """

    def __init__(self):
        super().__init__()
    
    def on_setup(
        self,
        logger: "SplatLogger",
        renderer: SplatBaseModule[SplatRenderPayloadT],
        data_provider: SplatBaseModule[SplatRenderPayloadT],
        loss_fn: SplatBaseModule[SplatRenderPayloadT],
        densification: SplatBaseModule[SplatRenderPayloadT],
        modules: Sequence[SplatBaseModule[SplatRenderPayloadT]], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        logger.info(f"Initialized renderer: {self.__class__.__name__}", module=self.module_name)
    
    @abstractmethod
    def render(
        self,
        splat_state: SplatTrainingState,
        cam_to_worlds: Tensor, # (..., 4, 4)
        Ks: Tensor, # (..., 3, 3)
        height: int,
        width: int,
        camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] | None = None,
        backgrounds: Tensor | None = None,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        sh_degree: int | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> Tuple[Tensor, SplatRenderPayloadT]:
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

    def get_visualization_options(self) -> Tuple[str, ...]:
        """
        Return available visualization options for this renderer.
        
        These are display modes for the interactive viewer, separate from
        training render modes. Each renderer defines what visualization
        modes make sense for its outputs.
        
        Returns:
            Tuple of visualization mode names (e.g., "rgb", "depth", "normal", "alpha")
        """
        return (
            "rgb",
            "alpha",
        )
    
    @abstractmethod
    def visualize(
        self,
        splat_state: SplatTrainingState,
        camera_state: "CameraState",
        width: int,
        height: int,
        visualization_mode: str = "rgb",
        sh_degree: int | None = None,
        backgrounds: Tensor | None = None,
        camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
        # Visualization-specific parameters
        normalize_nearfar: bool = False,
        near_plane: float = 1e-2,
        far_plane: float = 1e2,
        inverse: bool = False,
        colormap: str = "turbo",
    ) -> Tuple[np.ndarray, int]:
        """
        Generate visualization for interactive viewer (VISUALIZATION ONLY).
        
        This method is separate from render() which is used for training.
        It handles viewer-specific processing like depth normalization,
        colormap application, and format conversion for display.
        
        IMPORTANT: Implementations should use @torch.no_grad() decorator since
        visualization never needs gradients.
        
        Key Differences from render():
        - No gradients (use @torch.no_grad() decorator)
        - Returns numpy array instead of tensors
        - Applies human-readable colormaps
        - Single RGB output instead of multi-channel tensors
        - Used by viewer only, not training loop
        
        Args:
            splat_state: SplatTrainingState containing gaussian parameters
            camera_state: Camera state from viewer
            width: Image width
            height: Image height
            visualization_mode: Visualization option (from get_visualization_options())
            sh_degree: SH degree to use
            backgrounds: Background color
            camera_model: Camera model
            normalize_nearfar: Normalize depth with near/far planes
            near_plane: Near plane for depth normalization
            far_plane: Far plane for depth normalization
            inverse: Invert depth/alpha values
            colormap: Colormap name for depth/alpha visualization
        
        Returns:
            output: RGB image [H, W, 3] in range [0, 1] as numpy array
            rendered_gaussians: Number of gaussians rendered (visible)
        """
        pass