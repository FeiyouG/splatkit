from dataclasses import dataclass
from typing import Literal, Sequence, Tuple
import torch
from torch import Tensor

from gsplat.rendering import rasterization

from ..modules import SplatBaseModule, SplatRenderPayload
from ..splat.training_state import SplatTrainingState
from .base import SplatRenderer

@dataclass(frozen=True)
class Splat3dgsRenderPayload(SplatRenderPayload):
    """
    Metadata for 3D Gaussian Splatting rendered images.
    """
    means2d: Tensor # (..., H, W, 2)
    conics: Tensor # (..., H, W, 1)
    opacities: Tensor # (..., H, W, 1)
    gaussian_ids: Tensor | None = None # Visible gaussian indices for densification

class Splat3DGSRenderer(SplatRenderer[Splat3dgsRenderPayload]):
    """3D Gaussian Splatting renderer.
    """

    _camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"]
    _near_plane: float
    _far_plane: float
    _tile_size: int
    _sparse_grad: bool
    _radius_clip: float
    _antialiased: bool
    _eps2d: float
    _absgrad: bool
    _channel_chunk: int
    
    def __init__(
        self,
        # Camera settings
        camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        
        # Performance settings
        tile_size: int = 16,
        sparse_grad: bool = False,
        radius_clip: float = 0.0,
        
        # Quality settings
        antialiased: bool = False,
        eps2d: float = 0.3,
        
        # Advanced features
        absgrad: bool = False,
        channel_chunk: int = 32,
    ):
        """
        Initialize renderer with fixed configuration.
        
        Args:
            camera_model: Camera projection model
            near_plane: Near clipping plane distance
            far_plane: Far clipping plane distance
            packed: Use packed mode (memory efficient) but tradeoff for speed
            tile_size: Tile size for rasterization (typically 16)
            sparse_grad: Use sparse gradients (experimental)
            radius_clip: Skip gaussians with 2D radius <= this
            antialiased: Use Mip-Splatting antialiasing
            eps2d: Minimum 2D covariance eigenvalue, added to prevent projected GS to be too small
            absgrad: Compute absolute gradients (for AbsGS). If true then can be access from render_meta["means2d"].absgrad
            channel_chunk: Render in chunks if channels > this
        """
        
        # Store all as private attributes
        self._camera_model = camera_model
        self._near_plane = near_plane
        self._far_plane = far_plane
        self._tile_size = tile_size
        self._sparse_grad = sparse_grad
        self._radius_clip = radius_clip
        self._antialiased = antialiased
        self._eps2d = eps2d
        self._absgrad = absgrad
        self._channel_chunk = channel_chunk
    
    def render(
        self,
        splat_state: SplatTrainingState,
        cam_to_worlds: Tensor,
        Ks: Tensor,
        height: int,
        width: int,
        camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] | None = None,
        backgrounds: Tensor | None = None,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        sh_degree: int | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> Tuple[Tensor, Splat3dgsRenderPayload]:
        """
        Render splats from camera viewpoints.
        
        Args:
            splat_state: SplatTrainingState with gaussian parameters
            datasetItem: DataSetItem containing camera intrinsics, camera-to-world matrices, image, mask, points, depths
            sh_degree: SH degree to use (None = use splat_state.sh_degree)
            render_mode: What to render ("RGB", "D", "ED", "RGB+D", "RGB+ED")
            backgrounds: Background colors [B, C] (None = black)
            masks: Optional masks [B, H, W] to zero out regions
            world_rank: World rank
            world_size: World size
            **kwargs: Additional args (distortion coeffs, rolling shutter, etc.)
        
        Returns:
            renders: Rendered images [B, H, W, C]
            render_meta: Dict with alphas and intermediate results
        """
        # Extract gaussian parameters from SplatTrainingState
        means = splat_state.params["means"]
        quats = splat_state.params["quats"]
        scales = torch.exp(splat_state.params["scales"])
        opacities = torch.sigmoid(splat_state.params["opacities"])
        colors = splat_state.colors  # Property handles SH vs features

        distributed = world_size > 1
        
        # Use splat_state's sh_degree if not specified
        if sh_degree is None:
            sh_degree = splat_state.sh_degree

        if camera_model is None:
            camera_model = self._camera_model
        
        # Call gsplat's rasterization with private config
        renders, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(cam_to_worlds),
            Ks=Ks,
            width=width,
            height=height,
            near_plane=self._near_plane,
            far_plane=self._far_plane,
            radius_clip=self._radius_clip,
            eps2d=self._eps2d,
            sh_degree=sh_degree,
            tile_size=self._tile_size,
            backgrounds=backgrounds,
            render_mode=render_mode,
            sparse_grad=self._sparse_grad,
            absgrad=self._absgrad,
            rasterize_mode="antialiased" if self._antialiased else "classic",
            channel_chunk=self._channel_chunk,
            distributed=distributed,
            camera_model=camera_model,
            packed=False, # Never pack
        )
        
        # Build render_meta with alphas included
        outputs = Splat3dgsRenderPayload(
            renders=renders,
            alphas=alphas,
            n_cameras=info["n_cameras"],
            n_batches=info["n_batches"],
            radii=info["radii"],
            means2d=info["means2d"],
            depths=info["depths"],
            conics=info["conics"],
            opacities=info["opacities"],
            width=info["width"],
            height=info["height"],
            # gaussian_ids=info.get("gaussian_ids", None),  # Include gaussian_ids for densification
        )
        return renders, outputs


    
    # Setters for commonly adjusted parameters
    def set_antialiased(self, antialiased: bool):
        """Enable or disable antialiasing."""
        self._antialiased = antialiased
    
    def set_radius_clip(self, radius_clip: float):
        """Set radius clipping threshold."""
        self._radius_clip = radius_clip
    
    def set_absgrad(self, absgrad: bool):
        """Enable or disable absolute gradient computation."""
        self._absgrad = absgrad