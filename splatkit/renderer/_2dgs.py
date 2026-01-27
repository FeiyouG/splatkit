from dataclasses import dataclass
import math
from typing import Literal, Sequence, Tuple
import torch
from torch import Tensor

from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper

from ..modules import SplatBaseModule, SplatRenderPayload
from ..splat.training_state import SplatTrainingState
from .base import SplatRenderer

@dataclass(frozen=True)
class Splat2dgsRenderPayload(SplatRenderPayload):
    """
    Metadata for 2D Gaussian Splatting rendered images.
    """
    normals: Tensor  # (..., H, W, 3) - rendered normals
    normals_from_depth: Tensor  # (..., H, W, 3) - normals derived from depth
    render_distort: Tensor  # (..., H, W, 1) - distortion map
    depths_median: Tensor  # (..., H, W, 1) - median depth
    gradient_2dgs: Tensor  # Gradient for 2DGS densification

class Splat2DGSRenderer(SplatRenderer[Splat2dgsRenderPayload]):
    """2D Gaussian Splatting renderer.
    
    Uses gsplat's rasterization_2dgs or rasterization_2dgs_inria_wrapper
    for rendering 2D Gaussian splats with surface normals.
    """

    _camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"]
    _near_plane: float
    _far_plane: float
    _tile_size: int
    _sparse_grad: bool
    _radius_clip: float
    _eps2d: float
    _absgrad: bool
    _channel_chunk: int
    _distloss: bool
    
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
        eps2d: float = 0.3,
        
        # Advanced features
        absgrad: bool = False,
        channel_chunk: int = 32,
        
        # 2DGS specific
        distloss: bool = True,
    ):
        """
        Initialize 2DGS renderer with fixed configuration.
        
        Args:
            camera_model: Camera projection model
            near_plane: Near clipping plane distance
            far_plane: Far clipping plane distance
            tile_size: Tile size for rasterization (typically 16)
            sparse_grad: Use sparse gradients (experimental)
            radius_clip: Skip gaussians with 2D radius <= this
            eps2d: Minimum 2D covariance eigenvalue, added to prevent projected GS to be too small
            absgrad: Compute absolute gradients (for AbsGS)
            channel_chunk: Render in chunks if channels > this
            model_type: "2dgs" or "2dgs-inria" implementation
            distloss: Enable distortion loss computation
        """
        
        # Store all as private attributes
        self._camera_model = camera_model
        self._near_plane = near_plane
        self._far_plane = far_plane
        self._tile_size = tile_size
        self._sparse_grad = sparse_grad
        self._radius_clip = radius_clip
        self._eps2d = eps2d
        self._absgrad = absgrad
        self._channel_chunk = channel_chunk
        self._distloss = distloss
    
    def render(
        self,
        splat_state: SplatTrainingState,
        cam_to_worlds: Tensor,
        Ks: Tensor,
        height: int,
        width: int,
        camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] | None = None,
        backgrounds: Tensor | None = None,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB+ED",
        sh_degree: int | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> Tuple[Tensor, Splat2dgsRenderPayload]:
        """
        Render 2D Gaussian splats from camera viewpoints.
        
        Args:
            splat_state: SplatTrainingState with gaussian parameters
            cam_to_worlds: Camera-to-world matrices [B, 4, 4]
            Ks: Camera intrinsics [B, 3, 3]
            height: Image height
            width: Image width
            camera_model: Camera projection model
            backgrounds: Background colors [B, C] (None = black)
            render_mode: What to render ("RGB", "D", "ED", "RGB+D", "RGB+ED")
            sh_degree: SH degree to use (None = use splat_state.sh_degree)
            world_rank: World rank
            world_size: World size
        
        Returns:
            renders: Rendered images [B, H, W, C]
            render_payload: Splat2dgsRenderPayload with all outputs
        """
        # Extract gaussian parameters from SplatTrainingState
        means = splat_state.params["means"]
        quats = splat_state.params["quats"]
        scales = torch.exp(splat_state.params["scales"])
        opacities = torch.sigmoid(splat_state.params["opacities"])
        colors = splat_state.colors  # Property handles SH vs features
        viewmats = torch.linalg.inv(cam_to_worlds)
        
        # Use splat_state's sh_degree if not specified
        if sh_degree is None:
            sh_degree = splat_state.sh_degree

        if camera_model is None:
            camera_model = self._camera_model
        
        # Call gsplat's 2DGS rasterization

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
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
            packed=False,  # Never pack
            distloss=self._distloss,
            depth_mode="expected",
        )

        batch_dims = means.shape[:-2]
        num_batch_dims = len(batch_dims)
        B = math.prod(batch_dims)
        N = means.shape[-2]
        C = viewmats.shape[-3]
        
        expected_depths = None
        accumulated_depths = None
        if render_mode in ["D", "RGB+D"]:
            accumulated_depths = render_colors[..., -1:]
        elif render_mode in ["ED", "RGB+ED"]:
            expected_depths = render_colors[..., -1:]
        
        if render_mode in ["RGB+D", "RGB+ED"]:
            render_colors = render_colors[..., :-1]
            
        
        # Build render payload
        outputs = Splat2dgsRenderPayload(
            renders=render_colors,
            alphas=render_alphas,
            n_cameras=C,
            n_batches=B,
            radii=info["radii"],
            depths=info["depths"],
            depths_expected=expected_depths,
            depths_accumulated=accumulated_depths,
            depths_median=render_median,
            width=info["width"],
            height=info["height"],
            normals=render_normals,
            normals_from_depth=normals_from_depth,
            render_distort=render_distort,
            gradient_2dgs=info["gradient_2dgs"],
            means2d=info["means2d"],
            gaussian_ids=info.get("gaussian_ids", None),
        )
        return render_colors, outputs
    
    # Setters for commonly adjusted parameters
    def set_radius_clip(self, radius_clip: float):
        """Set radius clipping threshold."""
        self._radius_clip = radius_clip
    
    def set_absgrad(self, absgrad: bool):
        """Enable or disable absolute gradient computation."""
        self._absgrad = absgrad
    
    def set_distloss(self, distloss: bool):
        """Enable or disable distortion loss computation."""
        self._distloss = distloss

