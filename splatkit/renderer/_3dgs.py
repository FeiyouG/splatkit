from dataclasses import dataclass
from typing import Literal, Sequence, Tuple
import numpy as np
import torch
from torch import Tensor

from gsplat.rendering import rasterization

from ..modules import SplatBaseModule, SplatRenderPayload
from ..splat.training_state import SplatTrainingState
from .base import SplatRenderer

@dataclass(frozen=True)
class Splat3dgsRenderPayload(SplatRenderPayload):
    """
    Render output for 3D Gaussian Splatting (extends SplatRenderPayload).
    
    Adds 3DGS-specific outputs like conics (Gaussian covariance in 2D).
    """
    
    conics: Tensor
    """2D Gaussian covariance matrices, shape (..., C, N, 3) where C=cameras, N=gaussians. Upper triangular: [a, b, c] represents [[a, b], [b, c]]"""

class Splat3DGSRenderer(SplatRenderer[Splat3dgsRenderPayload]):
    """
    3D Gaussian Splatting renderer.

    NOTE:
            In 3DGS, depth images are volumetric (no true surface).
            depths_expected represents expected ray depth.
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
        render_colors, alphas, info = rasterization(
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

        expected_depths = None
        accumulated_depths = None
        if render_mode in ["D", "RGB+D"]:
            accumulated_depths = render_colors[..., -1:]
        elif render_mode in ["ED", "RGB+ED"]:
            expected_depths = render_colors[..., -1:]
        
        if render_mode in ["RGB+D", "RGB+ED"]:
            render_colors = render_colors[..., :-1]
            
            
        
        # Build render_meta with alphas included
        outputs = Splat3dgsRenderPayload(
            renders=render_colors,
            alphas=alphas,
            n_cameras=info["n_cameras"],
            n_batches=info["n_batches"],
            radii=info["radii"],
            means2d=info["means2d"],
            depths=info["depths"],
            depths_expected=expected_depths,
            depths_accumulated=accumulated_depths,
            conics=info["conics"],
            width=info["width"],
            height=info["height"],
            gaussian_ids=info.get("gaussian_ids", None),
        )
        return render_colors, outputs


    
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
    
    # Visualization methods
    def get_visualization_options(self) -> Tuple[str, ...]:
        """Return available visualization options for 3DGS renderer."""
        return ("rgb", "depth(accumulated)", "depth(expected)", "alpha")
    
    @torch.no_grad()
    def visualize(
        self,
        splat_state: SplatTrainingState,
        camera_state,  # CameraState from nerfview
        width: int,
        height: int,
        visualization_mode: str = "rgb",
        sh_degree: int | None = None,
        backgrounds: Tensor | None = None,
        camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
        normalize_nearfar: bool = False,
        near_plane: float = 1e-2,
        far_plane: float = 1e2,
        inverse: bool = False,
        colormap: str = "turbo",
    ) -> Tuple[np.ndarray, int]:
        """
        Generate 3DGS visualization for interactive viewer.
        
        Supports 3DGS-specific visualization modes including accumulated
        and expected depth rendering.
        """
        from nerfview import apply_float_colormap
        
        # Get camera parameters
        c2w = torch.from_numpy(camera_state.c2w).float().to(splat_state.device).unsqueeze(0)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(splat_state.device).unsqueeze(0)
        
        # Render with all data
        renders, payload = self.render(
            splat_state=splat_state,
            cam_to_worlds=c2w,
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree,
            render_mode="RGB+ED",  # Get all depth variants
            backgrounds=backgrounds,
            camera_model=camera_model,
        )
        
        # Calculate stats
        # radii shape: [1, 1, N, 2] where N=gaussians, 2=x/y (single camera in viewer)
        # Check if both x and y radii > 0 for each gaussian
        rendered_gaussians = int((payload.radii > 0).all(-1).sum().item())
        
        # Process based on visualization mode
        if visualization_mode == "rgb":
            output = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
            
        elif visualization_mode == "depth(expected)" and payload.depths_expected is not None:
            depth = payload.depths_expected[0, ..., 0]
            output = self._process_depth(depth, normalize_nearfar, near_plane, far_plane, inverse, colormap)
            
        elif visualization_mode == "depth(accumulated)" and payload.depths_accumulated is not None:
            depth = payload.depths_accumulated[0, ..., 0]
            output = self._process_depth(depth, normalize_nearfar, near_plane, far_plane, inverse, colormap)
            
        elif visualization_mode == "alpha":
            alpha = payload.alphas[0, ..., 0]
            if inverse:
                alpha = 1 - alpha
            output = apply_float_colormap(alpha.unsqueeze(-1), colormap).cpu().numpy()  # type: ignore
            
        else:
            # Fallback to RGB
            output = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
        
        return output, rendered_gaussians
    
    def _process_depth(
        self,
        depth: Tensor,
        normalize_nearfar: bool,
        near_plane: float,
        far_plane: float,
        inverse: bool,
        colormap: str,
    ) -> np.ndarray:
        """Process depth for visualization."""
        from nerfview import apply_float_colormap
        
        # Normalize depth
        if normalize_nearfar:
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
        else:
            depth_min = depth.min()
            depth_max = depth.max()
            depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-10)
        
        depth_norm = depth_norm.clamp(0, 1)
        
        if inverse:
            depth_norm = 1 - depth_norm
        
        return apply_float_colormap(depth_norm.unsqueeze(-1), colormap).cpu().numpy()  # type: ignore