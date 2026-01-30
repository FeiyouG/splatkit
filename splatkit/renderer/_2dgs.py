from dataclasses import dataclass
import math
from typing import Literal, Sequence, Tuple
import numpy as np
import torch
from torch import Tensor

from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper

from ..modules import SplatBaseModule, SplatRenderPayload
from ..splat.training_state import SplatTrainingState
from .base import SplatRenderer

@dataclass(frozen=True)
class Splat2dgsRenderPayload(SplatRenderPayload):
    """
    Render output for 2D Gaussian Splatting (extends SplatRenderPayload).
    
    Adds 2DGS-specific outputs like normals, distortion, and median depths.
    2DGS models surfaces more explicitly than 3DGS, enabling better normal estimation.
    """
    
    normals: Tensor
    """Rendered surface normals from 2D Gaussians, shape (..., H, W, 3)"""
    
    normals_from_depth: Tensor
    """Surface normals derived from depth gradients, shape (..., H, W, 3)"""
    
    render_distort: Tensor
    """Distortion map for regularization, shape (..., H, W, 1)"""
    
    depths_median: Tensor
    """Median depth values, shape (..., H, W, 1)"""
    
    gradient_2dgs: Tensor
    """Gradients for 2DGS-specific densification"""

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
        depth_mode: Literal["expected", "median"] = "expected",
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
        self._depth_mode: Literal["expected", "median"] = depth_mode
    
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

        disloss = self._distloss
        if render_mode in ["RGB"]:
            disloss = False
        
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
            distloss=disloss,
            depth_mode=self._depth_mode,
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
    
    # Visualization methods
    def get_visualization_options(self) -> Tuple[str, ...]:
        """Return available visualization options for 2DGS renderer."""
        return ("rgb", "depth(median)", "normal", "distortion", "alpha")
    
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
        Generate 2DGS visualization for interactive viewer.
        
        Supports 2DGS-specific visualization modes including surface normals,
        median depth, and distortion maps.
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
            render_mode="RGB+ED",  # Get all data
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
            
        elif visualization_mode == "depth(median)":
            depth = payload.depths_median[0, ..., 0]
            output = self._process_depth(depth, normalize_nearfar, near_plane, far_plane, inverse, colormap)
            
        elif visualization_mode == "normal":
            normals = payload.normals[0]
            normals = normals * 0.5 + 0.5  # Normalize to [0, 1]
            output = normals.cpu().numpy()
            
        elif visualization_mode == "distortion":
            distort = payload.render_distort[0, ..., 0]
            if inverse:
                distort = 1 - distort
            output = apply_float_colormap(distort.unsqueeze(-1), colormap).cpu().numpy()  # type: ignore
            
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