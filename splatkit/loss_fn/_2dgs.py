from typing import Generic, TYPE_CHECKING, Sequence
from typing_extensions import override

from ..utils.batched import normalize_batch_tensors
from ..splat.training_state import SplatTrainingState
from .base import SplatLossFn
import torch
import torch.nn.functional as F

from ..renderer._2dgs import Splat2dgsRenderPayload
from ..logger import SplatLogger
from ..modules import SplatRenderPayloadT
from ..modules.base import SplatBaseModule

class Splat2DGSLossFn(
    SplatLossFn[Splat2dgsRenderPayload],
):
    """Simple 2DGS loss: L1 + SSIM + normal consistency + distortion loss."""
    
    def __init__(
        self,
        ssim_lambda: float = 0.2,
        bg_lambda: float = 1.0,
        opacity_reg: float = 0.0,
        scale_reg: float = 0.0,
        normal_lambda: float = 0.05,
        normal_start_iter: int = 7000,
        dist_lambda: float = 0.01,
        dist_start_iter: int = 3000,
    ):
        """
        Initialize the Splat2DGSLossFn.
        
        Args:
            ssim_lambda: Weight for SSIM loss
            bg_lambda: Weight for background regularization
            opacity_reg: Prevents over-opaque splats and encouraging pruning of useless gaussians
            scale_reg: Prevent infinitely small gaussians or "fog clouds"; keeps spatial coherence
            normal_lambda: Weight for normal consistency loss
            normal_start_iter: Iteration to start applying normal loss
            dist_lambda: Weight for distortion loss
            dist_start_iter: Iteration to start applying distortion loss
        """
        self.ssim_lambda = ssim_lambda
        self.bg_lambda = bg_lambda
        self.opacity_reg = opacity_reg
        self.scale_reg = scale_reg
        self.normal_lambda = normal_lambda
        self.normal_start_iter = normal_start_iter
        self.dist_lambda = dist_lambda
        self.dist_start_iter = dist_start_iter
        self._current_step = 0

    @override
    def pre_compute_loss(
        self,
        logger: SplatLogger,
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor,
        target_frames: torch.Tensor,
        training_state: SplatTrainingState,
        rend_out: Splat2dgsRenderPayload,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """Update the current step for scheduled losses."""
        self._current_step = step
    
    def compute_loss(self, logger: SplatLogger, renders, targets, training_state, rend_out, masks=None):
        """
        Compute loss including normal consistency and distortion regularization.
        
        Args:
            renders: Rendered images [..., H, W, 3]
            targets: Target images [..., H, W, 3]
            training_state: Training state
            rend_out: Splat2dgsRenderPayload with normals and distortion
            masks: Optional masks [..., H, W]
            
        Returns:
            Scalar loss tensor
        """
        common_device = renders.device
        
        # Photometric loss (L1 + SSIM)
        photometric_loss = 0.0
        bg_loss = 0.0

        if masks is not None:
            masks = masks[..., None].to(common_device)  # (..., H, W) -> (..., H, W, 1)
            renders, targets, alphas, masks = normalize_batch_tensors(
                renders, targets, rend_out.alphas, masks, spatial_ndim=3
            )
            renders = renders * masks
            targets = targets * masks
            photometric_loss = self._photometric_loss(logger, renders, targets, self.ssim_lambda)

            bg_mask = ~masks
            bg_loss = self.bg_lambda * (alphas * bg_mask).mean()
        else:
            renders, targets = normalize_batch_tensors(renders, targets, spatial_ndim=3)
            photometric_loss = self._photometric_loss(logger, renders, targets, self.ssim_lambda)

        # Regularization
        opa_loss = self._opacity_reg(training_state.params["opacities"], self.opacity_reg)
        scale_loss = self._scale_reg(training_state.params["scales"], self.scale_reg)

        # Normal consistency loss (2DGS specific)
        normal_loss = self._compute_normal_loss(rend_out)
        
        # Distortion loss (2DGS specific)
        dist_loss = self._compute_distortion_loss(rend_out)

        # Combine all losses
        loss = photometric_loss + bg_loss + opa_loss + scale_loss + normal_loss + dist_loss
        
        return loss
    
    def _compute_normal_loss(self, rend_out: Splat2dgsRenderPayload) -> torch.Tensor:
        """
        Compute normal consistency loss between rendered normals and normals from depth.
        
        This regularization encourages the rendered surface normals to be consistent
        with the normals derived from the depth map.
        """
        if self._current_step < self.normal_start_iter or self.normal_lambda == 0.0:
            return torch.tensor(0.0, device=rend_out.normals.device)
        
        # Get normals
        normals = rend_out.normals  # [..., H, W, 3]
        normals_from_depth = rend_out.normals_from_depth  # [..., H, W, 3]
        alphas = rend_out.alphas  # [..., H, W, 1]
        
        # Weight normals from depth by alpha (opacity mask)
        normals_from_depth = normals_from_depth * alphas.detach()
        
        # Normalize to channel-first for computation
        # Reshape to [B, 3, H, W] if needed
        if len(normals.shape) == 4:  # [B, H, W, 3]
            normals_flat = normals.permute(0, 3, 1, 2)  # [B, 3, H, W]
            normals_from_depth_flat = normals_from_depth.permute(0, 3, 1, 2)  # [B, 3, H, W]
        else:  # [..., H, W, 3] - handle arbitrary batch dims
            original_shape = normals.shape
            normals_flat = normals.reshape(-1, *original_shape[-3:]).permute(0, 3, 1, 2)
            normals_from_depth_flat = normals_from_depth.reshape(-1, *original_shape[-3:]).permute(0, 3, 1, 2)
        
        # Compute cosine similarity loss: 1 - cos(θ)
        # Dot product along channel dimension
        dot_product = (normals_flat * normals_from_depth_flat).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        normal_error = 1.0 - dot_product
        
        # Mean over all pixels
        loss = self.normal_lambda * normal_error.mean()
        
        return loss
    
    def _compute_distortion_loss(self, rend_out: "Splat2dgsRenderPayload") -> torch.Tensor:
        """
        Compute distortion loss to encourage compact, well-separated gaussians.
        
        This regularization helps prevent overlapping gaussians and encourages
        better spatial distribution.
        """
        if self._current_step < self.dist_start_iter or self.dist_lambda == 0.0:
            return torch.tensor(0.0, device=rend_out.render_distort.device)
        
        # Get distortion map
        render_distort = rend_out.render_distort  # [..., H, W, 1] or [..., H, W]
        
        # Mean distortion
        loss = self.dist_lambda * render_distort.mean()
        
        return loss

