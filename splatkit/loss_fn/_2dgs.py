from typing import Generic, TYPE_CHECKING, Sequence
from typing_extensions import override

from ..utils.batched import normalize_batch_tensors
from ..splat.training_state import SplatTrainingState
import torch
import torch.nn.functional as F

from ..renderer._2dgs import Splat2dgsRenderPayload
from ..logger import SplatLogger
from ..modules import SplatRenderPayloadT
from ..modules.base import SplatBaseModule
from .base import SplatLossFn

class Splat2DGSLossFn(
    SplatLossFn[Splat2dgsRenderPayload],
):
    """Simple 2DGS loss: L1 + SSIM + normal consistency + distortion loss."""
    
    def __init__(
        self,
        ssim_lambda: float = 0.2,
        bg_lambda: float = 0.5,
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
    
    def compute_loss(self, logger: SplatLogger, step: int, renders, targets, K, training_state, rend_out, masks=None):
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
        normal_loss = self._compute_normal_loss(step, rend_out)
        
        # Distortion loss (2DGS specific)
        dist_loss = self._compute_distortion_loss(step, rend_out)

        # Combine all losses
        loss = photometric_loss + bg_loss + opa_loss + scale_loss
        
        return loss

    def _compute_normal_loss(self, step: int, rend_out: Splat2dgsRenderPayload) -> torch.Tensor:
        if step <= self.normal_start_iter or self.normal_lambda == 0.0:
            return torch.tensor(0.0, device=rend_out.normals.device)

        # [1, H, W, 3] -> [3, H, W]
        normals = rend_out.normals.squeeze(0).permute(2, 0, 1)
        normals_from_depth = rend_out.normals_from_depth.squeeze(0)
        alphas = rend_out.alphas.squeeze(0)

        # gsplat-style alpha masking
        normals_from_depth = normals_from_depth * alphas.detach()

        # [H, W, 3] -> [3, H, W]
        normals_from_depth = normals_from_depth.permute(2, 0, 1)

        # normal consistency loss
        normal_error = 1.0 - (normals * normals_from_depth).sum(dim=0, keepdim=True)
        loss = self.normal_lambda * normal_error.mean()

        return loss

    
    def _compute_distortion_loss(self, step: int, rend_out: "Splat2dgsRenderPayload") -> torch.Tensor:
        """
        Compute distortion loss to encourage compact, well-separated gaussians.
        
        This regularization helps prevent overlapping gaussians and encourages
        better spatial distribution.
        """
        if step < self.dist_start_iter or self.dist_lambda == 0.0:
            return torch.tensor(0.0, device=rend_out.render_distort.device)
        
        # Get distortion map
        render_distort = rend_out.render_distort  # [..., H, W, 1] or [..., H, W]
        
        # Mean distortion
        loss = self.dist_lambda * render_distort.mean()
        
        return loss

