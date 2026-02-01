from typing import Generic

from splatkit.logger import SplatLogger

from ..modules import SplatRenderPayload, SplatRenderPayload
from ..utils.batched import normalize_batch_tensors
from .base import SplatLossFn

class Splat3DGSLossFn(
    SplatLossFn[SplatRenderPayload],
):
    """Simple 3DGS loss: L1 + SSIM + regularization."""
    
    def __init__(
        self,
        ssim_lambda: float = 0.2,
        bg_lambda: float = 1.0,
        opacity_reg: float = 0.0,
        scale_reg: float = 0.0,
    ):
        """
        Initialize the Splat3DGSLossFn.
        
        Args:
            ssim_lambda: Weight for SSIM loss
            opacity_reg: Prevents over-opaque splats and encouring pruning of useless gaussians
            scale_reg: Prevent infintely small gaussians or "fog clouds"; keeps spatial coherence
        """
        self.ssim_lambda = ssim_lambda
        self.bg_lambda = bg_lambda
        self.opacity_reg = opacity_reg
        self.scale_reg = scale_reg
    
    def compute_loss(self, logger: SplatLogger, step: int, renders, targets, K, training_state, rend_out, masks=None):
        """
        Compute loss.
        """

        common_device = renders.device
        photometric_loss = 0.0
        bg_loss = 0.0
        if masks is not None:
            masks = masks[..., None].to(common_device) # (..., H, W) -> (..., H, W, 1)
            renders, targets, alphas, masks = normalize_batch_tensors(renders, targets, rend_out.alphas, masks, spatial_ndim=3)
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

        loss = photometric_loss + bg_loss + opa_loss + scale_loss
        
        return loss