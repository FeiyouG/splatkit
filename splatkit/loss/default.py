import torch.nn.functional as F

from ..utils.batched import normalize_batch_tensors
from .abc import SplatLoss

class DefaultSplatLoss(SplatLoss):
    """Simple 3DGS loss: L1 + SSIM + regularization."""
    
    def __init__(
        self,
        ssim_lambda: float = 0.2,
        opacity_reg: float = 0.0,
        scale_reg: float = 0.0,
    ):
        """
        Initialize the DefaultSplatLoss.
        
        Args:
            ssim_lambda: Weight for SSIM loss
            opacity_reg: Prevents over-opaque splats and encouring pruning of useless gaussians
            scale_reg: Prevent infintely small gaussians or "fog clouds"; keeps spatial coherence
        """
        self.ssim_lambda = ssim_lambda
        self.opacity_reg = opacity_reg
        self.scale_reg = scale_reg
    
    def compute(self, renders, targets, splat_state, rend_meta, target_meta):
        """
        Compute loss.
        """
        renders, targets = normalize_batch_tensors(renders, targets, spatial_ndim=3)
        photometric_loss = self._photometric_loss(renders, targets, self.ssim_lambda)

        # Regularization
        opa_loss = self._opacity_reg(splat_state.params["opacities"], self.opacity_reg)
        scale_loss = self._scale_reg(splat_state.params["scales"], self.scale_reg)

        loss = photometric_loss + opa_loss + scale_loss
        
        return loss