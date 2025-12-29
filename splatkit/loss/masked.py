from .abc import SplatLoss
from .default import DefaultSplatLoss
from ..utils.batched import normalize_batch_tensors

class MaskedSplatLoss(SplatLoss):
    """
    Masked Splat loss.
    """
    
    def __init__(
        self,
        ssim_lambda: float = 0.2,
        bg_lambda: float = 1.0,
        opacity_reg: float = 0.0,
        scale_reg: float = 0.0,
    ):
        """
        Initialize the DefaultSplatLoss.
        
        Args:
            ssim_lambda: Weight for SSIM loss
            bg_lambda: Weight for background loss
            opacity_reg: Prevents over-opaque splats and encouring pruning of useless gaussians
            scale_reg: Prevent infintely small gaussians or "fog clouds"; keeps spatial coherence
        """
        self.ssim_lambda = ssim_lambda
        self.opacity_reg = opacity_reg
        self.scale_reg = scale_reg
        self.bg_lambda = bg_lambda
    
    def compute(self, renders, targets, splat_state, rend_meta, target_meta):
        """
        Compute loss.
        """
        if "alpha" not in rend_meta:
            raise ValueError("MaskedSplatLoss requires alpha in meta")
        alpha = target_meta["alpha"]

        if "mask" not in target_meta:
            raise ValueError("MaskedSplatLoss requires mask in meta")
        mask = target_meta["mask"]

        mask = mask[..., None] # (..., H, W) -> (..., H, W, 1)
        renders, targets, alpha, mask = normalize_batch_tensors(renders, targets, alpha, mask, spatial_ndim=3)

        masked_renders = renders * mask
        masked_targets = targets * mask

        photometric_loss = self._photometric_loss(masked_renders, masked_targets, self.ssim_lambda)
        bg_loss = self.bg_lambda * (alpha * (1.0 - mask)).mean()

        opa_loss = self._opacity_reg(splat_state.params["opacities"], self.opacity_reg)
        scale_loss = self._scale_reg(splat_state.params["scales"], self.scale_reg)

        loss = photometric_loss + bg_loss + opa_loss + scale_loss

        return loss
