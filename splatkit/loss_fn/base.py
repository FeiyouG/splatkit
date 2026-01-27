from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Tuple, Generic
import torch.nn.functional as F

from ..splat.training_state import SplatTrainingState
from ..modules import SplatRenderPayloadT, SplatBaseModule

class SplatLossFn(
    SplatBaseModule[SplatRenderPayloadT],
    Generic[SplatRenderPayloadT],
    ABC
):
    """
    Base class for loss functions in Gaussian Splatting.
    
    Subclasses must implement:
        - compute_loss(): Calculate the loss between renders and targets

    Example:
        >>> from splatkit.loss_fn import Splat3DGSLossFn
        >>> loss_fn = Splat3DGSLossFn()
        >>> loss = loss_fn.compute_loss(
        ...     renders=rendered_images,
        ...     targets=ground_truth_images,
        ...     training_state=state,
        ...     rend_out=render_payload,
        ... )
        >>> loss.backward()
    """
    
    @abstractmethod
    def compute_loss(self,
        renders: torch.Tensor, # (..., H, W, 3)
        targets: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayloadT,
        masks: torch.Tensor | None = None, # (..., H, W)
    ) -> torch.Tensor:
        """
        Compute the training loss.
        
        Args:
            renders: Rendered RGB images, shape (..., H, W, 3), values in [0, 1]
                    (must have same shape as targets)
            targets: Ground truth RGB images, shape (..., H, W, 3), values in [0, 1]
            training_state: Current Gaussian parameters (e.g. SplatTrainingState)
            rend_out: Render payload (e.g. Splat3DGSRenderPayload, Splat2DGSRenderPayload)
            masks: Optional binary masks, shape (..., H, W)
                   If provided, only compute loss on masked pixels (1 = keep, 0 = ignore)
        
        Returns:
            Scalar loss tensor with gradients for backpropagation
        
        NOTE:
            - Images must be normalized to [0, 1] range, not [0, 255]
            - Masks are optional; if None, use all pixels
        """
        pass

    def _photometric_loss(self, renders: torch.Tensor, targets: torch.Tensor, ssim_lambda: float = 0.2) -> torch.Tensor:
        """
        Compute photometric loss (L1 + SSIM).
        
        Combines L1 loss (pixel-wise difference) with SSIM loss (structural similarity).
        SSIM captures perceptual quality better than L1 alone.
        
        Args:
            renders: Rendered images, shape (B, H, W, 3)
            targets: Target images, shape (B, H, W, 3)
            ssim_lambda: Weight for SSIM term (default: 0.2)
                        Final loss = (1-λ)*L1 + λ*SSIM
        
        Returns:
            Scalar photometric loss
        """
        from fused_ssim import fused_ssim

        l1_loss = F.l1_loss(renders, targets)
        ssim_loss = 1.0 - fused_ssim(
            renders.permute(0, 3, 1, 2), targets.permute(0, 3, 1, 2), padding="valid"
        )
        return l1_loss * (1 - ssim_lambda) + ssim_loss * ssim_lambda

    def _opacity_reg(self, opacities: torch.Tensor, opacity_reg: float = 0.0) -> torch.Tensor:
        """
        Compute opacity regularization.
        
        Penalizes high opacity values to encourage transparency and prevent
        Gaussians from blocking each other. Helps with over-reconstruction.
        
        Args:
            opacities: Gaussian opacity logits (before sigmoid)
            opacity_reg: Regularization weight (0 = disabled)
        
        Returns:
            Scalar regularization loss
        """
        if opacity_reg > 0:
            return opacity_reg * torch.sigmoid(opacities).mean()
        else:
            return opacities.new_zeros(())

    def _scale_reg(self, scales: torch.Tensor, scale_reg: float = 0.0) -> torch.Tensor:
        """
        Compute scale regularization.
        
        Penalizes large Gaussian scales to encourage finer details and prevent
        bloated Gaussians that cover too much area.
        
        Args:
            scales: Gaussian scale log-values (before exp)
            scale_reg: Regularization weight (0 = disabled)
        
        Returns:
            Scalar regularization loss
        """
        if scale_reg > 0:
            return scale_reg * torch.exp(scales).mean()
        else:
            return scales.new_zeros(())