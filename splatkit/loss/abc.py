from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Tuple, Generic
import torch.nn.functional as F
from fused_ssim import fused_ssim

from ..splat.training_state import SplatTrainingState
from ..renderer import SplatRendererOutputType

class SplatLoss(ABC, Generic[SplatRendererOutputType]):
    """
    Abstract base class for Splat losses.
    """
    
    @abstractmethod
    def compute(self,
        renders: torch.Tensor, # (..., H, W, 3)
        targets: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: SplatRendererOutputType,
        masks: torch.Tensor | None = None, # (..., H, W)
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            renders: Rendered images [,,,, H, W, 3]
            targets: Target images [..., H, W, 3] (has to be same shape as renders otherwise an error is raised)
            training_state: Training state
            rend_meta: Additional parameters related to the rendered image (from rasterizer)
            target_meta: Additional parameters related to the target (from dataset)
        
        Returns:
            Scalar loss tensor
        """
        pass

    def _photometric_loss(self, renders: torch.Tensor, targets: torch.Tensor, ssim_lambda: float = 0.2) -> torch.Tensor:
        """
        Compute photometric loss.
        """
        l1_loss = F.l1_loss(renders, targets)
        ssim_loss = 1.0 - fused_ssim(
            renders.permute(0, 3, 1, 2), targets.permute(0, 3, 1, 2), padding="valid"
        )
        return l1_loss * (1 - ssim_lambda) + ssim_loss * ssim_lambda

    def _opacity_reg(self, opacities: torch.Tensor, opacity_reg: float = 0.0) -> torch.Tensor:
        """
        Compute opacity regularization.
        """
        if opacity_reg > 0:
            return opacity_reg * torch.sigmoid(opacities).mean()
        else:
            return 0.0

    def _scale_reg(self, scales: torch.Tensor, scale_reg: float = 0.0) -> torch.Tensor:
        """
        Compute scale regularization.
        """
        if scale_reg > 0:
            return scale_reg * torch.exp(scales).mean()
        else:
            return 0.0