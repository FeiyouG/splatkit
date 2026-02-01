from abc import ABC, abstractmethod
from math import exp
import torch
from typing import Dict, Any, Tuple, Generic
import torch.nn.functional as F

from ..splat.training_state import SplatTrainingState
from ..logger import SplatLogger
from ..modules import SplatRenderPayloadT, SplatBaseModule
from torch.autograd import Variable

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
        logger: SplatLogger,
        step: int,
        renders: torch.Tensor, # (..., H, W, 3)
        targets: torch.Tensor, # (..., H, W, 3)
        K: torch.Tensor, # (..., 3, 3)
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayloadT,
        masks: torch.Tensor | None = None, # (..., H, W)
    ) -> torch.Tensor:
        """
        Compute the training loss.
        
        Args:
            logger: Logger for logging
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

    def _photometric_loss(self, logger: SplatLogger, renders: torch.Tensor, targets: torch.Tensor, ssim_lambda: float = 0.2) -> torch.Tensor:
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
        ssim_loss = self._ssim_loss(logger, renders, targets, ssim_lambda)
        l1_loss = self._l1_loss(logger, renders, targets, 1 - ssim_lambda)
        return l1_loss * (1 - ssim_lambda) + ssim_loss * ssim_lambda
    
    def _ssim_loss(self, logger: SplatLogger, renders: torch.Tensor, targets: torch.Tensor, ssim_lambda: float = 0.2) -> torch.Tensor:
        """
        Compute SSIM loss.
        """
        try:
            from fused_ssim import fused_ssim
            ssim_loss = 1.0 - fused_ssim(
                renders.permute(0, 3, 1, 2), targets.permute(0, 3, 1, 2), padding="valid"
            )
        except ImportError:
            logger.warning("fused-ssim is not installed, using ssim instead")
            ssim_loss = self._ssim(renders, targets)
        return ssim_loss * ssim_lambda
    
    def _l1_loss(self, logger: SplatLogger, renders: torch.Tensor, targets: torch.Tensor, l1_lambda: float = 1.0) -> torch.Tensor:
        """
        Compute L1 loss.
        """
        return l1_lambda * F.l1_loss(renders, targets)

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
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
        channel = img1.size(-3)
        window = self._create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
        

    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()


    def _smooth_loss(self, disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
        grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
        grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
        grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()


    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window