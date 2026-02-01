from typing import Callable
from typing_extensions import override

import torch
import torch.nn.functional as F

from splatkit.utils.batched import normalize_batch_tensors
from splatkit.splat.training_state import SplatTrainingState
from .renderer import SplatRaDeRenderPayload
from splatkit.logger import SplatLogger
from splatkit.loss_fn.base import SplatLossFn

class SplatRaDeLossFn(
    SplatLossFn[SplatRaDeRenderPayload],
):
    """RaDe-GS loss: L1 + DSSIM + depth-normal regularization."""
    
    def __init__(
        self,
        ssim_lambda: float = 0.2,
        depth_normal_lambda: float = 0.05,
        bg_lambda: float = 0.5,
        normal_start_step: int = 0,
        depth_ratio: float = 0.6,
        decoupled_appearance: bool = False,
    ):
        """
        Initialize the SplatRaDeLossFn.
        
        Args:
            ssim_lambda: Weight for DSSIM loss
            depth_normal_lambda: Weight for depth-normal regularization
            bg_lambda: Weight for background regularization
            normal_start_step: Step to start depth-normal loss
            depth_ratio: Blend ratio between expected/median depth normals
            decoupled_appearance: Enable appearance-decoupled L1 loss
        """
        self._ssim_lambda = ssim_lambda
        self._depth_normal_lambda = depth_normal_lambda
        self._bg_lambda = bg_lambda
        self._normal_start_step = normal_start_step
        self._depth_ratio = depth_ratio
        self._decoupled_appearance = decoupled_appearance

    def compute_loss(
        self, 
        logger: SplatLogger,
        step: int,
        renders: torch.Tensor,
        targets: torch.Tensor,
        K: torch.Tensor,
        training_state: SplatTrainingState,
        rend_out: SplatRaDeRenderPayload,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute loss following RaDe-GS formulation.
        
        Args:
            renders: Rendered images [..., H, W, 3]
            targets: Target images [..., H, W, 3]
            training_state: Training state
            rend_out: SplatRaDeRenderPayload with depth/normal outputs
            masks: Optional masks [..., H, W]
            
        Returns:
            Scalar loss tensor
        """
        common_device = renders.device

        bg_loss = 0.0
        if masks is not None:
            masks = masks[..., None].to(common_device)  # (..., H, W) -> (..., H, W, 1)
            renders, targets, alphas, masks = normalize_batch_tensors(
                renders, targets, rend_out.alphas, masks, spatial_ndim=3
            )
            renders = renders * masks
            targets = targets * masks
            bg_loss = self._bg_lambda * (alphas * masks).mean()
        else:
            renders, targets = normalize_batch_tensors(renders, targets, spatial_ndim=3)

        # if self._decoupled_appearance and self._appearance_loss_fn is not None:
        #     Ll1_render = self._appearance_loss_fn(renders, targets, training_state, rend_out)
        # else:
        Ll1_render = self._l1_loss(logger, renders, targets)
        ssim_loss = self._ssim_loss(logger, renders, targets)
        photometric_loss = (1.0 - self._ssim_lambda) * Ll1_render + self._ssim_lambda * ssim_loss

        height, width = renders.shape[1:3]
        if K.dim() == 3:
            fx = K[:, 0, 0]
            fy = K[:, 1, 1]
            cx = K[:, 0, 2]
            cy = K[:, 1, 2]
        else:
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

        depth_normal_loss = self._compute_depth_normal_loss(step, width, height, fx, fy, cx, cy, rend_out)
        loss = photometric_loss + depth_normal_loss + bg_loss

        return loss


    def _compute_depth_normal_loss(
        self,
        step: int,
        width: int,
        height: int,
        fx: torch.Tensor,
        fy: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        rend_out: SplatRaDeRenderPayload,
    ) -> torch.Tensor:

        if step < self._normal_start_step or self._depth_normal_lambda == 0.0:
            return rend_out.renders.new_zeros(())
        
        normals =  rend_out.expected_normals
        expected_depth = rend_out.depths_expected
        median_depth = rend_out.median_depths

        if normals is None or expected_depth is None or median_depth is None:
            raise ValueError(f"Rade-gs render output is missing required values: normals={normals is not None}, expected_depth={expected_depth is not None}, median_depth={median_depth is not None}")

        rendered_normal = self._normalize_rendered_normal(rend_out.expected_normals)
        depth_middepth_normal = self._depth_double_to_normal(width, height, fx, fy, cx, cy, expected_depth, median_depth)

        normal_error_map = 1.0 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=2)
        depth_normal_loss = (
            (1.0 - self._depth_ratio) * normal_error_map[0].mean()
            + self._depth_ratio * normal_error_map[1].mean()
        )
        return depth_normal_loss * self._depth_normal_lambda

    def _normalize_rendered_normal(self, rendered_normal: torch.Tensor) -> torch.Tensor:
        if rendered_normal.dim() == 4:
            # (B, H, W, 3) -> (B, 3, H, W)
            if rendered_normal.shape[-1] == 3:
                return rendered_normal.permute(0, 3, 1, 2)
            # (B, 3, H, W)
            if rendered_normal.shape[1] == 3:
                return rendered_normal
        if rendered_normal.dim() == 3:
            # (H, W, 3) -> (1, 3, H, W)
            if rendered_normal.shape[-1] == 3:
                return rendered_normal.permute(2, 0, 1).unsqueeze(0)
            # (3, H, W) -> (1, 3, H, W)
            if rendered_normal.shape[0] == 3:
                return rendered_normal.unsqueeze(0)
        raise ValueError(f"Unexpected normal tensor shape: {tuple(rendered_normal.shape)}")

    def _depth_double_to_normal(
        self,
        width: int,
        height: int,
        fx: torch.Tensor,
        fy: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
    ) -> torch.Tensor:
        points1, points2 = self._depths_double_to_points(width, height, fx, fy, cx, cy, depth1, depth2)
        return self._point_double_to_normal(points1, points2)

    def _point_double_to_normal(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        points = torch.stack([points1, points2], dim=0)
        output = torch.zeros_like(points)
        dx = points[..., 2:, 1:-1] - points[..., :-2, 1:-1]
        dy = points[..., 1:-1, 2:] - points[..., 1:-1, :-2]
        normal_map = F.normalize(torch.cross(dx, dy, dim=2), dim=2)
        output[..., 1:-1, 1:-1] = normal_map
        return output

    def _depths_double_to_points(
        self,
        width: int,
        height: int,
        fx: torch.Tensor,
        fy: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        depthmap1: torch.Tensor,
        depthmap2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if depthmap1.dim() == 4 and depthmap1.shape[-1] == 1:
            depthmap1 = depthmap1.squeeze(-1)
        if depthmap2.dim() == 4 and depthmap2.shape[-1] == 1:
            depthmap2 = depthmap2.squeeze(-1)
        if depthmap1.dim() == 2:
            depthmap1 = depthmap1.unsqueeze(0)
        if depthmap2.dim() == 2:
            depthmap2 = depthmap2.unsqueeze(0)

        batch_size = depthmap1.shape[0]
        grid_x, grid_y = torch.meshgrid(
            torch.arange(width, device=depthmap1.device) + 0.5,
            torch.arange(height, device=depthmap1.device) + 0.5,
            indexing="xy",
        )
        fx_t = fx.reshape(-1, 1, 1) if torch.is_tensor(fx) else torch.tensor(fx, device=depthmap1.device).reshape(1, 1, 1)
        fy_t = fy.reshape(-1, 1, 1) if torch.is_tensor(fy) else torch.tensor(fy, device=depthmap1.device).reshape(1, 1, 1)
        cx_t = cx.reshape(-1, 1, 1) if torch.is_tensor(cx) else torch.tensor(cx, device=depthmap1.device).reshape(1, 1, 1)
        cy_t = cy.reshape(-1, 1, 1) if torch.is_tensor(cy) else torch.tensor(cy, device=depthmap1.device).reshape(1, 1, 1)

        if fx_t.shape[0] == 1 and batch_size > 1:
            fx_t = fx_t.expand(batch_size, -1, -1)
            fy_t = fy_t.expand(batch_size, -1, -1)
            cx_t = cx_t.expand(batch_size, -1, -1)
            cy_t = cy_t.expand(batch_size, -1, -1)

        rays_d = torch.stack(
            [
                (grid_x - cx_t) / fx_t,
                (grid_y - cy_t) / fy_t,
                torch.ones_like(grid_x).expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        points1 = depthmap1.unsqueeze(1) * rays_d
        points2 = depthmap2.unsqueeze(1) * rays_d
        return points1, points2

    def _depths_double_to_points_from_grid(
        self, depthmap1: torch.Tensor, depthmap2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depthmap1 = depthmap1.squeeze()
        depthmap2 = depthmap2.squeeze()
        H, W = depthmap1.shape[-2], depthmap1.shape[-1]
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=depthmap1.device),
            torch.arange(H, device=depthmap1.device),
            indexing="xy",
        )
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).float()
        points1 = depthmap1.reshape(1, -1) * points.reshape(3, -1)
        points2 = depthmap2.reshape(1, -1) * points.reshape(3, -1)
        return points1.reshape(3, H, W), points2.reshape(3, H, W)

    # # function L1_loss_appearance is fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/train.py
    # def L1_loss_appearance(self, renders: torch.Tensor, targets: torch.Tensor, gaussians: torch.Tensor, view_idx: int, return_transformed_image: bool = False) -> torch.Tensor:
    #     appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    #     # center crop the image
    #     origH, origW = renders.shape[1:]
    #     H = origH // 32 * 32
    #     W = origW // 32 * 32
    #     left = origW // 2 - W // 2
    #     top = origH // 2 - H // 2
    #     crop_image = renders[:, top:top+H, left:left+W]
    #     crop_gt_image = targets[:, top:top+H, left:left+W]
        
    #     # down sample the image
    #     crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
        
    #     crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    #     mapping_image = gaussians.appearance_network(crop_image_down)
    #     transformed_image = mapping_image * crop_image
    #     if not return_transformed_image:
    #         return l1_loss(transformed_image, crop_gt_image)
    #     else:
    #         transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
    #         return transformed_image