from typing import Dict, Sequence, TYPE_CHECKING

import torch
from gsplat.strategy.ops import remove
from typing_extensions import override
from splatkit.data_provider import SplatDataProvider
from splatkit.densification import SplatDefaultDensification, SplatDensification
from splatkit.modules import SplatBaseModule, SplatRenderPayload
from splatkit.logger import SplatLogger
from gsplat.strategy import DefaultStrategy
from splatkit.data_provider.colmap.item import ColmapDataItem
from splatkit.splat import SplatTrainingState

if TYPE_CHECKING:
    from splatkit.renderer.base import SplatRenderer
    from splatkit.loss_fn.base import SplatLossFn
    from splatkit.densification.base import SplatDensification

class ObjectCentricDensification(SplatDensification):
    """
    Occlusion-aware densification strategy.
    
    Implementation of Object-Centric Training from 
        - Arxiv:https://arxiv.org/abs/2501.08174
        - Code: https://github.com/MarcelRogge/object-centric-2dgs
    """
   
    _base_densification: SplatDensification

    def __init__(
        self,
        baseDensification: SplatDensification,
    ):
        self._base_densification = baseDensification
        self._prune_unseen_intervals = 100
        self._seen: torch.Tensor | None = None
    

    @property
    def module_name(self) -> str:
        return "ObjectCentricDensification"
    
    @override
    def on_setup(self,
        logger: "SplatLogger",
        renderer: "SplatRenderer[SplatRenderPayload]",
        data_provider: "SplatDataProvider[SplatRenderPayload, ColmapDataItem]",
        loss_fn: "SplatLossFn[SplatRenderPayload]",
        densification: "SplatDensification[SplatRenderPayload]",
        modules: Sequence["SplatBaseModule[SplatRenderPayload]"], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):

        self._base_densification.on_setup(
            logger=SplatLogger(level="WARNING"), 
            renderer=renderer, 
            data_provider=data_provider, 
            loss_fn=loss_fn, 
            densification=densification, 
            modules=modules, 
            max_steps=max_steps, 
            world_rank=world_rank, 
            world_size=world_size, 
            scene_scale=scene_scale
        )
        
        # Per paper's suggestion,
        # Prune unseen gaussians once we iterate through all the training images twice
        self._prune_unseen_intervals = 2 * data_provider.get_train_data_size(world_rank=world_rank, world_size=world_size)

        logger.info(f"Successfully set up default densification strategy", module=self.module_name)
    
    @override
    def pre_compute_loss(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor, # (..., H, W, 3)
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayload,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):
        self._base_densification.pre_compute_loss(
            logger=logger,
            step=step,
            max_steps=max_steps,
            rendered_frames=rendered_frames,
            target_frames=target_frames,
            training_state=training_state,
            rend_out=rend_out,
            masks=masks,
            world_rank=world_rank,
            world_size=world_size,
        )
    
    def densify(self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor, # (..., H, W, 3)
        target_frames: torch.Tensor, # (..., H, W, 3)
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayload,
        masks: torch.Tensor | None = None, # (..., H, W)
        world_rank: int = 0,
        world_size: int = 1,
    ):

        # mark seen BEFORE default strategy potentially duplicates/splits/prunes
        self._sync_seen_length(training_state)
        self._update_seen(rend_out, masks)

        last_prune_mask = self._base_densification.densify(
            logger=logger,
            step=step,  
            max_steps=max_steps,
            rendered_frames=rendered_frames,
            target_frames=target_frames,
            training_state=training_state,
            rend_out=rend_out,
            masks=masks,
            world_rank=world_rank,
            world_size=world_size,
        )
        if last_prune_mask is not None:
            if self._seen is None:
                raise ValueError("Seen tensor is not initialized")
            if last_prune_mask.numel() != self._seen.numel():
                if last_prune_mask.numel() > self._seen.numel():
                    extra = torch.ones(
                        last_prune_mask.numel() - self._seen.numel(),
                        device=self._seen.device,
                        dtype=torch.bool,
                    )
                    self._seen = torch.cat([self._seen, extra], dim=0)
                    logger.warning(
                        "Seen length lagged prune mask; padding with seen=True",
                        module=self.module_name,
                    )
                else:
                    raise RuntimeError(f"Seen length exceeded prune mask: last_prune_mask.numel() = {last_prune_mask.numel()}, self._seen.numel() = {self._seen.numel()}")

            self._seen = self._seen[~last_prune_mask]

        # after strategy, params length may change, so resync
        self._sync_seen_length(training_state)

        if step >0 and step % self._prune_unseen_intervals == 0:
            n_pruned = self._prune_unseen(training_state)
            logger.info(f"Pruned {n_pruned} unseen gaussians", module=self.module_name)

    
    def _sync_seen_length(self, training_state: SplatTrainingState):
        num_gaussians = training_state.num_gaussians
        if self._seen is None:
            self._seen = torch.zeros(num_gaussians, dtype=torch.bool, device=training_state.device)
        
        cur_gaussians = self._seen.numel()
        if cur_gaussians < num_gaussians:
            # new gaussians were added
            extra = torch.ones(num_gaussians - cur_gaussians, device=self._seen.device, dtype=torch.bool)
            self._seen = torch.cat([self._seen, extra], dim=0)
        elif cur_gaussians > num_gaussians:
            # gaussians were removed
            raise RuntimeError("Pruned gaussians should be handled by desnsify function with last_prune_mask")

    def _update_seen(self, rend_out: SplatRenderPayload, masks: torch.Tensor | None = None):
        """
        Update per-Gaussian visibility (seen vector).

        A Gaussian is considered "seen" if:
        - it projects to a non-zero screen-space ellipse in at least one camera
        - AND (if masks are provided) that camera contains foreground pixels

        Notes:
        - self._seen has shape [N] (one entry per Gaussian)
        - rend_out.radii has shape [C, N, 2]
        - masks live in pixel space and therefore can only be applied at camera level
        """
        if self._seen is None:
            raise ValueError("Seen tensor is not initialized")

        gs_ids = rend_out.gaussian_ids
        if gs_ids is not None:
            if masks is None:
                self._seen[gs_ids] = True
            else:
                alphas = rend_out.alphas.flatten()          # [K]
                mask_flat = masks.flatten()                 # pixel space

                valid = (alphas > 0) & (mask_flat[:alphas.numel()] > 0)
                self._seen[gs_ids[valid]] = True
        else:
            radii = rend_out.radii              # [C, N, 2]
            visible = (radii > 0).all(dim=-1)    # [C, N]

            if masks is not None:
                fg_cam = masks.flatten(1).any(dim=1) # [C] - foreground cameras
                visible &= fg_cam[..., None]

            seen_filter = visible.any(dim=0) # [N]
            self._seen |= seen_filter
    
    def _prune_unseen(self, training_state: SplatTrainingState) -> int:
        if self._seen is None:
            raise ValueError("Seen tensor is not initialized")

        if self._seen.sum() == self._seen.numel():
            return 0
        
        unseen_filter = ~self._seen
        n_prunes = int(unseen_filter.sum().item())
        if n_prunes == 0:
            return 0

        remove(
            params=training_state.params,
            optimizers=training_state.optimizers,
            state=self._base_densification.state,
            mask=unseen_filter,
        )

        self._seen = self._seen[~unseen_filter]
        
        # reset seen for next visibility cycle (one full pass over training views)
        self._seen.zero_()
        return n_prunes
        
