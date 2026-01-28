Custom Loss Functions
==============================

Loss functions in splatkit compute the difference between rendered and ground truth images. You can create custom loss functions to implement new training objectives.

Creating a Custom Loss Function
--------------------------------

All loss functions must inherit from :class:`splatkit.loss_fn.base.SplatLossFn`:

Loss function itself is also a subclass of :class:`splatkit.modules.base.SplatBaseModule`,
and all of it hooks will be called during training just like any other modules.

Here's a complete loss function combining multiple terms:

.. code-block:: python

   from splatkit.loss_fn.base import SplatLossFn
   from splatkit.modules.frame import SplatRenderPayload
   from splatkit.splat.training_state import SplatTrainingState
   import torch
   from fused_ssim import fused_ssim
   
   class CombinedLoss(SplatLossFn[SplatRenderPayload]):
       """Combined L1 + SSIM loss with optional depth."""
       
       def __init__(
           self,
           lambda_l1: float = 0.8,
           lambda_ssim: float = 0.2,
           lambda_depth: float = 0.0,
       ):
           self.lambda_l1 = lambda_l1
           self.lambda_ssim = lambda_ssim
           self.lambda_depth = lambda_depth
       
       def compute_loss(
           self,
           rendered_frames: torch.Tensor,
           target_frames: torch.Tensor,
           training_state: SplatTrainingState,
           rend_out: SplatRenderPayload,
           masks: torch.Tensor | None = None,
           step: int = 0,
           max_steps: int = 30000,
       ) -> torch.Tensor:
           # L1 loss
           l1 = torch.abs(rendered_frames - target_frames)
           if masks is not None:
               l1 = l1 * masks.unsqueeze(-1)
               l1 = l1.sum() / (masks.sum() * 3 + 1e-8)
           else:
               l1 = l1.mean()
           
           # SSIM loss
           ssim_loss = 0.0
           if self.lambda_ssim > 0:
               # Reshape for fused_ssim: (B, 3, H, W)
               rendered = rendered_frames.permute(0, 3, 1, 2)
               target = target_frames.permute(0, 3, 1, 2)
               ssim_val = fused_ssim(rendered, target, padding="valid")
               ssim_loss = 1.0 - ssim_val.mean()
           
           # Depth loss (if available)
           depth_loss = 0.0
           if self.lambda_depth > 0 and hasattr(rend_out, 'depths'):
               if rend_out.depths is not None:
                   # Assume target depth is stored somewhere
                   # (this is dataset-dependent)
                   pass
           
           # Combined loss
           total_loss = (
               self.lambda_l1 * l1 +
               self.lambda_ssim * ssim_loss +
               self.lambda_depth * depth_loss
           )
           
           return total_loss

Using Your Loss Function
-------------------------

Pass your custom loss function to the trainer:

.. code-block:: python

   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   
   loss_fn = MyCustomLoss(lambda_l1=0.9, lambda_ssim=0.1)
   
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=30000),
       loss_fn=loss_fn,
       data_provider=data_provider,
   )
   trainer.run()

Best Practices
--------------

1. **Normalize losses**: Keep different loss terms in similar magnitude ranges
2. **Use masks carefully**: Handle both masked and unmasked cases
3. **Regularization**: Access ``training_state`` and ``rend_out`` for advanced terms

Advanced: Scheduled Losses
---------------------------

You can schedule loss weights based on training progress:

.. code-block:: python

   class ScheduledLoss(SplatLossFn[SplatRenderPayload]):
       """Loss with scheduled regularization."""
       
       def compute_loss(self, rendered_frames, target_frames,
                       training_state, rend_out, masks=None,
                       step=0, max_steps=30000):
           # Base image loss
           image_loss = torch.abs(rendered_frames - target_frames).mean()
           
           # Gradually increase regularization weight
           reg_weight = min(step / 10000, 1.0)
           
           # Some regularization term
           reg_loss = self.compute_regularization(training_state)
           
           return image_loss + reg_weight * reg_loss

See Also
--------

- :doc:`modules` - Writing custom training modules
- :doc:`renderers` - Writing custom renderers
- :class:`splatkit.loss_fn.base.SplatLossFn` - Base loss function API
- :class:`splatkit.loss_fn._3dgs.Splat3DGSLossFn` - 3DGS example
- :class:`splatkit.loss_fn._2dgs.Splat2DGSLossFn` - 2DGS example
