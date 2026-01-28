Custom Densification
================================

Densification strategies in splatkit control how Gaussians are added, split, and removed during training. You can create custom densification strategies to implement novel adaptive refinement techniques.

Creating a Custom Densification Strategy
-----------------------------------------

All densification strategies must inherit from :class:`splatkit.densification.base.SplatDensification`:

Densification itself is also a subclass of :class:`splatkit.modules.base.SplatBaseModule`,
and all of its hooks will be called during training just like any other modules.

Here's a minimal densification implementation:

.. code-block:: python

   from splatkit.densification.base import SplatDensification
   from splatkit.modules.frame import SplatRenderPayload
   from splatkit.splat.training_state import SplatTrainingState
   import torch
   
   class SimpleDensification(SplatDensification[SplatRenderPayload]):
       """Simple densification based on gradient thresholds."""
       
       def __init__(
           self,
           grow_grad2d: float = 0.0002,
           prune_opa: float = 0.005,
           refine_start_iter: int = 500,
           refine_stop_iter: int = 15000,
           refine_every: int = 100,
       ):
           self.grow_grad2d = grow_grad2d
           self.prune_opa = prune_opa
           self.refine_start_iter = refine_start_iter
           self.refine_stop_iter = refine_stop_iter
           self.refine_every = refine_every
       
       def densify(
           self,
           step: int,
           max_steps: int,
           rendered_frames: torch.Tensor,
           target_frames: torch.Tensor,
           training_state: SplatTrainingState,
           rend_out: SplatRenderPayload,
           masks: torch.Tensor | None = None,
           world_rank: int = 0,
           world_size: int = 1,
       ):
           # Only densify during refinement window
           if step < self.refine_start_iter or step > self.refine_stop_iter:
               return
           
           if step % self.refine_every != 0:
               return
           
           # Get 2D gradients from render output
           if not hasattr(rend_out, 'means2d'):
               return
           
           means2d = rend_out.means2d
           if means2d is None or means2d.grad is None:
               return
           
           # Compute gradient magnitude
           grad_2d = means2d.grad.norm(dim=-1)  # (N,)
           
           # Clone high-gradient Gaussians
           high_grad_mask = grad_2d > self.grow_grad2d
           if high_grad_mask.any():
               training_state.duplicate(high_grad_mask)
           
           # Prune low-opacity Gaussians
           opacities = training_state.params["opacities"].squeeze(-1)
           low_opa_mask = opacities < self.prune_opa
           if low_opa_mask.any():
               training_state.remove(low_opa_mask)

Key Components
--------------

When implementing custom densification, you typically need to:

1. **Track gradients**: Access 2D screen-space gradients from ``rend_out.means2d.grad``
2. **Identify candidates**: Find Gaussians that need refinement based on your criteria
3. **Add Gaussians**: Use ``training_state.duplicate()`` to clone or ``training_state.split()`` to subdivide
4. **Remove Gaussians**: Use ``training_state.remove()`` to prune unwanted Gaussians

Advanced: Using gsplat's Strategy
-----------------------------------

For more sophisticated densification, you can leverage gsplat's built-in strategies:

.. code-block:: python

   from gsplat.strategy import DefaultStrategy
   from splatkit.densification.base import SplatDensification
   from splatkit.modules.frame import SplatRenderPayload
   
   class GsplatBasedDensification(SplatDensification[SplatRenderPayload]):
       """Densification using gsplat's DefaultStrategy."""
       
       def __init__(
           self,
           prune_opa: float = 0.005,
           grow_grad2d: float = 0.0002,
           grow_scale3d: float = 0.01,
           prune_scale3d: float = 0.1,
       ):
           self.strategy = DefaultStrategy(
               prune_opa=prune_opa,
               grow_grad2d=grow_grad2d,
               grow_scale3d=grow_scale3d,
               prune_scale3d=prune_scale3d,
           )
           self.state = {}
       
       def on_setup(self, logger, renderer, data_provider, loss_fn,
                    densification, modules, max_steps, **kwargs):
           super().on_setup(logger, renderer, data_provider, loss_fn,
                           densification, modules, max_steps, **kwargs)
           # Initialize strategy state
           N = len(training_state.params["means"])
           self.state = self.strategy.initialize_state(scene_scale=1.0)
       
       def densify(self, step, max_steps, rendered_frames, target_frames,
                   training_state, rend_out, masks=None, **kwargs):
           # Update strategy with current gradients and info
           self.strategy.step_post_backward(
               params=training_state.params,
               optimizers=training_state.optimizers,
               state=self.state,
               step=step,
               info=rend_out.info,  # Contains radii, etc.
           )

Distributed Training Considerations
------------------------------------

If your densification strategy needs to work with distributed training:

.. code-block:: python

   def densify(self, step, max_steps, rendered_frames, target_frames,
               training_state, rend_out, masks=None,
               world_rank=0, world_size=1):
       # Only the leader rank decides which Gaussians to add/remove
       if world_rank == 0:
           # Compute masks for cloning/pruning
           clone_mask = self.compute_clone_mask(...)
           prune_mask = self.compute_prune_mask(...)
           
           # Apply operations
           if clone_mask.any():
               training_state.duplicate(clone_mask)
           if prune_mask.any():
               training_state.remove(prune_mask)
       
       # SplatTrainingState automatically syncs across ranks

Using Your Densification Strategy
-----------------------------------

Pass your custom densification strategy to the trainer:

.. code-block:: python

   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   
   densification = MyCustomDensification(
       grow_grad2d=0.0005,
       prune_opa=0.01,
   )
   
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=30000),
       densification=densification,
       renderer=renderer,
       data_provider=data_provider,
   )
   trainer.run()

Best Practices
--------------

1. **Schedule refinement**: Only densify within a specific iteration range (e.g., 500-15000)
2. **Refine periodically**: Don't densify every step; use intervals (e.g., every 100 steps)
3. **Balance growth and pruning**: Too aggressive growth wastes memory; too aggressive pruning loses detail
4. **Use scene scale**: Scale thresholds based on scene size for consistent behavior across datasets
5. **Monitor counts**: Log Gaussian counts to ensure your strategy is working as expected

Advanced: Custom Refinement Criteria
--------------------------------------

You can implement sophisticated refinement strategies based on multiple criteria:

.. code-block:: python

   class MultiCriterionDensification(SplatDensification[SplatRenderPayload]):
       """Densification using multiple refinement signals."""
       
       def densify(self, step, max_steps, rendered_frames, target_frames,
                   training_state, rend_out, masks=None, **kwargs):
           # Criterion 1: High 2D gradient (under-reconstruction)
           grad_2d = rend_out.means2d.grad.norm(dim=-1)
           high_grad = grad_2d > self.grad_threshold
           
           # Criterion 2: Low depth confidence (uncertainty)
           if hasattr(rend_out, 'depth_variance'):
               high_uncertainty = rend_out.depth_variance > self.var_threshold
           else:
               high_uncertainty = torch.zeros_like(high_grad)
           
           # Criterion 3: Large screen-space footprint
           radii = rend_out.radii  # (N,)
           large_footprint = radii > self.radius_threshold
           
           # Combine criteria
           should_split = high_grad & large_footprint
           should_clone = high_grad & ~large_footprint
           should_prune = (training_state.params["opacities"] < 0.005).squeeze()
           
           # Apply operations
           if should_split.any():
               training_state.split(should_split)
           if should_clone.any():
               training_state.duplicate(should_clone)
           if should_prune.any():
               training_state.remove(should_prune)

See Also
--------

- :doc:`modules` - Writing custom training modules
- :doc:`renderers` - Writing custom renderers
- :doc:`loss_functions` - Writing custom loss functions
- :class:`splatkit.densification.base.SplatDensification` - Base densification API
- :class:`splatkit.densification.default.SplatDefaultDensification` - Default strategy example
- :class:`splatkit.densification.mcmc.SplatMCMCDensification` - MCMC strategy example
