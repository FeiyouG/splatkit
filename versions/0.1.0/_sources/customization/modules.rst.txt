Custom Modules
======================

Modules in splatkit provide hooks into the training loop, allowing you to add custom functionality like logging, visualization, evaluation, or any other training-time operations.

Creating a Custom Module
-------------------------

All modules must inherit from :class:`splatkit.modules.base.SplatBaseModule`:

Here's a complete example of a module that saves checkpoints:

.. code-block:: python

   import os
   from pathlib import Path
   from splatkit.modules.base import SplatBaseModule
   from splatkit.modules.frame import SplatRenderPayload
   
   class CheckpointModule(SplatBaseModule[SplatRenderPayload]):
       """Save training checkpoints at regular intervals."""
       
       def __init__(self, output_dir: str, save_every: int = 1000):
           self.output_dir = Path(output_dir)
           self.save_every = save_every
       
       def on_setup(self, logger, renderer, data_provider, loss_fn,
                    densification, modules, max_steps, **kwargs):
           """Create output directory."""
           self.output_dir.mkdir(parents=True, exist_ok=True)
           logger.info(f"Checkpoints will be saved to {self.output_dir}")
       
       def post_step(self, logger, step, max_steps, loss, training_state,
                     rend_out, target_frames, **kwargs):
           """Save checkpoint at regular intervals."""
           if step % self.save_every == 0 or step == max_steps:
               ckpt_path = self.output_dir / f"step_{step:06d}.pth"
               training_state.save_ckpt(str(ckpt_path))
               logger.info(f"Saved checkpoint to {ckpt_path}")

Using Your Module
-----------------

Add your custom module to the trainer:

.. code-block:: python

   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   
   modules = [
       MyCustomModule(my_param=20),
       CheckpointModule(output_dir="checkpoints", save_every=500),
       # ... other modules
   ]
   
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=30000),
       data_provider=data_provider,
       modules=modules,
   )
   trainer.run()

Best Practices
--------------

1. **Keep hooks lightweight**: Avoid expensive operations in frequently-called hooks like ``pre_step``
2. **Use world_rank**: Only log or save from rank 0 in distributed training
3. **Handle failures gracefully**: Use try-except blocks for non-critical operations
4. **Provide clear parameters**: Make your module configurable via ``__init__`` parameters
5. **Initialize resources in on_setup**: Avoid initializing resources in ``__init__`` so the modules can be pickled and used in distributed training.

See Also
--------

- :doc:`renderers` - Writing custom renderers
- :doc:`loss_functions` - Writing custom loss functions
- :class:`splatkit.modules.base.SplatBaseModule` - Base module API reference
