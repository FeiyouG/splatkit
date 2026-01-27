Base Module
===========


``SplatBaseModule`` is the abstract base class for all modules in splatkit.
It defines the lifecycle hooks that modules can implement to integrate with
the training loop.

All modules are generic over the render payload type, allowing them to work
with different renderers (3DGS, 2DGS, etc.).


.. currentmodule:: splatkit.modules.base

.. autoclass:: SplatBaseModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__


Subclassing Example
----------------------

.. code-block:: python

   from splatkit.modules.base import SplatBaseModule
   from splatkit.modules.frame import SplatRenderPayload
   
   class MyModule(SplatBaseModule[SplatRenderPayload]):
       def __init__(self, param: int = 10):
           self.param = param
       
       def on_setup(self, logger, renderer, data_provider, loss_fn,
                    densification, modules, max_steps, **kwargs):
           logger.info(f"MyModule initialized with param={self.param}")
       
       def post_step(self, logger, step, max_steps, loss, training_state,
                     rend_out, target_frames, **kwargs):
           if step % 100 == 0:
               logger.info(f"Step {step}: Loss = {loss.item():.4f}")

See Also
--------

- :doc:`../../customization/modules` - Guide to writing custom modules
- :doc:`viewer` - Real-time visualization module
- :doc:`progress_tracker` - Progress bar module
- :doc:`tensorboard` - TensorBoard logging module
