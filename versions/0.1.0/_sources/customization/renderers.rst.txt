Custom Renderers
========================

Renderers in splatkit are responsible for rasterizing 3D Gaussians into 2D images. You can create custom renderers to implement new splatting techniques or rendering algorithms.

Creating a Custom Renderer
---------------------------

All renderers must inherit from :class:`splatkit.renderer.base.SplatRenderer`:

Renderer itself is also a subclass of :class:`splatkit.modules.base.SplatBaseModule`,
and all of it hooks will be called during training just like any other modules.

Here's a minimal renderer implementation:

.. code-block:: python

   from splatkit.renderer.base import SplatRenderer
   from splatkit.modules.frame import SplatRenderPayload
   from splatkit.splat.training_state import SplatTrainingState
   import torch
   import gsplat
   
   class SimpleRenderer(SplatRenderer[SplatRenderPayload]):
       """Simple 3D Gaussian Splatting renderer."""
       
       def __init__(self, background_color: tuple = (0, 0, 0)):
           super().__init__()
           self.background = torch.tensor(
               background_color, dtype=torch.float32
           )
       
       def render(
           self,
           training_state: SplatTrainingState,
           Ks: torch.Tensor,
           cam_to_worlds: torch.Tensor,
           image_height: int,
           image_width: int,
           sh_degree: int,
           **kwargs
       ) -> SplatRenderPayload:
           # Get Gaussian parameters
           means = training_state.params["means"]  # (N, 3)
           scales = training_state.params["scales"]  # (N, 3)
           quats = training_state.params["quats"]  # (N, 4)
           opacities = training_state.params["opacities"]  # (N, 1)
           colors = training_state.colors(sh_degree)  # (N, 3)
           
           # Rasterize using gsplat
           renders, alphas, info = gsplat.rasterization(
               means=means,
               scales=scales,
               quats=quats,
               opacities=opacities,
               colors=colors,
               viewmats=torch.linalg.inv(cam_to_worlds),
               Ks=Ks,
               width=image_width,
               height=image_height,
               backgrounds=self.background.to(means.device),
           )
           
           return SplatRenderPayload(
               renders=renders,
               alphas=alphas,
               radii=info["radii"],
               depths=info.get("depths"),
           )

Custom Render Payloads & Type Safety
-------------------------------------

For more advanced rendering, you can subclass :class:`SplatRenderPayload` to include 
custom data (e.g., normal maps, depth buffers, auxiliary outputs):

.. code-block:: python

   from dataclasses import dataclass
   from splatkit.modules.frame import SplatRenderPayload
   
   @dataclass
   class MyCustomPayload(SplatRenderPayload):
       """Custom payload with additional data."""
       normals: torch.Tensor | None = None
       auxiliary_output: torch.Tensor | None = None

Subclassing :class:`SplatRenderPayload` provides **type hints for all other modules** 
used in the same training loop (loss functions, custom modules, etc.). Your IDE and 
type checker can now catch errors before runtime, making your code more robust.

.. note::
   Python's weak type system doesn't enforce the payload type inheritance at compile 
   time, but :class:`SplatTrainer` includes **runtime checks** to verify that your 
   renderer's output matches the expected payload type for a given trainer.
   If there's a mismatch, you'll get a clear error message during trainer initialization.

Using Your Renderer
-------------------

Pass your custom renderer to the trainer:

.. code-block:: python

   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   from splatkit.data_provider import ColmapDataItem
   
   renderer = MyCustomRenderer(my_param=2.0)
   
   trainer = SplatTrainer[ColmapDataItem, MyCustomRenderPayload](
       config=SplatTrainerConfig(max_steps=30000),
       renderer=renderer,
       ...
   ).run()

Best Practices
--------------

1. **Use custom payloads for type safety**: Subclass :class:`SplatRenderPayload` when adding custom outputs to get type hints across your entire training loop
2. **Gradient compatibility**: Ensure all operations support backpropagation
3. **Device handling**: Keep tensors on the correct device

See Also
--------

- :doc:`modules` - Writing custom training modules  
- :doc:`loss_functions` - Writing custom loss functions
- :class:`splatkit.renderer.base.SplatRenderer` - Base renderer API
- :class:`splatkit.renderer._3dgs.Splat3DGSRenderer` - 3DGS example
- :class:`splatkit.renderer._2dgs.Splat2DGSRenderer` - 2DGS example
