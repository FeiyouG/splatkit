Quick Start
===========

This guide shows you how to train your first Gaussian Splatting model.

.. note::
   Make sure you've installed splatkit first! See :doc:`installation` for instructions.

Training Your First Model
-------------------------

3D Gaussian Splatting
~~~~~~~~~~~~~~~~~~~~~

Here's a complete example to train a 3D Gaussian Splatting model on COLMAP data:

.. code-block:: python

   from splatkit.trainer import Trainer, TrainerConfig
   from splatkit.data_provider import ColmapDataProvider
   from splatkit.renderer import Splat3DGSRenderer
   from splatkit.loss_fn import Splat3DGSLossFn
   from splatkit.densification import SplatDefaultDensification

   # Configure the trainer
   config = TrainerConfig(
       output_dir="outputs/my_scene",
       max_steps=30000,
   )

   # Set up data provider
   data_provider = ColmapDataProvider(
       data_dir="path/to/colmap/data",
       data_factor=1,
   )

   # Create and run trainer
   trainer = Trainer(
        config=config, 
        renderer=Splat3DGSRenderer(),
        loss_fn=Splat3DGSLossFn(),
        data_provider=data_provider,
        densification=SplatDefaultDensification(),
    )
   trainer.train()

The trainer will automatically use the default 3DGS renderer and loss function.

2D Gaussian Splatting
~~~~~~~~~~~~~~~~~~~~~

To use 2D Gaussian Splatting instead, specify the 2DGS components:

.. code-block:: python

   from splatkit.trainer import Trainer, TrainerConfig
   from splatkit.renderer import Splat2DGSRenderer
   from splatkit.loss_fn import Splat2DGSLossFn
   from splatkit.data_provider import ColmapDataProvider
   from splatkit.densification import SplatDefaultDensification

   config = TrainerConfig(
       output_dir="outputs/my_2dgs_scene",
       max_steps=30000,
   )

   data_provider = ColmapDataProvider(
       data_dir="path/to/colmap/data",
       data_factor=1,
   )

   trainer = Trainer(
       config=config,
       renderer=Renderer2DGS(),
       loss_fn=Loss2DGS(),
       data_provider=data_provider,
       densification=SplatDefaultDensification(),
   )
   trainer.train()

With Viewer, Progress, and TensorBoard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a full-featured training setup with visualization:

.. code-block:: bash

   # Install with all features
   pip install -e ".[viewer,progress,tensorboard]"
   # or
   uv pip install -e ".[viewer,progress,tensorboard]"

.. code-block:: python

   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   from splatkit.data_provider import SplatColmapDataProvider, SplatColmapDataProviderConfig
   from splatkit.renderer import Splat3DGSRenderer
   from splatkit.loss_fn import Splat3DGSLossFn
   from splatkit.densification import SplatDefaultDensification
   from splatkit.modules import (
       SplatViewer,          # Real-time visualization
       SplatProgressTracker, # Progress bars
       SplatTensorboard,     # TensorBoard logging
   )
   
   config = SplatTrainerConfig(
       max_steps=30000,
       output_dir="outputs/my_scene",
   )
   
   data_provider = SplatColmapDataProvider(
       config=SplatColmapDataProviderConfig(
           colmap_dir="data/sparse/0",
           images_dir="data/images",
           factor=1,
           normalize=True,
       )
   )
   
   modules = [
       SplatProgressTracker(update_every=10),
       SplatViewer(port=8080, mode="training"),
       SplatTensorboard(log_dir="outputs/logs"),
   ]
   
   trainer = SplatTrainer(
       config=config,
       data_provider=data_provider,
       renderer=Splat3DGSRenderer(),
       loss_fn=Splat3DGSLossFn(),
       densification=SplatDefaultDensification(),
       modules=modules,
   )
   trainer.run()

Then:
- Open ``http://localhost:8080`` for real-time viewer
- Run ``tensorboard --logdir outputs/logs`` for metrics
- Watch progress bar in terminal

Next Steps
----------

* Check out the ``examples/`` directory for complete examples
* Learn how to write custom components in the :doc:`customization/index` guide
* Explore the :doc:`api/index` for detailed API documentation
* Customize training parameters in :doc:`api/trainer`
