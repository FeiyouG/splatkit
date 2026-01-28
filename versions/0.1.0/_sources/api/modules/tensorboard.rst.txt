TensorBoard Module
==================

.. currentmodule:: splatkit.modules.tensorboard

.. autoclass:: SplatTensorboard
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

``SplatTensorboard`` logs training metrics to TensorBoard for visualization and analysis.
It automatically logs:

- Training loss
- Learning rates for all parameter groups
- Gaussian count over time
- Custom metrics from your modules

View the logs by running ``tensorboard --logdir <log_dir>`` and opening
``http://localhost:6006`` in your browser.

Installation
------------

TensorBoard is included with PyTorch, but you can install it explicitly:

.. code-block:: bash

   pip install -e ".[tensorboard]"
   # or
   uv pip install -e ".[tensorboard]"

This installs: ``tensorboard``.

Quick Start
-----------

.. code-block:: python

   from splatkit.modules import SplatTensorboard
   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   
   # Create TensorBoard logger
   tensorboard = SplatTensorboard(
       log_dir="logs/experiment_1",
       log_every=10,  # Log every 10 steps
   )
   
   # Add to trainer
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=30000),
       data_provider=data_provider,
       renderer=renderer,
       loss_fn=loss_fn,
       densification=densification,
       modules=[tensorboard],
   )
   trainer.run()
   
   # In another terminal, start TensorBoard:
   # tensorboard --logdir logs/

Logged Metrics
--------------

The module automatically logs:

**Scalars**
  - ``train/loss``: Training loss value
  - ``train/lr_means``: Learning rate for Gaussian positions
  - ``train/lr_scales``: Learning rate for Gaussian scales
  - ``train/lr_quats``: Learning rate for Gaussian rotations
  - ``train/lr_opacities``: Learning rate for Gaussian opacities
  - ``train/lr_sh0``: Learning rate for 0th order SH coefficients
  - ``train/lr_shN``: Learning rate for higher order SH coefficients
  - ``train/num_gaussians``: Total number of Gaussians

**Images** (optional)
  - Rendered images at specified intervals
  - Ground truth images for comparison

Configuration Options
---------------------

``log_dir``
  Directory to save TensorBoard logs (required)
  
  Example: ``"logs/3dgs_experiment"``

``log_every``
  Log metrics every N steps (default: 10)
  
  - Lower = more detailed logs, larger file size
  - Higher = less overhead, less detail

``log_images``
  Log rendered and target images (default: False)
  
  Warning: Image logging can significantly increase log file size.

``log_image_every``
  Log images every N steps (default: 1000)
  
  Only used if ``log_images=True``.

Usage Notes
-----------

**Viewing Logs**

Start TensorBoard from the command line:

.. code-block:: bash

   tensorboard --logdir logs/
   # Then open http://localhost:6006

**Multiple Experiments**

Organize multiple experiments in subdirectories:

.. code-block:: text

   logs/
   ├── experiment_1/
   ├── experiment_2/
   └── experiment_3/

TensorBoard will show all experiments in the same interface:

.. code-block:: bash

   tensorboard --logdir logs/

**Distributed Training**

Only rank 0 writes to TensorBoard in distributed training.

**Log File Size**

TensorBoard logs grow over time. To keep file sizes manageable:

- Increase ``log_every`` to log less frequently
- Set ``log_images=False`` to disable image logging
- Delete old experiments you don't need

Examples
--------

**Basic Usage**

.. code-block:: python

   from splatkit.modules import SplatTensorboard
   
   tensorboard = SplatTensorboard(log_dir="logs/run1")
   # Add to trainer's modules list

**With Image Logging**

.. code-block:: python

   tensorboard = SplatTensorboard(
       log_dir="logs/with_images",
       log_every=10,
       log_images=True,
       log_image_every=500,  # Log images every 500 steps
   )

**Multiple Experiments**

.. code-block:: python

   # Experiment 1: High learning rate
   tensorboard_1 = SplatTensorboard(log_dir="logs/lr_high")
   
   # Experiment 2: Low learning rate
   tensorboard_2 = SplatTensorboard(log_dir="logs/lr_low")
   
   # Compare in TensorBoard:
   # tensorboard --logdir logs/

**With Other Modules**

.. code-block:: python

   from splatkit.modules import (
       SplatTensorboard,
       SplatProgressTracker,
       SplatViewer,
   )
   
   modules = [
       SplatProgressTracker(update_every=10),
       SplatTensorboard(log_dir="logs/", log_every=10),
       SplatViewer(port=8080, update_interval=5),
   ]

Custom Metrics
--------------

You can log custom metrics from your own modules:

.. code-block:: python

   from splatkit.modules.base import SplatBaseModule
   
   class MyModule(SplatBaseModule):
       def post_step(self, logger, step, **kwargs):
           # Log custom metric
           logger.log_scalar("my_metric", value, step)

See Also
--------

- :doc:`progress_tracker` - Terminal progress bars
- :doc:`viewer` - Real-time 3D visualization
- :doc:`../../customization/modules` - Writing custom modules
