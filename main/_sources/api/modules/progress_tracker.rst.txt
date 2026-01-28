Progress Tracker Module
=======================

.. currentmodule:: splatkit.modules.progress_tracker

.. autoclass:: SplatProgressTracker
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

``SplatProgressTracker`` displays a progress bar in the terminal during training
using tqdm. It shows:

- Current step and total steps
- Training progress percentage
- Current loss value
- Steps per second (training speed)
- Estimated time remaining

Installation
------------

Progress tracker requires tqdm:

.. code-block:: bash

   pip install -e ".[progress]"
   # or
   uv pip install -e ".[progress]"

This installs: ``tqdm``.

Quick Start
-----------

.. code-block:: python

   from splatkit.modules import SplatProgressTracker
   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   
   # Create progress tracker
   progress = SplatProgressTracker(
       update_every=10,  # Update every 10 steps
   )
   
   # Add to trainer
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=30000),
       data_provider=data_provider,
       renderer=renderer,
       loss_fn=loss_fn,
       densification=densification,
       modules=[progress],
   )
   trainer.run()

Output Example
--------------

The progress bar displays information like this:

.. code-block:: text

   Training: 45%|████▌     | 13500/30000 [02:15<02:45, 99.8it/s, loss=0.0234]

Where:
- ``45%``: Progress percentage
- ``13500/30000``: Current step / Total steps
- ``02:15<02:45``: Elapsed time < Remaining time
- ``99.8it/s``: Steps per second
- ``loss=0.0234``: Current loss value

Configuration Options
---------------------

``update_every``
  Update progress bar every N steps (default: 10)
  
  - Lower = more frequent updates, more accurate ETA
  - Higher = less overhead, smoother training

``desc``
  Progress bar description (default: "Training")

Usage Notes
-----------

**Jupyter Notebooks**

The progress bar automatically adapts for Jupyter notebooks, displaying
a graphical progress widget.

**Distributed Training**

Progress bars only display on rank 0 in distributed training. Other ranks
run silently.

**Multiple Progress Bars**

You can create multiple progress trackers with different update rates:

.. code-block:: python

   modules = [
       SplatProgressTracker(update_every=1, desc="Fast"),    # Every step
       SplatProgressTracker(update_every=100, desc="Slow"),  # Every 100 steps
   ]

However, this is rarely useful and can clutter the terminal.

Examples
--------

**Basic Usage**

.. code-block:: python

   from splatkit.modules import SplatProgressTracker
   
   progress = SplatProgressTracker(update_every=10)
   # Add to trainer's modules list

**Custom Description**

.. code-block:: python

   progress = SplatProgressTracker(
       update_every=5,
       desc="3DGS Training",
   )

**With Other Modules**

.. code-block:: python

   from splatkit.modules import (
       SplatProgressTracker,
       SplatViewer,
       SplatTensorboard,
   )
   
   modules = [
       SplatProgressTracker(update_every=10),
       SplatViewer(port=8080, update_interval=1),
       SplatTensorboard(log_dir="logs/"),
   ]

**Frequent Updates**

For very detailed progress tracking:

.. code-block:: python

   progress = SplatProgressTracker(update_every=1)  # Update every step

Note: Frequent updates may slightly slow down training due to terminal I/O.

See Also
--------

- :doc:`viewer` - Real-time 3D visualization
- :doc:`tensorboard` - TensorBoard logging
- :doc:`../../customization/modules` - Writing custom modules
