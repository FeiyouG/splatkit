Viewer Module
=============

.. currentmodule:: splatkit.modules.viewer

.. autoclass:: SplatViewer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

``SplatViewer`` provides real-time 3D visualization during training using
viser and nerfview. It opens a web interface where you can:

- Navigate the scene from any angle
- Switch between visualization modes (RGB, depth, alpha)
- Adjust rendering parameters (SH degree, background, etc.)
- View live training statistics

The viewer runs in a separate thread and doesn't block training.

Installation
------------

The viewer requires additional dependencies:

.. code-block:: bash

   pip install -e ".[viewer]"
   # or
   uv pip install -e ".[viewer]"

This installs: ``nerfview``, ``viser``, and ``splines``.

Quick Start
-----------

.. code-block:: python

   from splatkit.modules import SplatViewer
   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   
   # Create viewer module
   viewer = SplatViewer(
       port=8080,                 # Web server port
       update_interval=1,         # Update every step
       mode="training",           # Training mode
   )
   
   # Add to trainer
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=30000),
       data_provider=data_provider,
       renderer=renderer,
       loss_fn=loss_fn,
       densification=densification,
       modules=[viewer],  # Include viewer
   )
   trainer.run()
   
   # Open http://localhost:8080 in browser

Features
--------

**Navigation**
  - Free-flight camera controls
  - Orbit around scene center
  - Smooth interpolation

**Visualization Modes**
  - RGB rendering (default)
  - Depth visualization with colormaps
  - Alpha channel visualization
  - Custom modes from your renderer

**Controls**
  - Adjust SH degree in real-time
  - Change background color
  - Toggle depth normalization
  - Select colormaps (turbo, viridis, etc.)

**Statistics**
  - Frames per second (FPS)
  - Gaussian count (total and visible)
  - Rays per second
  - Current training step

Configuration Options
---------------------

``port``
  Web server port (default: 8080). Access at ``http://localhost:{port}``

``output_dir``
  Directory for screenshots and exports (default: "viewer_output")

``update_interval``
  Update viewer every N training steps (default: 1)
  
  - Lower = smoother visualization, slower training
  - Higher = faster training, less frequent updates

``mode``
  Viewer mode:
  
  - ``"training"``: Updates during training, shows live progress
  - ``"rendering"``: Static viewing, no training updates

``verbose``
  Print viser server debug logs (default: False)

Usage Notes
-----------

**Performance**

The viewer renders on GPU and may compete with training for memory.
If you experience out-of-memory errors:

- Increase ``update_interval`` to update less frequently
- Use a smaller viewport resolution in the browser
- Close the viewer tab when not needed

**Distributed Training**

The viewer only runs on rank 0 in distributed training. Other ranks
ignore the viewer module.

**Port Conflicts**

If port 8080 is already in use, change the port number:

.. code-block:: python

   viewer = SplatViewer(port=8081)  # Use different port

Examples
--------

**Minimal Setup**

.. code-block:: python

   from splatkit.modules import SplatViewer
   
   viewer = SplatViewer(port=8080)
   # Add to trainer's modules list

**Custom Configuration**

.. code-block:: python

   viewer = SplatViewer(
       port=8888,
       output_dir="outputs/viewer",
       update_interval=10,  # Update every 10 steps
       mode="training",
       verbose=True,  # Show server logs
   )

**With Multiple Modules**

.. code-block:: python

   from splatkit.modules import SplatViewer, SplatProgressTracker, SplatTensorboard
   
   modules = [
       SplatProgressTracker(update_every=10),
       SplatViewer(port=8080, update_interval=5),
       SplatTensorboard(log_dir="logs/"),
   ]
   
   trainer = SplatTrainer(
       config=config,
       data_provider=data_provider,
       renderer=renderer,
       loss_fn=loss_fn,
       densification=densification,
       modules=modules,
   )

See Also
--------

- :doc:`progress_tracker` - Terminal progress bars
- :doc:`tensorboard` - TensorBoard logging
- :doc:`../../customization/modules` - Writing custom modules
