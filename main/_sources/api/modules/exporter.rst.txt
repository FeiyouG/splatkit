Exporter Module
===============

.. currentmodule:: splatkit.modules.exporter

.. autoclass:: SplatExporter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

``SplatExporter`` automatically saves trained models and training checkpoints
at specified steps. It can export:

- **PLY files**: Compact trained Gaussian models for visualization and rendering
- **Checkpoints**: Full training state for resuming training

Exports are saved to organized subdirectories with step numbers in filenames.

Quick Start
-----------

.. code-block:: python

   from splatkit.modules import SplatExporter
   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   
   # Create exporter
   exporter = SplatExporter(
       output_dir="results/output",
       export_steps=[7_000, 15_000, 30_000],
       save_splat=True,   # Save PLY files
       save_ckpt=True,    # Save checkpoints
   )
   
   # Add to trainer
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=30000),
       data_provider=data_provider,
       renderer=renderer,
       loss_fn=loss_fn,
       densification=densification,
       modules=[exporter],
   )
   trainer.run()
   
   # Files saved to:
   #   results/output/splats/step_007000.ply
   #   results/output/splats/step_015000.ply
   #   results/output/splats/step_030000.ply
   #   results/output/ckpts/step_007000.ckpt
   #   results/output/ckpts/step_015000.ckpt
   #   results/output/ckpts/step_030000.ckpt

Output Files
------------

**PLY Files** (``splats/step_{step:06d}.ply``)

Compact Gaussian splat models containing:
- Gaussian positions, scales, rotations
- Opacities and SH coefficients
- All parameters needed for rendering

PLY files can be viewed in:
- Standard 3D viewers (MeshLab, CloudCompare)
- Web-based Gaussian splat viewers
- Custom rendering pipelines

Typical size: 10-100 MB depending on Gaussian count

**Checkpoint Files** (``ckpts/step_{step:06d}.ckpt``)

Full training state including:
- All Gaussian parameters
- Optimizer states (momentum, etc.)
- Training step counter
- Random number generator state

Use checkpoints to resume training from a specific step.

Typical size: 100-500 MB (larger than PLY due to optimizer state)

Configuration Options
---------------------

``output_dir``
  Directory to save exports (required)
  
  Creates subdirectories: ``splats/`` and ``ckpts/``

``export_steps``
  List of training steps at which to export (default: [])
  
  Example: ``[5000, 10000, 30000]``
  
  Tip: Include your final step to save the trained model

``save_splat``
  Save PLY files (default: True)
  
  Recommended: Always enable to save trained models

``save_ckpt``
  Save training checkpoints (default: True)
  
  Enable if you might need to resume training

``splat_format``
  Export format for splat models (default: "ply")
  
  Currently only "ply" is supported

Usage Notes
-----------

**Disk Space**

Both PLY files and checkpoints can be large. For a 30k step training run
with exports at [7k, 15k, 30k]:

- PLY files: ~30-300 MB total
- Checkpoints: ~300-1500 MB total

Monitor disk space if exporting frequently.

**Distributed Training**

Only rank 0 saves files in distributed training. Other ranks skip export.

**Resuming Training**

To resume from a checkpoint:

.. code-block:: python

   trainer = SplatTrainer(
       config=config,
       data_provider=data_provider,
       renderer=renderer,
       loss_fn=loss_fn,
       densification=densification,
       ckpt_path="results/output/ckpts/step_015000.ckpt",  # Resume from here
   )
   trainer.run()

**Loading PLY Files**

To load a trained PLY model:

.. code-block:: python

   from splatkit.splat import SplatModel
   
   model = SplatModel.load_ply("results/output/splats/step_030000.ply")
   
   # Convert to training state if needed
   training_state = model.to_training_state(
       learning_rates={...},
       device="cuda:0"
   )

Examples
--------

**Basic Usage**

.. code-block:: python

   from splatkit.modules import SplatExporter
   
   exporter = SplatExporter(
       output_dir="results/my_scene",
       export_steps=[30000],  # Only export final model
   )

**Frequent Exports**

For detailed checkpointing:

.. code-block:: python

   exporter = SplatExporter(
       output_dir="results/frequent",
       export_steps=list(range(5000, 30001, 5000)),  # Every 5000 steps
       save_splat=True,
       save_ckpt=True,
   )

**PLY Only**

Save disk space by only exporting PLY files:

.. code-block:: python

   exporter = SplatExporter(
       output_dir="results/ply_only",
       export_steps=[7000, 15000, 30000],
       save_splat=True,   # Save PLY
       save_ckpt=False,   # Don't save checkpoints
   )

**Checkpoints Only**

For experiments where you only need to resume training:

.. code-block:: python

   exporter = SplatExporter(
       output_dir="results/ckpt_only",
       export_steps=[10000, 20000, 30000],
       save_splat=False,   # Don't save PLY
       save_ckpt=True,     # Save checkpoints
   )

**With Other Modules**

.. code-block:: python

   from splatkit.modules import (
       SplatExporter,
       SplatEvaluator,
       SplatProgressTracker,
   )
   
   modules = [
       SplatProgressTracker(update_every=10),
       SplatEvaluator(
           output_dir="results/eval",
           eval_steps=[7000, 15000, 30000],
       ),
       SplatExporter(
           output_dir="results/export",
           export_steps=[7000, 15000, 30000],
           save_splat=True,
           save_ckpt=True,
       ),
   ]

Best Practices
--------------

1. **Always save final model**: Include your ``max_steps`` in ``export_steps``

2. **Balance frequency**: More exports = more disk space but better recovery options

3. **Keep PLY files**: Delete old checkpoints but keep PLY files for rendering

4. **Organize by experiment**: Use descriptive ``output_dir`` names

5. **Backup important models**: Copy PLY files to safe storage after training

See Also
--------

- :doc:`evaluator` - Compute quality metrics
- :doc:`../../customization/modules` - Writing custom modules
- :class:`splatkit.splat.model.SplatModel` - Model loading and saving
