Evaluator Module
================

.. currentmodule:: splatkit.modules.evaluator

.. autoclass:: SplatEvaluator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

``SplatEvaluator`` computes reconstruction quality metrics on validation data
at specified training steps. It measures:

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better, measures pixel-level accuracy
- **SSIM** (Structural Similarity Index): 0-1 scale, measures perceptual similarity
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better, perceptual distance

Results are saved to JSON files and optionally printed to the console.

Quick Start
-----------

.. code-block:: python

   from splatkit.modules import SplatEvaluator
   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   
   # Create evaluator
   evaluator = SplatEvaluator(
       output_dir="results/eval",
       eval_steps=[7_000, 15_000, 30_000],  # When to evaluate
       save_images=True,   # Save comparison images
       save_stats=True,    # Save metrics to JSON
   )
   
   # Add to trainer
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=30000),
       data_provider=data_provider,
       renderer=renderer,
       loss_fn=loss_fn,
       densification=densification,
       modules=[evaluator],
   )
   trainer.run()
   
   # Results saved to:
   #   results/eval/stats/metrics_007000.json
   #   results/eval/stats/metrics_015000.json
   #   results/eval/stats/metrics_030000.json

Output Format
-------------

**Metrics JSON** (``stats/metrics_{step}.json``):

.. code-block:: javascript

   {
     "step": 30000,
     "psnr_mean": 28.45,
     "ssim_mean": 0.923,
     "lpips_mean": 0.089,
     "per_image": [
       {
         "image_name": "image_001.jpg",
         "psnr": 29.12,
         "ssim": 0.931,
         "lpips": 0.082
       }
       // ... more images
     ]
   }

**Comparison Images** (``images/{step}/{image_name}.png``):

Side-by-side images showing Ground Truth | Rendered output.

Configuration Options
---------------------

``output_dir``
  Directory to save evaluation results (required)

``eval_steps``
  List of training steps at which to evaluate (default: [])
  
  Example: ``[7000, 15000, 30000]``

``save_images``
  Save GT vs Rendered comparison images (default: False)
  
  Warning: Can create large amounts of data if validation set is large.

``save_stats``
  Save metrics to JSON files (default: True)

``lpips_net``
  LPIPS network backbone (default: "alex")
  
  - ``"alex"``: AlexNet (faster, good performance)
  - ``"vgg"``: VGG16 (slower, slightly better)

``log_to_console``
  Print metrics to terminal (default: True)

``ckpt_path``
  Optional checkpoint path for standalone evaluation (default: None)

Usage Notes
-----------

**Evaluation Overhead**

Evaluation runs on the GPU and renders all validation images. This can be slow
for large validation sets. Consider:

- Fewer eval_steps for faster training
- Smaller validation set
- Disabling image saving (``save_images=False``)

**LPIPS Weights**

On first use, LPIPS will download pretrained weights (~100MB). This happens
automatically and is cached for future runs.

**Distributed Training**

Evaluation only runs on rank 0 in distributed training.

Examples
--------

**Basic Usage**

.. code-block:: python

   from splatkit.modules import SplatEvaluator
   
   evaluator = SplatEvaluator(
       output_dir="results/eval",
       eval_steps=[7000, 30000],
   )

**With Image Saving**

.. code-block:: python

   evaluator = SplatEvaluator(
       output_dir="results/eval",
       eval_steps=[10000, 20000, 30000],
       save_images=True,      # Save comparisons
       save_stats=True,       # Save JSON metrics
       log_to_console=True,   # Print to terminal
   )

**Frequent Evaluation**

For detailed tracking:

.. code-block:: python

   evaluator = SplatEvaluator(
       output_dir="results/eval",
       eval_steps=list(range(1000, 30001, 1000)),  # Every 1000 steps
       save_images=False,  # Don't save images (too many)
   )

**VGG Network**

For potentially better LPIPS scores:

.. code-block:: python

   evaluator = SplatEvaluator(
       output_dir="results/eval",
       eval_steps=[30000],
       lpips_net="vgg",  # Use VGG instead of AlexNet
   )

**Standalone Evaluation**

Evaluate a trained checkpoint without training:

.. code-block:: python

   evaluator = SplatEvaluator(
       output_dir="results/eval",
       eval_steps=[0],  # Evaluate immediately
       ckpt_path="checkpoints/model.ckpt",
   )
   
   # Run evaluation
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=0),  # Don't train
       data_provider=data_provider,
       renderer=renderer,
       loss_fn=loss_fn,
       densification=densification,
       modules=[evaluator],
   )
   trainer.run()

Interpreting Metrics
--------------------

**PSNR (Peak Signal-to-Noise Ratio)**
  - Typical range: 20-35 dB for good reconstructions
  - Higher is better
  - Measures pixel-level accuracy
  - Sensitive to noise and blur

**SSIM (Structural Similarity)**
  - Range: 0-1
  - Higher is better (1 = perfect match)
  - Measures perceptual similarity
  - Better correlates with human perception than PSNR

**LPIPS (Learned Perceptual Image Patch Similarity)**
  - Typical range: 0.0-0.3 for good reconstructions
  - Lower is better (0 = perfect match)
  - Best correlates with human perception
  - Uses deep learning features

See Also
--------

- :doc:`exporter` - Save checkpoints and models
- :doc:`progress_tracker` - Training progress bars
- :doc:`../../customization/modules` - Writing custom modules
