# SplatKit

A modular framework for 3D Gaussian Splatting training powered by [gsplat](https://github.com/nerfstudio-project/gsplat). SplatKit is designed to make it easy to share, recreate, and extend experiments from research papers.

## What is SplatKit?

SplatKit provides a composable architecture for training 3D Gaussian Splatting models. Instead of a monolithic training script, SplatKit breaks down the training pipeline into modular components that can be easily swapped, extended, or customized for your research needs.

**Key Features:**
- 🧩 **Modular Design**: Swap out data providers, renderers, loss functions, and densification strategies
- 🔬 **Research-Friendly**: Easy to reproduce and share experiments from papers
- ⚡ **Powered by gsplat**: Built on top of the fast gsplat library
- 📦 **Batteries Included**: Default implementations for common use cases

## How to Use

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/splatkit.git
cd splatkit

# Install dependencies (requires Python >=3.10)
uv sync
```

### Quick Start

See the [`examples/`](examples/) folder for complete examples. Here's a minimal example:

```python
from splatkit.trainer import SplatTrainer
from splatkit.data_provider import SplatColmapDataProvider, SplatColmapDataProviderConfig
from splatkit.renderer import Splat3DGSRenderer
from splatkit.loss_fn import SplatDefaultLossFn
from splatkit.densification import SplatDefaultDensification
from splatkit.modules import SplatExporter, SplatProgressTracker

# Configure data provider
data_provider = SplatColmapDataProvider(
    config=SplatColmapDataProviderConfig(
        colmap_dir="path/to/colmap/sparse/0",
        images_dir="path/to/images",
        factor=1,
        normalize=True,
        train_test_ratio=0.8,
    )
)

# Set up training components
renderer = Splat3DGSRenderer()
loss_func = SplatDefaultLossFn()
densification = SplatDefaultDensification()

# Add optional modules
modules = [
    SplatProgressTracker(update_every=1),
    SplatExporter(
        splat_dir="output/splat",
        ckpt_dir="output/ckpt",
        splat_save_on=[0, 1000, 5000, 10000],
    ),
]

# Create trainer and run
trainer = SplatTrainer(
    renderer=renderer,
    loss_fn=loss_func,
    data_provider=data_provider,
    densification=densification,
    modules=modules,
)
trainer.run()
```

## Architecture Overview

SplatKit is built around these core components:

### Core Components

- **`SplatTrainer`**: Orchestrates the training loop
- **`DataProvider`**: Handles data loading and preprocessing
  - `SplatColmapDataProvider`: Load scenes from COLMAP format
- **`Renderer`**: Renders gaussians to images
  - `Splat3DGSRenderer`: Standard 3D Gaussian Splatting renderer
- **`LossFn`**: Computes training loss
  - `SplatDefaultLossFn`: Default loss function (L1 + SSIM)
- **`Densification`**: Manages gaussian densification strategy
  - `SplatDefaultDensification`: Standard densification and pruning

### Optional Modules

Modules provide additional functionality during training:

- **`SplatProgressTracker`**: Display training progress with tqdm
- **`SplatExporter`**: Save checkpoints and splat files at specified iterations
- **`SplatEvaluator`**: Evaluate model on test set with PSNR, SSIM, LPIPS metrics
- **`SplatTensorboarder`**: Log training and evaluation metrics to TensorBoard (auto-integrates with evaluator)
- **`SplatViewer`**: Interactive 3D viewer during training
- **`SplatFrame`**: Base class for custom training hooks

You can create custom modules by extending the base classes in `splatkit.modules`.

## Project Structure

```
splatkit/
├── data_provider/     # Data loading and preprocessing
├── densification/     # Gaussian densification strategies
├── loss_fn/          # Loss function implementations
├── modules/          # Training modules and hooks
├── renderer/         # Rendering implementations
├── splat/            # Splat model and training state
├── trainer/          # Training orchestration
└── utils/            # Utility functions

examples/             # Example scripts
└── simple_3dgs.py    # Basic 3DGS training example
```