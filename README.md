# splatkit

A modular toolkit for Gaussian Splatting training, built on top of [gsplat](https://github.com/nerfstudio-project/gsplat).

## Why splatkit?

**Built on gsplat**</br>
Standing on the shoulders of giants (and their highly optimized CUDA kernels). SplatKit handles the boilerplate so you can focus on the fun parts.

**Reproducibility & Clarity**  </br>
Clear abstractions and configuration objects make your experiments easier to understand, share, and reproduce.

**Rapid Experimentation**</br>
Swap renderers, loss functions, or densification strategies in seconds. Write new modules with minimal boilerplate to test research ideas faster.

**Production Readiness**</br>
The same modular design ensures consistent, reproducible training for production. Checkpoint management, distributed training, and evaluation metrics included.

## Installation

**Step 1:** Install PyTorch with CUDA support (required, not included):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Step 2:** Install splatkit:

```bash
pip install -e ".[all]"  # or use: uv pip install -e ".[all]"
```

See the [installation guide](docs/source/installation.rst) for more options.

## Quick Example

Train a 3D Gaussian Splatting model:

```python
from splatkit.trainer import SplatTrainer, SplatTrainerConfig
from splatkit.data_provider import SplatColmapDataProvider, SplatColmapDataProviderConfig
from splatkit.renderer import Splat3DGSRenderer
from splatkit.loss_fn import Splat3DGSLossFn
from splatkit.densification import SplatDefaultDensification

# Configure training
config = SplatTrainerConfig(
    max_steps=30000,
    output_dir="outputs/my_scene",
)

# Set up COLMAP data
data_provider = SplatColmapDataProvider(
    config=SplatColmapDataProviderConfig(
        colmap_dir="data/sparse/0",
        images_dir="data/images",
        normalize=True,
    )
)

# Create and run trainer
trainer = SplatTrainer(
    config=config,
    data_provider=data_provider,
    renderer=Splat3DGSRenderer(),
    loss_fn=Splat3DGSLossFn(),
    densification=SplatDefaultDensification(),
)
trainer.run()
```

## Documentation

📚 **[Full Documentation](docs/)** — Installation, guides, API reference, and customization examples.

Check out `examples/` folder for more:
- `examples/3dgs/simple_3dgs.py` — 3D Gaussian Splatting
- `examples/2dgs/simple_2dgs.py` — 2D Gaussian Splatting
