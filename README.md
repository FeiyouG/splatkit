# splatkit

A modular toolkit for Gaussian Splatting training, built on top of [gsplat](https://github.com/nerfstudio-project/gsplat).

## Installation

**Step 1:** Install PyTorch with CUDA support (required, not included):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(Optional) For fused SSIM support:

```bash
pip install git+https://github.com/rahul-goel/fused-ssim@98126b7781f9e563234c92d2bf08ee0994f4f175
```

**Step 2:** Install splatkit:

```bash
# From PyPI (once published)
pip install splatkit[all]

# or using uv
uv add splatkit --extra all

# For development (from source)
git clone https://github.com/veristic/splatkit.git
cd splatkit
pip install -e ".[all]"  # or: uv pip install -e ".[all]"
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

Check out `examples/` folder for more:
- `examples/3dgs/simple_3dgs.py` — 3D Gaussian Splatting
- `examples/2dgs/simple_2dgs.py` — 2D Gaussian Splatting


## Documentation

📚 **[Full Documentation](https://feiyoug.github.io/splatkit/main/index.html)** — Installation, guides, API reference, and customization examples.
