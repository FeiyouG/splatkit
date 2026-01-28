"""
splatkit: A modular toolkit for 3D and 2D Gaussian Splatting

Built on top of gsplat, splatkit provides a clean, modular framework for
training Gaussian Splatting models with reproducibility and extensibility in mind.
"""

__version__ = "0.1.0"
__author__ = "Feiyou Guo"
__license__ = "Apache-2.0"

# Import main components for convenience
from .trainer import SplatTrainer, SplatTrainerConfig
from .data_provider import (
    SplatDataProvider,
    SplatColmapDataProvider,
    SplatColmapDataProviderConfig,
)
from .renderer import Splat3DGSRenderer, Splat2DGSRenderer
from .loss_fn import Splat3DGSLossFn, Splat2DGSLossFn
from .densification import SplatDefaultDensification

__all__ = [
    "__version__",
    "SplatTrainer",
    "SplatTrainerConfig",
    "SplatDataProvider",
    "SplatColmapDataProvider",
    "SplatColmapDataProviderConfig",
    "Splat3DGSRenderer",
    "Splat2DGSRenderer",
    "Splat3DGSLossFn",
    "Splat2DGSLossFn",
    "SplatDefaultDensification",
]
