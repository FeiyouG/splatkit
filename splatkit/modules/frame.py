from dataclasses import dataclass
from typing import Any, KeysView, ItemsView, ValuesView, TypeVar
from torch import Tensor

@dataclass(frozen=True)
class SplatRenderPayload:
    """
    Metadata for rendered images.
    """
    renders: Tensor  # (..., H, W, 3) - rendered images
    alphas: Tensor  # (..., H, W, 1) - rendered alpha
    depths: Tensor # (..., C, N) - per-gaussian depths (C=cameras, N=gaussians)
    radii: Tensor  # (..., C, N, 2) - per-gaussian radii in x/y (C=cameras, N=gaussians)
    n_cameras: int
    n_batches: int
    width: int
    height: int
    means2d: Tensor # (..., C, N, 2) - per-gaussian 2D means (C=cameras, N=gaussians)

    gaussian_ids: Tensor | None # Visible gaussian indices for densification
    depths_expected: Tensor | None # (..., H, W, 1) - expected depths; will be None if render_mode is not "RGB+D" or "RGB+ED"
    depths_accumulated: Tensor | None # (..., H, W, 1) - accumulated depths; will be None if render_mode is not "D" or "RGB+D"

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    def values(self) -> ValuesView[Any]:
        return self.__dict__.values()

    def items(self) -> ItemsView[str, Any]:
        return self.__dict__.items()

    def to_dict(self) -> dict:
        return dict(self.__dict__)

# We use contravariant to allow for sub-classes of SplatRenderedOutput to be used as the type parameter.
SplatRenderPayloadT = TypeVar("SplatRenderPayloadT", bound="SplatRenderPayload", contravariant=True)