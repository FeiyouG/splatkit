from dataclasses import dataclass
from typing import Any, KeysView, ItemsView, ValuesView, TypeVar
from torch import Tensor

@dataclass(frozen=True)
class SplatRenderPayload:
    """
    Metadata for rendered images.
    """
    renders: Tensor  # (..., H, W, 3)
    alphas: Tensor  # (..., H, W, 1)
    radii: Tensor  # (..., H, W, 1)
    depths: Tensor # (..., H, W, 1)
    n_cameras: int
    n_batches: int
    width: int
    height: int

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