from dataclasses import dataclass

@dataclass
class SplatColmapDataProviderConfig:
    """
    Configuration for COLMAP data provider.
    """
    colmap_dir: str
    images_dir: str
    factor: int = 1
    normalize: bool = False
    load_depth: bool = False
    masks_dir: str | None = None
    train_test_ratio: float = 0.8