from dataclasses import dataclass

@dataclass
class SplatColmapDataProviderConfig:
    """
    Configuration for loading COLMAP sparse reconstruction data.
    
    Args:
        colmap_dir: Path to COLMAP sparse directory (contains cameras.bin, images.bin, points3D.bin)
                    Typically "data/sparse/0" from COLMAP reconstruction
        images_dir: Path to directory containing RGB images
                    Images referenced by COLMAP reconstruction
        
        factor: Image downscale factor. 1 = original resolution, 2 = half resolution, 4 = quarter resolution (default: 1)
        
        normalize: Normalize scene to unit sphere (default: True)
                   If True, centers cameras at origin and scales to fit in unit sphere
        
        load_depth: Projected points to image plane to get depths (default: False)
        
        masks_dir: Optional path to mask directory (default: None)
                   Masks should be binary and have same filenames and resolution as its corresponding images
        
        train_test_ratio: Fraction of images for training (default: 0.8)
                          0.8 = 80% train, 20% validation
        
        batch_size: Number of images per training step (default: 1)
                    Higher = more GPU memory, potentially faster
        
        num_workers: DataLoader worker processes (default: 4)
                     More workers = faster data loading, more CPU/RAM usage
    
    Example:
        >>> config = SplatColmapDataProviderConfig(
        ...     colmap_dir="data/sparse/0",
        ...     images_dir="data/images",
        ...     factor=2,  # Half resolution
        ...     normalize=True,
        ... )
    """
    colmap_dir: str
    images_dir: str
    factor: int = 1
    normalize: bool = True
    load_depth: bool = False
    masks_dir: str | None = None
    train_test_ratio: float = 0.8

    batch_size: int = 1
    num_workers: int = 4