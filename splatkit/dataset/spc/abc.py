from abc import ABC, abstractmethod
from typing import Callable

from torch.utils.data import Dataset
import numpy as np

from ..item import DataSetItem

class SparsePointCloudDataset(ABC, Dataset):
    """
    Abstract base class for datasets that provide sparse 3D point clouds.
    
    Any dataset that can initialize a SplatModel must implement this interface.
    """
    
    @property
    @abstractmethod
    def points(self) -> np.ndarray:
        """
        Sparse 3D point cloud positions.
        
        Returns:
            Array of shape [N, 3] with xyz coordinates (float32)
        """
        pass
    
    @property
    @abstractmethod
    def points_rgb(self) -> np.ndarray:
        """
        RGB colors for sparse point cloud.
        
        Returns:
            Array of shape [N, 3] with RGB values in range [0, 255] (uint8)
        """
        pass
    
    @property
    @abstractmethod
    def scene_scale(self) -> float:
        """
        Estimate of scene scale (approximate radius from center).
        
        Used for:
        - Learning rate scaling
        - Densification thresholds
        - Initialization extent
        
        Returns:
            Scene scale as a positive float
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Number of points in the dataset.
        
        Returns:
            Number of points as an integer
        """
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> DataSetItem:
        """
        Get a single item from the dataset.
        
        Args:
            index: The index of the item to get
        
        Returns:
            A DataSetItem containing the item data
        """
        pass
    
    @abstractmethod
    def split(self, predicate: Callable[[int, str], bool]) -> 'SparsePointCloudDataset':
        """
        Create a new dataset containing only items that match the predicate.
        
        Args:
            predicate: A function that takes an index and an image name and returns True/False.
                      Items returning True will be included in the new dataset.
        
        Returns:
            A new ColmapDataset containing only the filtered items (shares data with parent).
        
        Raises:
            ValueError: If no items match the predicate.
        
        Examples:
            >>> # Split into train/test based on every 8th image
            >>> train = dataset.split(lambda index, image_name: index % 8 != 0)
            >>> test = dataset.split(lambda index, image_name: index % 8 == 0)
            
            >>> # Split based on filename pattern
            >>> outdoor = dataset.split(lambda index, image_name: 'outdoor' in image_name)
            
            >>> # Filter by image brightness
            >>> bright = dataset.split(lambda index, image_name: image.mean() > 128)
        """
        pass