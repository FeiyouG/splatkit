import torch
from torch import Tensor
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import overload

def knn(x: np.ndarray, K: int = 4) -> np.ndarray:
    """
    Find K nearest neighbors for each point.
    
    Args:
        x: Points, shape [N, D]
        K: Number of neighbors (including self)
    
    Returns:
        Distances to K nearest neighbors, shape [N, K]
    """
    
    # Fit KNN model
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x)
    distances, _ = model.kneighbors(x)
    
    return distances