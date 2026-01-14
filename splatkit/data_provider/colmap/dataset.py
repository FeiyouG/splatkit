import os
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path

import numpy as np
import imageio
import torch
import pycolmap
from torch.utils.data import Dataset

from .item import ColmapDataItem
from .config import SplatColmapDataProviderConfig

class SplatColmapDataset(Dataset[ColmapDataItem]):
    """
    Torch-compatible COLMAP dataset.
    """

    _colmap_dir: str
    _images_dir: str
    _factor: int
    _normalize: bool
    _load_depth: bool
    _masks_dir: str | None

    _image_names: List[str] = []
    _image_paths: List[str] = []
    _world_to_cam: np.ndarray = np.array([], dtype=np.float32) # (N, 4, 4)
    _cam_to_world: np.ndarray = np.array([], dtype=np.float32) # (N, 4, 4)
    _Ks: np.ndarray = np.array([], dtype=np.float32) # (N, 3, 3)
    _image_sizes: List[Tuple[int, int]] = []
    _mask_paths: List[str] | None
    _points: np.ndarray = np.array([], dtype=np.float32) # (N, 3)
    _points_rgb: np.ndarray = np.array([], dtype=np.uint8) # (N, 3)
    _points_err: np.ndarray = np.array([], dtype=np.float32) # (N,)
    _point_indices: Dict[str, np.ndarray] = {} # {image_name: np.ndarray[int32]}mo
    _transform: pycolmap.Sim3d | None = None
    _scene_scale: float = 0.0
    _camera_models: List[str] = []

    # Index mapping for split datasets
    _parent: Optional['SplatColmapDataset']
    _valid_indices: np.ndarray

    def __init__(
        self,
        config: SplatColmapDataProviderConfig,
    ):
        self._colmap_dir = config.colmap_dir
        self._images_dir = config.images_dir
        self._factor = config.factor
        self._normalize = config.normalize
        self._load_depth = config.load_depth
        self._masks_dir = config.masks_dir
        self._parent = None

        # Load COLMAP reconstruction
        if not os.path.exists(self._colmap_dir):
            raise FileNotFoundError(f"COLMAP directory not found: {self._colmap_dir}")

        recon = pycolmap.Reconstruction(self._colmap_dir)
        if recon.num_reg_images == 0:
            raise ValueError("No registered images in COLMAP reconstruction")

        if self._normalize:
            self._transform = recon.normalize()
            recon.transform(self._transform)
        else:
            self._transform = None

        # Validate image directory
        if not os.path.exists(self._images_dir):
            raise FileNotFoundError(f"Image directory not found: {self._images_dir}")
        
        if self._masks_dir is not None:
            if not os.path.exists(self._masks_dir):
                raise FileNotFoundError(f"Mask directory not found: {self._masks_dir}")

        image_names: List[str] = []
        image_paths: List[str] = []
        mask_paths: List[str] = []
        world_to_cam: List[np.ndarray] = []
        cam_to_world: List[np.ndarray] = []
        Ks: List[np.ndarray] = []
        image_sizes: List[Tuple[int, int]] = []
        camera_models: List[str] = []

        # Extract per-image data
        image: pycolmap.Image
        for image in recon.images.values():
            camera = image.camera

            if camera is None:
                raise ValueError(f"Camera not found for image: {image.name}")

            name = image.name
            img_path = os.path.join(self._images_dir, name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            w2c_3x4 = image.cam_from_world().matrix().astype(np.float32)
            c2w_3x4 = image.cam_from_world().inverse().matrix().astype(np.float32)
            
            # Promote pycolmap w2c 3x4 matrix to 4x4 matrix
            w2c = np.vstack((w2c_3x4, np.array([0, 0, 0, 1]))).astype(np.float32)
            c2w = np.vstack((c2w_3x4, np.array([0, 0, 0, 1]))).astype(np.float32)

            K = camera.calibration_matrix().astype(np.float32)
            K[:2, :] /= self._factor

            width = int(camera.width // self._factor)
            height = int(camera.height // self._factor)

            image_names.append(name)
            image_paths.append(img_path)
            world_to_cam.append(w2c)
            cam_to_world.append(c2w)
            Ks.append(K)
            image_sizes.append((width, height))
            camera_models.append(camera.model.name)

            if self._masks_dir is not None:
                stem = Path(name).stem  # filename without extension

                matches = list(Path(self._masks_dir).glob(f"{stem}.*"))
                if not matches or len(matches) == 0:
                    raise FileNotFoundError(f"Mask not found for {name} in {self._masks_dir}")
                if len(matches) > 1:
                    raise RuntimeError(f"Multiple masks found for {name}: {matches}")

                mask_path = matches[0]  # or enforce exactly one
                mask_paths.append(str(mask_path))

        # Sort by filename (stable)
        order = np.argsort(image_names)

        self._image_names = [image_names[i] for i in order]
        self._image_paths = [image_paths[i] for i in order]
        self._image_sizes = [image_sizes[i] for i in order]
        self._mask_paths = [mask_paths[i] for i in order] if self._masks_dir is not None else None
        self._camera_models = [camera_models[i] for i in order]

        self._world_to_cam = np.stack([world_to_cam[i] for i in order], axis=0)
        assert self._world_to_cam.shape[1:] == (4, 4)

        self._cam_to_world = np.stack([cam_to_world[i] for i in order], axis=0)
        assert self._cam_to_world.shape[1:] == (4, 4)

        self._Ks = np.stack([Ks[i] for i in order], axis=0)
        assert self._Ks.shape[1:] == (3, 3)

        # Sparse point cloud
        self._points = np.array([p.xyz for p in recon.points3D.values()], dtype=np.float32)
        self._points_rgb = np.array([p.color for p in recon.points3D.values()], dtype=np.uint8)
        self._points_err = np.array([p.error for p in recon.points3D.values()], dtype=np.float32)

        point_indices: Dict[str, list[int]] = {}
        image_id_to_name = {i.image_id: i.name for i in recon.images.values()}
        point_id_to_index = {
            pid: idx for idx, pid in enumerate(recon.points3D.keys())
        }

        for point_id, point3D in recon.points3D.items():
            pidx = point_id_to_index[point_id]
            for element in point3D.track.elements:
                name = image_id_to_name[element.image_id]
                point_indices.setdefault(name, []).append(pidx)

        # Convert to numpy
        for k, v in point_indices.items():
            self._point_indices[k] = np.asarray(v, dtype=np.int32)

        # Scene scale
        camera_centers = self._cam_to_world[:, :3, 3]
        center = camera_centers.mean(axis=0)
        self._scene_scale = np.max(np.linalg.norm(camera_centers - center, axis=1))

        # Initialize valid indices to include all images
        self._valid_indices = np.arange(len(self._image_names), dtype=np.int32)

        # Free COLMAP memory
        del recon

    @classmethod
    def _from_parent(cls, parent: 'SplatColmapDataset', indices: np.ndarray) -> 'SplatColmapDataset':
        """
        Private constructor for creating a split dataset from a parent.
        
        Args:
            parent: The parent ColmapDataset to share data with
            indices: The indices (in parent's valid index space) to include
            
        Returns:
            A new ColmapDataset instance that shares data with parent
        """
        # Create an empty instance without calling __init__
        instance = cls.__new__(cls)
        
        # Store reference to parent (or parent's parent if parent is already a split)
        instance._parent = parent
        
        # Share all data from the root parent
        root = instance._parent
        instance._colmap_dir = root._colmap_dir
        instance._images_dir = root._images_dir
        instance._factor = root._factor
        instance._normalize = root._normalize
        instance._masks_dir = root._masks_dir
        instance._load_depth = root._load_depth
        instance._transform = root._transform
        instance._scene_scale = root._scene_scale
        
        # Share all arrays (no copying!)
        instance._image_names = root._image_names
        instance._image_paths = root._image_paths
        instance._world_to_cam = root._world_to_cam
        instance._cam_to_world = root._cam_to_world
        instance._Ks = root._Ks
        instance._image_sizes = root._image_sizes
        instance._mask_paths = root._mask_paths
        instance._camera_models = root._camera_models
        
        instance._points = root._points
        instance._points_rgb = root._points_rgb
        instance._points_err = root._points_err
        instance._point_indices = root._point_indices
        
        # Only thing that's different: the valid indices
        # If parent is already a split, map through parent's indices
        instance._valid_indices = parent._valid_indices[indices]
        
        return instance


    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, index: int) -> ColmapDataItem:
        actul_index = self._valid_indices[index]

        image = imageio.imread(self._image_paths[actul_index])

        mask: torch.Tensor | None = None
        if self._masks_dir is not None:
            if self._mask_paths is None:
                raise ValueError("Mask paths are not set")

            mask_np = imageio.imread(self._mask_paths[actul_index])
            if mask_np.ndim == 3:
                # RGB or RGBA - take first channel only
                mask_np = mask_np[..., 0] # (H, W, 3) -> (H, W)

            mask = torch.from_numpy(mask_np).bool() # (H, W)
        
        
        points: torch.Tensor | None = None
        depths: torch.Tensor | None = None
        if self._load_depth:
            world_to_cam = self._world_to_cam[actul_index]
            image_name = self._image_names[actul_index]
             
            if image_name in self._point_indices:
                pidx = self._point_indices[image_name]
                points_world = self._points[pidx]

                points_cam = (
                    world_to_cam[:3, :3] @ points_world.T + 
                    world_to_cam[:3, 3:4]
                ).T

                points_proj = (self._Ks[actul_index] @ points_cam.T).T
                xy = points_proj[:, :2] / points_proj[:, 2:3]
                depths_np = points_cam[:, 2]

                H, W = image.shape[:2]
                valid = (
                    (xy[:, 0] >= 0) & (xy[:, 0] < W) &
                    (xy[:, 1] >= 0) & (xy[:, 1] < H) &
                    (depths_np > 0)
                )

                points = torch.from_numpy(xy[valid]).float()
                depths = torch.from_numpy(depths_np[valid]).float()
        

        item: ColmapDataItem = ColmapDataItem(
            id=actul_index,
            image_name=self._image_names[actul_index],
            camera_model=self._camera_models[actul_index],

            image=torch.from_numpy(image[..., :3]).float(),
            K=torch.from_numpy(self._Ks[actul_index]).float(),
            cam_to_world=torch.from_numpy(self._cam_to_world[actul_index]).float(),
            
            mask=mask,
            points=points,
            depths=depths,
        )

        return item

    def split(self, predicate: Callable[[int, str], bool]) -> 'SplatColmapDataset':
        selected_indices = []
        
        for idx in range(len(self)):
            if predicate(idx, self._image_names[idx]):
                selected_indices.append(idx)
        
        if len(selected_indices) == 0:
            raise ValueError("No items matched the predicate")
        
        indices = np.array(selected_indices, dtype=np.int32)
        
        # Use the private class method to create the split dataset
        return self._from_parent(self, indices)
    
    @property
    def points(self) -> np.ndarray:
        return self._points
    
    @property
    def points_rgb(self) -> np.ndarray:
        return self._points_rgb

    @property
    def scene_scale(self) -> float:
        return self._scene_scale