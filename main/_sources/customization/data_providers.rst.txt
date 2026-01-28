Custom Data Providers
==============================

Data providers load training data from various formats (COLMAP, NeRF, custom datasets).
You can create custom data providers to support new formats or implement specialized data loading.

Creating a Custom Data Provider
--------------------------------

All data providers must inherit from :class:`splatkit.data_provider.base.SplatDataProvider`:

Data provider itself is also a subclass of :class:`splatkit.modules.base.SplatBaseModule`,
and all of it hooks will be called during training just like any other modules.

Here's a complete example for loading NeRF synthetic datasets:

.. code-block:: python

   import json
   import torch
   import imageio.v3 as imageio
   from pathlib import Path
   from splatkit.data_provider.base import SplatDataProvider, SplatDataItem
   from splatkit.modules.frame import SplatRenderPayload
   
   class NerfSyntheticDataProvider(SplatDataProvider[SplatRenderPayload, SplatDataItem]):
       """Data provider for NeRF Synthetic dataset format."""
       
       def __init__(self, data_dir: str, split: str = "train", white_bg: bool = True):
           super().__init__()
           self.data_dir = Path(data_dir)
           self.split = split
           self.white_bg = white_bg
           self.frames = []
       
       def load_data(self) -> float:
           """Load NeRF synthetic data."""
           # Load transforms JSON
           json_path = self.data_dir / f"transforms_{self.split}.json"
           with open(json_path, 'r') as f:
               meta = json.load(f)
           
           # Extract camera parameters
           H, W = 800, 800
           focal = 0.5 * W / torch.tan(0.5 * meta['camera_angle_x'])
           
           # Load frames
           for frame in meta['frames']:
               # Load image
               img_path = self.data_dir / f"{frame['file_path']}.png"
               img = imageio.imread(img_path)
               img = torch.from_numpy(img).float() / 255.0  # (H, W, 4)
               
               # Handle alpha channel
               rgb = img[..., :3]
               alpha = img[..., 3:4]
               if self.white_bg:
                   rgb = rgb * alpha + (1 - alpha)  # Composite on white
               
               # Build intrinsics
               K = torch.tensor([
                   [focal, 0, W/2],
                   [0, focal, H/2],
                   [0, 0, 1],
               ], dtype=torch.float32)
               
               # Load pose (c2w)
               pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
               
               self.frames.append({
                   'image': rgb,
                   'K': K,
                   'pose': pose,
                   'name': frame['file_path'],
               })
           
           # Compute scene scale from camera positions
           positions = torch.stack([f['pose'][:3, 3] for f in self.frames])
           scene_scale = positions.norm(dim=-1).max().item()
           
           return scene_scale
       
       def next_train_data(self, step: int) -> SplatDataItem:
           """Sample random training frame."""
           idx = torch.randint(0, len(self.frames), (1,)).item()
           frame = self.frames[idx]
           
           return SplatDataItem(
               id=idx,
               image_name=frame['name'],
               camera_model="pinhole",
               K=frame['K'].unsqueeze(0),
               cam_to_world=frame['pose'].unsqueeze(0),
               image=frame['image'].unsqueeze(0),
           )
       
       def get_init_point_cloud(self) -> tuple[torch.Tensor, torch.Tensor]:
           """Generate random point cloud in scene bounds."""
           N = 10000
           # Sample random points in unit sphere
           points = torch.randn(N, 3)
           points = points / points.norm(dim=-1, keepdim=True)
           points = points * torch.rand(N, 1) * 0.5  # Radius 0.5
           
           # Random colors
           colors = torch.rand(N, 3)
           
           return points, colors

Using Your Data Provider
-------------------------

Add your custom data provider to the trainer:

.. code-block:: python

   from splatkit.trainer import SplatTrainer, SplatTrainerConfig
   
   data_provider = NerfSyntheticDataProvider(
       data_dir="data/nerf_synthetic/lego",
       split="train",
       white_bg=True,
   )
   
   trainer = SplatTrainer(
       config=SplatTrainerConfig(max_steps=30000),
       data_provider=data_provider,
       renderer=renderer,
       loss_fn=loss_fn,
       densification=densification,
   )
   trainer.run()

Best Practices
--------------

1. **Image Format**: Always return images as float32 in [0, 1] range
2. **Batch Dimension**: Add batch dimension even for single images
3. **Memory Management**: Don't load all images into memory if dataset is large
4. **Validation Set**: Implement ``next_test_data()`` for evaluation

Advanced: DataLoader Integration
---------------------------------

For large datasets, use PyTorch DataLoader:

.. code-block:: python

   from torch.utils.data import Dataset, DataLoader
   
   class MyDataset(Dataset):
       def __init__(self, data_dir):
           # Load metadata only, not images
           self.image_paths = self._find_images(data_dir)
           self.cameras = self._load_cameras(data_dir)
       
       def __len__(self):
           return len(self.image_paths)
       
       def __getitem__(self, idx):
           # Load image on-demand
           image = self._load_image(self.image_paths[idx])
           return {
               'image': image,
               'K': self.cameras[idx]['K'],
               'pose': self.cameras[idx]['pose'],
           }
   
   class MyDataProvider(SplatDataProvider):
       def load_data(self) -> float:
           self.dataset = MyDataset(self.data_dir)
           self.dataloader = DataLoader(
               self.dataset,
               batch_size=1,
               shuffle=True,
               num_workers=4,
           )
           self.data_iter = iter(self.dataloader)
           return self._compute_scene_scale()
       
       def next_train_data(self, step: int) -> SplatDataItem:
           try:
               batch = next(self.data_iter)
           except StopIteration:
               self.data_iter = iter(self.dataloader)
               batch = next(self.data_iter)
           
           return SplatDataItem(
               id=0,
               image_name="batch",
               camera_model="pinhole",
               K=batch['K'],
               cam_to_world=batch['pose'],
               image=batch['image'],
           )

See Also
--------

- :doc:`modules` - Writing custom training modules
- :doc:`renderers` - Writing custom renderers
- :doc:`loss_functions` - Writing custom loss functions
- :class:`splatkit.data_provider.base.SplatDataProvider` - Base class API
- :class:`splatkit.data_provider.colmap.provider.SplatColmapDataProvider` - COLMAP example
