# RaDe-GS: Mesh Extraction from Gaussian Splatting

This example demonstrates mesh extraction from Gaussian Splatting models using the marching tetrahedra algorithm, adapted from [Gaussian Opacity Fields (GOF)](https://github.com/autonomousvision/gaussian-opacity-fields).

## Overview

The mesh extraction pipeline converts a trained 3D Gaussian Splatting model into a triangle mesh using:
- **Marching Tetrahedra**: A volumetric surface extraction algorithm
- **Alpha Integration**: Computing signed distance function (SDF) by integrating opacity
- **Binary Search Refinement**: Precisely locating the surface by iterative bisection

## Features

- ✅ Automatic mesh extraction at specified training steps
- ✅ Multi-view alpha integration for robust SDF computation
- ✅ Binary search refinement for high-quality surfaces
- ✅ Scale-based filtering to remove noise
- ✅ Compatible with splatkit's modular architecture

## Installation

### Required Dependencies

```bash
# Core dependencies (installed with splatkit)
pip install torch numpy trimesh tqdm

# Optional: tetra-nerf for faster triangulation
pip install git+https://github.com/jkulhanek/tetra-nerf
```

**Note**: Without `tetra-nerf`, the implementation falls back to scipy's Delaunay triangulation (slower but functional).

## Usage

### Basic Example

```python
from mesh_extractor import SplatMeshExtractor

# Create mesh extractor
mesh_extractor = SplatMeshExtractor(
    output_dir="./output",
    export_steps=[7000, 15000, 30000],  # Extract at these training steps
    kernel_size=0.1,                     # Alpha integration kernel
    n_binary_steps=8,                    # Refinement iterations
)

# Add to trainer (will automatically extract meshes during training)
trainer = SplatTrainer(...)
trainer.add_module(mesh_extractor)
trainer.train()
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_dir` | Directory to save extracted meshes | Required |
| `export_steps` | List of training steps to extract meshes | `[]` |
| `kernel_size` | 3D filter radius for alpha integration | `0.1` |
| `n_binary_steps` | Number of binary search refinement steps | `8` |
| `extract_full_mesh` | Whether to extract full mesh (vs. just save model) | `True` |

## How It Works

### 1. Tetrahedra Generation

For each Gaussian:
- Create 8 corner points of a bounding box scaled by 3σ (99.7% coverage)
- Add the Gaussian center as the 9th point
- These 9 points form tetrahedra vertices

```python
# Each Gaussian generates 9 vertices
vertices = [corner_1, corner_2, ..., corner_8, center]
# Scaled by 3 * gaussian_scale
```

### 2. SDF Evaluation

For each vertex, compute signed distance function:
- Project vertex to all training camera views
- Integrate transmittance (accumulated opacity) along ray to vertex
- Take minimum alpha across all views (conservative estimate)
- Convert: `SDF = 0.5 - alpha` (inside if alpha > 0.5)

```python
for camera in training_cameras:
    alpha = integrate_transmittance(vertex, camera, gaussians)
    min_alpha = min(min_alpha, alpha)
sdf = 0.5 - min_alpha  # Negative inside, positive outside
```

### 3. Marching Tetrahedra

- Find tetrahedra where SDF changes sign (contains surface)
- Extract edges that cross the zero-level
- Create initial mesh from crossing edges

### 4. Binary Search Refinement

For each edge crossing the surface:
```python
while n_iterations < n_binary_steps:
    mid_point = (left + right) / 2
    mid_sdf = evaluate_sdf(mid_point)
    
    if sign(mid_sdf) == sign(left_sdf):
        left = mid_point
    else:
        right = mid_point
```

### 5. Filtering

Remove noisy faces:
- Filter edges longer than combined Gaussian radii
- This removes connections between distant/unrelated Gaussians

## Output Format

Meshes are saved as PLY files:
```
output_dir/
└── meshes/
    ├── 7000.ply
    ├── 15000.ply
    └── 30000.ply
```

Each mesh contains:
- Vertex positions (refined by binary search)
- Triangle faces (from marching tetrahedra)

## Tips & Tricks

### For Better Meshes

1. **Use more training views**: More cameras = more accurate SDF
2. **Increase binary search steps**: More refinement = smoother surface
3. **Adjust kernel size**: Smaller = sharper details, larger = smoother
4. **Filter low opacity Gaussians**: Uncomment filtering in `_get_tetra_points()` for cleaner backgrounds

### Performance Optimization

1. **Reduce point batching**: Decrease batch size in `_integrate_alpha()` if GPU runs out of memory
2. **Skip some views**: Sample subset of training cameras for faster (but less accurate) extraction
3. **Use tetra-nerf**: Much faster triangulation than scipy

## Comparison with GOF

This implementation closely follows GOF with splatkit adaptations:

| Aspect | GOF | This Implementation |
|--------|-----|---------------------|
| Core algorithm | Marching tetrahedra + binary search | ✅ Same |
| Alpha integration | Custom CUDA kernel | PyTorch (compatible with any renderer) |
| Triangulation | tetra-nerf | tetra-nerf + scipy fallback |
| Module system | Standalone script | Integrated SplatBaseModule |
| Data loading | Custom Scene class | splatkit DataProvider |

## References

- **Gaussian Opacity Fields**: [Paper](https://arxiv.org/abs/2404.10772) | [Code](https://github.com/autonomousvision/gaussian-opacity-fields)
- **Tetra-NeRF**: [Code](https://github.com/jkulhanek/tetra-nerf)
- **Marching Tetrahedra**: Doi & Koide, 1991

## Troubleshooting

**Q: "tetra-nerf not found" warning**
A: Install with `pip install git+https://github.com/jkulhanek/tetra-nerf` or use scipy fallback

**Q: Out of memory during alpha integration**
A: Reduce batch size in `_integrate_alpha()` method (default 10000)

**Q: Mesh has holes or artifacts**
A: Try increasing `n_binary_steps` or adjusting `kernel_size`

**Q: Extraction is very slow**
A: Install tetra-nerf for faster triangulation, or reduce number of training views

## License

Adapted from Gaussian Opacity Fields (GOF), which is licensed for non-commercial research use.
