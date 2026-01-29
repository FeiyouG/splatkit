# Object-Centric 2D Gaussian Splatting

Implementation of object-centric training for 2D Gaussian Splatting with background removal and occlusion-aware pruning.

## Overview

This example demonstrates:
- **Background removal** via masked photometric loss and background regularization
- **Occlusion-aware pruning** to remove Gaussians that are never visible in training views
- Compact object-only 3D models suitable for downstream tasks (AR, robotics, asset creation)

## Paper & Repository

**Paper**: [Object-Centric 2D Gaussian Splatting: Background Removal and Occlusion-Aware Pruning for Compact Object Models](https://www.scitepress.org/PublicationsDetail.aspx?ID=OQ/8fxj3bNI=&t=1)  
**GitHub**: [MarcelRogge/object-centric-2dgs](https://github.com/MarcelRogge/object-centric-2dgs)


```bibtex
@conference{RoggeOC2DGS2025,
    author={Marcel Rogge and Didier Stricker},
    title={Object-Centric 2D Gaussian Splatting: Background Removal and Occlusion-Aware Pruning for Compact Object Models},
    booktitle={Proceedings of the 14th International Conference on Pattern Recognition Applications and Methods - ICPRAM},
    year={2025},
    pages={519-530},
    publisher={SciTePress},
    organization={INSTICC},
    doi={10.5220/0013305500003905},
    isbn={978-989-758-730-6},
    issn={2184-4313}
}
```

## Usage

```bash
python -m examples.object-centric.main \
    --colmap_dir /path/to/sparse/0 \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --output_dir ./output \
    --max_steps 30000
```

**Requirements**: Binary object masks (0=background, 255=foreground) matching each training image.