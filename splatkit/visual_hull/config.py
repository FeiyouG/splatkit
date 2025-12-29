from pydantic import BaseModel, Field
from typing import Tuple, Literal

class VisualHullConfig(BaseModel):
    # ---- voxel grid ----
    grid_resolution: int = Field(
        192, ge=32, le=1024, description="Voxels per axis"
    )

    bounds_min: Tuple[float, float, float] = (-1.2, -1.2, -1.2)
    bounds_max: Tuple[float, float, float] = ( 1.2,  1.2,  1.2)


    # ---- mask semantics ----
    mask_foreground_value: Literal[1] | Literal[255] = 1
    carve_on_background: bool = True

    # ---- output ----
    extract_mesh: bool = True
    mesh_level: float = 0.5

    class Config:
        frozen = True  # immutable
