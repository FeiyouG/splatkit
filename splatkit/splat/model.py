import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from scipy.special import logit

from ..utils.sh import rgb_to_sh, sh_to_K
from ..utils.knn import knn

@dataclass
class SplatModel:
    """
    Immutable model representing trained Gaussian Splat parameters.
    
    Used for saving to disk, loading checkpoints, and converting to/from
    training state.
    
    Unlike SplatTrainingState (PyTorch, GPU, mutable), SplatModel is:
    - NumPy arrays on CPU (lightweight, portable)
    - Immutable (frozen dataclass)
    - Self-contained (includes all parameters and SH degree)
    
    Attributes:
        _sh_degree: Maximum spherical harmonics degree
        _points: Gaussian centers (N, 3) in world space
        _scales: Gaussian sizes (N, 3) in log-space
        _quats: Gaussian rotations (N, 4) as quaternions
        _opacities: Gaussian opacities (N,) in logit-space
        _sh0: 0th order SH coefficients (N, 1, 3)
        _shN: Higher order SH coefficients (N, K, 3)
    
    Example:
        >>> # Save trained model
        >>> model = SplatModel.from_training_state(training_state)
        >>> model.save_ply("output.ply")
        >>> 
        >>> # Load and resume training
        >>> model = SplatModel.load_ply("checkpoint.ply")
        >>> training_state = model.to_training_state(
        ...     learning_rates={...},
        ...     device="cuda:0"
        ... )
    
    NOTE:
        - Use property accessors (e.g., model.points) not _points
        - Create from factory methods, don't construct directly
    """
    
    # Private fields - access via properties only
    _sh_degree: int
    _points: np.ndarray       # [N, 3] - world space
    _scales: np.ndarray       # [N, 3] - log space
    _quats: np.ndarray        # [N, 4] - quaternions (not normalized)
    _opacities: np.ndarray    # [N,] - logit space
    _sh0: np.ndarray          # [N, 1, 3] - DC component
    _shN: np.ndarray          # [N, K-1, 3] - higher order SH

    def __post_init__(self):
        """Validate state after initialization"""
        self.__validate__()
    
    def __validate__(self):
        """Validate shapes, types, and consistency of all parameters"""
        N = len(self._points)
        
        # Type checks
        for name, arr in [
            ("points", self._points),
            ("scales", self._scales),
            ("quats", self._quats),
            ("opacities", self._opacities),
            ("sh0", self._sh0),
            ("shN", self._shN),
        ]:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} must be numpy array")
            if arr.dtype != np.float32:
                raise ValueError(f"{name} must be float32, got {arr.dtype}")
        
        # Shape checks
        if self._points.shape != (N, 3):
            raise ValueError(f"points shape mismatch: {self._points.shape} != ({N}, 3)")
        if self._scales.shape != (N, 3):
            raise ValueError(f"scales shape mismatch: {self._scales.shape} != ({N}, 3)")
        if self._quats.shape != (N, 4):
            raise ValueError(f"quats shape mismatch: {self._quats.shape} != ({N}, 4)")
        if self._opacities.shape != (N,):
            raise ValueError(f"opacities shape mismatch: {self._opacities.shape} != ({N},)")
        if self._sh0.shape != (N, 1, 3):
            raise ValueError(f"sh0 shape mismatch: {self._sh0.shape} != ({N}, 1, 3)")
        if self._sh_degree < 0 or self._sh_degree > 3:
            raise ValueError(f"sh_degree must be in [0, 3], got {self._sh_degree}")
        
        # Validate SH degree consistency    
        K = sh_to_K(self._sh_degree)
        if self._shN.shape != (N, K - 1, 3):
            raise ValueError(f"shN shape mismatch: {self._shN.shape} != ({N}, {K}, 3)")
        
    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        colors: np.ndarray,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
    ) -> 'SplatModel':
        """
        Initialize from point cloud with colors.
        
        Args:
            points: Point positions [N, 3]
            colors: Point colors [N, 3] in range [0, 1]
            sh_degree: Maximum SH degree (0-3)
            init_opacity: Initial opacity before logit transform
            init_scale: Scale multiplier for initial Gaussian sizes
        
        Returns:
            SplatModel initialized from points
        """
        if points.shape[0] != colors.shape[0]:
            raise ValueError(f"points and colors must have same length, got {points.shape[0]} and {colors.shape[0]}")
        if points.shape[1] != 3:
            raise ValueError(f"points must be [N, 3], got {points.shape[1]}")
        if colors.shape[1] != 3:
            raise ValueError(f"colors must be [N, 3], got {colors.shape[1]}")

        # Ensure float32 and make copies
        points = np.asarray(points, dtype=np.float32).copy()
        colors = np.asarray(colors, dtype=np.float32).copy()
        N = points.shape[0]
        
        # Compute initial scales from KNN distances
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(axis=-1)  # [N,]
        dist_avg = np.sqrt(dist2_avg)
        scales = np.log(dist_avg * init_scale + 1e-8).astype(np.float32)
        scales = np.broadcast_to(scales[:, None], (len(scales), 3)).copy()  # [N, 3]
        
        # Random quaternions
        quats = np.random.rand(N, 4).astype(np.float32)
        
        # Initialize opacities in logit space
        opacities = logit(np.clip(init_opacity, 1e-6, 1 - 1e-6))
        opacities = np.full((N,), opacities, dtype=np.float32)
        
        # Initialize SH coefficients
        K = sh_to_K(sh_degree)
        colors_sh = np.zeros((N, K, 3), dtype=np.float32)
        colors_sh[:, 0, :] = rgb_to_sh(colors)
        
        sh0 = colors_sh[:, :1, :]    # [N, 1, 3]
        shN = colors_sh[:, 1:, :]    # [N, K-1, 3]
        
        return cls(
            _sh_degree=sh_degree,
            _points=points,
            _scales=scales,
            _quats=quats,
            _opacities=opacities,
            _sh0=sh0,
            _shN=shN,
        )
    
    @classmethod
    def from_random(
        cls,
        num_points: int = 100_000,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        init_extent: float = 3.0,
        init_scale: float = 1.0,
    ) -> 'SplatModel':
        """
        Initialize with random points (for testing).
        
        Args:
            num_points: Number of Gaussians
            sh_degree: Maximum SH degree
            init_opacity: Initial opacity
            init_extent: Spatial extent of initialization
            init_scale: Scale multiplier
        
        Returns:
            SplatModel with random parameters
        """
        points = (np.random.rand(num_points, 3) * 2 - 1) * init_extent
        colors = np.random.rand(num_points, 3)
        
        return cls.from_points(points, colors, sh_degree, init_opacity, init_scale)
    
    def save_ply(self, path: str):
        """
        Export model as PLY file for viewing/inference.
        
        Args:
            path: Output path (e.g., "output.ply")
        """
        from gsplat import export_splats
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to torch tensors
        means = torch.from_numpy(self._points)
        scales = torch.from_numpy(self._scales)
        quats = torch.from_numpy(self._quats)
        opacities = torch.from_numpy(self._opacities)
        sh0 = torch.from_numpy(self._sh0)
        shN = torch.from_numpy(self._shN)
        
        export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=path,
        )
    
    @classmethod
    def load_ply(cls, path: str) -> 'SplatModel':
        """
        Load model from PLY file.
        
        Args:
            path: Path to PLY file
        
        Returns:
            SplatModel loaded from PLY
        """
        from plyfile import PlyData
        
        plydata = PlyData.read(path)
        element = plydata.elements[0]
        
        # Parse positions
        points = np.stack([
            np.asarray(element["x"]),
            np.asarray(element["y"]),
            np.asarray(element["z"]),
        ], axis=1).astype(np.float32)
        
        # Parse opacities (stored in logit space in PLY)
        opacities = np.asarray(element["opacity"]).astype(np.float32)
        
        # Parse scales (stored in log space in PLY)
        scales = np.stack([
            np.asarray(element[f"scale_{i}"]) for i in range(3)
        ], axis=1).astype(np.float32)
        
        # Parse quaternions
        quats = np.stack([
            np.asarray(element[f"rot_{i}"]) for i in range(4)
        ], axis=1).astype(np.float32)
        
        # Parse SH DC component
        sh0 = np.stack([
            np.asarray(element[f"f_dc_{i}"]) for i in range(3)
        ], axis=1).astype(np.float32)
        sh0 = sh0[:, np.newaxis, :]  # [N, 1, 3]
        
        # Parse higher-order SH
        extra_f_names = [p.name for p in element.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        
        if len(extra_f_names) > 0:
            features_extra = np.stack([
                np.asarray(element[name]) for name in extra_f_names
            ], axis=1).astype(np.float32)
            
            # Infer sh_degree
            num_sh_rest = len(extra_f_names) // 3
            sh_degree = int(np.sqrt(num_sh_rest + 1)) - 1
            
            # Reshape to [N, K-1, 3]
            K = (sh_degree + 1) ** 2 - 1
            shN = features_extra.reshape((points.shape[0], K, 3))
        else:
            sh_degree = 0
            shN = np.zeros((points.shape[0], 0, 3), dtype=np.float32)
        
        model = cls(
            _sh_degree=sh_degree,
            _points=points,
            _scales=scales,
            _quats=quats,
            _opacities=opacities,
            _sh0=sh0,
            _shN=shN,
        )
        
        print(f"Loaded PLY: {path} ({model.num_points} Gaussians, sh_degree={sh_degree})")
        return model
    
    # Properties (read-only access)
    @property
    def sh_degree(self) -> int:
        """Maximum spherical harmonics degree"""
        return self._sh_degree
    
    @property
    def points(self) -> np.ndarray:
        """Gaussian positions [N, 3]"""
        return self._points
    
    @property
    def scales(self) -> np.ndarray:
        """Gaussian scales in log-space [N, 3]"""
        return self._scales
    
    @property
    def quats(self) -> np.ndarray:
        """Gaussian quaternions [N, 4]"""
        return self._quats
    
    @property
    def opacities(self) -> np.ndarray:
        """Gaussian opacities in logit-space [N,]"""
        return self._opacities
    
    @property
    def sh0(self) -> np.ndarray:
        """0th order spherical harmonics [N, 1, 3]"""
        return self._sh0
    
    @property
    def shN(self) -> np.ndarray:
        """Higher order spherical harmonics [N, K, 3]"""
        return self._shN
    
    @property
    def num_points(self) -> int:
        """Number of Gaussians"""
        return len(self._points)
    
    def __repr__(self) -> str:
        return (
            f"SplatModel(\n"
            f"  num_points={self.num_points},\n"
            f"  sh_degree={self.sh_degree},\n"
            f"  points: {self._points.shape} {self._points.dtype},\n"
            f"  scales: {self._scales.shape} {self._scales.dtype},\n"
            f"  quats: {self._quats.shape} {self._quats.dtype},\n"
            f"  opacities: {self._opacities.shape} {self._opacities.dtype},\n"
            f"  sh0: {self._sh0.shape} {self._sh0.dtype},\n"
            f"  shN: {self._shN.shape} {self._shN.dtype}\n"
            f")"
        )