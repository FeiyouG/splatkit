import numpy as np


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to 0th order spherical harmonics.
    
    Args:
        rgb: RGB colors in range [0, 1], shape [..., 3]
    
    Returns:
        SH coefficients, same shape as input
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def sh_to_K(sh_degree: int) -> int:
    """
    Convert spherical harmonics degree to number of coefficients.
    
    Args:
        sh_degree: Spherical harmonics degree
    
    Returns:
        Number of coefficients
    """
    return (sh_degree + 1) ** 2