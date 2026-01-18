import numpy as np


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to 0th order spherical harmonics (DC component).
    
    The DC coefficient for spherical harmonics represents the average radiance.
    The formula converts RGB [0,1] to SH space by:
    1. Shifting to [-0.5, 0.5] range 
    2. Dividing by C0 constant (sqrt(1/(4*pi)))
    
    Args:
        rgb: RGB colors in range [0, 1], shape [..., 3]
    
    Returns:
        SH DC coefficients, same shape as input
    """
    # SH constant for l=0, m=0: Y_0^0 = 1/(2*sqrt(pi)) = 0.28209479177387814
    C0 = 0.28209479177387814
    # Convert RGB from [0,1] to SH space
    # Center around 0.5 to represent gray as zero SH coefficient
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