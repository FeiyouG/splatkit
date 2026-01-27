from dataclasses import dataclass, field

from gsplat.strategy import DefaultStrategy, MCMCStrategy

@dataclass
class SplatTrainerConfig:
    """
    Training configuration for Gaussian Splatting.
    
    Controls training hyperparameters, optimization settings, and logging.
    
    Args:
        max_steps: Total number of training iterations to run
        batch_size: Number of images to render per training step (default: 1)
        
        sh_degree: Maximum spherical harmonics degree (0-3, default: 3)
                   Higher degrees = better view-dependent effects but slower
        sh_degree_interval: Steps between SH degree increases (default: 1000)
                           Gradually increases from 0 to sh_degree
        
        init_opacity: Initial opacity for new Gaussians (0-1, default: 0.1)
        init_scale: Initial scale multiplier for new Gaussians (default: 1.0)
        
        num_workers: DataLoader worker processes (default: 4)
        global_scale: Scene scale factor, affects densification (default: 1.0)
    
    """
    max_steps: int

    batch_size: int = 1

    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opacity: float = 0.1
    init_scale: float = 1.0
    
    num_workers: int = 4
    global_scale: float = 1.0