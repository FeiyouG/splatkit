from dataclasses import dataclass, field

from gsplat.strategy import DefaultStrategy, MCMCStrategy

@dataclass
class SplatTrainerConfig:
    """Training configuration"""
    max_steps: int
    batch_size: int = 1
    
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    
    log_steps: int = 100
    result_dir: str = "results/default"
    
    strategy: DefaultStrategy | MCMCStrategy = field(
        default_factory=lambda: DefaultStrategy(verbose=True)
    )
    
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opacity: float = 0.1
    init_scale: float = 1.0
    
    num_workers: int = 4
    global_scale: float = 1.0