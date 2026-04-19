from dataclasses import dataclass
from typing import Optional


@dataclass
class BlockDiffusionSMCConfig:
    # power-smc parameters
    alpha: float = 2.0
    n_particles: int = 16
    ess_threshold: float = 0.5

    # Block diffusion parameters
    gen_length: int = 128  
    block_length: int = 128
    steps_per_block: int = 128
    temperature: float = 0.,
    remasking: str  = 'low_confidence', 
    mask_id: int = 126336, 
    
    threshold: Optional[float] = None  
    factor: Optional[float] = None 