import torch
from transformers import PreTrainedModel, AutoTokenizer

from .data_classes import BlockDiffusionSMCConfig


@torch.no_grad()
def smc_block_diffusion(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,   # (1, prompt_len)
    cfg: BlockDiffusionSMCConfig,
):
    """
    Run Power-SMC over a block diffusion model.
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1

    assert cfg.gen_length % cfg.block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = cfg.gen_length // cfg.block_length

    assert cfg.steps_per_block % num_blocks == 0, "steps_per_block must be divisible by num_blocks"
    cfg.steps_per_block = cfg.steps_per_block // num_blocks
    
    device = input_ids.device
    N = cfg.n_particles
    prompt_len = input_ids.size(1)
    num_blocks = cfg.gen_length // cfg.block_length
    g = torch.Generator(device=device)