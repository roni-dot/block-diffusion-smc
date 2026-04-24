# ──────────────────────────────────────────────────────────────────────────────
#     KV-cache utilities
#     Adapted from power_smc/Power-SMC/smc_samp_utils.py
#     Simplified for the tuple-of-tuples format used by LLaDA / HuggingFace.
#     Each entry: past_key_values[layer] = (keys, values)
#                 keys / values shape: (batch, n_heads, seq_len, head_dim)
# ──────────────────────────────────────────────────────────────────────────────

from click import Tuple
import torch


def truncate_kv_to_prefix(
    past_key_values: Tuple, prefix_len: int
) -> Tuple:
    """Keep only the first `prefix_len` positions of each KV tensor."""
    return tuple(
        tuple(t[:, :, :prefix_len, :] for t in layer_kv)
        for layer_kv in past_key_values
    )


def reorder_kv(past_key_values: Tuple, idx: torch.Tensor) -> Tuple:
    """
    Reorder the batch dimension of KV caches according to `idx`.
    Used after resampling to align caches with new particle order.
    idx: 1-D long tensor of length N.
    """
    return tuple(
        tuple(t[idx] for t in layer_kv)
        for layer_kv in past_key_values
    )


def expand_kv(past_key_values: Tuple, N: int) -> Tuple:
    """
    Broadcast a batch-1 KV cache to N particles.
    Useful for sharing the initial prompt cache.
    """
    return tuple(
        tuple(t.expand(N, -1, -1, -1).contiguous() for t in layer_kv)
        for layer_kv in past_key_values
    )