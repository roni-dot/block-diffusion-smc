# ──────────────────────────────────────────────────────────────────────────────
#     Weight update  — NEW  (Eq. 8 from the paper)
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F


def compute_block_log_weight(
    logits: torch.Tensor,  # (N, L, V) — may be full sequence or pre-sliced block
    s: int,                # start index into logits (0 when logits is already sliced)
    e: int,                # end index into logits (exclusive)
    alpha: float,
) -> torch.Tensor:
    """
    Log weight update for one block under the Mean Field Approximation.

    From Eq. 8:
        w_update = ∏_{i=s}^{e-1}  ∑_{v ∈ V}  p(x_i=v | prefix)^α

    In log space:
        log w_update = ∑_{i=s}^{e-1}  logsumexp( α · log p(x_i | prefix) )

    Args:
        logits: full-sequence logits from the initial forward pass of this block.
                Shape (N, full_seq_len, V).
        s, e  : start and end (exclusive) positions of the current block.
        alpha : sharpening exponent.

    Returns:
        log_w_update: shape (N,)
    """
    block_logits = logits[:, s:e, :].float()             # (N, B, V)
    log_probs = F.log_softmax(block_logits, dim=-1)      # (N, B, V)
    # logsumexp(α · log p) = log( ∑_v p^α )  for each position
    per_pos = torch.logsumexp(alpha * log_probs, dim=-1)  # (N, B)
    return per_pos.sum(dim=-1)                            # (N,)