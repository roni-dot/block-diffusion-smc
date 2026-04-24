# ──────────────────────────────────────────────────────────────────────────────
#     Block generation helpers
#     Copied from fast_dllm/Fast-dLLM/llada/generate.py
# ──────────────────────────────────────────────────────────────────────────────

from typing import Optional, Tuple
import torch.nn.functional as F
import torch
import numpy as np

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Gumbel-max trick for categorical sampling.
    temperature=0 → pure argmax (no noise).
    Uses float64 for numerical stability (per Fast-dLLM).
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(
    block_mask_index: torch.Tensor, steps: int
) -> torch.Tensor:
    """
    Compute per-step token transfer schedule for the current block.
    block_mask_index: (N, block_length) bool — which positions are still masked.
    Returns: (N, steps) int — how many tokens to unmask at each step.
    """
    device = block_mask_index.device
    total = block_mask_index.sum(dim=1)              # (N,)
    base = torch.div(total, steps, rounding_mode="floor")  # (N,)
    rem = total - base * steps                       # (N,)

    num_transfer = base.unsqueeze(1).expand(-1, steps).to(torch.long)
    cols = torch.arange(steps, device=device).unsqueeze(0)  # (1, steps)
    add_mask = cols < rem.unsqueeze(1)               # (N, steps)
    return num_transfer + add_mask.to(torch.long)


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (N, L) bool
    x: torch.Tensor,            # (N, L) long
    num_transfer_tokens,        # (N,) long tensor, or None when threshold is used
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decide which masked positions to unmask this step.
    Returns:
        x0            : (N, L) long — proposed tokens
        transfer_index: (N, L) bool — positions to update
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (N, L)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float32)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(
        torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype
    )
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (N, L)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens required when threshold is None.")

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(
        dtype=torch.long, device=confidence.device
    ).clamp(min=0)

    N, L = confidence.shape
    _, idx_sort = torch.sort(confidence, dim=1, descending=True)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(N, L)
    k_exp = num_transfer_tokens.unsqueeze(1).expand(N, L)
    select_sorted = cols < k_exp

    transfer_int = torch.zeros(N, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx_sort, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index

# ──────────────────────────────────────────────────────────────────────────────
# 7.  Factor-based parallel decoding helper  (Fast-dLLM §3.3 extension)
#     Included for completeness; mirrors get_transfer_index_dynamic in
#     fast_dllm/Fast-dLLM/llada/generate.py
# ──────────────────────────────────────────────────────────────────────────────

def get_transfer_index_factor(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factor-based parallel decoding: find largest n such that (n+1)(1 - c^(n)) < factor.
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float32)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, dtype=x0_p.dtype, device=x0_p.device))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    num_masked = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        n_tok = int(num_masked[j].item())
        if n_tok == 0:
            continue
        ns = list(range(1, n_tok + 1))
        threshs = [1.0 - factor / (n + 1) for n in ns]
        threshs[0] = -1.0  # always unmask at least one token

        sorted_conf = torch.sort(
            confidence[j][mask_index[j]], descending=True
        )[0]
        top_i = 0
        for top_i in range(len(threshs)):
            if sorted_conf[top_i] < threshs[top_i]:
                break
        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, sel_idx = torch.topk(confidence[j], k=top_i)
        transfer_index[j, sel_idx] = True

    return x0, transfer_index