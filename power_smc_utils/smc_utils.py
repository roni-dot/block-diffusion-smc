# ──────────────────────────────────────────────────────────────────────────────
#     SMC primitives
#     Copied verbatim from power_smc/Power-SMC/smc_samp_utils.py
# ──────────────────────────────────────────────────────────────────────────────

import torch


def effective_sample_size(w: torch.Tensor, eps: float = 1e-12) -> float:
    """ESS = 1 / ∑ w_i^2  (normalized weights)."""
    w = w.clamp_min(eps)
    return float(1.0 / torch.sum(w * w).item())


def systematic_resample(w: torch.Tensor, generator=None) -> torch.Tensor:
    """
    Systematic (low-variance) resampling.
    w: normalized weights, shape (N,).
    Returns index tensor of length N.
    """
    N = w.numel()
    device = w.device
    if generator is None:
        u0 = torch.rand((), device=device)
    else:
        u0 = torch.rand((), device=device, generator=generator)
    positions = (u0 + torch.arange(N, device=device)) / N
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0
    idx = torch.searchsorted(cdf, positions, right=False)
    return idx.clamp_max(N - 1).to(torch.long)