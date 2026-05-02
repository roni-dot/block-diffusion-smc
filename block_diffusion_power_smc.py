"""
block_diffusion_power_smc.py

Power-SMC applied to Block Diffusion Language Models (LLaDA / Dream).

Implements the weight update derived in "Applying Power-SMC to Block Diffusion":

    w_update = ∏_{i=n+1}^{n+B} ( ∑_{v ∈ V} p(x_i=v | x_{1:n})^α )   [Eq. 8]

Under the Mean Field Approximation the complexity drops from O(|V|^B) to O(B·|V|).

High-level algorithm (one SMC run):
    For each block m = 0 … K-1:
        1.  Full-sequence forward pass on all N particles (batched).
            → produces marginal logits p(x_i | prefix) for every block position.
        2.  Compute log weight update (Eq. 8) from those marginals.
        3.  Run block denoising (Fast-dLLM style, prefix-KV cache) to commit
            block tokens for every particle.
        4.  Resample if ESS < threshold · N.
    Return weighted particle set; draw final answer.

Code sources:
    KV-cache utilities and SMC primitives adapted from:
        power_smc/Power-SMC/smc_samp_utils.py
    Block generation helpers copied from:
        fast_dllm/Fast-dLLM/llada/generate.py
    Weight update (compute_block_log_weight) is new.
"""

from __future__ import annotations

import torch
from typing import Any, Dict

from data_classes import BlockDiffusionSMCConfig
from power_smc_utils import (
    effective_sample_size,
    systematic_resample,
)
from fast_dllm_utils.block_generation_utils import (
    get_transfer_index, 
    get_num_transfer_tokens, 
    get_transfer_index_factor,
)
from block_diffusion_power_smc_utils import compute_block_log_weight


# ──────────────────────────────────────────────────────────────────────────────
#    Main SMC function
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def smc_block_diffusion(
    model,
    input_ids: torch.Tensor,   # (1, prompt_len)
    cfg: BlockDiffusionSMCConfig,
) -> Dict[str, Any]:
    """
    Run Power-SMC over a block diffusion model.

    Returns a dict with:
        sequences     : (N, prompt_len + gen_length) — all particle sequences
        log_w         : (N,) — unnormalized log weights
        w             : (N,) — normalized weights
        chosen_idx    : int  — index of the sampled output particle
        chosen_sequence: (prompt_len + gen_length,)
        stats         : dict with ESS history etc.
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    assert cfg.gen_length % cfg.block_length == 0, \
        "gen_length must be divisible by block_length"

    device = input_ids.device
    N = cfg.n_particles
    prompt_len = input_ids.size(1)
    num_blocks = cfg.gen_length // cfg.block_length
    g = torch.Generator(device=device)

    print(f"\n{'='*60}")
    print(f"  Power-SMC  |  N={N} particles  α={cfg.alpha}  blocks={num_blocks}x{cfg.block_length}tok")
    print(f"  prompt_len={prompt_len}  gen_length={cfg.gen_length}  steps/block={cfg.steps_per_block}")
    print(f"  ESS threshold={cfg.ess_threshold*N:.1f}  temperature={cfg.temperature}")
    print(f"{'='*60}\n")

    # ── Initialise particle sequences (all answer positions are [MASK]) ────
    # x: (N, prompt_len + gen_length)
    x = torch.full(
        (N, prompt_len + cfg.gen_length),
        cfg.mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = input_ids.expand(N, -1)

    log_w = torch.zeros(N, device=device)

    stats: Dict[str, Any] = {
        "ess_history": [],
        "mean_logw_history": [],
        "max_logw_history": [],
        "resample_count": 0,
        "resample_at_blocks": [],
    }

    # ── Outer loop: one iteration per block ───────────────────────────────
    for nb in range(num_blocks):
        s = prompt_len + nb * cfg.block_length   # block start (absolute)
        e = s + cfg.block_length                 # block end   (exclusive)

        print(f"┌─ Block {nb+1}/{num_blocks}  (positions {s}–{e-1}) {'─'*30}")

        # ── 6a. Full-sequence forward pass (batched over N particles) ─────
        print(f"│  [fwd] full-sequence forward pass (batch={N}) …", end=" ", flush=True)
        out = model(x, use_cache=True)
        print("done")

        # Slice block logits immediately and free the full output to save GPU memory.
        # Weight update and step-0 denoising only need positions [s, e).
        logits_block = out.logits[:, s:e, :].contiguous()  # (N, B, V)
        full_kv = out.past_key_values  # full cache; dual-cache steps patch [s,e) in-place
        del out  # release full logits; PyTorch reclaims memory lazily

        # ── 6b. Compute log weight update (Eq. 8) ────────────────────────
        log_w_update = compute_block_log_weight(logits_block, 0, cfg.block_length, cfg.alpha)
        log_w = log_w + log_w_update

        stats["mean_logw_history"].append(log_w_update.mean().item())
        stats["max_logw_history"].append(log_w_update.max().item())
        print(f"│  [wt]  log_w_update  mean={log_w_update.mean():.3f}  "
              f"min={log_w_update.min():.3f}  max={log_w_update.max():.3f}")
        print(f"│        per-particle: {log_w_update.tolist()}")

        # ── 6c. Block denoising — step 0 (using block logits) ────────────
        block_mask_index = (x[:, s:e] == cfg.mask_id)           # (N, B)
        num_transfer = get_num_transfer_tokens(
            block_mask_index, cfg.steps_per_block
        )  # (N, steps_per_block)

        x_block = x[:, s:e].clone()  # (N, B)
        if cfg.factor is not None:
            x0_blk, transfer_blk = get_transfer_index_factor(
                logits_block, cfg.temperature, cfg.remasking,
                block_mask_index, x_block, cfg.factor
            )
        else:
            x0_blk, transfer_blk = get_transfer_index(
                logits_block, cfg.temperature, cfg.remasking,
                block_mask_index, x_block,
                num_transfer[:, 0] if cfg.threshold is None else None,
                cfg.threshold,
            )
        x[:, s:e] = torch.where(transfer_blk, x0_blk, x_block)
        del logits_block, x_block

        tokens_unmasked_step0 = transfer_blk.sum(dim=1)
        print(f"│  [den] step 0: unmasked {tokens_unmasked_step0.tolist()} tokens per particle")

        # ── 6d. Block denoising — steps 1 … steps_per_block-1 ────────────
        # Dual-cache: pass only x[:, s:e] (block_length tokens) each step.
        # replace_position tells the model to patch positions [s, e) in full_kv
        # in-place, so attention sees the correct full-sequence context.
        replace_position = torch.zeros((N, x.shape[1]), dtype=torch.bool, device=device)
        replace_position[:, s:e] = True

        steps_run = 1
        for step_i in range(1, cfg.steps_per_block):
            if (x[:, s:e] == cfg.mask_id).sum() == 0:
                break  # block fully unmasked — skip remaining steps

            x_block = x[:, s:e].clone()          # (N, B)
            block_mask_idx = (x_block == cfg.mask_id)  # (N, B)

            logits_blk = model(
                x_block,
                past_key_values=full_kv,
                use_cache=False,
                replace_position=replace_position,
            ).logits  # (N, B, V)

            if cfg.factor is not None:
                x0_blk, transfer_blk = get_transfer_index_factor(
                    logits_blk, cfg.temperature, cfg.remasking,
                    block_mask_idx, x_block, cfg.factor
                )
            else:
                x0_blk, transfer_blk = get_transfer_index(
                    logits_blk, cfg.temperature, cfg.remasking,
                    block_mask_idx, x_block,
                    num_transfer[:, step_i] if cfg.threshold is None else None,
                    cfg.threshold,
                )

            x[:, s:e] = torch.where(transfer_blk, x0_blk, x_block)
            steps_run += 1

        print(f"│  [den] denoising complete after {steps_run}/{cfg.steps_per_block} steps")

        # ── 6e. Resampling ────────────────────────────────────────────────
        lw = log_w - torch.logsumexp(log_w, dim=0)
        w = torch.exp(lw)
        ess = effective_sample_size(w)
        stats["ess_history"].append(ess)

        print(f"│  [wt]  cumulative log_w: {log_w.tolist()}")
        print(f"│  [wt]  norm weights:     {[f'{v:.3f}' for v in w.tolist()]}")
        print(f"│  [ess] ESS={ess:.2f}/{N}  (threshold={cfg.ess_threshold*N:.1f})", end="")

        if ess < cfg.ess_threshold * N:
            idx_rs = systematic_resample(w, generator=g)
            x = x[idx_rs]
            log_w = torch.zeros(N, device=device)
            stats["resample_count"] += 1
            stats["resample_at_blocks"].append(nb)
            print(f"  → RESAMPLING  indices={idx_rs.tolist()}")
        else:
            print("  → no resample")

        print(f"└{'─'*55}\n")

    # ── Final weighted draw ────────────────────────────────────────────────
    lw_final = log_w - torch.logsumexp(log_w, dim=0)
    w_final = torch.exp(lw_final)
    chosen_idx = int(torch.multinomial(w_final, 1, generator=g).item())
    print(f"[final] weights: {[f'{v:.3f}' for v in w_final.tolist()]}")
    print(f"[final] chose particle {chosen_idx}\n")

    return {
        "sequences": x,
        "log_w": log_w,
        "w": w_final,
        "chosen_idx": chosen_idx,
        "chosen_sequence": x[chosen_idx],
        "stats": stats,
    }
