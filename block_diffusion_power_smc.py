"""
block_diffusion_power_smc.py

Power-SMC applied to Block Diffusion Language Models (LLaDA / Dream).

Weight update — committed-token log-probability (replaces Eq. 8 Rényi entropy):

    log w_update = α × ∑_{i=n+1}^{n+B} log p(x_i_committed | x_{1:n})

Computed AFTER denoising, using the initial block forward-pass logits as the
marginal approximation.  This directly rewards particles that committed
high-probability tokens, rather than measuring how peaked the marginals were.

Original Eq. 8 (Rényi entropy / partition function) is preserved in
block_diffusion_power_smc_utils.compute_block_log_weight for reference.

High-level algorithm (one SMC run):
    For each block m = 0 … K-1:
        1.  Forward pass on x[:, prompt_len:] with fixed prompt KV cache.
            LLaDA uses BIDIRECTIONAL attention: every token's K/V depends on
            ALL positions, so we must recompute from the full generation range
            each block to get correct K/V for previously-committed tokens.
            → produces marginal logits p(x_i | current sequence) for block m.
        2.  Run block denoising (Fast-dLLM dual-cache style) to commit
            block tokens for every particle.
        3.  Compute log weight update from the committed tokens' log-probs.
        4.  Resample if ESS < threshold · N.
    Return weighted particle set; draw final answer.

Code sources:
    KV-cache utilities and SMC primitives adapted from:
        power_smc/Power-SMC/smc_samp_utils.py
    Block generation helpers copied from:
        fast_dllm/Fast-dLLM/llada/generate.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any, Dict

from data_classes import BlockDiffusionSMCConfig
from power_smc_utils import (
    expand_kv,
    effective_sample_size,
    systematic_resample,
)
from fast_dllm_utils.block_generation_utils import (
    get_transfer_index,
    get_num_transfer_tokens,
    get_transfer_index_factor,
)
# Original Eq. 8 weight (Rényi entropy) — kept for reference / easy revert.
from block_diffusion_power_smc_utils import compute_block_log_weight  # noqa: F401


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

    v = cfg.verbose
    if v:
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

    # ── Prompt KV cache: computed once at batch=1, expanded to N ─────────
    # Prompt tokens never change — compute their K/V once and reuse every block.
    # kv_prompt_N is NEVER modified: all particles share the same prompt, so
    # resampling only needs to reorder x, not the KV cache.
    if v: print(f"│  [fwd] prompt forward pass (batch=1) …", end=" ", flush=True)
    out_prompt = model(x[:1, :prompt_len], use_cache=True)
    prompt_kv = out_prompt.past_key_values
    del out_prompt
    torch.cuda.empty_cache()
    kv_prompt_N = expand_kv(prompt_kv, N)  # (N, prompt_len) K/V — fixed for all blocks
    del prompt_kv
    if v: print("done")

    # ── Outer loop: one iteration per block ───────────────────────────────
    for nb in range(num_blocks):
        s = prompt_len + nb * cfg.block_length   # block start (absolute)
        e = s + cfg.block_length                 # block end   (exclusive)

        if v: print(f"┌─ Block {nb+1}/{num_blocks}  (positions {s}–{e-1}) {'─'*30}")

        # ── 6a. Forward pass on x[:, prompt_len:] with fixed prompt KV ──────
        # LLaDA uses BIDIRECTIONAL attention: K/V at every position depends on
        # ALL other positions. We cannot carry forward a KV cache built when
        # future blocks were [MASK] — committed tokens would have stale K/V.
        # Fix: always recompute the full generation range from the fixed prompt
        # KV, so every committed token gets fresh K/V given the current sequence.
        gen_start = nb * cfg.block_length   # offset within generation range
        if v: print(f"│  [fwd] forward pass on x[:, prompt_len:] (batch={N}) …", end=" ", flush=True)
        out = model(x[:, prompt_len:], past_key_values=kv_prompt_N, use_cache=True)
        if v: print("done")

        # logits for block nb are at gen_start..gen_start+B in the output.
        logits_block = out.logits[:, gen_start:gen_start+cfg.block_length, :].contiguous()  # (N, B, V)
        full_kv = out.past_key_values  # covers 0..total_len-1; dual-cache steps patch [s,e) in-place
        del out

        # ── 6b. Block denoising — step 0 (using block logits) ────────────
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
        del x_block  # keep logits_block alive — needed for weight after full denoising

        if v:
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

        if v: print(f"│  [den] denoising complete after {steps_run}/{cfg.steps_per_block} steps")

        # ── 6c. Committed-token log weight ───────────────────────────────
        # log w = α × Σ_j log p(x_j_committed | prefix_with_MASK_block)
        # Uses logits_block from the initial forward pass (all MASK) as the
        # marginal approximation — same compute, no extra forward pass needed.
        # Rewards particles that committed high-probability tokens directly.
        log_probs = F.log_softmax(logits_block.float(), dim=-1)   # (N, B, V)
        del logits_block
        committed_log_p = log_probs.gather(
            -1, x[:, s:e].unsqueeze(-1)
        ).squeeze(-1)                                              # (N, B)
        del log_probs
        log_w_update = cfg.alpha * committed_log_p.sum(dim=-1)    # (N,)
        log_w = log_w + log_w_update

        stats["mean_logw_history"].append(log_w_update.mean().item())
        stats["max_logw_history"].append(log_w_update.max().item())
        if v:
            print(f"│  [wt]  log_w_update  mean={log_w_update.mean():.3f}  "
                  f"min={log_w_update.min():.3f}  max={log_w_update.max():.3f}")
            print(f"│        per-particle: {log_w_update.tolist()}")

        # full_kv is not carried forward — next block recomputes from kv_prompt_N.
        del full_kv

        # ── 6e. Resampling ────────────────────────────────────────────────
        lw = log_w - torch.logsumexp(log_w, dim=0)
        w = torch.exp(lw)
        ess = effective_sample_size(w)
        stats["ess_history"].append(ess)

        if v:
            print(f"│  [wt]  cumulative log_w: {log_w.tolist()}")
            print(f"│  [wt]  norm weights:     {[f'{w_:.3f}' for w_ in w.tolist()]}")
            print(f"│  [ess] ESS={ess:.2f}/{N}  (threshold={cfg.ess_threshold*N:.1f})", end="")

        if ess < cfg.ess_threshold * N:
            idx_rs = systematic_resample(w, generator=g)
            x = x[idx_rs]
            # kv_prompt_N is identical for all particles (same prompt) — no reorder needed.
            log_w = torch.zeros(N, device=device)
            stats["resample_count"] += 1
            stats["resample_at_blocks"].append(nb)
            if v: print(f"  → RESAMPLING  indices={idx_rs.tolist()}")
        else:
            if v: print("  → no resample")

        if v: print(f"└{'─'*55}\n")

    # ── Final weighted draw ────────────────────────────────────────────────
    lw_final = log_w - torch.logsumexp(log_w, dim=0)
    w_final = torch.exp(lw_final)
    chosen_idx = int(torch.multinomial(w_final, 1, generator=g).item())
    if v:
        print(f"[final] weights: {[f'{w_:.3f}' for w_ in w_final.tolist()]}")
        print(f"[final] chose particle {chosen_idx}\n")

    return {
        "sequences": x,
        "log_w": log_w,
        "w": w_final,
        "chosen_idx": chosen_idx,
        "chosen_sequence": x[chosen_idx],
        "stats": stats,
    }
