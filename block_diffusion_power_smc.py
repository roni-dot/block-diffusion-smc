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

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from .types import BlockDiffusionSMCConfig
from .power_smc_utils.kv_cache_utils import truncate_kv_to_prefix, reorder_kv, expand_kv


from .power_smc_utils.smc_utils import effective_sample_size, systematic_resample


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Block generation helpers
#     Copied from fast_dllm/Fast-dLLM/llada/generate.py
# ──────────────────────────────────────────────────────────────────────────────

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
# 5.  Weight update  — NEW  (Eq. 8 from the paper)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main SMC function
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def smc_block_diffusion(
    model,
    tokenizer,
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
        "log_w_update_history": [],
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
        prefix_kv = truncate_kv_to_prefix(out.past_key_values, s)
        del out  # release full logits; PyTorch reclaims memory lazily

        # ── 6b. Compute log weight update (Eq. 8) ────────────────────────
        log_w_update = compute_block_log_weight(logits_block, 0, cfg.block_length, cfg.alpha)
        log_w = log_w + log_w_update

        stats["log_w_update_history"].append(log_w_update.mean().item())
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
            x0_blk, transfer_blk = _get_transfer_index_factor(
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
        steps_run = 1
        for step_i in range(1, cfg.steps_per_block):
            remaining = (x[:, s:e] == cfg.mask_id).sum()
            if remaining == 0:
                break  # block fully unmasked — skip remaining steps

            # Input to the model: current block onwards (prefix handled by KV)
            block_input = x[:, s:]            # (N, gen_length - nb*block_length)
            block_mask_idx = (block_input == cfg.mask_id)
            block_mask_idx[:, cfg.block_length:] = False  # restrict to current block

            logits_blk = model(
                block_input,
                past_key_values=prefix_kv,
                use_cache=False,
            ).logits

            if cfg.factor is not None:
                x0_blk, transfer_blk = _get_transfer_index_factor(
                    logits_blk, cfg.temperature, cfg.remasking,
                    block_mask_idx, block_input, cfg.factor
                )
            else:
                x0_blk, transfer_blk = get_transfer_index(
                    logits_blk, cfg.temperature, cfg.remasking,
                    block_mask_idx, block_input,
                    num_transfer[:, step_i] if cfg.threshold is None else None,
                    cfg.threshold,
                )

            block_input_updated = torch.where(transfer_blk, x0_blk, block_input)
            x = torch.cat([x[:, :s], block_input_updated], dim=1)
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


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Factor-based parallel decoding helper  (Fast-dLLM §3.3 extension)
#     Included for completeness; mirrors get_transfer_index_dynamic in
#     fast_dllm/Fast-dLLM/llada/generate.py
# ──────────────────────────────────────────────────────────────────────────────

def _get_transfer_index_factor(
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


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Quick sanity-check / demo
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    """
    Minimal smoke-test (requires a GPU and the LLaDA model weights).
    Run with:  python block_diffusion_power_smc.py
    """
    from transformers import AutoTokenizer, AutoModel

    model_name = "GSAI-ML/LLaDA-8B-Instruct"

    print(f"Loading {model_name} …")
   
    from llada.model import LLaDAModelLM
    

    model = (
        LLaDAModelLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompt = "What is 157 multiplied by 34?"
    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = torch.tensor(
        tokenizer(prompt_str)["input_ids"],
        device=next(model.parameters()).device,
    ).unsqueeze(0)

    cfg = BlockDiffusionSMCConfig(
        alpha=2.0,
        n_particles=16,
        gen_length=64,
        block_length=32,
        steps_per_block=32,
        temperature=2.0,
    )

    print(f"Running SMC with N={cfg.n_particles} particles, α={cfg.alpha} …")
    result = smc_block_diffusion(model, tokenizer, input_ids, cfg)

    answer_ids = result["chosen_sequence"][input_ids.shape[1]:]
    eos_id = tokenizer.eos_token_id
    eos_pos = (answer_ids == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_pos):
        answer_ids = answer_ids[:eos_pos[0]]
    print("Answer:", tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True))
    print("Stats:", result["stats"])


if __name__ == "__main__":
    _demo()
