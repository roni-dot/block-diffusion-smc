#!/usr/bin/env bash
# run_eval.sh — Evaluate Power-SMC block diffusion on GSM8K / MATH / HumanEval
#
# Usage:  bash run_eval.sh [task]   (default: gsm8k)
#
# Prerequisites:
#   pip install lm-eval   (lm-evaluation-harness)
#   CUDA_VISIBLE_DEVICES=0 selects the RTX 3090

set -euo pipefail

MODEL="GSAI-ML/LLaDA-8B-Instruct"
TASK="${1:-gsm8k}"
DEVICE="cuda"

# ── Common generation settings ────────────────────────────────────────────────
GEN_LENGTH=256        # total tokens to generate
BLOCK_LENGTH=32       # tokens per block  (8 blocks of 32)
STEPS_PER_BLOCK=32    # denoising steps per block
TEMPERATURE=0.5       # must be > 0 for particle diversity
REMASKING="low_confidence"

# ── SMC settings ──────────────────────────────────────────────────────────────
N_PARTICLES=32
ALPHA=2.0
ESS_THRESHOLD=0.5     # resample when ESS < 0.5 * N

# ── Baseline settings (standard block diffusion) ──────────────────────────────
N_PARTICLES_BASE=1
ALPHA_BASE=1.0

# ─────────────────────────────────────────────────────────────────────────────
# Helper: shared model_args string (everything except N and alpha)
# ─────────────────────────────────────────────────────────────────────────────
common_args() {
  echo "model_path=${MODEL},temperature=${TEMPERATURE},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},steps_per_block=${STEPS_PER_BLOCK},remasking=${REMASKING},device=${DEVICE}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Run a single evaluation
#   $1 = label (for output dir)
#   $2 = n_particles
#   $3 = alpha
# ─────────────────────────────────────────────────────────────────────────────
run_eval() {
  local label="$1"
  local n="$2"
  local alpha="$3"
  local outdir="results/${TASK}/${label}"

  echo ""
  echo "══════════════════════════════════════════════════════"
  echo "  Task: ${TASK}   Run: ${label}   N=${n}  α=${alpha}"
  echo "══════════════════════════════════════════════════════"

  CUDA_VISIBLE_DEVICES=0 python -m lm_eval \
    --model smc_block_diffusion \
    --model_args "$(common_args),n_particles=${n},alpha=${alpha},ess_threshold=${ESS_THRESHOLD},save_dir=${outdir}/predictions" \
    --tasks "${TASK}" \
    --num_fewshot 5 \
    --output_path "${outdir}" \
    --include_path "$(pwd)"   # so lm_eval can find eval_smc.py
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
echo "Task: ${TASK}"
echo "Model: ${MODEL}"

# 1. Baseline
run_eval "baseline_N1_alpha1" "${N_PARTICLES_BASE}" "${ALPHA_BASE}"

# 2. Power-SMC
run_eval "smc_N${N_PARTICLES}_alpha${ALPHA}" "${N_PARTICLES}" "${ALPHA}"

echo ""
echo "Results written to results/${TASK}/"
