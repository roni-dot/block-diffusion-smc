#!/usr/bin/env bash
# run_eval.sh — Evaluate Power-SMC block diffusion on GSM8K / MATH / HumanEval
#
# Usage:  bash run_eval.sh [task]   (default: gsm8k)
#
# Prerequisites:
#   pip install lm-eval   (lm-evaluation-harness)

set -euo pipefail

MODEL="GSAI-ML/LLaDA-8B-Instruct"
TASK="${1:-gsm8k}"

# ── Common generation settings ────────────────────────────────────────────────
GEN_LENGTH=256        # total tokens to generate (matches Fast-dLLM paper setting)
BLOCK_LENGTH=8       # tokens per block  (32 blocks of 8)
STEPS_PER_BLOCK=32    # denoising steps per block
REMASKING="low_confidence"

# ── SMC settings ──────────────────────────────────────────────────────────────
N_PARTICLES=8
ALPHA=2.0
ALPHA_POWERSMC=4.0    # matches Power-SMC paper: alpha = 1/temp = 1/0.25
ESS_THRESHOLD=0.5     # resample when ESS < 0.5 * N
TEMPERATURE=0.5
TEMPERATURE_POWERSMC=0.25  # matches Power-SMC paper default

# ── Baseline settings (standard block diffusion) ──────────────────────────────
N_PARTICLES_BASE=1
ALPHA_BASE=1.0
TEMPERATURE_BASE=0.0

# ─────────────────────────────────────────────────────────────────────────────
# Helper: shared model_args string (everything except N and alpha)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE=200              # number of examples to evaluate (passed as --limit to lm_eval)

common_args() {
  local outdir="$1"
  echo "model_path=${MODEL},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},steps_per_block=${STEPS_PER_BLOCK},remasking=${REMASKING},save_dir=${outdir}/predictions"
}

# ─────────────────────────────────────────────────────────────────────────────
# Run a single evaluation
#   $1 = label (for output dir)
#   $2 = n_particles
#   $3 = alpha
#   $4 = temperature
# ─────────────────────────────────────────────────────────────────────────────
run_eval() {
  local label="$1"
  local n="$2"
  local alpha="$3"
  local temperature="$4"
  local outdir="results/${TASK}/${label}"

  echo ""
  echo "══════════════════════════════════════════════════════"
  echo "  Task: ${TASK}   Run: ${label}   N=${n}  α=${alpha}"
  echo "══════════════════════════════════════════════════════"

  python eval_smc.py \
    --model smc_block_diffusion \
    --model_args "$(common_args "${outdir}"),n_particles=${n},alpha=${alpha},ess_threshold=${ESS_THRESHOLD},temperature=${temperature}" \
    --tasks "${TASK}" \
    --num_fewshot 5 \
    --limit "${SAMPLE}" \
    --output_path "${outdir}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
echo "Task: ${TASK}"
echo "Model: ${MODEL}"

# # 1. Baseline (temperature=0, greedy — matches Fast-dLLM)
# run_eval "corrected_kv_cache_baseline_N1_alpha1_T0_S${SAMPLE}_BL${BLOCK_LENGTH}" "${N_PARTICLES_BASE}" "${ALPHA_BASE}" "${TEMPERATURE_BASE}"

# # 2. Baseline (temperature=0.5 — isolates temperature effect)
# run_eval "baseline_N1_alpha1_T05_S${SAMPLE}_BL${BLOCK_LENGTH}" "${N_PARTICLES_BASE}" "${ALPHA_BASE}" "${TEMPERATURE}"

# # 3. Baseline (temperature=0.25, same temp as SMC — isolates temperature effect)
# run_eval "baseline_N1_alpha1_T025_S${SAMPLE}_BL${BLOCK_LENGTH}" "${N_PARTICLES_BASE}" "${ALPHA_BASE}" "${TEMPERATURE_POWERSMC}"

# # 4. Power-SMC N=4 temp=0.5
# run_eval "equation8_smc_N4_alpha${ALPHA}_T05_S${SAMPLE}_BL${BLOCK_LENGTH}" 4 "${ALPHA}" "${TEMPERATURE}"

# # 5. Power-SMC N=4 (matches Power-SMC paper: temp=0.25, alpha=4.0)
# run_eval "equation8_smc_N4_alpha${ALPHA_POWERSMC}_T025_S${SAMPLE}_BL${BLOCK_LENGTH}" 4 "${ALPHA_POWERSMC}" "${TEMPERATURE_POWERSMC}"

# 6. Power-SMC N=8 T05
run_eval "equation8_smc_N${N_PARTICLES}_alpha${ALPHA}_T05_S${SAMPLE}_BL${BLOCK_LENGTH}" "${N_PARTICLES}" "${ALPHA}" "${TEMPERATURE}"

# 7. Power-SMC N=8 (matches Power-SMC paper: temp=0.25, alpha=4.0)
run_eval "equation8_smc_N${N_PARTICLES}_alpha${ALPHA_POWERSMC}_T025_S${SAMPLE}_BL${BLOCK_LENGTH}" "${N_PARTICLES}" "${ALPHA_POWERSMC}" "${TEMPERATURE_POWERSMC}"

# 6. Power-SMC N=12 T05 (added N=12 to see if more particles helps)
run_eval "equation8_smc_N12_alpha${ALPHA}_T05_S${SAMPLE}_BL${BLOCK_LENGTH}" 12 "${ALPHA}" "${TEMPERATURE}"

# 7. Power-SMC N=12 (matches Power-SMC paper: temp=0.25, alpha=4.0)
run_eval "equation8_smc_N12_alpha${ALPHA_POWERSMC}_T025_S${SAMPLE}_BL${BLOCK_LENGTH}" 12 "${ALPHA_POWERSMC}" "${TEMPERATURE_POWERSMC}"

echo ""
echo "Results written to results/${TASK}/"
