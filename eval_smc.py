"""
eval_smc.py

lm-evaluation-harness wrapper for Power-SMC block diffusion.

Registers the model as "smc_block_diffusion" so it can be evaluated with:

    lm_eval --model smc_block_diffusion \
            --model_args model_path=GSAI-ML/LLaDA-8B-Instruct,n_particles=32,alpha=2.0,temperature=0.5,gen_length=256,block_length=32,steps_per_block=32 \
            --tasks gsm8k \
            --num_fewshot 5 \
            --output_path results/

Baseline (standard block diffusion, no SMC):
    --model_args ...,n_particles=1,alpha=1.0,temperature=0.5,...

Adapted from fast_dllm/Fast-dLLM/llada/eval_llada.py (NVIDIA / LLaDA).
Key change: generate_until calls smc_block_diffusion instead of generate().
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer

from block_diffusion_power_smc import smc_block_diffusion
from data_classes import BlockDiffusionSMCConfig
from llada.model import LLaDAModelLM


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model wrapper ─────────────────────────────────────────────────────────────

@register_model("smc_block_diffusion")
class SMCBlockDiffusionHarness(LM):
    """
    lm-eval harness wrapper for Power-SMC block diffusion (LLaDA).

    smc_block_diffusion processes one prompt at a time (input_ids: (1, L)),
    running N particles internally — so there is no outer batch loop here.
    """

    def __init__(
        self,
        model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
        # SMC parameters
        n_particles: int = 32,
        alpha: float = 2.0,
        ess_threshold: float = 0.5,
        temperature: float = 0.5,
        # Block diffusion parameters
        gen_length: int = 256,
        block_length: int = 32,
        steps_per_block: int = 32,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        threshold=None,
        factor=None,
        # Harness config
        device: str = "cuda",
        save_dir: str = None,
        seed: int = 42,
        sample: int = None,   # randomly sample this many examples (None = use all / lm_eval --limit)
        **kwargs,
    ):
        super().__init__()

        set_seed(seed)
        self.sample = int(sample) if sample is not None else None
        self.seed = seed

        print(f"Loading {model_path} …")
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            # Split model evenly so each GPU has headroom for batch activations.
            mem_per_gpu = f"{20 // n_gpus}GiB"
            max_memory = {i: mem_per_gpu for i in range(n_gpus)}
        else:
            max_memory = None
        self.model = (
            LLaDAModelLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
            )
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = torch.device(device)

        self.is_instruct = "instruct" in model_path.lower()

        # Build SMC config once — reused for every example.
        self.cfg = BlockDiffusionSMCConfig(
            alpha=float(alpha),
            n_particles=int(n_particles),
            ess_threshold=float(ess_threshold),
            gen_length=int(gen_length),
            block_length=int(block_length),
            steps_per_block=int(steps_per_block),
            temperature=float(temperature),
            remasking=remasking,
            mask_id=int(mask_id),
            threshold=float(threshold) if threshold is not None else None,
            factor=float(factor) if factor is not None else None,
            verbose=False,  # suppress per-block output during eval
        )

        self.save_dir = save_dir

        print(
            f"\nSMC config: N={self.cfg.n_particles}  α={self.cfg.alpha}  "
            f"temp={self.cfg.temperature}  "
            f"gen={self.cfg.gen_length}tok  block={self.cfg.block_length}tok  "
            f"steps/block={self.cfg.steps_per_block}\n"
        )

    # ── lm-eval required properties ───────────────────────────────────────────

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    # ── Not used for generation tasks ─────────────────────────────────────────

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for SMC harness.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    # ── Main generation loop ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:
        # Random subsample: shuffle with fixed seed, pick first `sample` indices,
        # process only those, return empty string for the rest.
        # lm_eval computes accuracy only over the returned non-empty outputs that
        # it sent us, so we must return one string per request in original order.
        rng = random.Random(self.seed)
        all_indices = list(range(len(requests)))
        rng.shuffle(all_indices)
        if self.sample is not None and self.sample < len(requests):
            active_indices = set(all_indices[: self.sample])
            print(f"Random subsample: processing {self.sample} / {len(requests)} examples (seed={self.seed})")
        else:
            active_indices = set(all_indices)

        output = [""] * len(requests)   # pre-fill; only active slots get real answers
        processed_count = 0
        save_path = None

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, "predictions.jsonl")
            if os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8") as f:
                    saved = [json.loads(line) for line in f]
                    processed_count = len(saved)
                    # Restore already-generated answers into the output list.
                    # Saved entries are in the order they were processed (active_indices order).
                    active_list = sorted(active_indices)
                    for idx, entry in zip(active_list[:processed_count], saved):
                        output[idx] = entry["answer"]
                print(f"Resuming from {processed_count} saved predictions.")

        total_resample_events = 0
        total_time = 0.0
        n_done = 0

        for i, req in enumerate(tqdm(requests, desc="SMC generation")):
            if i not in active_indices:
                continue  # not in random sample — leave output[i] as ""

            if output[i] != "":
                continue  # already restored from save file

            question: str = req.args[0]
            stop_tokens: list = req.args[1].get("until", [])

            # ── Encode prompt ──────────────────────────────────────────────
            if self.is_instruct:
                messages = [{"role": "user", "content": question}]
                prompt_str = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                prompt_str = question

            input_ids = torch.tensor(
                self.tokenizer(prompt_str)["input_ids"],
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)  # (1, prompt_len)

            # ── Run SMC ────────────────────────────────────────────────────
            t0 = time.time()
            result = smc_block_diffusion(self.model, input_ids, self.cfg)
            elapsed = time.time() - t0
            total_time += elapsed
            n_done += 1

            stats = result["stats"]
            total_resample_events += stats["resample_count"]

            # ── Decode answer ──────────────────────────────────────────────
            answer_ids = result["chosen_sequence"][input_ids.shape[1]:]

            eos_id = self.tokenizer.eos_token_id
            eos_positions = (answer_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions):
                answer_ids = answer_ids[: eos_positions[0]]

            answer_text = self.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True)

            for stop_seq in stop_tokens:
                if stop_seq in answer_text:
                    answer_text = answer_text.split(stop_seq)[0]

            output[i] = answer_text

            # ── Per-example log ────────────────────────────────────────────
            print(
                f"\n[{n_done}/{len(active_indices)}] idx={i}  "
                f"particle={result['chosen_idx']}  "
                f"weight={result['w'][result['chosen_idx']]:.3f}  "
                f"resamples={stats['resample_count']}  "
                f"time={elapsed:.1f}s"
            )
            print(f"  answer: {answer_text[:400]}")
            print(f"  ESS history:  {[f'{v:.1f}' for v in stats['ess_history']]}")

            # ── Incremental save ───────────────────────────────────────────
            if save_path is not None:
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "idx": i,
                                "answer": answer_text,
                                "chosen_idx": result["chosen_idx"],
                                "chosen_weight": float(result["w"][result["chosen_idx"]]),
                                "resample_count": stats["resample_count"],
                                "ess_history": stats["ess_history"],
                                "time_s": round(elapsed, 2),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        if n_done > 0:
            print(
                f"\n── Generation complete ──\n"
                f"  Examples processed: {n_done}\n"
                f"  Total resamplings:  {total_resample_events}\n"
                f"  Avg time / example: {total_time / n_done:.1f}s\n"
            )

        return output


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli_evaluate()
