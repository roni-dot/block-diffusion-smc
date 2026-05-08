# ──────────────────────────────────────────────────────────────────────────────
# 8.  Quick sanity-check / demo
# ──────────────────────────────────────────────────────────────────────────────

from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torch

from block_diffusion_power_smc import smc_block_diffusion
from data_classes import BlockDiffusionSMCConfig
from llada.model import LLaDAModelLM


def _demo():
    """
    Minimal smoke-test (requires a GPU and the LLaDA model weights).
    Run with:  python block_diffusion_power_smc.py
    """
    

    model_name = "GSAI-ML/LLaDA-8B-Instruct"

    print(f"Loading {model_name} …")
   
    model = (
        LLaDAModelLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        .eval()
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
    result = smc_block_diffusion(model, input_ids, cfg)

    answer_ids = result["chosen_sequence"][input_ids.shape[1]:]
    eos_id = tokenizer.eos_token_id
    eos_pos = (answer_ids == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_pos):
        answer_ids = answer_ids[:eos_pos[0]]
    print("Answer:", tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True))
    print("Stats:", result["stats"])


if __name__ == "__main__":
    _demo()