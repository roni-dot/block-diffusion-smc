"""
compute_accuracy.py

Compute GSM8K accuracy from saved predictions.jsonl files.
Replicates lm_eval's flexible-extract and strict-match metrics.

Usage:
    python compute_accuracy.py results/gsm8k/baseline_N1_alpha1/predictions/predictions.jsonl
    python compute_accuracy.py results/gsm8k/smc_N8_alpha2.0/predictions/predictions.jsonl

    # Compare two runs:
    python compute_accuracy.py \
        results/gsm8k/baseline_N1_alpha1/predictions/predictions.jsonl \
        results/gsm8k/smc_N8_alpha2.0/predictions/predictions.jsonl
"""

import json
import re
import sys
from datasets import load_dataset


def extract_strict(text: str) -> str | None:
    """Extract answer after '#### ' (GSM8K chain-of-thought format)."""
    m = re.search(r"#### (-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    return None


def extract_flexible(text: str) -> str | None:
    """Extract the last number in text (lm_eval flexible-extract)."""
    matches = re.findall(r"-?[\$0-9][0-9,\.]*|[-]?[0-9]+", text)
    if matches:
        return matches[-1].replace(",", "").replace("$", "").rstrip(".")
    return None


def normalize(s: str) -> str:
    return s.replace(",", "").replace("$", "").rstrip(".").strip()


def evaluate(jsonl_path: str, dataset) -> dict:
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    # Deduplicate by idx (keep last occurrence in case of resume duplicates)
    by_idx = {}
    for e in entries:
        by_idx[e["idx"]] = e

    strict_correct = 0
    flexible_correct = 0
    n = len(by_idx)

    for idx, entry in by_idx.items():
        gold_full = dataset[idx]["answer"]   # e.g. "... #### 42"
        gold = extract_strict(gold_full)
        if gold is None:
            continue
        gold = normalize(gold)

        pred = entry["answer"]
        pred_strict = extract_strict(pred)
        pred_flex = extract_flexible(pred)

        if pred_strict is not None and normalize(pred_strict) == gold:
            strict_correct += 1
        if pred_flex is not None and normalize(pred_flex) == gold:
            flexible_correct += 1

    avg_resample = sum(e.get("resample_count", 0) for e in by_idx.values()) / n if n else 0
    avg_time = sum(e.get("time_s", 0) for e in by_idx.values()) / n if n else 0

    return {
        "n": n,
        "strict_match": strict_correct / n if n else 0,
        "flexible_extract": flexible_correct / n if n else 0,
        "avg_resamples_per_example": round(avg_resample, 3),
        "avg_time_s": round(avg_time, 1),
    }


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python compute_accuracy.py <predictions.jsonl> [<predictions2.jsonl> ...]")
        sys.exit(1)

    print("Loading GSM8K test split …")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    for path in paths:
        print(f"\n{'='*60}")
        print(f"  {path}")
        print(f"{'='*60}")
        r = evaluate(path, dataset)
        print(f"  Examples evaluated:       {r['n']}")
        print(f"  Strict-match accuracy:    {r['strict_match']:.1%}")
        print(f"  Flexible-extract accuracy:{r['flexible_extract']:.1%}")
        print(f"  Avg resamples/example:    {r['avg_resamples_per_example']}")
        print(f"  Avg time/example:         {r['avg_time_s']}s")

    if len(paths) == 2:
        print(f"\n{'='*60}")
        print("  COMPARISON")
        print(f"{'='*60}")
        r0 = evaluate(paths[0], dataset)
        r1 = evaluate(paths[1], dataset)
        delta_strict = r1["strict_match"] - r0["strict_match"]
        delta_flex = r1["flexible_extract"] - r0["flexible_extract"]
        print(f"  Strict-match:     {r0['strict_match']:.1%} -> {r1['strict_match']:.1%}  (delta {delta_strict:+.1%})")
        print(f"  Flexible-extract: {r0['flexible_extract']:.1%} -> {r1['flexible_extract']:.1%}  (delta {delta_flex:+.1%})")


if __name__ == "__main__":
    main()
