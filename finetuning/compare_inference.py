"""
Compare inference quality of the base model vs. the fine-tuned (LoRA) model.

For each prompt from a held-out slice of the AKS combined dataset (or a custom
prompt file), this script generates a response from:
  1. The base Hugging Face model.
  2. The base model + LoRA adapter produced by `finetune_with_tune.py`
     (either a local checkpoint dir or a HF Hub repo id).

It then reports per-prompt and aggregate metrics:
  - Generation latency (seconds) and tokens/sec
  - Output token count
  - Perplexity of the reference answer under each model
  - ROUGE-L F1 between generated output and reference answer

Usage:
    python compare_inference.py \\
        --base-model Qwen/Qwen2.5-3B \\
        --adapter /tmp/ray_results/llm-finetune-tune/<trial>/checkpoint_xxx \\
        --num-samples 20 \\
        --output-json comparison.json

    # Or pull the adapter from the Hub:
    python compare_inference.py \\
        --base-model Qwen/Qwen2.5-3B \\
        --adapter chengliangli/qwen2.5-3b-aks-tsg \\
        --num-samples 20

By default the held-out prompts are read from
``$DATASET_DIR/aks_combined.val.jsonl`` (DATASET_DIR defaults to
``/home/ray/dataset``). Override with ``--eval-file`` to point at any
Alpaca-style JSONL file with ``instruction``/``input``/``output`` fields.
"""

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEFAULT_DATASET_DIR = os.environ.get("DATASET_DIR", "/home/ray/dataset")
DEFAULT_VAL_FILE = "aks_combined.val.jsonl"


def format_prompt(instruction: str, input_text: str = "") -> str:
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def load_eval_samples(
    num_samples: int,
    seed: int = 123,
    eval_file: str = None,
) -> List[Dict[str, str]]:
    """Load held-out evaluation prompts from an Alpaca-style JSONL file.

    Defaults to ``$DATASET_DIR/aks_combined.val.jsonl``. The val file is
    already a held-out split, so we just shuffle and take ``num_samples``.
    """
    if eval_file is None:
        eval_file = os.path.join(DEFAULT_DATASET_DIR, DEFAULT_VAL_FILE)
    if not os.path.isfile(eval_file):
        raise FileNotFoundError(
            f"Evaluation JSONL not found: {eval_file}. "
            "Pass --eval-file or set DATASET_DIR."
        )
    ds = load_dataset("json", data_files=eval_file, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    return [
        {
            "instruction": ex["instruction"],
            "input": ex.get("input") or "",
            "reference": ex["output"],
        }
        for ex in ds
    ]


def load_prompts_from_file(path: str) -> List[Dict[str, str]]:
    with open(path, "r") as f:
        data = json.load(f)
    # Accept either a list of strings or a list of {instruction,input,reference}.
    out = []
    for item in data:
        if isinstance(item, str):
            out.append({"instruction": item, "input": "", "reference": ""})
        else:
            out.append(
                {
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "reference": item.get("reference", item.get("output", "")),
                }
            )
    return out


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - start
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    n = int(new_tokens.shape[0])
    return {
        "text": text,
        "latency_s": elapsed,
        "new_tokens": n,
        "tokens_per_s": (n / elapsed) if elapsed > 0 else 0.0,
    }


@torch.no_grad()
def perplexity(model, tokenizer, prompt: str, reference: str) -> float:
    """Perplexity of `reference` conditioned on `prompt`."""
    if not reference:
        return float("nan")
    full = prompt + reference
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100  # only score the reference tokens
    out = model(input_ids=full_ids, labels=labels)
    return float(math.exp(out.loss.item()))


def rouge_l_f1(pred: str, ref: str) -> float:
    """Simple ROUGE-L F1 on whitespace tokens (no external deps)."""
    if not pred or not ref:
        return 0.0
    a = pred.split()
    b = ref.split()
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


def _is_peft_adapter(path_or_repo: str) -> bool:
    """Detect whether `path_or_repo` is a PEFT adapter (vs a full model)."""
    import os
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, "adapter_config.json"))
    # HF Hub repo: probe for adapter_config.json.
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=path_or_repo, filename="adapter_config.json")
        return True
    except Exception:
        return False


def load_models(base_model: str, adapter: str, dtype: torch.dtype):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    base.eval()

    is_adapter = _is_peft_adapter(adapter)
    if is_adapter:
        print(f"Loading LoRA adapter: {adapter}")
        base_for_ft = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        finetuned = PeftModel.from_pretrained(base_for_ft, adapter)
    else:
        print(f"Loading full fine-tuned model: {adapter}")
        finetuned = AutoModelForCausalLM.from_pretrained(
            adapter,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    finetuned.eval()

    return tokenizer, base, finetuned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="HF id of the base model.")
    ap.add_argument(
        "--adapter",
        required=True,
        help="Path or HF repo id of the LoRA adapter to compare against.",
    )
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--eval-file",
        type=str,
        default=None,
        help="Path to an Alpaca-style JSONL file with held-out prompts "
             "(default: $DATASET_DIR/aks_combined.val.jsonl).",
    )
    ap.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional JSON file of prompts (overrides --eval-file).",
    )
    ap.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the full comparison report.",
    )
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]

    samples = (
        load_prompts_from_file(args.prompts_file)
        if args.prompts_file
        else load_eval_samples(args.num_samples, eval_file=args.eval_file)
    )

    tokenizer, base_model, ft_model = load_models(args.base_model, args.adapter, dtype)

    results = []
    agg = {
        "base": {"latency": 0.0, "tokens": 0, "ppl": [], "rouge": []},
        "finetuned": {"latency": 0.0, "tokens": 0, "ppl": [], "rouge": []},
    }

    for i, ex in enumerate(samples, 1):
        prompt = format_prompt(ex["instruction"], ex["input"])
        ref = ex["reference"]

        base_gen = generate(base_model, tokenizer, prompt, args.max_new_tokens)
        ft_gen = generate(ft_model, tokenizer, prompt, args.max_new_tokens)

        base_ppl = perplexity(base_model, tokenizer, prompt, ref)
        ft_ppl = perplexity(ft_model, tokenizer, prompt, ref)

        base_rouge = rouge_l_f1(base_gen["text"], ref)
        ft_rouge = rouge_l_f1(ft_gen["text"], ref)

        agg["base"]["latency"] += base_gen["latency_s"]
        agg["base"]["tokens"] += base_gen["new_tokens"]
        agg["finetuned"]["latency"] += ft_gen["latency_s"]
        agg["finetuned"]["tokens"] += ft_gen["new_tokens"]
        if not math.isnan(base_ppl):
            agg["base"]["ppl"].append(base_ppl)
            agg["finetuned"]["ppl"].append(ft_ppl)
        agg["base"]["rouge"].append(base_rouge)
        agg["finetuned"]["rouge"].append(ft_rouge)

        print("=" * 72)
        print(f"[{i}/{len(samples)}] Instruction: {ex['instruction'][:120]}")
        if ex["input"]:
            print(f"Input: {ex['input'][:120]}")
        print(f"-- BASE      (ppl={base_ppl:.2f}, rougeL={base_rouge:.3f}, "
              f"{base_gen['tokens_per_s']:.1f} tok/s):\n{base_gen['text']}")
        print(f"-- FINETUNED (ppl={ft_ppl:.2f}, rougeL={ft_rouge:.3f}, "
              f"{ft_gen['tokens_per_s']:.1f} tok/s):\n{ft_gen['text']}")
        if ref:
            print(f"-- REFERENCE:\n{ref}")

        results.append(
            {
                "instruction": ex["instruction"],
                "input": ex["input"],
                "reference": ref,
                "base": {**base_gen, "perplexity": base_ppl, "rouge_l": base_rouge},
                "finetuned": {**ft_gen, "perplexity": ft_ppl, "rouge_l": ft_rouge},
            }
        )

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "num_samples": len(samples),
        "base": {
            "avg_latency_s": agg["base"]["latency"] / len(samples),
            "avg_tokens_per_s": agg["base"]["tokens"] / agg["base"]["latency"]
            if agg["base"]["latency"] > 0 else 0.0,
            "avg_perplexity": mean(agg["base"]["ppl"]),
            "avg_rouge_l": mean(agg["base"]["rouge"]),
        },
        "finetuned": {
            "avg_latency_s": agg["finetuned"]["latency"] / len(samples),
            "avg_tokens_per_s": agg["finetuned"]["tokens"] / agg["finetuned"]["latency"]
            if agg["finetuned"]["latency"] > 0 else 0.0,
            "avg_perplexity": mean(agg["finetuned"]["ppl"]),
            "avg_rouge_l": mean(agg["finetuned"]["rouge"]),
        },
    }
    summary["delta"] = {
        "perplexity_drop": summary["base"]["avg_perplexity"]
        - summary["finetuned"]["avg_perplexity"],
        "rouge_l_gain": summary["finetuned"]["avg_rouge_l"]
        - summary["base"]["avg_rouge_l"],
    }

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"\nWrote full report to {args.output_json}")


if __name__ == "__main__":
    main()
