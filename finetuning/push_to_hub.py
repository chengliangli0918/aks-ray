"""
Standalone tool to merge a LoRA adapter into a base model and push the merged
model to the Hugging Face Hub.

Usage:
    python push_to_hub.py \
        --adapter-dir /tmp/ray_results/llm-finetune-tune/<best_trial>/checkpoint_000000 \
        --base-model Qwen/Qwen2.5-3B \
        --repo-id user/repo-name \
        --private

Reads HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) from environment.
"""

import argparse
import glob
import os
import sys

import torch
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def find_adapter_dir(path: str) -> str:
    """Accept either an adapter dir directly or a Ray checkpoint dir containing checkpoint-*."""
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    candidates = sorted(glob.glob(os.path.join(path, "checkpoint-*")))
    if candidates:
        return candidates[-1]
    # Some Ray checkpoints nest under an extra dir.
    nested = sorted(glob.glob(os.path.join(path, "*", "adapter_config.json")))
    if nested:
        return os.path.dirname(nested[-1])
    raise FileNotFoundError(f"No adapter_config.json found under {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True, help="Local path to LoRA adapter (or Ray checkpoint dir).")
    parser.add_argument("--base-model", required=True, help="HF id of the base model.")
    parser.add_argument("--repo-id", required=True, help="Target HF repo (user/name).")
    parser.add_argument("--private", action="store_true", help="Create repo as private.")
    parser.add_argument("--adapter-only", action="store_true", help="Upload adapter only (skip merge).")
    parser.add_argument("--commit-message", default="Upload fine-tuned model from Ray Train + Tune")
    args = parser.parse_args()

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) must be set.")

    adapter_dir = find_adapter_dir(args.adapter_dir)
    print(f"Adapter directory: {adapter_dir}")
    print(f"Target repo: {args.repo_id} (private={args.private}, adapter_only={args.adapter_only})")

    create_repo(args.repo_id, token=token, private=args.private, exist_ok=True)
    api = HfApi(token=token)

    if args.adapter_only:
        api.upload_folder(
            folder_path=adapter_dir,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
        )
    else:
        print(f"Loading base model {args.base_model} in bf16...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("Merging LoRA adapter...")
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

        print("Pushing merged model + tokenizer to Hub...")
        merged.push_to_hub(args.repo_id, token=token, commit_message=args.commit_message)
        tokenizer.push_to_hub(args.repo_id, token=token, commit_message=args.commit_message)

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
