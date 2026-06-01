"""
LLM Fine-tuning with Ray Train + Ray Tune on AKS
=================================================

This script demonstrates how to fine-tune a Large Language Model (LLM) using:
- Ray Train: For distributed training across multiple GPUs
- Ray Tune: For hyperparameter optimization
- PEFT/LoRA: For parameter-efficient fine-tuning
- QLoRA: For memory-efficient 4-bit quantization

The script uses the Alpaca dataset format and supports various base models
from Hugging Face.

Usage:
    python finetune_with_tune.py [--model MODEL_NAME] [--num-samples N]

Reference:
    - Ray Train + Transformers: https://docs.ray.io/en/latest/train/getting-started-transformers.html
    - Ray Tune: https://docs.ray.io/en/latest/tune/index.html
"""

import os
import argparse
import tempfile
from typing import Dict, Any

import ray
from ray import tune, train
from ray.tune import RunConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import Checkpoint

# Workaround for Ray 2.52.x bug: get_air_verbosity() crashes on `verbose.value`
# when the verbosity arrives as a plain str (default code path inside Tuner).
# https://github.com/ray-project/ray/issues/49454-adjacent
import ray.tune.experimental.output as _ray_output  # noqa: E402
_orig_get_air_verbosity = _ray_output.get_air_verbosity
def _patched_get_air_verbosity(verbose):  # noqa: D401
    if isinstance(verbose, str):
        try:
            return _ray_output.AirVerbosity(int(verbose))
        except (ValueError, KeyError):
            return _ray_output.AirVerbosity.DEFAULT
    if verbose is None:
        return _ray_output.AirVerbosity.DEFAULT
    return _orig_get_air_verbosity(verbose)
_ray_output.get_air_verbosity = _patched_get_air_verbosity

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# =============================================================================
# Configuration
# =============================================================================

# Default model and dataset configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-3B"  # Smaller model for faster experimentation
DEFAULT_DATASET = "tatsu-lab/alpaca"
MAX_SEQ_LENGTH = 512

# Hyperparameter search space for Ray Tune
TUNE_CONFIG = {
    "lora_r": tune.choice([8, 16, 32, 64]),
    "lora_alpha": tune.choice([16, 32, 64, 128]),
    "lora_dropout": tune.uniform(0.0, 0.1),
    "learning_rate": tune.loguniform(1e-5, 5e-4),
    "per_device_train_batch_size": tune.choice([1, 2, 4]),
    "gradient_accumulation_steps": tune.choice([4, 8, 16]),
    "num_train_epochs": tune.choice([1, 2, 3]),
    "warmup_ratio": tune.uniform(0.01, 0.1),
    "weight_decay": tune.uniform(0.0, 0.1),
}


def format_alpaca_prompt(example: Dict[str, str]) -> str:
    """Format data in Alpaca instruction format."""
    if example.get("input", ""):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""


def prepare_dataset(tokenizer, max_length: int = MAX_SEQ_LENGTH):
    """Load and preprocess the Alpaca dataset."""
    dataset = load_dataset(DEFAULT_DATASET, split="train")
    
    # Take a subset for faster training (adjust for full training)
    dataset = dataset.shuffle(seed=42).select(range(min(10000, len(dataset))))
    
    def preprocess_function(examples):
        texts = [format_alpaca_prompt({"instruction": i, "input": inp, "output": o}) 
                 for i, inp, o in zip(examples["instruction"], 
                                       examples["input"], 
                                       examples["output"])]
        return {"text": texts}
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return dataset


def train_func(config: Dict[str, Any]):
    """
    Tune trainable: fine-tune one trial with the supplied hyperparameters.

    Runs as a single-process trainable; Ray Tune allocates the GPUs via the
    `resources_per_trial` argument on the Tuner. Metrics and a LoRA-adapter
    checkpoint are reported back to Tune through `ray.train.report`.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    try:
        from trl import SFTConfig as _SFTConfig
    except ImportError:
        _SFTConfig = None

    model_name = config.get("model_name", DEFAULT_MODEL)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = prepare_dataset(tokenizer)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    trial_output_dir = tempfile.mkdtemp(prefix="hf_trainer_")

    _args_cls = _SFTConfig if _SFTConfig is not None else TrainingArguments
    _args_kwargs = dict(
        output_dir=trial_output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    import inspect as _inspect_args
    _args_params = _inspect_args.signature(_args_cls.__init__).parameters
    if _SFTConfig is not None:
        for _k, _v in (("dataset_text_field", "text"), ("max_seq_length", MAX_SEQ_LENGTH), ("packing", False)):
            if _k in _args_params:
                _args_kwargs[_k] = _v
    training_args = _args_cls(**_args_kwargs)

    class TuneReportCallback(TrainerCallback):
        """Forward HF Trainer eval metrics to Ray Tune."""

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return
            payload = {
                "eval_loss": float(metrics.get("eval_loss", float("nan"))),
                "epoch": float(metrics.get("epoch", state.epoch or 0.0)),
                "step": int(state.global_step),
            }
            tune.report(payload)

    sft_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[TuneReportCallback()],
    )
    import inspect as _inspect
    _sft_params = _inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in _sft_params:
        sft_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in _sft_params:
        sft_kwargs["tokenizer"] = tokenizer
    # Legacy trl: these were SFTTrainer kwargs; modern trl: they live on SFTConfig.
    for _k, _v in (
        ("dataset_text_field", "text"),
        ("max_seq_length", MAX_SEQ_LENGTH),
        ("packing", False),
    ):
        if _k in _sft_params and _SFTConfig is None:
            sft_kwargs[_k] = _v
    trainer = SFTTrainer(**sft_kwargs)

    trainer.train()

    # Final eval + checkpoint with adapter weights.
    final_metrics = trainer.evaluate()
    with tempfile.TemporaryDirectory() as ckpt_dir:
        trainer.save_model(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        # Inline-push: when INLINE_PUSH_REPO_ID is set, publish this trial's
        # adapter directly to the HF Hub from inside the worker, before the
        # tempdir is destroyed. This bypasses Ray Tune's checkpoint persistence
        # (which silently drops files when storage_path is node-local on a
        # multi-node cluster).
        inline_repo = os.environ.get("INLINE_PUSH_REPO_ID")
        if inline_repo:
            try:
                _inline_push_to_hub(
                    adapter_dir=ckpt_dir,
                    base_model_name=model_name,
                    repo_id=inline_repo,
                    private=os.environ.get("INLINE_PUSH_PRIVATE", "1") == "1",
                    merge_adapter=os.environ.get("INLINE_PUSH_MERGE", "0") == "1",
                )
            except Exception as e:
                print(f"[inline-push] FAILED: {e}")

        tune.report(
            {
                "eval_loss": float(final_metrics.get("eval_loss", float("nan"))),
                "epoch": float(final_metrics.get("epoch", 0.0)),
                "final": True,
            },
            checkpoint=Checkpoint.from_directory(ckpt_dir),
        )


def _inline_push_to_hub(adapter_dir, base_model_name, repo_id, private, merge_adapter):
    """Push LoRA adapter (or merged model) to HF Hub from inside a Tune trial."""
    import torch
    from huggingface_hub import HfApi, create_repo
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HUGGING_FACE_HUB_TOKEN env var required for inline push.")

    print(f"[inline-push] repo={repo_id} private={private} merge={merge_adapter}")
    create_repo(repo_id, token=token, private=private, exist_ok=True)
    api = HfApi(token=token)

    if not merge_adapter:
        api.upload_folder(folder_path=adapter_dir, repo_id=repo_id,
                          commit_message="Inline push adapter from Ray Tune trial")
    else:
        print(f"[inline-push] loading base {base_model_name} in bf16 for merge...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        merged.push_to_hub(repo_id, token=token,
                           commit_message="Inline push merged model from Ray Tune trial")
        tok.push_to_hub(repo_id, token=token,
                        commit_message="Inline push tokenizer from Ray Tune trial")
    print(f"[inline-push] done: https://huggingface.co/{repo_id}")


def _make_run_config(name: str) -> RunConfig:
    return RunConfig(
        name=name,
        storage_path="/tmp/ray_results",
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    )


def _resources_per_trial(gpus_per_worker: int) -> Dict[str, float]:
    return {"CPU": 8, "GPU": float(gpus_per_worker)}


def run_hyperparameter_tuning(
    model_name: str = DEFAULT_MODEL,
    num_samples: int = 10,
    num_workers: int = 1,
    use_gpu: bool = True,
    gpus_per_worker: int = 4,
):
    """Run hyperparameter tuning with Ray Tune (Train V2 compatible)."""
    if not ray.is_initialized():
        ray.init()

    print(f"Ray initialized with {ray.available_resources()}")

    tune_config = TUNE_CONFIG.copy()
    tune_config["model_name"] = model_name

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=20,
        grace_period=2,
        reduction_factor=3,
        brackets=1,
    )
    search_alg = OptunaSearch(metric="eval_loss", mode="min")

    trainable = tune.with_resources(
        train_func, resources=_resources_per_trial(gpus_per_worker if use_gpu else 0)
    )

    tuner = tune.Tuner(
        trainable,
        param_space=tune_config,
        tune_config=tune.TuneConfig(
            metric="eval_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=_make_run_config("llm-finetune-tune"),
    )

    print("=" * 60)
    print("Starting hyperparameter tuning...")
    print(f"Model: {model_name}")
    print(f"Number of trials: {num_samples}")
    print(f"GPUs per trial: {gpus_per_worker}")
    print("=" * 60)

    results = tuner.fit()
    best_result = results.get_best_result(metric="eval_loss", mode="min")

    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Complete!")
    print("=" * 60)
    print("\nBest trial config:")
    for key, value in best_result.config.items():
        if key != "model_name":
            print(f"  {key}: {value}")
    print(f"\nBest trial final eval_loss: {best_result.metrics.get('eval_loss', 'N/A')}")
    print(f"Best checkpoint path: {best_result.checkpoint}")

    return best_result


def run_single_training(
    model_name: str = DEFAULT_MODEL,
    num_workers: int = 1,
    use_gpu: bool = True,
    gpus_per_worker: int = 4,
):
    """Run a single fine-tuning trial via Tune with fixed hyperparameters."""
    if not ray.is_initialized():
        ray.init()

    print(f"Ray initialized with {ray.available_resources()}")

    default_config = {
        "model_name": model_name,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 1,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
    }

    trainable = tune.with_resources(
        train_func, resources=_resources_per_trial(gpus_per_worker if use_gpu else 0)
    )

    tuner = tune.Tuner(
        trainable,
        param_space=default_config,
        tune_config=tune.TuneConfig(metric="eval_loss", mode="min", num_samples=1),
        run_config=_make_run_config("llm-finetune-single"),
    )

    print("=" * 60)
    print("Starting single training run...")
    print(f"Model: {model_name}")
    print(f"GPUs per trial: {gpus_per_worker}")
    print("=" * 60)

    results = tuner.fit()
    result = results[0]

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final eval_loss: {result.metrics.get('eval_loss', 'N/A')}")
    print(f"Checkpoint path: {result.checkpoint}")

    return result


def push_checkpoint_to_hub(
    checkpoint,
    base_model_name: str,
    repo_id: str,
    private: bool = True,
    merge_adapter: bool = False,
    commit_message: str = "Upload fine-tuned model from Ray Train + Tune",
):
    """
    Publish a Ray Train checkpoint (containing a PEFT/LoRA adapter) to the
    Hugging Face Hub.

    Args:
        checkpoint: ray.train.Checkpoint returned from training/tuning.
        base_model_name: HF id of the base model the adapter was trained on.
        repo_id: Target HF repo, e.g. "your-user/qwen2.5-3b-alpaca-lora".
        private: Create the repo as private.
        merge_adapter: If True, merge LoRA weights into the base model and
            push a full standalone model. Requires enough memory to load the
            base model in fp16/bf16 (no 4-bit). If False, push only the
            lightweight adapter.
        commit_message: Commit message for the upload.
    """
    import os
    import glob
    import torch
    from huggingface_hub import HfApi, create_repo
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) env var is required to push to the Hub."
        )

    # Materialize the checkpoint to a local directory.
    with checkpoint.as_directory() as ckpt_dir:
        # SFTTrainer/HF saves a checkpoint-* subdir; pick the latest if present.
        adapter_dir = ckpt_dir
        candidates = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint-*")))
        if candidates:
            adapter_dir = candidates[-1]

        print(f"Publishing checkpoint from: {adapter_dir}")
        print(f"Target repo: {repo_id} (private={private}, merge={merge_adapter})")

        create_repo(repo_id, token=token, private=private, exist_ok=True)
        api = HfApi(token=token)

        if not merge_adapter:
            # Push just the LoRA adapter directory (small, fast).
            api.upload_folder(
                folder_path=adapter_dir,
                repo_id=repo_id,
                commit_message=commit_message,
            )
        else:
            # Load base model in bf16, merge adapter weights, push full model.
            print("Loading base model for adapter merge...")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True
            )

            merged.push_to_hub(repo_id, token=token, commit_message=commit_message)
            tokenizer.push_to_hub(repo_id, token=token, commit_message=commit_message)

    print(f"Successfully published to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM with Ray Train + Tune on AKS"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face model to fine-tune (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of hyperparameter combinations to try (default: 10)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Ray Train workers per trial (default: 1)",
    )
    parser.add_argument(
        "--gpus-per-worker",
        type=int,
        default=4,
        help="Number of GPUs per worker (default: 4)",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run a single training instead of hyperparameter tuning",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the best checkpoint to the Hugging Face Hub after training.",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="Target HF repo id, e.g. 'your-user/qwen2.5-3b-alpaca-lora'.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create the HF repo as private.",
    )
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge LoRA adapter into the base model and push a full model "
             "(otherwise pushes only the adapter).",
    )
    
    args = parser.parse_args()
    
    if args.single_run:
        result = run_single_training(
            model_name=args.model,
            num_workers=args.num_workers,
            gpus_per_worker=args.gpus_per_worker,
        )
        best_checkpoint = result.checkpoint if result is not None else None
    else:
        best_result = run_hyperparameter_tuning(
            model_name=args.model,
            num_samples=args.num_samples,
            num_workers=args.num_workers,
            gpus_per_worker=args.gpus_per_worker,
        )
        best_checkpoint = best_result.checkpoint if best_result is not None else None

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise SystemExit("--hub-repo-id is required when --push-to-hub is set.")
        if best_checkpoint is None:
            raise SystemExit("No checkpoint available to publish.")
        push_checkpoint_to_hub(
            checkpoint=best_checkpoint,
            base_model_name=args.model,
            repo_id=args.hub_repo_id,
            private=args.hub_private,
            merge_adapter=args.merge_adapter,
        )


if __name__ == "__main__":
    main()
