# LLM Fine-tuning with Ray Train + Ray Tune on AKS Automatic

This sample demonstrates how to fine-tune Large Language Models (LLMs) on AKS Automatic using Ray Train for distributed training and Ray Tune for hyperparameter optimization. The example fine-tunes `Qwen/Qwen2.5-3B` on a local AKS-domain instruction dataset built from AKS public documentation, the AKS troubleshooting guide (TSG).

## Overview

The sample provides:
- **Distributed Training**: Scale fine-tuning across multiple A100 GPUs using Ray Train
- **Hyperparameter Tuning**: Automatically find optimal hyperparameters using Ray Tune
- **Memory Efficiency**: QLoRA (4-bit quantization) + LoRA for efficient fine-tuning
- **AKS Automatic**: Simplified cluster management with automatic GPU node provisioning

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AKS Automatic Cluster                        │
├─────────────────────────────────────────────────────────────────┤
│                      GPU Workload Pool                           │
│                    (A100 GPU Nodes)                              │
│                  Label: accelerator=nvidia                       │
│                                                                  │
│  ┌────────────────────┐    ┌───────────────────────────────┐    │
│  │ Ray Head Pod       │    │ Ray Worker Pod (GPU)          │    │
│  │ - Dashboard (8265) │◄───│ - Training Workers            │    │
│  │ - GCS (6379)       │    │ - QLoRA Fine-tuning           │    │
│  │ - Tune Controller  │    │ - 4x A100 GPUs                │    │
│  │ - 8 CPU, 32Gi RAM  │    │ - 48 CPU, 192Gi RAM           │    │
│  └────────────────────┘    └───────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Ray Tune                                                │    │
│  │  - ASHA Scheduler (Early stopping)                      │    │
│  │  - Optuna Search Algorithm                              │    │
│  │  - Parallel trial execution                             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Ray Train
- Distributed data-parallel training
- Automatic gradient synchronization
- Checkpoint management
- Fault tolerance with automatic recovery

### Ray Tune
- **ASHA Scheduler**: Early stops poorly performing trials
- **Optuna Search**: Smart hyperparameter sampling
- **Parallel Execution**: Run multiple trials concurrently

### QLoRA Fine-tuning
- 4-bit quantization for memory efficiency
- LoRA adapters for parameter-efficient training
- Supports models up to 70B parameters on A100 GPUs

## Hyperparameters Tuned

| Parameter | Search Space | Description |
|-----------|-------------|-------------|
| `lora_r` | [8, 16, 32, 64] | LoRA rank |
| `lora_alpha` | [16, 32, 64, 128] | LoRA scaling factor |
| `lora_dropout` | [0.0, 0.1] | LoRA dropout rate |
| `learning_rate` | [1e-5, 5e-4] | Learning rate (log scale) |
| `batch_size` | [1, 2, 4] | Per-device batch size |
| `gradient_accumulation` | [4, 8, 16] | Gradient accumulation steps |
| `epochs` | [1, 2, 3] | Number of training epochs |
| `warmup_ratio` | [0.01, 0.1] | Warmup ratio |
| `weight_decay` | [0.0, 0.1] | Weight decay |

## Prerequisites

1. Azure subscription with GPU quota for A100 VMs
2. Azure CLI, kubectl, Helm, and kubelogin installed
3. [Hugging Face](https://huggingface.co/) account and API token
4. **Azure Kubernetes Service RBAC Cluster Admin** role
5. A local copy of the AKS combined dataset (Alpaca-style JSONL files):
   - `aks_combined.train.jsonl` (~12k records)
   - `aks_combined.val.jsonl` (~1.4k records)

   Each record has `instruction` / `input` / `output` fields built from AKS
   public docs, the AKS TSG, and IcM incident summaries. Point `DATASET_SRC_DIR`
   at the directory that contains these two files before running the setup
   script (default: `/home/charlili/go/src/github.com/chengliangli0918/aks-tsg/dataset`).

## Quick Start

### 1. Configure the Setup Script

Edit `setup-aks-raytune-finetune.sh`:

```bash
SUBSCRIPTION_ID="your-subscription-id"
RESOURCE_GROUP="your-resource-group"
CLUSTER_NAME="your-cluster-name"
LOCATION="your-region"  # e.g., italynorth, eastus2
HF_TOKEN="your-huggingface-token"
DATASET_SRC_DIR="/abs/path/to/aks-combined-dataset"  # contains the two JSONL files
```

### 2. Run the Setup Script

```bash
cd finetuning
chmod +x setup-aks-raytune-finetune.sh
./setup-aks-raytune-finetune.sh
```

This will:
1. Create an AKS Automatic cluster with GPU nodes
2. Install the KubeRay operator
3. Create a `hf-token` Secret and a `finetune-script` ConfigMap
4. Provision an `aks-dataset` Azure Files PVC (ReadWriteMany) and upload
   `aks_combined.train.jsonl` / `aks_combined.val.jsonl` into it via
   `kubectl cp`
5. Deploy a Ray cluster for fine-tuning (with the PVC mounted at
   `/home/ray/dataset` on the head and worker pods)

### 3. Submit a Fine-tuning Job

```bash
# Submit the Ray Job for hyperparameter tuning
kubectl apply -f ray-job.finetune.yaml

# Monitor the job
kubectl logs -f -l ray.io/job-name=llm-finetune-job
```

### 4. Access Ray Dashboard

```bash
# Get the dashboard URL
DASHBOARD_IP=$(kubectl get svc ray-finetune-dashboard-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Ray Dashboard: http://${DASHBOARD_IP}"
```

## Configuration Options

### Fine-tuning Script Arguments

```bash
python finetune_with_tune.py [OPTIONS]

Options:
  --model MODEL          Hugging Face model ID (default: Qwen/Qwen2.5-3B)
  --num-samples N        Number of hyperparameter combinations (default: 10)
  --num-workers N        Workers per trial (default: 1)
  --gpus-per-worker N    GPUs per worker (default: 4)
  --single-run           Run single training (no tuning)
```

### Custom Hyperparameter Search

Edit `finetune_with_tune.py` to customize the search space:

```python
TUNE_CONFIG = {
    "lora_r": tune.choice([8, 16, 32, 64]),
    "lora_alpha": tune.choice([16, 32, 64, 128]),
    "learning_rate": tune.loguniform(1e-5, 5e-4),
    # Add more hyperparameters...
}
```

### Custom Dataset

The fine-tuning script reads pre-split Alpaca-style JSONL files from
`$DATASET_DIR` (default `/home/ray/dataset`, populated from the
`aks-dataset` PVC). To swap in a different dataset:

1. Replace the contents of the `aks-dataset` PVC with your own
   `*.train.jsonl` / `*.val.jsonl` files (each record must have
   `instruction` / `input` / `output`).
2. Optionally override the filenames at runtime via the
   `DATASET_TRAIN_FILE` and `DATASET_VAL_FILE` env vars on the RayJob.

For a deeper change (different prompt template, multi-turn chat format,
etc.) edit `prepare_dataset()` in `finetune_with_tune.py`.

## Files

| File | Description |
|------|-------------|
| `setup-aks-raytune-finetune.sh` | AKS cluster setup + dataset PVC bootstrap script |
| `ray-cluster.finetune.yaml` | Ray cluster configuration (mounts `aks-dataset` PVC at `/home/ray/dataset`) |
| `ray-job.finetune.yaml` | Ray Job for fine-tuning (inline-pushes the merged model to `chengliangli/qwen2.5-3b-aks-tsg`) |
| `ray-job.push.yaml` | Ray Job to push an existing checkpoint to the HF Hub |
| `ray-job.compare.yaml` | Ray Job to compare base vs fine-tuned model on the AKS val split |
| `finetune_with_tune.py` | Fine-tuning script with Tune (reads `$DATASET_DIR/aks_combined.{train,val}.jsonl`; inline HF push via `INLINE_PUSH_*` envs) |
| `push_to_hub.py` | Standalone CLI to merge a LoRA adapter and push to HF Hub |
| `compare_inference.py` | Compare base vs adapter / merged model on the AKS val split (or any Alpaca-style JSONL via `--eval-file`) |
| `comparison.json` | Base vs fine-tuned eval on 20 held-out AKS val samples (output of the latest `ray-job.compare.yaml` run) |

## Fine-tuning Approach

The end-to-end pipeline implemented by this sample:

1. **Model + method.** `Qwen/Qwen2.5-3B` loaded in 4-bit (NF4, double-quant) via
   `bitsandbytes`, with LoRA adapters injected on the attention + MLP projections
   (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
2. **Data.** `aks_combined.train.jsonl` / `aks_combined.val.jsonl` (Alpaca-style
   `instruction` / `input` / `output` records built from AKS public docs, the
   AKS TSG are uploaded to an Azure Files PVC (`aks-dataset`, RWX) and mounted
   at `/home/ray/dataset` on the Ray head + worker pods.
3. **Distributed training.** `finetune_with_tune.py` wraps a Hugging Face
   `Trainer` inside a Ray Train `TorchTrainer`. By default it runs as a
   `--single-run` (one trial, fixed best-known hyperparameters) on 1 worker
   with `--gpus-per-worker 1`; pass plain `python finetune_with_tune.py` to
   launch the full Ray Tune sweep (ASHA + Optuna) over the table above.
4. **Inline merge + push.** Because Ray Tune's default checkpoint persistence
   fails on multi-node clusters without shared object storage, the trainer
   merges the LoRA adapter into fp16 weights and pushes them to the Hugging
   Face Hub from inside `train_func` *before* `tune.report()`. This is driven
   by env vars on the RayJob:

   | Env var | Purpose |
   |---|---|
   | `INLINE_PUSH_REPO_ID` | Target HF repo (`chengliangli/qwen2.5-3b-aks-tsg`) |
   | `INLINE_PUSH_PRIVATE=1` | Create the repo as private |
   | `INLINE_PUSH_MERGE=1` | Merge LoRA into the base weights before push |

   The merged fp16 model (~5.7 GB) lands on the Hub at the end of the run;
   you don't need a shared filesystem for checkpoints.
5. **Evaluation.** `ray-job.compare.yaml` runs `compare_inference.py`, which
   loads the base model and the merged fine-tuned model side-by-side, samples
   `--num-samples` records from `aks_combined.val.jsonl`, generates completions
   with both, and writes per-sample perplexity / ROUGE-L / latency to
   `/tmp/comparison.json` on the head pod.

Best hyperparameters used by `--single-run` (verified against the full sweep):

| Hyperparameter | Value |
|---|---|
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `learning_rate` | 2e-4 |
| `per_device_train_batch_size` | 8 |
| `gradient_accumulation_steps` | 2 |
| `epochs` | 1 |
| `warmup_ratio` | 0.03 |
| `weight_decay` | 0.01 |

A full `--single-run` on 4× A100 (80 GB) completes in ~95 minutes and
reaches a final `eval_loss` of ~1.64.

## Results

Fine-tuned `Qwen/Qwen2.5-3B` on `aks_combined.train.jsonl` (≈12k
Alpaca-style records covering AKS public docs, AKS TSG, and IcM incident
summaries) with QLoRA (4-bit) + LoRA, then merged and pushed to
[`chengliangli/qwen2.5-3b-aks-tsg`](https://huggingface.co/chengliangli/qwen2.5-3b-aks-tsg)
(private, 5.67 GB fp16).

Evaluation: 20 held-out samples from `aks_combined.val.jsonl`, generated
with greedy decoding on the merged fp16 model. Full per-sample data lives
in [comparison.json](./comparison.json).

| Metric | Base `Qwen2.5-3B` | Fine-tuned `qwen2.5-3b-aks-tsg` | Δ |
|---|---|---|---|
| Avg perplexity | 12.56 | **4.51** | **−64%** |
| Avg ROUGE-L vs reference | 0.090 | **0.195** | **+117%** (≈2.2×) |
| Per-sample perplexity win-rate | — | **20 / 20** | every sample improved |
| Avg latency / sample | 6.96 s | 7.81 s | +0.85 s (longer outputs) |
| Avg throughput | 23.4 tok/s | 23.5 tok/s | ≈ unchanged |

The largest wins are on AKS-specific prompts where the base model
hallucinates plausible-sounding but wrong commands (e.g. sample 0:
ppl 41.6 → 1.3; sample 3: 63.5 → 2.2; sample 7: 24.2 → 1.1). Qualitatively
the fine-tuned model emits real AKS doc patterns —
`az provider register --namespace Microsoft.ContainerService --wait`,
`azurecli-interactive` fences, reference-style links — instead of generic
Kubernetes prose.

To reproduce end-to-end:

```bash
# 1. fine-tune + inline-push merged model to HF
#    (INLINE_PUSH_REPO_ID=chengliangli/qwen2.5-3b-aks-tsg is set in ray-job.finetune.yaml)
kubectl apply -f ray-job.finetune.yaml

# 2. once SUCCEEDED, compare base vs your fine-tuned model on the val split
kubectl delete rayjob llm-finetune-job  # free the GPU worker
kubectl apply -f ray-job.compare.yaml
HEAD=$(kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
SID=$(kubectl get rayjob llm-compare-inference -o jsonpath='{.status.jobId}')
kubectl exec "$HEAD" -c ray-head -- ray job logs "$SID" | tail -50

# 3. pull the per-sample comparison report out of the head pod
kubectl cp "$HEAD":/tmp/comparison.json ./comparison.json -c ray-head
```

## Monitoring

### Ray Dashboard

The Ray Dashboard provides:
- **Jobs**: View running and completed fine-tuning jobs
- **Tune**: Monitor hyperparameter trials and results
- **Cluster**: Check node and resource utilization
- **Logs**: View training logs for each trial

**Jobs** — live status of the fine-tuning RayJob and its Tune trials:

![Ray Dashboard Jobs](../docs/images/ray-dashboard-joblist.png)

**Cluster** — Ray head and GPU worker nodes registered with the cluster:

![Ray Dashboard Nodes](../docs/images/ray-dashboard-nodelist.png)

### Kubernetes Logs

```bash
# View Ray head logs
kubectl logs -l ray.io/cluster=ray-finetune-cluster,ray.io/node-type=head

# View Ray worker logs
kubectl logs -l ray.io/cluster=ray-finetune-cluster,ray.io/node-type=worker

# View specific job logs
kubectl logs -l ray.io/job-name=llm-finetune-job
```

## Estimated Resources

| Component | CPU | Memory | GPU |
|-----------|-----|--------|-----|
| Ray Head | 4-8 | 16-32 Gi | 0 |
| Ray Worker | 32-48 | 128-192 Gi | 4x A100 |
| Per Trial | - | ~40 Gi | 4x A100 |

**Note**: For parallel trials, multiply worker resources by the number of concurrent trials.

## Troubleshooting

### Out of Memory

Reduce batch size or gradient accumulation:
```python
"per_device_train_batch_size": tune.choice([1]),
"gradient_accumulation_steps": tune.choice([16, 32]),
```

### Slow Training

- Increase `max_seq_length` in the script
- Use a smaller model for initial testing
- Reduce `num_samples` for faster exploration

### GPU Not Found

```bash
# Verify GPU nodes are ready
kubectl get nodes -l accelerator=nvidia

# Check GPU resources
kubectl describe node <gpu-node-name> | grep nvidia
```

## References

- [Ray Train with Transformers](https://docs.ray.io/en/latest/train/getting-started-transformers.html)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [PEFT/LoRA Guide](https://huggingface.co/docs/peft/main/en/index)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [AKS Automatic](https://learn.microsoft.com/azure/aks/intro-aks-automatic)
