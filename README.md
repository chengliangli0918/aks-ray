# KubeRay on AKS Automatic

This repository demonstrates how to run [Ray](https://www.ray.io/) workloads on [Azure Kubernetes Service (AKS) Automatic](https://learn.microsoft.com/azure/aks/intro-aks-automatic) using the [KubeRay](https://github.com/ray-project/kuberay) operator, with GPU-accelerated samples for LLM inference and fine-tuning.

## Overview

AKS Automatic simplifies Kubernetes cluster management by automating cluster setup, node management, scaling, and security configurations. Combined with KubeRay, this provides a powerful platform for running distributed AI/ML workloads on Azure.

## Samples

| Sample | Description | Status |
|--------|-------------|--------|
| [Inferencing](./inferencing/README.md) | Serve LLMs with Ray Serve + vLLM on A100 GPUs, exposing an OpenAI-compatible API | Available |
| [Fine-tuning](./finetuning/README.md) | QLoRA fine-tuning of Qwen2.5-3B with Ray Train + Ray Tune, including HF Hub push and base-vs-tuned benchmarking | Available |
| Training | Distributed model training with Ray Train | Coming Soon |

See each sample's README for prerequisites, setup, and usage details.

## Resources

- [Ray on Kubernetes](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [KubeRay Project](https://github.com/ray-project/kuberay)
- [AKS Automatic](https://learn.microsoft.com/azure/aks/intro-aks-automatic)
- [AKS Managed GPU Driver](https://learn.microsoft.com/azure/aks/gpu-cluster)
- [kubelogin](https://github.com/Azure/kubelogin)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
