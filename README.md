# KubeRay on AKS Automatic

This sample demonstrates how to deploy a [Ray](https://www.ray.io/) cluster on [Azure Kubernetes Service (AKS) Automatic](https://learn.microsoft.com/azure/aks/intro-aks-automatic) using the [KubeRay](https://github.com/ray-project/kuberay) operator. It showcases GPU-accelerated LLM inference using Ray Serve with an OpenAI-compatible API endpoint.

## Overview

AKS Automatic simplifies Kubernetes cluster management by automating cluster setup, node management, scaling, and security configurations. Combined with KubeRay, this provides a powerful platform for running distributed AI/ML workloads on Azure.

### Current Samples

| Sample | Description | Status |
|--------|-------------|--------|
| [Inferencing](./inferencing/) | Deploy LLM inference with Ray Serve on GPU nodes | ✅ Available |
| Training | Distributed model training with Ray Train | 🚧 Coming Soon |
| Model Tuning | Hyperparameter tuning with Ray Tune | 🚧 Coming Soon |

## Inferencing Sample

The inferencing sample deploys a complete LLM serving stack on AKS Automatic with:

- **Model**: Qwen2.5-7B-Instruct (configurable)
- **Serving Framework**: Ray Serve with vLLM backend
- **GPU**: NVIDIA A100 (Standard_ND96amsr_A100_v4)
- **API**: OpenAI-compatible REST endpoints (`/v1/chat/completions`, `/v1/models`)
- **Features**: Autoscaling, streaming responses, GPU memory optimization

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AKS Automatic Cluster                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────────────────────┐    │
│  │   System Pool   │    │        GPU Workload Pool         │    │
│  │  (Default Node) │    │     (A100 GPU Nodes)             │    │
│  │                 │    │                                  │    │
│  │  ┌───────────┐  │    │  ┌────────────────────────────┐  │    │
│  │  │Ray Head   │◄─┼────┼──│ Ray Worker (GPU)           │  │    │
│  │  │Pod        │  │    │  │ - vLLM Engine              │  │    │
│  │  │- Dashboard│  │    │  │ - Qwen2.5-7B Model         │  │    │
│  │  │- GCS      │  │    │  │ - 4x A100 GPUs             │  │    │
│  │  └───────────┘  │    │  └────────────────────────────┘  │    │
│  └─────────────────┘    └──────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LoadBalancer Services                                   │    │
│  │  - Inference API (port 80 → 8000)                       │    │
│  │  - Ray Dashboard (port 80 → 8265)                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Ray Dashboard

The Ray Dashboard provides real-time monitoring of your Ray cluster, including:
- Cluster health and node status
- Running jobs and actors
- Resource utilization (CPU, GPU, memory)
- Serve deployment metrics

![Ray Dashboard Overview](docs/images/ray-dashboard-overview.png)

## Getting Started

Clone the repository to your local machine, then make sure you have all the prerequisites installed.

### Prerequisites

1. An Azure subscription. If you don't have an Azure subscription, you can create a free account [here](https://azure.microsoft.com/free/).
2. The Azure CLI installed on your local machine. You can install the Azure CLI by following the instructions [here](https://docs.microsoft.com/cli/azure/install-azure-cli).
3. [kubectl](https://kubernetes.io/docs/tasks/tools/) must be installed.
4. [Helm](https://helm.sh/docs/intro/install/) must be installed.
5. A [Hugging Face](https://huggingface.co/) account and API token for model access.
6. Sufficient Azure GPU quota for A100 VMs in your target region.

### Quickstart

1. Clone this repository:
   ```bash
   git clone https://github.com/chengliangli0918/aks-ray.git
   cd aks-ray
   ```

2. Update the configuration in `inferencing/setup-aks-rayserve-llm.sh`:
   ```bash
   SUBSCRIPTION_ID="your-subscription-id"
   RESOURCE_GROUP="your-resource-group"
   CLUSTER_NAME="your-cluster-name"
   LOCATION="your-region"  # e.g., italynorth, eastus2
   HF_TOKEN="your-huggingface-token"
   ```

3. Run the setup script:
   ```bash
   cd inferencing
   chmod +x setup-aks-rayserve-llm.sh
   ./setup-aks-rayserve-llm.sh
   ```

4. Test the inference endpoint:
   ```bash
   SERVE_IP=$(kubectl get svc ray-serve-llm-serve-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
   
   curl -X POST "http://${SERVE_IP}/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen2.5-7b-instruct",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 100
     }'
   ```

### What the Setup Script Does

1. **Creates AKS Automatic Cluster** - Provisions a managed Kubernetes cluster with automatic scaling
2. **Adds GPU Node Pool** - Deploys A100 GPU nodes with custom tolerations
3. **Installs NVIDIA GPU Operator** - Sets up GPU drivers and device plugins (in kube-system namespace to work with AKS Automatic security policies)
4. **Deploys KubeRay Operator** - Installs the Kubernetes operator for managing Ray clusters
5. **Creates Ray Service** - Deploys Ray Serve with vLLM for LLM inference
6. **Configures LoadBalancers** - Exposes inference API and Ray Dashboard externally

## Usage Examples

### List Available Models
```bash
$ curl -s http://${SERVE_IP}/v1/models | jq .
```

**Response:**
```json
{
  "data": [
    {
      "id": "qwen2.5-7b-instruct",
      "object": "model",
      "owned_by": "organization-owner",
      "permission": [],
      "metadata": {
        "model_id": "qwen2.5-7b-instruct",
        "input_modality": "text",
        "max_request_context_length": 1024
      }
    }
  ],
  "object": "list"
}
```

### Chat Completion
```bash
$ curl -s -X POST "http://${SERVE_IP}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "What is Kubernetes in one sentence?"}],
    "max_tokens": 50
  }' | jq .
```

**Response:**
```json
{
  "id": "chatcmpl-cb938086-16ec-4665-9847-2981946d0c00",
  "object": "chat.completion",
  "created": 1779076935,
  "model": "qwen2.5-7b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Kubernetes is an open-source container orchestration system for automating the deployment, scaling, and management of containerized applications."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 36,
    "total_tokens": 62,
    "completion_tokens": 26
  }
}
```

### Streaming Response
```bash
$ curl -s -X POST "http://${SERVE_IP}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "Write a haiku about AKS"}],
    "stream": true
  }'
```

**Response (streamed):**
```
data: {"choices":[{"delta":{"role":"assistant","content":""}}]}
data: {"choices":[{"delta":{"content":"Cloud"}}]}
data: {"choices":[{"delta":{"content":" clusters"}}]}
data: {"choices":[{"delta":{"content":" bloom"}}]}
data: {"choices":[{"delta":{"content":" bright"}}]}
data: {"choices":[{"delta":{"content":",\n"}}]}
data: {"choices":[{"delta":{"content":"AKS"}}]}
data: {"choices":[{"delta":{"content":" orchestrates"}}]}
data: {"choices":[{"delta":{"content":" with"}}]}
data: {"choices":[{"delta":{"content":" grace"}}]}
data: {"choices":[{"delta":{"content":",\n"}}]}
data: {"choices":[{"delta":{"content":"Shadows"}}]}
data: {"choices":[{"delta":{"content":" of"}}]}
data: {"choices":[{"delta":{"content":" scalable"}}]}
data: {"choices":[{"delta":{"content":"."}}]}
data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}
data: [DONE]
```

**Generated Haiku:**
> *Cloud clusters bloom bright,*  
> *AKS orchestrates with grace,*  
> *Shadows of scalable.*

### Python Client Example
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<SERVE_IP>/v1",
    api_key="not-needed"  # Ray Serve doesn't require API key by default
)

response = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Azure Kubernetes Service briefly."}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Useful Commands

```bash
# Check Ray Service status
kubectl get rayservice ray-serve-llm

# Check Ray cluster pods
kubectl get pods -l ray.io/serve=true

# View Ray Dashboard
DASHBOARD_IP=$(kubectl get svc ray-serve-llm-dashboard-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Dashboard: http://${DASHBOARD_IP}"

# Check GPU availability
kubectl get nodes -l nvidia.com/gpu.present=true -o custom-columns=NAME:.metadata.name,GPUs:.status.capacity.nvidia\\.com/gpu

# View Ray head logs
kubectl logs -l ray.io/node-type=head -c ray-head --tail=100

# View Ray worker logs  
kubectl logs -l ray.io/node-type=worker -c ray-worker --tail=100
```

## Customization

### Change the Model

Edit `inferencing/ray-service.llm-serve.yaml` to use a different model:

```yaml
llm_configs:
- model_loading_config:
    model_id: your-model-id
    model_source: HuggingFace/Model-Name
  engine_kwargs:
    dtype: bfloat16
    max_model_len: 2048
    gpu_memory_utilization: 0.85
```

### Scale Workers

Adjust replicas in the RayService configuration:

```yaml
workerGroupSpecs:
- replicas: 2        # Number of worker pods
  minReplicas: 1
  maxReplicas: 4
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pod not ready | Check `kubectl describe pod <pod-name>` for events |
| LoadBalancer timeout | Verify endpoints: `kubectl get endpoints <svc-name>` |
| GPU not detected | Ensure GPU Operator pods are running: `kubectl get pods -n kube-system -l app=nvidia-driver-daemonset` |
| Model download fails | Verify HF_TOKEN secret: `kubectl get secret hf-token` |

## Resources

- [Ray on Kubernetes](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [KubeRay Project](https://github.com/ray-project/kuberay)
- [Ray Serve LLM](https://docs.ray.io/en/latest/serve/llm/serving-llms.html)
- [AKS Automatic](https://learn.microsoft.com/azure/aks/intro-aks-automatic)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.