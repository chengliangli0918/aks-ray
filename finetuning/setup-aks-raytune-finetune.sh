#!/bin/bash

# =============================================================================
# AKS Ray Train + Tune LLM Fine-tuning Setup Script
# =============================================================================
# This script creates an AKS Automatic cluster with GPU nodes for LLM
# fine-tuning using Ray Train with Ray Tune for hyperparameter optimization.
#
# Key features:
# - Creates AKS Automatic cluster with GPU node pool (A100 GPUs)
# - Uses AKS managed GPU driver (--enable-managed-gpu=true)
# - Installs KubeRay operator for Ray cluster management
# - Deploys Ray cluster for distributed fine-tuning with Tune
# - Supports LoRA/QLoRA fine-tuning with hyperparameter search
#
# Reference: https://docs.ray.io/en/latest/train/getting-started-transformers.html
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration Variables
# -----------------------------------------------------------------------------
SUBSCRIPTION_ID="c582d154-33c4-47e4-a3c2-0632d20b12eb"
RESOURCE_GROUP="chengliangli-rg"
CLUSTER_NAME="chengliangli-auto"
LOCATION="italynorth"
GPU_VM_SIZE="Standard_ND96amsr_A100_v4"  # A100 GPU optimized for AI workloads
HF_TOKEN="${HF_TOKEN:-}"  # Hugging Face API token — export HF_TOKEN=hf_xxx before running

# Storage configuration for checkpoints and datasets
STORAGE_ACCOUNT_NAME=""  # Azure Storage account name (optional)
STORAGE_CONTAINER_NAME="raycheckpoints"  # Container for Ray checkpoints

# -----------------------------------------------------------------------------
# Step 1: Set Azure Subscription
# -----------------------------------------------------------------------------
echo "==> Setting Azure subscription..."
az account set --subscription "$SUBSCRIPTION_ID"

# -----------------------------------------------------------------------------
# Step 2: Create Resource Group (if not exists)
# -----------------------------------------------------------------------------
if az group show --name "$RESOURCE_GROUP" &>/dev/null; then
    echo "==> Resource group $RESOURCE_GROUP already exists, skipping..."
else
    echo "==> Creating resource group: $RESOURCE_GROUP in $LOCATION..."
    az group create \
        --name "$RESOURCE_GROUP" \
        --location "$LOCATION"
fi

# -----------------------------------------------------------------------------
# Step 3: Create AKS Automatic Cluster (if not exists)
# -----------------------------------------------------------------------------
if az aks show --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" &>/dev/null; then
    echo "==> AKS cluster $CLUSTER_NAME already exists, skipping..."
else
    echo "==> Creating AKS automatic cluster: $CLUSTER_NAME..."
    az aks create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CLUSTER_NAME" \
        --location "$LOCATION" \
        --sku automatic
fi

# Wait for cluster provisioning to complete (check provisioningState = Succeeded)
echo "==> Waiting for cluster provisioning to complete..."
while true; do
    STATE=$(az aks show --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" --query "provisioningState" -o tsv 2>/dev/null)
    if [[ "$STATE" == "Failed" ]]; then
        echo "==> ERROR: Cluster provisioning failed. Exiting..."
        exit 1
    fi
    if [[ "$STATE" == "Succeeded" ]]; then
        # Double-check no in-progress operations by waiting a few seconds
        sleep 10
        STATE=$(az aks show --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" --query "provisioningState" -o tsv 2>/dev/null)
        if [[ "$STATE" == "Succeeded" ]]; then
            echo "==> Cluster provisioning completed."
            break
        fi
    fi
    echo "==> Cluster provisioning state: $STATE. Waiting..."
    sleep 30
done

# -----------------------------------------------------------------------------
# Step 4: Add GPU Workload Node Pool (if not exists)
# -----------------------------------------------------------------------------
# Uses AKS managed GPU driver which automatically installs and manages
# NVIDIA drivers on GPU nodes. The GPU nodes will have the label
# 'accelerator=nvidia' for scheduling GPU workloads.
# -----------------------------------------------------------------------------
if az aks nodepool show --resource-group "$RESOURCE_GROUP" --cluster-name "$CLUSTER_NAME" --name workload &>/dev/null; then
    echo "==> Node pool 'workload' already exists, skipping..."
else
    echo "==> Adding GPU workload node pool with $GPU_VM_SIZE..."
    az aks nodepool add \
        --resource-group "$RESOURCE_GROUP" \
        --cluster-name "$CLUSTER_NAME" \
        --name workload \
        --node-vm-size "$GPU_VM_SIZE" \
        --node-count 1 \
        --enable-managed-gpu=true
fi

# Wait for node pool to be ready
echo "==> Waiting for GPU node pool to be ready..."
az aks nodepool wait \
    --resource-group "$RESOURCE_GROUP" \
    --cluster-name "$CLUSTER_NAME" \
    --name workload \
    --created \
    --interval 30 \
    --timeout 1800

# -----------------------------------------------------------------------------
# Step 5: Get AKS Credentials
# -----------------------------------------------------------------------------
echo "==> Getting AKS credentials..."
az aks get-credentials \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CLUSTER_NAME" \
    --overwrite-existing

# Convert kubeconfig for Azure CLI authentication (required for Azure RBAC)
echo "==> Converting kubeconfig for Azure CLI authentication..."
kubelogin convert-kubeconfig -l azurecli

# Verify GPUs are available (AKS Automatic uses 'accelerator=nvidia' label)
echo "==> Verifying GPU availability..."
GPU_COUNT=$(kubectl get nodes -l accelerator=nvidia -o jsonpath='{.items[*].status.capacity.nvidia\.com/gpu}' 2>/dev/null | tr ' ' '+' | bc 2>/dev/null || echo "0")
echo "==> Total GPUs available: $GPU_COUNT"

# -----------------------------------------------------------------------------
# Step 6: Install KubeRay Operator
# -----------------------------------------------------------------------------
echo "==> Installing KubeRay operator..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install KubeRay operator v1.6.0
helm upgrade --install kuberay-operator kuberay/kuberay-operator \
    --version 1.6.0 \
    --namespace kuberay-system \
    --create-namespace \
    --wait

echo "==> Waiting for KubeRay operator to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/kuberay-operator -n kuberay-system

# -----------------------------------------------------------------------------
# Step 7: Create Hugging Face Token Secret
# -----------------------------------------------------------------------------
echo "==> Creating Hugging Face token secret..."
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running this script:"
    echo "       export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    exit 1
fi
kubectl create secret generic hf-token \
    --from-literal=hf_token="$HF_TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -

# -----------------------------------------------------------------------------
# Step 8: Create ConfigMap for Fine-tuning Script
# -----------------------------------------------------------------------------
echo "==> Creating ConfigMap for fine-tuning script..."
kubectl create configmap finetune-script \
    --from-file=finetune_with_tune.py \
    --from-file=compare_inference.py \
    --from-file=push_to_hub.py \
    --dry-run=client -o yaml | kubectl apply -f -

# -----------------------------------------------------------------------------
# Step 9: Deploy Ray Cluster for Fine-tuning
# -----------------------------------------------------------------------------
echo "==> Deploying Ray Cluster for fine-tuning..."
kubectl apply -f ray-cluster.finetune.yaml

echo "==> Waiting for Ray Cluster to be ready..."
kubectl wait --for=condition=RayClusterProvisioned --timeout=600s raycluster/ray-finetune-cluster || true

# -----------------------------------------------------------------------------
# Step 10: Create LoadBalancer Service for Ray Dashboard
# -----------------------------------------------------------------------------
echo "==> Creating LoadBalancer service for Ray Dashboard..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: ray-finetune-dashboard-lb
  labels:
    app: ray-finetune
spec:
  type: LoadBalancer
  selector:
    ray.io/cluster: ray-finetune-cluster
    ray.io/node-type: head
  ports:
  - name: dashboard
    protocol: TCP
    port: 80
    targetPort: 8265
EOF

# -----------------------------------------------------------------------------
# Step 11: Display Access Information
# -----------------------------------------------------------------------------
echo ""
echo "============================================================================="
echo "                    Ray Fine-tuning Cluster Setup Complete!"
echo "============================================================================="
echo ""
echo "Wait for LoadBalancer IP to be assigned, then access:"
echo ""
echo "Ray Dashboard:"
echo "  kubectl get svc ray-finetune-dashboard-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}'"
echo ""
echo "Submit fine-tuning job:"
echo "  kubectl apply -f ray-job.finetune.yaml"
echo ""
echo "Monitor fine-tuning progress:"
echo "  kubectl logs -f -l ray.io/cluster=ray-finetune-cluster,ray.io/node-type=head"
echo ""
echo "View Ray Tune results:"
echo "  Access Ray Dashboard -> Jobs -> View Results"
echo ""
echo "============================================================================="
