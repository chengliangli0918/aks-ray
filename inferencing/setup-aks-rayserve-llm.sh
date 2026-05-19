#!/bin/bash

# =============================================================================
# AKS Ray Serve LLM Inferencing Setup Script
# =============================================================================
# This script creates an AKS Automatic cluster with GPU nodes for Ray Serve
# LLM inferencing workloads.
#
# Key features:
# - Creates AKS Automatic cluster with GPU node pool (A100 GPUs)
# - Uses AKS managed GPU driver (--enable-managed-gpu=true)
# - Installs KubeRay operator for Ray cluster management
# - Deploys Ray Serve for LLM inference
#
# Reference: https://docs.ray.io/en/latest/cluster/kubernetes/examples/rayserve-llm-example.html
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
HF_TOKEN="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"  # Hugging Face API token

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
    echo "==> Adding GPU workload node pool with $GPU_VM_SIZE (skipping AKS GPU driver)..."
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
kubectl create secret generic hf-token \
    --from-literal=hf_token="$HF_TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -

# -----------------------------------------------------------------------------
# Step 8: Deploy Ray Service for LLM Inference
# -----------------------------------------------------------------------------
echo "==> Deploying Ray Service for LLM inference..."
kubectl apply -f ray-service.llm-serve.yaml

echo "==> Waiting for Ray Service to be ready..."
kubectl wait --for=condition=Available --timeout=600s rayservice/ray-serve-llm || true

# -----------------------------------------------------------------------------
# Step 9: Create LoadBalancer Services for External Access
# -----------------------------------------------------------------------------
echo "==> Creating LoadBalancer services for external access..."

# Wait for RayCluster to be created and get the actual cluster name
# KubeRay adds a random suffix to cluster names (e.g., ray-serve-llm-xznmx)
echo "==> Waiting for RayCluster to be created..."
for i in {1..60}; do
    RAY_CLUSTER_NAME=$(kubectl get raycluster -l ray.io/served-by=ray-serve-llm -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [[ -n "$RAY_CLUSTER_NAME" ]]; then
        echo "==> Found RayCluster: $RAY_CLUSTER_NAME"
        break
    fi
    echo "==> Waiting for RayCluster... (attempt $i/60)"
    sleep 5
done

if [[ -z "$RAY_CLUSTER_NAME" ]]; then
    echo "==> ERROR: RayCluster not found after 5 minutes. Using fallback selector."
    RAY_CLUSTER_NAME="ray-serve-llm"
fi

# Create LoadBalancer service for Ray Serve inference endpoint (port 80 -> 8000)
# Uses the same selector as the internal ray-serve-llm-serve-svc service
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: ray-serve-llm-serve-lb
  labels:
    app: ray-serve-llm
spec:
  type: LoadBalancer
  selector:
    ray.io/serve: "true"
    ray.io/cluster: ${RAY_CLUSTER_NAME}
  ports:
  - name: serve
    port: 80
    targetPort: 8000
    protocol: TCP
EOF

# Create LoadBalancer service for Ray Dashboard (port 80 -> 8265)
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: ray-serve-llm-dashboard-lb
  labels:
    app: ray-serve-llm
spec:
  type: LoadBalancer
  selector:
    ray.io/node-type: head
    ray.io/cluster: ${RAY_CLUSTER_NAME}
  ports:
  - name: dashboard
    port: 80
    targetPort: 8265
    protocol: TCP
EOF

# -----------------------------------------------------------------------------
# Step 10: Wait for LoadBalancer IPs and Display Access Information
# -----------------------------------------------------------------------------
echo "==> Waiting for LoadBalancer IPs to be assigned..."
sleep 30

echo ""
echo "============================================================================="
echo "                         DEPLOYMENT COMPLETE                                 "
echo "============================================================================="
echo ""

# Get Serve endpoint IP
SERVE_IP=$(kubectl get svc ray-serve-llm-serve-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
echo "Ray Serve Inference Endpoint: http://${SERVE_IP}"
echo "  - OpenAI-compatible API: http://${SERVE_IP}/v1/chat/completions"
echo "  - Models endpoint: http://${SERVE_IP}/v1/models"

# Get Dashboard IP
DASHBOARD_IP=$(kubectl get svc ray-serve-llm-dashboard-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
echo ""
echo "Ray Dashboard: http://${DASHBOARD_IP}"

echo ""
echo "============================================================================="
echo "                         USEFUL COMMANDS                                     "
echo "============================================================================="
echo ""
echo "# Check Ray Service status:"
echo "kubectl get rayservice ray-serve-llm"
echo ""
echo "# Check pods:"
echo "kubectl get pods -l ray.io/cluster=${RAY_CLUSTER_NAME}"
echo ""
echo "# Check LoadBalancer IPs:"
echo "kubectl get svc ray-serve-llm-serve-lb ray-serve-llm-dashboard-lb"
echo ""
echo "# Test inference (replace IP with actual serve endpoint IP):"
echo 'curl -X POST http://<SERVE_IP>/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"model": "qwen2.5-7b-instruct", "messages": [{"role": "user", "content": "Hello!"}]}'"'"
echo ""
echo "# View logs:"
echo "kubectl logs -l ray.io/node-type=head -c ray-head --tail=100"
echo ""
echo "============================================================================="
