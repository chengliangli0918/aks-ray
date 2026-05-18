#!/bin/bash

# =============================================================================
# AKS Ray Serve LLM Inferencing Setup Script
# =============================================================================
# This script creates an AKS Automatic cluster with GPU nodes for Ray Serve
# LLM inferencing workloads.
#
# Key features:
# - Creates AKS Automatic cluster with GPU node pool (A100 GPUs)
# - Skips AKS built-in GPU driver and uses NVIDIA GPU Operator instead
# - Installs GPU Operator in kube-system (exempted from AKS security policies)
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
# TODO: For AKS Automatic clusters, we skip AKS's built-in GPU driver installation
# (--gpu-driver none) and use NVIDIA GPU Operator instead. This is because:
# 1. AKS Automatic has strict security policies that can interfere with CSE
# 2. GPU Operator provides more flexibility and is installed in kube-system
#    namespace which is exempted from AKS Automatic security policies
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
        --gpu-driver none \
        --node-taints "sku=gpu:NoSchedule" \
        --labels "workload=rayserve" \
                 "feature.node.kubernetes.io/pci-10de.present=true" \
                 "nvidia.com/gpu.present=true" \
                 "feature.node.kubernetes.io/system-os_release.ID=ubuntu" \
                 "feature.node.kubernetes.io/system-os_release.VERSION_ID=22.04" \
                 "nvidia.com/gpu.deploy.driver=true" \
                 "nvidia.com/gpu.deploy.container-toolkit=true" \
                 "nvidia.com/gpu.deploy.device-plugin=true" \
                 "nvidia.com/gpu.deploy.gpu-feature-discovery=true" \
                 "nvidia.com/gpu.deploy.operator-validator=true" \
                 "nvidia.com/gpu.deploy.dcgm-exporter=true"
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

# -----------------------------------------------------------------------------
# Step 6: Install NVIDIA GPU Operator
# -----------------------------------------------------------------------------
# NOTE: We install GPU Operator in kube-system namespace which is exempted from
# AKS Automatic's baseline security policies (hostPath volumes, privileged containers).
# We also disable NFD (Node Feature Discovery) because it requires nodes/proxy RBAC
# which AKS blocks, and instead use manual labels added via nodepool configuration.
# -----------------------------------------------------------------------------
echo "==> Installing NVIDIA GPU Operator..."

# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update nvidia

# Check if GPU operator is already installed
if helm status gpu-operator -n kube-system &>/dev/null; then
    echo "==> GPU Operator already installed, upgrading..."
    HELM_CMD="upgrade"
else
    echo "==> Installing GPU Operator in kube-system namespace..."
    HELM_CMD="install"
fi

# Install/upgrade GPU Operator with AKS Automatic compatible settings
helm $HELM_CMD gpu-operator nvidia/gpu-operator \
    --namespace kube-system \
    --set driver.enabled=true \
    --set driver.manager.enabled=false \
    --set toolkit.enabled=true \
    --set devicePlugin.enabled=true \
    --set migManager.enabled=false \
    --set dcgmExporter.enabled=true \
    --set nfd.enabled=false \
    --set operator.defaultRuntime=containerd \
    --set 'daemonsets.tolerations[0].key=sku' \
    --set 'daemonsets.tolerations[0].operator=Equal' \
    --set 'daemonsets.tolerations[0].value=gpu' \
    --set 'daemonsets.tolerations[0].effect=NoSchedule' \
    --set driver.upgradePolicy.autoUpgrade=false \
    --wait --timeout 15m

# Remove k8s-driver-manager init container from driver daemonset
# This init container tries to modify node labels which AKS blocks
echo "==> Patching nvidia-driver-daemonset to remove init containers..."
sleep 10  # Wait for daemonset to be created
kubectl patch daemonset nvidia-driver-daemonset -n kube-system \
    --type='json' \
    -p='[{"op": "remove", "path": "/spec/template/spec/initContainers"}]' 2>/dev/null || true

# Wait for GPU driver to be installed
echo "==> Waiting for NVIDIA driver installation (this may take 5-10 minutes)..."
kubectl rollout status daemonset/nvidia-driver-daemonset -n kube-system --timeout=600s || true

# Wait for device plugin to be ready
echo "==> Waiting for NVIDIA device plugin..."
kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n kube-system --timeout=300s || true

# Verify GPUs are available
echo "==> Verifying GPU availability..."
GPU_COUNT=$(kubectl get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[*].status.capacity.nvidia\.com/gpu}' 2>/dev/null | tr ' ' '+' | bc 2>/dev/null || echo "0")
echo "==> Total GPUs available: $GPU_COUNT"

# -----------------------------------------------------------------------------
# Step 7: Install KubeRay Operator
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
# Step 8: Create Hugging Face Token Secret
# -----------------------------------------------------------------------------
echo "==> Creating Hugging Face token secret..."
kubectl create secret generic hf-token \
    --from-literal=hf_token="$HF_TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -

# -----------------------------------------------------------------------------
# Step 9: Deploy Ray Service for LLM Inference
# -----------------------------------------------------------------------------
echo "==> Deploying Ray Service for LLM inference..."
kubectl apply -f ray-service.llm-serve.yaml

echo "==> Waiting for Ray Service to be ready..."
kubectl wait --for=condition=Available --timeout=600s rayservice/ray-serve-llm || true

# -----------------------------------------------------------------------------
# Step 10: Create LoadBalancer Services for External Access
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
# Step 11: Wait for LoadBalancer IPs and Display Access Information
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
