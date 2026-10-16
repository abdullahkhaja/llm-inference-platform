#!/usr/bin/env bash
# Prepare a RunPod GPU pod for Phase 2 smoke testing.
# Does NOT create or start a pod — run this ON an already-rented machine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== Phase 2 RunPod setup (manual; costs money while the pod is running) ==="
echo "Repo: $ROOT"
echo "This script is for the rented GPU pod only (not local mock)."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found — this host has no NVIDIA driver/GPU" >&2
  exit 1
fi

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found" >&2
  exit 1
fi

if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "WARNING: docker may lack NVIDIA runtime — GPU containers can fail"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN before setup (private adapter + gated Llama)" >&2
  exit 2
fi

if [[ -z "${ADAPTER_PATH:-}" ]]; then
  echo "ERROR: export ADAPTER_PATH=your-hf-user/your-private-adapter-repo" >&2
  exit 2
fi

: "${MODEL_PATH:=meta-llama/Llama-3.1-8B-Instruct}"
: "${ADAPTER_SUBFOLDER:=sft/final}"
: "${VLLM_MODEL_NAME:=spider-sft}"
: "${MAX_MODEL_LEN:=2048}"
: "${GPU_MEMORY_UTILIZATION:=0.85}"

mkdir -p logs ops/runpod/smoke_results

echo "Building GPU compose stack (pulls pinned vllm/vllm-openai)..."
docker compose -f docker-compose.gpu.yml build

echo "Setup complete. Next: ./ops/runpod/start.sh"
echo "Remember: record the hourly GPU rate before leaving the pod running."
