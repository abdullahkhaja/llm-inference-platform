#!/usr/bin/env bash
# Start vLLM (+ optional FastAPI) on the GPU pod. Fail-fast with timeouts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${HF_TOKEN:?HF_TOKEN required}"
: "${ADAPTER_PATH:?ADAPTER_PATH required}"
: "${VLLM_READY_TIMEOUT_S:=600}"

echo "Starting docker compose GPU stack..."
docker compose -f docker-compose.gpu.yml up -d

echo "Waiting for vLLM /v1/models (timeout=${VLLM_READY_TIMEOUT_S}s)..."
deadline=$((SECONDS + VLLM_READY_TIMEOUT_S))
until curl -fsS "http://127.0.0.1:8001/v1/models" >/tmp/vllm_models.json 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: vLLM not ready in time — dumping logs" >&2
    docker compose -f docker-compose.gpu.yml logs --tail=200 vllm >&2 || true
    exit 1
  fi
  sleep 5
done

echo "vLLM models:"
cat /tmp/vllm_models.json
echo
if ! grep -q "${VLLM_MODEL_NAME:-spider-sft}" /tmp/vllm_models.json; then
  echo "ERROR: LoRA model name ${VLLM_MODEL_NAME:-spider-sft} not listed in /v1/models" >&2
  exit 1
fi

echo "Stack is up. Run ./ops/runpod/smoke_test.sh next."
echo "Teardown when done: ./ops/runpod/cleanup.sh  (and TERMINATE the RunPod pod in the console)."
