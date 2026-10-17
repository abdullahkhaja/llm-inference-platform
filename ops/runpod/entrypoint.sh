#!/usr/bin/env bash
# Start vLLM with a locally resolved LoRA adapter. Runs on the GPU pod only.
set -euo pipefail

: "${VLLM_MODEL:=meta-llama/Llama-3.1-8B-Instruct}"
: "${VLLM_MODEL_NAME:=spider-sft}"
: "${ADAPTER_LOCAL_DIR:=/models/adapters/spider-sft}"
: "${ADAPTER_PATH:=}"
: "${ADAPTER_SUBFOLDER:=sft/final}"
: "${MAX_MODEL_LEN:=2048}"
: "${GPU_MEMORY_UTILIZATION:=0.85}"
: "${MAX_LORA_RANK:=16}"
: "${PORT:=8001}"
: "${HF_HOME:=/models/hf-cache}"
: "${ENABLE_PREFIX_CACHING:=true}"

mkdir -p "$HF_HOME" "$(dirname "$ADAPTER_LOCAL_DIR")"

if [[ ! -f "$ADAPTER_LOCAL_DIR/adapter_config.json" ]]; then
  if [[ -z "$ADAPTER_PATH" ]]; then
    echo "ERROR: ADAPTER_LOCAL_DIR missing adapter_config.json and ADAPTER_PATH unset" >&2
    exit 1
  fi
  echo "Resolving private adapter from Hugging Face (credentials never printed)..."
  PYTHON="$(command -v python3 || command -v python)"
  "$PYTHON" /usr/local/bin/resolve_adapter.py \
    --repo-id "$ADAPTER_PATH" \
    --subfolder "$ADAPTER_SUBFOLDER" \
    --out-dir "$ADAPTER_LOCAL_DIR"
fi

if [[ ! -f "$ADAPTER_LOCAL_DIR/adapter_config.json" ]]; then
  echo "ERROR: adapter_config.json not found under $ADAPTER_LOCAL_DIR" >&2
  exit 1
fi

PYTHON="$(command -v python3 || command -v python)"
FLAGS_PY="/usr/local/bin/vllm_flags.py"
if [[ ! -f "$FLAGS_PY" ]]; then
  FLAGS_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vllm_flags.py"
fi
CACHE_ARG="$("$PYTHON" "$FLAGS_PY" "$ENABLE_PREFIX_CACHING")"
echo "Starting vLLM OpenAI server"
echo "  interpreter=$PYTHON"
echo "  base_model=$VLLM_MODEL"
echo "  lora_name=$VLLM_MODEL_NAME"
echo "  adapter_dir=$ADAPTER_LOCAL_DIR"
echo "  max_model_len=$MAX_MODEL_LEN"
echo "  gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "  port=$PORT"
echo "  prefix_caching_arg=$CACHE_ARG"

exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL" \
  --served-model-name "$VLLM_MODEL" \
  --enable-lora \
  --lora-modules "${VLLM_MODEL_NAME}=${ADAPTER_LOCAL_DIR}" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "$CACHE_ARG" \
  --host 0.0.0.0 \
  --port "$PORT" \
  "$@"
