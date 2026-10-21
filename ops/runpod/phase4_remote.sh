#!/usr/bin/env bash
# Phase 4 measurements on a pod that already has the repo, an L4, and Prometheus.
# Does not create or terminate the pod.
set -u
ROOT="${ROOT:-/work/src}"
OUT="${OUT:-/work/phase4}"
REQ="${PHASE4_REQUESTS:-1000}"
READY_TIMEOUT="${VLLM_READY_TIMEOUT_S:-600}"
mkdir -p "$OUT" /models/hf-cache /usr/local/bin
cd "$ROOT"
export HF_HOME="${HF_HOME:-/models/hf-cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export VLLM_MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-spider-sft}"
export ADAPTER_SUBFOLDER="${ADAPTER_SUBFOLDER:-sft/final}"
export ADAPTER_LOCAL_DIR="${ADAPTER_LOCAL_DIR:-/models/adapters/spider-sft}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
export PORT=8001
export PYTHONPATH="$ROOT"

fail() { echo "FAIL: $*" | tee -a "$OUT/status.log"; exit 1; }
log() { echo "$*" | tee -a "$OUT/status.log"; }

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv > "$OUT/nvidia.csv" || fail "nvidia-smi"
cp -f ops/runpod/resolve_adapter.py ops/runpod/vllm_flags.py /usr/local/bin/
cp -f ops/runpod/entrypoint.sh /usr/local/bin/vllm-lora-entrypoint.sh
chmod +x /usr/local/bin/resolve_adapter.py /usr/local/bin/vllm_flags.py /usr/local/bin/vllm-lora-entrypoint.sh

if [[ ! -x /work/api-venv/bin/python ]]; then
  python3 -m venv /work/api-venv
  /work/api-venv/bin/pip install -q -r requirements.txt
fi

stop_vllm() {
  if [[ -f "$OUT/vllm.pid" ]]; then kill "$(cat "$OUT/vllm.pid")" 2>/dev/null || true; fi
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 3
}
wait_model() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:8001/v1/models" -o "$OUT/models.json" 2>/dev/null; then
      if grep -q "spider-sft" "$OUT/models.json"; then
        log "model-ready"
        return 0
      fi
    fi
    sleep 5
  done
  fail "vLLM did not list spider-sft"
}
start_vllm() {
  stop_vllm
  export ENABLE_PREFIX_CACHING="$1"
  env -u VLLM_API_KEY nohup /usr/local/bin/vllm-lora-entrypoint.sh > "$OUT/vllm_$1.log" 2>&1 &
  echo $! > "$OUT/vllm.pid"
  wait_model
}
stop_api() {
  if [[ -f "$OUT/api.pid" ]]; then kill "$(cat "$OUT/api.pid")" 2>/dev/null || true; fi
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 1
}
start_api() {
  stop_api
  export BACKEND=vllm
  export VLLM_BASE_URL=http://127.0.0.1:8001/v1
  export VLLM_MODEL_NAME=spider-sft
  export VLLM_API_KEY=EMPTY
  export VLLM_TIMEOUT_S=180
  export VLLM_READY_TIMEOUT_S=60
  export ADMISSION_ENABLED="$1"
  export MAX_CONCURRENT_REQUESTS="${2:-4}"
  export MAX_WAITING_REQUESTS="${3:-8}"
  export MAX_QUEUE_WAIT_MS="${4:-2000}"
  export TRACING_ENABLED=false
  export LOG_PROMPTS=false
  nohup /work/api-venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 > "$OUT/api.log" 2>&1 &
  echo $! > "$OUT/api.pid"
  local i
  for i in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8000/healthz -o "$OUT/healthz.json"; then
      log "api-ready admission=$1"
      return 0
    fi
    sleep 1
  done
  fail "API healthz"
}

run_cond() {
  local id="$1"; shift
  if [[ -f "$OUT/$id/summary.json" ]]; then
    log "resume-skip $id"
    return 0
  fi
  log "start $id"
  /work/api-venv/bin/python -m bench.run_phase4 \
    --mode full --base-url http://127.0.0.1:8000 --out "$OUT" \
    --experiment-id "$id" --requests "$REQ" --gpu "NVIDIA L4" "$@" \
    || fail "$id"
}

start_prometheus() {
  if curl -fsS http://127.0.0.1:9090/-/ready >/dev/null 2>&1; then
    log "prometheus already ready"
    return 0
  fi
  if [[ ! -x /tmp/prometheus/prometheus ]]; then
    curl -fsSL -o /tmp/prom.tgz "https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.linux-amd64.tar.gz"
    mkdir -p /tmp/prometheus
    tar -xzf /tmp/prom.tgz -C /tmp/prometheus --strip-components=1
  fi
  cat > /tmp/prometheus.yml <<'EOF'
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: llm-api
    static_configs:
      - targets: ["127.0.0.1:8000"]
  - job_name: vllm
    static_configs:
      - targets: ["127.0.0.1:8001"]
EOF
  nohup /tmp/prometheus/prometheus \
    --config.file=/tmp/prometheus.yml \
    --web.listen-address=127.0.0.1:9090 \
    --storage.tsdb.path=/work/promdata \
    --storage.tsdb.retention.time=24h \
    > "$OUT/prometheus.log" 2>&1 &
  echo $! > "$OUT/prometheus.pid"
  sleep 3
  curl -fsS http://127.0.0.1:9090/-/ready >/dev/null || fail "prometheus"
  log "prometheus ready on 127.0.0.1:9090"
}

log "phase4 begin requests=$REQ"
mkdir -p /work/promdata
start_prometheus
start_vllm false
start_api false
run_cond A-repeated-off --workload repeated-schema --concurrency 4 --stream --prefix-caching off --warmup 20
run_cond A-unrelated-off --workload unrelated-schema --concurrency 4 --stream --prefix-caching off --warmup 20
start_vllm true
start_api false
run_cond A-repeated-on --workload repeated-schema --concurrency 4 --stream --prefix-caching on --warmup 20
run_cond A-unrelated-on --workload unrelated-schema --concurrency 4 --stream --prefix-caching on --warmup 20

# Admission. Below / near / above are open-loop rates chosen from a short probe.
start_api false
if [[ ! -f "$OUT/B-probe/summary.json" ]]; then
  /work/api-venv/bin/python -m bench.run_phase4 --mode full --base-url http://127.0.0.1:8000 \
    --out "$OUT" --experiment-id B-probe --pattern open --workload short \
    --arrival-rate 2 --duration 20 --requests 40 --warmup 4 --admission off || fail "B-probe"
fi
run_cond B-below-off --pattern open --workload short --arrival-rate 1 --duration 180 --warmup 8 --admission off
run_cond B-above-off --pattern open --workload short --arrival-rate 6 --duration 60 --warmup 4 --admission off
start_api true 4 8 2000
run_cond B-below-on --pattern open --workload short --arrival-rate 1 --duration 180 --warmup 8 --admission on
run_cond B-near-on --pattern open --workload short --arrival-rate 2 --duration 180 --warmup 8 --admission on
run_cond B-above-on --pattern open --workload short --arrival-rate 6 --duration 60 --warmup 4 --admission on

start_api false
for c in 1 2 4 8 16; do
  run_cond "C-c${c}" --workload short --concurrency "$c" --warmup 8 --admission off
done
log "phase4 conditions complete"
