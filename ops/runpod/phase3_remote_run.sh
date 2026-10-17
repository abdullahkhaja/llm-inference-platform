#!/usr/bin/env bash
# Bounded Phase 3 checks on a pod that already has this repo and an L4.
# Does not create or terminate the pod.
set -u
ROOT="${ROOT:-/work/src}"
OUT="${OUT:-/work/out}"
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

fail() { echo "FAIL: $*" | tee -a "$OUT/status.log"; exit 1; }

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv > "$OUT/nvidia.csv" || fail "nvidia-smi"
cp -f ops/runpod/resolve_adapter.py ops/runpod/vllm_flags.py /usr/local/bin/
cp -f ops/runpod/entrypoint.sh /usr/local/bin/vllm-lora-entrypoint.sh
chmod +x /usr/local/bin/resolve_adapter.py /usr/local/bin/vllm_flags.py /usr/local/bin/vllm-lora-entrypoint.sh

if [[ ! -x /work/api-venv/bin/python ]]; then
  python3 -m venv /work/api-venv
  /work/api-venv/bin/pip install -q -r requirements.txt
fi

python3 - <<'PY' >> "$OUT/status.log"
import os
t = os.environ.get("HF_TOKEN") or ""
print("hf_token_len", len(t))
print("hf_token_placeholder", t.startswith("{{"))
PY

stop_vllm() {
  if [[ -f "$OUT/vllm.pid" ]]; then
    kill "$(cat "$OUT/vllm.pid")" 2>/dev/null || true
  fi
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 3
}

wait_model() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:8001/v1/models" -o "$OUT/models.json" 2>/dev/null; then
      if grep -q "spider-sft" "$OUT/models.json"; then
        echo "model-ready" | tee -a "$OUT/status.log"
        return 0
      fi
    fi
    sleep 5
  done
  fail "vLLM did not list spider-sft"
}

start_vllm() {
  local flag="$1"
  stop_vllm
  export ENABLE_PREFIX_CACHING="$flag"
  # VLLM_API_KEY is an API setting. If it leaks into this process, vLLM
  # requires a bearer token and the localhost readiness check gets 401.
  env -u VLLM_API_KEY nohup /usr/local/bin/vllm-lora-entrypoint.sh > "$OUT/vllm_${flag}.log" 2>&1 &
  echo $! > "$OUT/vllm.pid"
  wait_model
}

stop_api() {
  if [[ -f "$OUT/api.pid" ]]; then
    kill "$(cat "$OUT/api.pid")" 2>/dev/null || true
  fi
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 1
}

start_api() {
  stop_api
  local admission="$1"
  local tracing="$2"
  export BACKEND=vllm
  export VLLM_BASE_URL=http://127.0.0.1:8001/v1
  export VLLM_MODEL_NAME=spider-sft
  export VLLM_API_KEY=EMPTY
  export VLLM_TIMEOUT_S=180
  export VLLM_READY_TIMEOUT_S=60
  export ADMISSION_ENABLED="$admission"
  export MAX_CONCURRENT_REQUESTS="${3:-4}"
  export MAX_WAITING_REQUESTS="${4:-8}"
  export MAX_QUEUE_WAIT_MS="${5:-2000}"
  export TRACING_ENABLED="$tracing"
  export TRACE_LOG_PATH="$OUT/traces.jsonl"
  export REQUEST_LOG_PATH="$OUT/requests.jsonl"
  export LOG_PROMPTS=false
  nohup /work/api-venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 > "$OUT/api.log" 2>&1 &
  echo $! > "$OUT/api.pid"
  local i
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if curl -fsS http://127.0.0.1:8000/healthz -o "$OUT/healthz.json"; then
      echo "api-ready admission=$admission tracing=$tracing" | tee -a "$OUT/status.log"
      return 0
    fi
    sleep 1
  done
  fail "API healthz"
}

BODY='{"question":"List employee names.","schema":"CREATE TABLE employees (id INTEGER, name TEXT);","max_new_tokens":32}'

echo "=== cache off ===" | tee -a "$OUT/status.log"
if [[ "${SKIP_CACHE_OFF:-}" == "1" ]]; then
  echo "skip cache off; artifacts already collected" | tee -a "$OUT/status.log"
else
start_vllm false
start_api false false
curl -fsS http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d "$BODY" -o "$OUT/generate_cache_off.json" || fail "generate cache off"
curl -fsS -N --max-time 120 http://127.0.0.1:8000/generate/stream -H 'Content-Type: application/json' -d "$BODY" -o "$OUT/stream_cache_off.txt" || fail "stream cache off"
/work/api-venv/bin/python -m bench.load_test --base-url http://127.0.0.1:8000 --pattern closed --concurrency 1 --requests 2 --workload repeated-schema --label auto --out-dir "$OUT/warmup_off" --experiment-id phase3-warmup-off --prefix-caching false --gpu "NVIDIA L4"
/work/api-venv/bin/python -m bench.load_test --base-url http://127.0.0.1:8000 --pattern closed --concurrency 1 --requests 4 --workload repeated-schema --label auto --out-dir "$OUT/bench" --experiment-id phase3-cache-off-repeated --prefix-caching false --gpu "NVIDIA L4" --latency-objective 30
/work/api-venv/bin/python -m bench.load_test --base-url http://127.0.0.1:8000 --pattern closed --concurrency 1 --requests 4 --workload unrelated-schema --label auto --out-dir "$OUT/bench" --experiment-id phase3-cache-off-unrelated --prefix-caching false --gpu "NVIDIA L4"
curl -fsS http://127.0.0.1:8001/metrics -o "$OUT/vllm_metrics_cache_off.prom" || fail "metrics cache off"
fi

echo "=== cache on ===" | tee -a "$OUT/status.log"
start_vllm true
start_api false false
/work/api-venv/bin/python -m bench.load_test --base-url http://127.0.0.1:8000 --pattern closed --concurrency 1 --requests 2 --workload repeated-schema --label auto --out-dir "$OUT/warmup_on" --experiment-id phase3-warmup-on --prefix-caching true --gpu "NVIDIA L4"
/work/api-venv/bin/python -m bench.load_test --base-url http://127.0.0.1:8000 --pattern closed --concurrency 1 --requests 4 --workload repeated-schema --label auto --out-dir "$OUT/bench" --experiment-id phase3-cache-on-repeated --prefix-caching true --gpu "NVIDIA L4" --latency-objective 30
/work/api-venv/bin/python -m bench.load_test --base-url http://127.0.0.1:8000 --pattern closed --concurrency 1 --requests 4 --workload unrelated-schema --label auto --out-dir "$OUT/bench" --experiment-id phase3-cache-on-unrelated --prefix-caching true --gpu "NVIDIA L4"
curl -fsS http://127.0.0.1:8001/metrics -o "$OUT/vllm_metrics_cache_on.prom" || fail "metrics cache on"

echo "=== admission ===" | tee -a "$OUT/status.log"
start_api false false
/work/api-venv/bin/python -m bench.load_test --base-url http://127.0.0.1:8000 --pattern closed --concurrency 2 --requests 4 --workload short --label auto --out-dir "$OUT/bench" --experiment-id phase3-admission-off --admission false --gpu "NVIDIA L4"
/work/api-venv/bin/python -m bench.load_test --base-url http://127.0.0.1:8000 --pattern open --arrival-rate 1 --duration 4 --concurrency 1 --requests 1 --workload short --label auto --out-dir "$OUT/bench" --experiment-id phase3-open-loop --gpu "NVIDIA L4"
start_api true false 1 0 200
/work/api-venv/bin/python - <<'PY'
import json, urllib.request
from concurrent.futures import ThreadPoolExecutor
body = json.dumps({"question":"List employee names.","schema":"CREATE TABLE employees (id INTEGER, name TEXT);","max_new_tokens":16}).encode()
def one():
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=body, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, resp.headers.get("Retry-After")
    except Exception as exc:
        status = getattr(exc, "code", None)
        headers = getattr(exc, "headers", None)
        retry = headers.get("Retry-After") if headers else None
        return status, retry
with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(lambda _: one(), range(2)))
open("/work/out/admission_overload.json","w").write(json.dumps({"results": results}))
print(results)
PY

echo "=== tracing ===" | tee -a "$OUT/status.log"
rm -f "$OUT/traces.jsonl"
start_api true true 4 8 2000
curl -fsS http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d "$BODY" -o "$OUT/generate_traced.json"
curl -fsS -N --max-time 120 http://127.0.0.1:8000/generate/stream -H 'Content-Type: application/json' -d "$BODY" -o "$OUT/stream_traced.txt"
/work/api-venv/bin/python - <<'PY'
import json
from pathlib import Path
names=set()
traces=set()
for line in Path("/work/out/traces.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec=json.loads(line)
    blob=json.dumps(rec)
    if "hf_" in blob or "CREATE TABLE" in blob:
        raise SystemExit("trace leaked prompt or token")
    names.add(rec["name"])
    traces.add(rec["trace_id"])
Path("/work/out/trace_names.json").write_text(json.dumps({"names":sorted(names),"trace_count":len(traces)}, indent=2), encoding="utf-8")
print(sorted(names))
PY

if ! command -v prometheus >/dev/null 2>&1 && [[ ! -x /tmp/prometheus/prometheus ]]; then
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
pkill -f "/tmp/prometheus/prometheus" 2>/dev/null || true
nohup /tmp/prometheus/prometheus --config.file=/tmp/prometheus.yml --web.listen-address=127.0.0.1:9090 --storage.tsdb.path=/tmp/promdata > "$OUT/prometheus.log" 2>&1 &
sleep 8
/work/api-venv/bin/python - <<'PY'
import json, urllib.parse, urllib.request
from pathlib import Path
dash = json.loads(Path("/work/src/ops/grafana/dashboards/phase3_serving.json").read_text(encoding="utf-8"))
exprs=[]
for panel in dash["panels"]:
    for target in panel.get("targets") or []:
        exprs.append((panel["title"], target["expr"]))
rows=[]
for title, expr in exprs:
    url="http://127.0.0.1:9090/api/v1/query?"+urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(url, timeout=20) as resp:
        data=json.loads(resp.read().decode())
    result=data.get("data",{}).get("result") or []
    rows.append({"panel": title, "expr": expr, "series": len(result), "status": data.get("status")})
Path("/work/out/grafana_query_check.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("queries", len(rows), "with_series", sum(1 for r in rows if r["series"]))
PY
curl -fsS http://127.0.0.1:9090/api/v1/targets -o "$OUT/prometheus_targets.json" || true
echo "remote-run-complete" | tee -a "$OUT/status.log"
