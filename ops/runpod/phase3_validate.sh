#!/usr/bin/env bash
# Optional Phase 3 check on an already-rented GPU pod.
# Does not create or start a RunPod pod.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/ops/runpod/smoke_results}"
mkdir -p "$OUT"

: "${VLLM_BASE:=http://127.0.0.1:8001}"
: "${API_BASE:=http://127.0.0.1:8000}"
: "${VALIDATE_REQUESTS:=4}"

echo "Phase 3 validation (bounded). This script does not rent a GPU."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Run this on the GPU pod, not a CPU dev machine." >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee "$OUT/phase3_nvidia.csv"

echo "--- vLLM /metrics (on-pod; same port as the API) ---"
curl -fsS "$VLLM_BASE/metrics" -o "$OUT/vllm_metrics.prom"
# Names from app/vllm_metrics_catalog.py required_for_phase3_check().
for name in vllm:num_requests_running vllm:num_requests_waiting vllm:gpu_cache_usage_perc; do
  if ! grep -q "$name" "$OUT/vllm_metrics.prom"; then
    echo "MISSING $name" >&2
    exit 1
  fi
done
echo "required vLLM series present"

echo "--- prefix caching flag is explicit in the process args ---"
if ! grep -q "enable-prefix-caching" "$OUT/vllm_metrics.prom" && ! tr '\0' '\n' < /proc/1/cmdline 2>/dev/null | grep -q "prefix-caching"; then
  echo "NOTE: confirm the vLLM command line contains --enable-prefix-caching or --no-enable-prefix-caching"
fi

echo "--- API health ---"
curl -fsS "$API_BASE/healthz" | tee "$OUT/phase3_healthz.json"

echo "--- tiny closed-loop sample (not a benchmark) ---"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m bench.load_test \
  --base-url "$API_BASE" \
  --concurrency 1,2 \
  --requests "$VALIDATE_REQUESTS" \
  --workload repeated-schema \
  --out-dir "$OUT/phase3_bench" \
  --label auto \
  --experiment-id phase3-validate

echo "Copy $OUT off the pod, log the session in ops/cost_tracker.md, then terminate the pod."
echo "cleanup.sh stops containers only. It does not stop RunPod billing."
