#!/usr/bin/env bash
# Short real-GPU smoke test (few requests). Not a benchmark sweep.
# Saves artifacts under ops/runpod/smoke_results/ (copy off-pod before terminate).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/ops/runpod/smoke_results"
mkdir -p "$OUT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUT/smoke_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

: "${VLLM_MODEL_NAME:=spider-sft}"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8001/v1}"

echo "=== Phase 2 smoke test @ $STAMP ==="
echo "API_BASE=$API_BASE VLLM_BASE=$VLLM_BASE model=$VLLM_MODEL_NAME"

echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv || true

echo "--- /v1/models ---"
curl -fsS "$VLLM_BASE/models" | tee "$OUT/models_${STAMP}.json"
grep -q "$VLLM_MODEL_NAME" "$OUT/models_${STAMP}.json"

echo "--- non-stream chat/completions (direct vLLM) ---"
curl -fsS "$VLLM_BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$VLLM_MODEL_NAME\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You translate a question into one read-only SQLite SELECT query. Return SQL only, no markdown or explanation.\"},
      {\"role\": \"user\", \"content\": \"Database schema:\\nCREATE TABLE employees (id INTEGER, name TEXT, salary INTEGER);\\n\\nQuestion: List employees earning more than 50000.\\nSQL:\"}
    ],
    \"max_tokens\": 64,
    \"temperature\": 0
  }" | tee "$OUT/direct_generate_${STAMP}.json"

python - <<'PY' "$OUT/direct_generate_${STAMP}.json"
import json, sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
text = data["choices"][0]["message"]["content"]
usage = data.get("usage") or {}
assert text and text.strip(), "empty model output"
assert "usage" in data, "missing usage"
assert int(usage.get("completion_tokens") or 0) > 0
print("direct SQL preview:", text.strip()[:200])
print("usage:", usage)
PY

echo "--- FastAPI /healthz ---"
curl -fsS "$API_BASE/healthz" | tee "$OUT/healthz_${STAMP}.json"
python - <<'PY' "$OUT/healthz_${STAMP}.json"
import json, sys
h = json.load(open(sys.argv[1], encoding="utf-8"))
assert h.get("ready") is True
assert h.get("simulated") is False
assert h.get("backend") == "vllm"
print("health ok")
PY

echo "--- FastAPI /generate ---"
curl -fsS "$API_BASE/generate" \
  -H "Content-Type: application/json" \
  -H "x-request-id: smoke-$STAMP" \
  -d '{
    "question": "List employees earning more than 50000.",
    "schema": "CREATE TABLE employees (id INTEGER, name TEXT, salary INTEGER);",
    "max_new_tokens": 64
  }' | tee "$OUT/generate_${STAMP}.json"

python - <<'PY' "$OUT/generate_${STAMP}.json"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("simulated") is False, d
assert d.get("backend") == "vllm"
assert d.get("output_tokens", 0) > 0
assert "SIMULATED" not in (d.get("sql") or "")
print("generate ok; sql=", (d.get("sql") or "")[:200])
PY

echo "--- FastAPI /generate/stream (short) ---"
curl -fsS -N "$API_BASE/generate/stream" \
  -H "Content-Type: application/json" \
  -H "x-request-id: smoke-stream-$STAMP" \
  -d '{
    "question": "List employee names.",
    "schema": "CREATE TABLE employees (id INTEGER, name TEXT);",
    "max_new_tokens": 32
  }' | tee "$OUT/stream_${STAMP}.sse"

grep -q "event: token" "$OUT/stream_${STAMP}.sse"
grep -q "event: done" "$OUT/stream_${STAMP}.sse"

echo "=== SMOKE PASS ==="
echo "Artifacts in $OUT — copy them off the pod BEFORE terminate."
echo "Next: ./ops/runpod/cleanup.sh then TERMINATE the RunPod pod."
