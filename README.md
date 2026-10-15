# LLM Inference Serving & Benchmarking Platform

Combined project for **Spider text-to-SQL fine-tuning** (already completed) and a modular **inference serving + observability** stack.

Phase 1 ships a **CPU mock backend** so the FastAPI server, JSONL logging, Prometheus metrics, Grafana dashboards, Docker Compose, and load tests can be developed at **$0 GPU spend**. No Llama weights, vLLM, or RunPod resources are used in this phase.

---

## Original fine-tuning results (model quality — not serving benchmarks)

These numbers come from the prior Llama 3.1 8B Instruct QLoRA experiment on Spider 1.0 (1,034 eval examples). They measure **SQL accuracy**, not tokens/sec or latency:

| Metric | Base → Fine-tuned |
| --- | --- |
| Exact-match accuracy | 40.5% → **73.3%** |
| Execution accuracy | 49.9% → **75.7%** |

Phase 1 mock benchmark RPS / latency figures are **simulated** and must not be compared to these accuracy results or treated as real model throughput.

---

## Architecture

| Area | Path | Responsibility |
| --- | --- | --- |
| Training | `training/` | QLoRA fine-tuning + Spider evaluation |
| Serving | `app/` | FastAPI inference API, metrics, logging |
| Backends | `app/backends/` | `MockBackend` now; `vllm` stub for Phase 2 |
| Benchmarks | `bench/` | Async load tests, CSV export, plots |
| Monitoring | `ops/` | Prometheus scrape config + Grafana provisioning |

**Hard separation:** the inference server never imports or launches training. Training-only deps (TRL, PEFT, bitsandbytes) stay out of the Phase 1 Docker image.

```
training/finetune_spider.py  →  adapter artifacts (local / private HF)
                                      ↓  (Phase 2+)
app/ (vLLM backend)  ←── MODEL_PATH + ADAPTER_PATH
```

---

## Repository layout

```
training/
  finetune_spider.py          # Converted from Llama31_Spider_QLoRA_GRPO_Colab.ipynb
  requirements-training.txt   # Separate GPU training env
app/
  main.py                     # FastAPI routes
  config.py                   # Env-based settings
  schemas.py                  # Pydantic models
  serving.py                  # Prompt build + inference service
  metrics.py                  # Prometheus instrumentation
  logging_config.py           # JSONL request logs
  backends/
    base.py                   # Backend interface
    mock.py                   # Serialized CPU mock (Phase 1)
bench/
  load_test.py
  analyze.py
  results/
ops/
  prometheus.yml
  grafana/
tests/
Dockerfile
docker-compose.yml
requirements.txt              # Phase 1 serving only
```

---

## Training script

`training/finetune_spider.py` is a faithful conversion of the Colab notebook used for the Spider QLoRA run. It preserves model config, dataset validation, LoRA hyperparameters (`r=16`, `alpha=32`, …), SFT settings, adapter save/upload logic, official evaluation hooks, and optional GRPO.

**Training uses a separate environment** (see `training/requirements-training.txt`). Do not `pip install` those packages into the Phase 1 serving venv/image.

Example (GPU machine, after preparing Spider data):

```bash
pip install -r training/requirements-training.txt
python training/finetune_spider.py \
  --spider-zip /path/to/spider.zip \
  --spider-eval-dir /path/to/spider \
  --work-dir ./spider_project
```

---

## Phase 1 setup (local / Docker)

### Option A — Docker Compose (recommended)

```bash
cp .env.example .env
docker compose up --build -d
```

| Service | URL |
| --- | --- |
| FastAPI | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin by default) |

```bash
docker compose ps
curl http://localhost:8000/healthz
```

### Option B — local Python

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API examples

**Health**

```bash
curl http://localhost:8000/healthz
```

**Generate (simulated SQL)**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "x-request-id: demo-1" \
  -d '{
    "question": "List all employees earning more than 50000.",
    "schema": "CREATE TABLE employees (id INTEGER, name TEXT, salary INTEGER);",
    "max_new_tokens": 128
  }'
```

Responses include `"simulated": true` and SQL text marked `[SIMULATED MOCK RESPONSE]`.

**Prometheus metrics**

```bash
curl http://localhost:8000/metrics
```

---

## Benchmarks

Default is a **smoke** workload. p99 from small samples is unreliable — use `--requests 1000+` for stronger tail-latency estimates.

```bash
# Smoke (default)
python -m bench.load_test --base-url http://127.0.0.1:8000 --analyze

# Stronger tail-latency signal
python -m bench.load_test \
  --base-url http://127.0.0.1:8000 \
  --concurrency 1,2,4,8,16 \
  --requests 1000 \
  --max-new-tokens 64 \
  --analyze
```

Outputs land in `bench/results/` as CSVs, JSON summaries, and a latency-vs-concurrency plot — all prefixed / labeled **simulated**.

---

## Prometheus & Grafana

1. Start the stack with `docker compose up --build -d`.
2. Open Grafana at http://localhost:3000.
3. Open dashboard **Phase 1 — Simulated Mock Serving** (folder LLM).
4. Run a load test so panels populate (RPS, simulated tok/s, p50/p95/p99, queue depth, queue wait, inference duration, error rate).

Datasource and dashboard are auto-provisioned from `ops/grafana/`.

---

## Phase 1 limitations (simulated inference)

- No real LLM weights are loaded; SQL is a labeled placeholder.
- Timing is a synthetic prefill + per-token model behind a **single serialized worker**.
- No dynamic batching, KV-cache, or GPU utilization telemetry.
- Mock tokens/sec and latency are for pipeline validation only.

---

## Future phases (not implemented)

| Phase | Scope |
| --- | --- |
| **2** | Load QLoRA adapter via vLLM / GPU; streaming |
| **3** | Prefix caching, admission control, deeper latency breakdowns |
| **4** | RunPod L4 (or similar) real benchmarks; **total cloud GPU spend ≤ $1** |

Phase 1 GPU expenditure: **$0**. Prepare images/scripts before any paid pod.

---

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

---

## What to provide later for real model serving (Phase 2)

1. Path (or private HF repo) to the **QLoRA adapter** from `training/finetune_spider.py` (`sft/final`).
2. Hugging Face access to **meta-llama/Llama-3.1-8B-Instruct** (gated).
3. Prefer a GPU with enough VRAM for Llama 3.1 8B (quantized); the 8 GB RTX 4060 may need an alternate quant / merge path.
4. Confirm whether to serve adapter-on-base via vLLM LoRA or a merged checkpoint.

Set `BACKEND=vllm`, `MODEL_PATH=...`, and `ADAPTER_PATH=...` when Phase 2 is implemented — no redesign of HTTP routes should be required.
