# RunPod deployment

Runbook for serving Llama 3.1 8B Instruct plus the Spider SFT adapter on a RunPod GPU pod. Scripts in this directory assume you already rented the machine. They do not create a pod.

Session costs: [`../cost_tracker.md`](../cost_tracker.md).

## Verified smoke (15 October 2026)

| Item | Value |
| --- | --- |
| GPU | NVIDIA L4, Secure Cloud, EU-RO-1 |
| Image | `vllm/vllm-openai:v0.8.5` ([`Dockerfile.gpu`](../../Dockerfile.gpu)) |
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| LoRA name | `spider-sft` |
| Context | `max_model_len=2048`, `gpu_memory_utilization=0.85`, `max_lora_rank=16` |
| Result | `/v1/models` listed `spider-sft`. Local FastAPI (`BACKEND=vllm`) returned SQL with `"simulated": false`. |

That run used a handful of short requests from a local FastAPI process against the pod proxy. It is not a throughput or tail-latency benchmark, and it was not an on-pod `docker compose` smoke.

## 1. Hardware and image

Prefer an NVIDIA L4 (24 GB). Confirm the live hourly price in the RunPod console before you start the pod.

The GPU image is the official vLLM image plus [`resolve_adapter.py`](resolve_adapter.py) and [`entrypoint.sh`](entrypoint.sh). [`docker-compose.gpu.yml`](../../docker-compose.gpu.yml) publishes vLLM on port **8001** and an optional FastAPI service on port **8000**.

Local mock development uses [`docker-compose.yml`](../../docker-compose.yml) and `BACKEND=mock`. Do not start the GPU compose file on a machine without an NVIDIA GPU.

## 2. Model and adapter

| Variable | Role |
| --- | --- |
| `MODEL_PATH` | Base model id. Default `meta-llama/Llama-3.1-8B-Instruct` |
| `ADAPTER_PATH` | Hugging Face repo id that contains the adapter |
| `ADAPTER_SUBFOLDER` | Default `sft/final` |
| `VLLM_MODEL_NAME` | LoRA module name. Default `spider-sft` |
| `HF_TOKEN` | Read token for the gated base model and the adapter repo |

`entrypoint.sh` downloads the subfolder to `/models/adapters/spider-sft` when `adapter_config.json` is not already there, then starts:

```text
python3 -m vllm.entrypoints.openai.api_server
  --enable-lora
  --lora-modules spider-sft=/models/adapters/spider-sft
  --host 0.0.0.0 --port 8001
```

The stock `vllm/vllm-openai` image provides `python3`, not `python`. The entrypoint selects `python3` first.

## 3. Hugging Face access

On the pod, before setup:

```bash
export HF_TOKEN=...
export ADAPTER_PATH=your-hf-user/your-adapter-repo
export ADAPTER_SUBFOLDER=sft/final
export MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export VLLM_MODEL_NAME=spider-sft
```

The token’s Hugging Face account must be allowed to read Llama 3.1 8B Instruct and the adapter repo. Do not commit the token.

## 4. Deploy on the pod

1. Rent the GPU and note the hourly rate.
2. Clone this repository on the pod. Docker with the NVIDIA runtime must be available.
3. Export the variables above.
4. Build and start:

```bash
./ops/runpod/setup.sh
./ops/runpod/start.sh
```

`setup.sh` checks `nvidia-smi`, Docker, `HF_TOKEN`, and `ADAPTER_PATH`, then builds the GPU compose stack.

`start.sh` runs `docker compose -f docker-compose.gpu.yml up -d` and polls `http://127.0.0.1:8001/v1/models` until `VLLM_READY_TIMEOUT_S` (default 600 seconds). It exits if `spider-sft` is missing from the model list.

Expose **8001/http** on the pod if a machine outside the pod will call vLLM. The proxy form is:

```text
https://<pod-id>-8001.proxy.runpod.net/v1
```

## 5. Smoke test

On the pod, with both compose services up (API on port 8000, vLLM on port 8001):

```bash
./ops/runpod/smoke_test.sh
```

Defaults: `API_BASE=http://127.0.0.1:8000`, `VLLM_BASE=http://127.0.0.1:8001/v1`.

The script checks:

- `nvidia-smi`
- `/v1/models` contains `spider-sft`
- one non-streaming `chat/completions` call returns text and `completion_tokens > 0`
- FastAPI `/healthz` is `ready`, `backend=vllm`, `simulated=false`
- `POST /generate` returns SQL with `simulated=false` and `output_tokens > 0`
- `POST /generate/stream` emits `event: token` and `event: done`

Artifacts land in `ops/runpod/smoke_results/` (gitignored except `.gitkeep`). Copy that directory off the pod before you terminate it.

## 6. Local FastAPI against the pod

Leave FastAPI off the pod and point a local process at the proxy:

```bash
# .env
BACKEND=vllm
VLLM_BASE_URL=https://<pod-id>-8001.proxy.runpod.net/v1
VLLM_MODEL_NAME=spider-sft
VLLM_READY_TIMEOUT_S=60
VLLM_TIMEOUT_S=180
```

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/healthz
```

`/healthz` stays not-ready until vLLM lists `VLLM_MODEL_NAME`.

## 7. Shutdown

`cleanup.sh` only stops containers on the machine. The RunPod pod keeps billing until you stop or terminate it in the console. Copy `smoke_results/` first. Persistent or network volumes can keep billing after the GPU is released. Confirm in the RunPod UI that nothing you intended to keep is still running, then add a row to [`../cost_tracker.md`](../cost_tracker.md).

```bash
./ops/runpod/cleanup.sh
```

## 8. Troubleshooting

Problems hit during the Phase 2 smoke, and the fix that worked.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Pod create returns `402 Payment Required` | Account balance too low to place the GPU | Add credits, then create again. No pod is billed when create is rejected. |
| Container exits immediately: `python: command not found`. Proxy stays 404. | `vllm/vllm-openai:v0.8.5` has `python3` only. A boot script called `python`. | Call `python3`, or `command -v python3`. [`entrypoint.sh`](entrypoint.sh) does this. |
| Proxy returns `502` for several minutes after the container starts | Base model is still downloading or loading. Nothing is listening on port 8001 yet. | Poll `/v1/models`. Read `docker compose -f docker-compose.gpu.yml logs vllm` if the ready timeout in `start.sh` fires. |
