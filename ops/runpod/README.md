# RunPod deployment

Runbook for serving Llama 3.1 8B Instruct plus the Spider SFT adapter on a RunPod GPU pod. The scripts assume the machine already exists. They do not create a pod.

Session costs: [`../cost_tracker.md`](../cost_tracker.md).

## Verified environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA L4, 24 GB, Secure Cloud, EU-RO-1 |
| Quoted rate | $0.59/h at pod create. Check the live price before renting again |
| Image | `vllm/vllm-openai:v0.8.5` |
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| LoRA name | `spider-sft`, files from subfolder `sft/final` |
| Context | `max_model_len=2048`, `gpu_memory_utilization=0.85`, `max_lora_rank=16` |
| Decoding | temperature 0 |
| Phase 2 smoke | `/v1/models` listed `spider-sft`. FastAPI returned SQL with `"simulated": false` |
| Phase 4 | Same image and GPU. Benchmark request records are real (`simulated: false`) |

The Phase 2 smoke was a handful of short requests. It is not the Phase 4 throughput result.

The base model is gated and the adapter repo may be private. The pod needs `HF_TOKEN` with read access. The token’s Hugging Face account must have accepted the Llama 3.1 license. Do not commit the token.

## Two ways to start vLLM

**What Phase 4 actually ran.** The stock vLLM image has no Docker daemon. vLLM, FastAPI, and Prometheus were processes inside that container. Grafana and the image renderer were started later, also as processes, and only after the measured requests finished. Do not treat Docker-in-Docker as the Phase 4 setup.

**Optional Compose path.** [`docker-compose.gpu.yml`](../../docker-compose.gpu.yml) is for a host that already has Docker and an NVIDIA GPU. [`setup.sh`](setup.sh) and [`start.sh`](start.sh) build that stack. Do not start it on a machine without an NVIDIA GPU. Local mock development uses [`docker-compose.yml`](../../docker-compose.yml) and `BACKEND=mock`.

## Model process

[`entrypoint.sh`](entrypoint.sh) downloads the adapter to `/models/adapters/spider-sft` when `adapter_config.json` is missing, then starts vLLM. A raw repo id is not passed to `--lora-modules`.

```text
python3 -m vllm.entrypoints.openai.api_server
  --enable-lora
  --lora-modules spider-sft=/models/adapters/spider-sft
  --enable-prefix-caching | --no-enable-prefix-caching
  --host 0.0.0.0 --port 8001
```

`ENABLE_PREFIX_CACHING` selects the flag. vLLM 0.8.5 V1 turns prefix caching on when the flag is omitted, so the entrypoint always passes one of the two. Restart vLLM to switch. Unset `VLLM_API_KEY` in the vLLM process. If that variable is `EMPTY`, vLLM 0.8.5 requires a bearer token and the localhost readiness check returns 401.

The image provides `python3`, not `python`.

| Variable | Role |
| --- | --- |
| `MODEL_PATH` | Base model id. Default `meta-llama/Llama-3.1-8B-Instruct` |
| `ADAPTER_PATH` | Hugging Face repo id that contains the adapter |
| `ADAPTER_SUBFOLDER` | Default `sft/final` |
| `VLLM_MODEL_NAME` | LoRA module name. Default `spider-sft` |
| `HF_TOKEN` | Read token. Never commit it |

```bash
export HF_TOKEN=...
export ADAPTER_PATH=your-hf-user/your-adapter-repo
export ADAPTER_SUBFOLDER=sft/final
export MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export VLLM_MODEL_NAME=spider-sft
```

FastAPI, when used, listens on `127.0.0.1:8000` with `BACKEND=vllm` and `VLLM_BASE_URL=http://127.0.0.1:8001/v1`. Admission defaults: off, then 4 active, 8 waiting, 2,000 ms. Tracing defaults off.

Direct TCP SSH needs port `22/tcp` and a registered public key. The `ssh.runpod.io` proxy is not the port-forward path. Do not publish Grafana (`3000`), the renderer (`8081`), or Prometheus (`9090`).

## Phase 4 benchmark

[`phase4_remote.sh`](phase4_remote.sh) does not create or terminate a pod. It expects the repo at `/work/src`, an L4, and `HF_TOKEN` plus `ADAPTER_PATH` in the environment. It installs a virtualenv for the API, starts Prometheus 2.54.1 on `127.0.0.1:9090`, and runs vLLM and FastAPI as processes.

`PHASE4_REQUESTS` defaults to 1000. A finished condition is skipped when `summary.json` already exists.

```bash
export ROOT=/work/src
export OUT=/work/phase4
export PHASE4_REQUESTS=1000
bash ops/runpod/phase4_remote.sh
```

The script runs, in order:

1. Prefix cache off, then on. Each flag is a vLLM restart. Streaming closed loop, concurrency 4, repeated-schema and unrelated-schema.
2. Admission off and on. Open loop at 1 req/s and 6 req/s, plus a 2 req/s admission-on run and a 20 s probe.
3. Closed-loop concurrency 1, 2, 4, 8, and 16 on the `short` workload. Concurrency 32 is not in the script.

From a laptop, against an API that is already up, these match [`bench/run_phase4.py`](../../bench/run_phase4.py):

```bash
python -m bench.experiments --out bench/results/phase3_experiment_plan.json
python -m bench.run_phase4 --mode dry-run --experiment-id A-repeated-off --workload repeated-schema --concurrency 4 --stream
python -m bench.run_phase4 --mode smoke --base-url http://127.0.0.1:8000 --experiment-id smoke --workload short --concurrency 1 --requests 4
python -m bench.run_phase4 --mode full --base-url http://127.0.0.1:8000 --experiment-id C-c4 --workload short --concurrency 4 --requests 1000 --out bench/results/phase4
```

Dry-run prints the plan and does not open a socket. Smoke caps the run at 4 requests. Full mode writes `bench/results/phase4/<experiment-id>/requests.csv` and `summary.json`, including `measured_start` and `ended_at`. Those directories are gitignored.

One condition at a time is the supported interface. The shell script is the campaign.

## Grafana on the pod

[`phase4_grafana.sh`](phase4_grafana.sh) runs after the measured requests, not during them. It does not start vLLM.

Verified on the Phase 4 pod:

| Piece | Version |
| --- | --- |
| Grafana | 12.1.0, bound to `127.0.0.1:3000` |
| Image renderer | `grafana-image-renderer` v5.12.7, `127.0.0.1:8081` |
| Browser | Google Chrome, `--disable-gpu --use-gl=swiftshader` |
| Datasource | `http://127.0.0.1:9090` |
| Dashboard | `ops/grafana/dashboards/phase3_serving.json`, uid `phase3-serving` |

Grafana is configured with `GF_RENDERING_SERVER_URL=http://127.0.0.1:8081/render`, `GF_RENDERING_CALLBACK_URL=http://127.0.0.1:3000/`, and a matching renderer token generated on the pod. Authentication stays on. The script refuses to start the renderer when available memory is under 4 GB.

Export uses the recorded measurement windows, not “last 5 minutes”:

```bash
export GRAFANA_URL=http://127.0.0.1:3000
export GRAFANA_USER=admin
export GRAFANA_PASSWORD=...   # pod-local file, not committed
export PHASE4_RESULTS=/work/phase4
export PHASE4_SHOTS=/work/src/docs/images/phase4
python -m bench.export_phase4_shots
```

[`bench/capture_grafana.py`](../../bench/capture_grafana.py) calls Grafana’s `/render` endpoint. The PNG must be a real image. The first export of this session was Grafana’s “no image renderer” panel, because Grafana had been started before the renderer. Restart Grafana after the renderer is up, then export again. Committed PNGs are the second export.

## SSH to Grafana

Use direct TCP SSH, not the RunPod HTTP proxy and not `ssh.runpod.io`.

```bash
ssh -N -L 127.0.0.1:13000:127.0.0.1:3000 -p <public-port> -i <private-key> root@<public-host>
```

Then open `http://localhost:13000`. The dashboard uid path is `http://localhost:13000/d/phase3-serving`. The forward listens on localhost only. Host-key checking stays on. The private key and the Grafana password stay out of git.

The URL works only while the pod and the tunnel are up. After terminate, use the PNGs in [`docs/images/phase4/`](../../docs/images/phase4/).

## Artifacts and shutdown

Copy results off the pod before terminate:

- `/work/phase4/<experiment-id>/requests.csv` and `summary.json`
- `/work/src/docs/images/phase4/*.png` and `manifest.json`

`cleanup.sh` only stops Compose services. It does not stop billing. A pod that is still running, including one where vLLM has been killed, is still a rental. Terminate it in the RunPod API or console and confirm it is gone. This project did not attach a network volume. Container disk is removed on terminate.

Add a row to [`../cost_tracker.md`](../cost_tracker.md). Mark the dollars as an estimate unless an invoice posted the charge.

## Optional Compose smoke

On a Docker-capable GPU host, after `setup.sh` and `start.sh`:

```bash
./ops/runpod/smoke_test.sh
```

The script checks `nvidia-smi`, `spider-sft` on `/v1/models`, one chat completion, and FastAPI `simulated: false`. Artifacts go to `ops/runpod/smoke_results/`.

A smaller Phase 3 check, also without creating a pod:

```bash
export VLLM_BASE=http://127.0.0.1:8001
export API_BASE=http://127.0.0.1:8000
export VALIDATE_REQUESTS=4
./ops/runpod/phase3_validate.sh
```

`VALIDATE_REQUESTS` defaults to 4. It is a tooling smoke, not the Phase 4 campaign.

If something outside the pod must call vLLM, expose `8001/http`. The proxy form is `https://<pod-id>-8001.proxy.runpod.net/v1`. A proxy to 8001 also exposes `/metrics` on this vLLM build.

Point a local API at a running pod without loading weights locally:

```bash
# .env
BACKEND=vllm
VLLM_BASE_URL=https://<pod-id>-8001.proxy.runpod.net/v1
VLLM_MODEL_NAME=spider-sft
```

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/healthz
```

## Troubleshooting

Problems hit during the Phase 2 smoke, and the fix that worked.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Pod create returns `402 Payment Required` | Account balance too low to place the GPU | Add credits, then create again. No pod is billed when create is rejected |
| Container exits immediately: `python: command not found` | `vllm/vllm-openai:v0.8.5` has `python3` only | Call `python3`. [`entrypoint.sh`](entrypoint.sh) does this |
| Proxy returns `502` for several minutes | The base model is still downloading. Nothing is listening on 8001 yet | Poll `/v1/models` until `spider-sft` is listed |
| `/v1/models` returns 401 | `VLLM_API_KEY` leaked into the vLLM process | Start vLLM with that variable unset |
| Grafana render PNGs say the image renderer is missing | Grafana was started before the renderer, and the old process kept port 3000 | Restart Grafana after the renderer is listening on 8081 |
