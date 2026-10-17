"""Prometheus metrics for the inference server.

Phase 1 mock timings remain available. Phase 2 adds backend/mode labels and
real-inference series (TTFT, remote errors, output tok/s). Do not interpret
``mode=simulated`` series as GPU telemetry.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# Histogram buckets suitable for p50/p95/p99 (seconds).
LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    13.0,
    21.0,
    34.0,
    55.0,
    90.0,
)

REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total /generate and /generate/stream requests",
    labelnames=("status", "backend", "mode"),  # mode: simulated | real
)

IN_FLIGHT = Gauge(
    "llm_requests_in_flight",
    "Number of inference requests currently being handled",
    labelnames=("backend", "mode"),
)

QUEUE_DEPTH = Gauge(
    "llm_queue_depth",
    "Requests waiting at the API (mock lock or admission queue). Not the vLLM scheduler queue.",
    labelnames=("backend",),
)

INFLIGHT = Gauge(
    "llm_inflight_requests",
    "HTTP requests currently inside the backend client. Not queued.",
    labelnames=("backend",),
)

ADMISSION_ACTIVE = Gauge(
    "llm_admission_active",
    "Requests that hold an admission slot",
)

ADMISSION_WAITING = Gauge(
    "llm_admission_waiting",
    "Requests waiting for an admission slot",
)

ADMISSION_REJECTED = Counter(
    "llm_admission_rejected_total",
    "Requests rejected by API admission control",
    labelnames=("reason",),
)

ADMISSION_WAIT = Histogram(
    "llm_admission_wait_seconds",
    "Time from admission acquire start until a slot is held",
    buckets=LATENCY_BUCKETS,
)

ADMISSION_TIMEOUTS = Counter(
    "llm_admission_timeouts_total",
    "Admission waits that exceeded MAX_QUEUE_WAIT_MS",
)

QUEUE_WAIT = Histogram(
    "llm_queue_wait_seconds",
    "Time spent waiting for the inference worker (mock serialization only)",
    labelnames=("backend", "mode"),
    buckets=LATENCY_BUCKETS,
)

INFERENCE_DURATION = Histogram(
    "llm_inference_duration_seconds",
    "Inference execution time (mock prefill+decode, or remote HTTP round-trip)",
    labelnames=("backend", "mode"),
    buckets=LATENCY_BUCKETS,
)

REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "End-to-end /generate latency including queue wait",
    labelnames=("backend", "mode"),
    buckets=LATENCY_BUCKETS,
)

TTFT = Histogram(
    "llm_ttft_seconds",
    "Time to first token for streaming requests",
    labelnames=("backend", "mode"),
    buckets=LATENCY_BUCKETS,
)

GENERATED_TOKENS = Counter(
    "llm_generated_tokens_total",
    "Output tokens produced",
    labelnames=("backend", "mode"),
)

INPUT_TOKENS = Counter(
    "llm_input_tokens_total",
    "Input tokens (estimated for mock; from engine usage for vllm)",
    labelnames=("backend", "mode"),
)

# Gauge updated per successful request: instantaneous tokens/sec
SIM_OUTPUT_TOKENS_PER_SEC = Gauge(
    "llm_simulated_output_tokens_per_second",
    "Output tokens / inference_seconds for the last successful mock request",
)

REAL_OUTPUT_TOKENS_PER_SEC = Gauge(
    "llm_real_output_tokens_per_second",
    "Output tokens / inference_seconds for the last successful real request",
)

REMOTE_ERRORS = Counter(
    "llm_remote_errors_total",
    "Remote vLLM / transport errors",
    labelnames=("category",),
)

STREAM_REQUESTS = Counter(
    "llm_stream_requests_total",
    "Streaming /generate/stream requests",
    labelnames=("status", "backend", "mode"),
)

APP_INFO = Info(
    "llm_app",
    "Static application metadata",
)


def set_app_info(*, version: str, backend: str, mode: str) -> None:
    APP_INFO.info(
        {
            "version": version,
            "backend": backend,
            "phase": "2" if backend == "vllm" else "1",
            "measurements": mode,
        }
    )


def mode_label(simulated: bool) -> str:
    return "simulated" if simulated else "real"
