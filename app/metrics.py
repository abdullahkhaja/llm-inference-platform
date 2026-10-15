"""Prometheus metrics for the inference server.

All latency / throughput series from Phase 1 reflect *simulated* mock timings.
Do not interpret them as real GPU telemetry.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# Histogram buckets suitable for p50/p95/p99 on mock latencies (seconds).
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
)

REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total /generate requests",
    labelnames=("status",),  # success | error
)

IN_FLIGHT = Gauge(
    "llm_requests_in_flight",
    "Number of /generate requests currently being handled",
)

QUEUE_DEPTH = Gauge(
    "llm_queue_depth",
    "Requests waiting for the serialized inference worker",
)

QUEUE_WAIT = Histogram(
    "llm_queue_wait_seconds",
    "Time spent waiting for the inference worker (simulated serialization)",
    buckets=LATENCY_BUCKETS,
)

INFERENCE_DURATION = Histogram(
    "llm_inference_duration_seconds",
    "Simulated inference execution time (prefill + decode)",
    buckets=LATENCY_BUCKETS,
)

REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "End-to-end /generate latency including queue wait",
    buckets=LATENCY_BUCKETS,
)

GENERATED_TOKENS = Counter(
    "llm_generated_tokens_total",
    "Simulated output tokens produced",
)

INPUT_TOKENS = Counter(
    "llm_input_tokens_total",
    "Simulated input tokens estimated from prompt length",
)

# Gauge updated per successful request: instantaneous simulated tokens/sec
SIM_OUTPUT_TOKENS_PER_SEC = Gauge(
    "llm_simulated_output_tokens_per_second",
    "Simulated output tokens / inference_seconds for the last successful request",
)

APP_INFO = Info(
    "llm_app",
    "Static application metadata",
)


def set_app_info(*, version: str, backend: str) -> None:
    APP_INFO.info(
        {
            "version": version,
            "backend": backend,
            "phase": "1",
            "measurements": "simulated_mock",
        }
    )
