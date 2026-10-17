"""Metric names documented for vLLM v0.8.5.

The Phase 2 smoke log showed a V1 engine. V0 and V1 both expose several of
these names. On the v0.8.5 V1 scrape, prefix-cache counters are
`vllm:gpu_prefix_cache_hits_total` and `vllm:gpu_prefix_cache_queries_total`.
A missing series means it was not scraped or not emitted
by that engine build. Do not substitute zero.
"""

from __future__ import annotations

# name -> short meaning. Presence is verified only on a live engine scrape.
VLLM_085_METRICS: dict[str, str] = {
    "vllm:num_requests_running": "Requests in a model execution batch (engine queue, not the API admission queue).",
    "vllm:num_requests_waiting": "Requests waiting inside the vLLM scheduler.",
    "vllm:gpu_cache_usage_perc": "Fraction of GPU KV-cache blocks in use.",
    "vllm:gpu_prefix_cache_hits_total": "Prefix-cache hits. The 0.8.5 scrape adds the Prometheus _total suffix.",
    "vllm:gpu_prefix_cache_queries_total": "Prefix-cache lookups. The 0.8.5 scrape adds the Prometheus _total suffix.",
    "vllm:prompt_tokens_total": "Prompt tokens processed by the engine.",
    "vllm:generation_tokens_total": "Generated tokens processed by the engine.",
    "vllm:time_to_first_token_seconds": "Engine TTFT histogram.",
    "vllm:time_per_output_token_seconds": "Engine inter-token latency histogram.",
    "vllm:e2e_request_latency_seconds": "Engine end-to-end request latency histogram.",
    "vllm:request_queue_time_seconds": "Time spent in the engine queue.",
    "vllm:request_prefill_time_seconds": "Engine prefill time histogram.",
    "vllm:request_decode_time_seconds": "Engine decode time histogram.",
    "vllm:num_preemptions_total": "Prefill/decode preemptions, when the build emits the counter.",
    "vllm:cache_config_info": "Cache configuration, including whether prefix caching is enabled.",
}


def required_for_phase3_check() -> tuple[str, ...]:
    """Names a short GPU validation should find on /metrics. Others may be absent."""
    return (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
    )
