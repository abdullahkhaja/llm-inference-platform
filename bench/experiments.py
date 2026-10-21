"""Phase 4 experiment plans. These describe runs; they do not start a GPU.

Compare caching or admission only when the other variables stay fixed.
Do not compare a cold start with a warmed-up server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def plans() -> dict[str, Any]:
    common = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter": "spider-sft",
        "adapter_subfolder": "sft/final",
        "vllm": "v0.8.5",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.85,
        "max_new_tokens": 64,
        "warmup_requests": 8,
        "measured_requests_per_condition": 1000,
        "note": (
            "Plans only. Phase 3 does not execute these on a GPU. "
            "Headline latency and throughput numbers come from Phase 4."
        ),
    }
    return {
        "common": common,
        "A_prefix_caching": {
            "question": "Does an explicit prefix cache change TTFT and throughput on repeated schemas?",
            "keep_fixed": ["gpu", "model", "adapter", "prompts", "max_new_tokens", "concurrency"],
            "conditions": [
                {"enable_prefix_caching": False, "workload": "repeated-schema"},
                {"enable_prefix_caching": True, "workload": "repeated-schema"},
                {"enable_prefix_caching": False, "workload": "unrelated-schema"},
                {"enable_prefix_caching": True, "workload": "unrelated-schema"},
            ],
            "restart_vllm_between_cache_flags": True,
            "warmup_each_condition": True,
            "record": [
                "ttft",
                "latency",
                "output_tokens_per_sec",
                "vllm:gpu_prefix_cache_hits_total",
                "vllm:gpu_prefix_cache_queries_total",
                "vllm:gpu_cache_usage_perc",
            ],
        },
        "B_admission": {
            "question": "Does API admission change accepted-request latency by rejecting overflow?",
            "keep_fixed": ["gpu", "model", "adapter", "workload"],
            "conditions": [
                {"admission_enabled": False, "offered": "below_capacity"},
                {"admission_enabled": False, "offered": "above_capacity"},
                {"admission_enabled": True, "offered": "above_capacity"},
            ],
            "record": [
                "successes",
                "rejections",
                "timeouts",
                "goodput",
                "p95_accepted",
                "p99_accepted",
                "admission_waiting",
                "admission_wait",
            ],
            "warning": "A lower accepted p99 with more 503s is not higher capacity.",
        },
        "C_concurrency": {
            "question": "How do throughput and tail latency move with concurrency?",
            "concurrency": [1, 2, 4, 8, 16, 32],
            "pattern": "closed-loop",
            "record": [
                "requests_per_sec",
                "output_tokens_per_sec",
                "p50",
                "p95",
                "p99",
                "ttft",
                "vllm:gpu_cache_usage_perc",
                "failures",
            ],
        },
    }


def write_plans(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plans(), indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write Phase 4 experiment plans")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench/results/phase3_experiment_plan.json"),
    )
    args = parser.parse_args()
    write_plans(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
