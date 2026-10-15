"""Benchmark helper calculation accuracy."""

from __future__ import annotations

from bench.load_test import RequestRecord, percentile, summarize


def test_percentile_basic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 5.0
    assert percentile(values, 50) == 3.0


def test_percentile_empty() -> None:
    import math

    assert math.isnan(percentile([], 50))


def test_summarize_metrics() -> None:
    records = [
        RequestRecord(1, 0, True, 200, 0.10, 1.0, 90.0, 10, 20, "", True),
        RequestRecord(1, 1, True, 200, 0.20, 2.0, 180.0, 10, 20, "", True),
        RequestRecord(1, 2, True, 200, 0.30, 3.0, 270.0, 10, 20, "", True),
        RequestRecord(1, 3, False, 500, 1.00, 0.0, 0.0, 0, 0, "boom", True),
        RequestRecord(1, 4, False, 0, 2.00, 0.0, 0.0, 0, 0, "timeout: x", True),
    ]
    summary = summarize(records, wall_s=1.0)
    assert summary["simulated"] is True
    assert summary["requests"] == 5
    assert summary["successes"] == 3
    assert summary["failures"] == 2
    assert summary["timeouts"] == 1
    assert abs(summary["rps"] - 3.0) < 1e-9
    assert abs(summary["simulated_output_tokens_per_sec"] - 30.0) < 1e-9
    assert abs(summary["mean_latency_s"] - 0.2) < 1e-9
    assert abs(summary["p50_latency_s"] - 0.2) < 1e-9
    assert "p99_caveat" in summary
