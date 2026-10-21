"""CPU-only Phase 4 checks. These tests never start vLLM or CUDA."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from bench.load_test import RequestRecord, first_content_token_ttft_s, summarize
from bench.phase4 import (
    cost_per_million_output_tokens,
    cost_per_successful_request,
    grafana_range_ms,
    is_png,
    json_safe,
    percent_change,
    redact_secrets,
    window_cost_usd,
)
from bench.experiments import plans
from bench.capture_grafana import render_path, save_png


def test_ttft_ignores_meta_event():
    blocks = [
        (0.01, 'event: meta\ndata: {"request_id": "abc"}'),
        (0.04, 'event: token\ndata: {"text": ""}'),
        (0.12, 'event: token\ndata: {"text": "SELECT"}'),
    ]
    assert first_content_token_ttft_s(blocks) == 0.12


def test_open_loop_offered_rate_excludes_drain():
    records = [
        RequestRecord(0, i, True, 200, 2.0, None, None, 8, 20, "", False)
        for i in range(10)
    ]
    summary = summarize(records, wall_s=12.0, arrival_window_s=1.0)
    assert summary["offered_rps"] == 10.0
    assert summary["rps"] == 10 / 12


def test_percentile_and_null_ttft():
    records = [
        RequestRecord(1, 0, True, 200, 0.2, None, None, 4, 10, "", False, 100.0),
        RequestRecord(1, 1, True, 200, 0.4, None, None, 4, 10, "", False, 300.0),
    ]
    summary = summarize(records, 1.0)
    assert abs(summary["p50_latency_s"] - 0.3) < 1e-9
    assert summary["mean_ttft_ms"] == 200.0
    empty = summarize([], 1.0)
    assert empty["mean_ttft_ms"] is None


def test_cost_math_uses_measurement_window_only():
    cost = window_cost_usd(0.59, 3600)
    assert cost == 0.59
    assert cost_per_successful_request(0.59, 100) == 0.0059
    assert cost_per_successful_request(0.59, 0) is None
    assert cost_per_million_output_tokens(0.50, 250_000) == 2.0


def test_cache_plan_has_four_conditions():
    conditions = plans()["A_prefix_caching"]["conditions"]
    assert len(conditions) == 4
    assert {c["workload"] for c in conditions} == {"repeated-schema", "unrelated-schema"}


def test_grafana_range_and_png(tmp_path: Path):
    start = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    start_ms, end_ms = grafana_range_ms(start, end)
    assert end_ms - start_ms == 300_000
    path = render_path("phase3-serving", 7, start_ms, end_ms)
    assert "from=" in path and "panelId=7" in path
    blob = b"\x89PNG\r\n\x1a\n" + b"x" * 40
    dest = tmp_path / "panel.png"
    save_png(dest, blob)
    assert is_png(dest.read_bytes())


def test_redaction_and_nan_null():
    assert "hf_REDACTED" in redact_secrets("token hf_abcdefghijklmnopqrstuvwxyz")
    assert json_safe({"p99": float("nan"), "ok": 1})["p99"] is None


def test_percent_change():
    assert percent_change(100.0, 80.0) == -20.0
    assert percent_change(0.0, 1.0) is None
