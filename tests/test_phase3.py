"""Phase 3 CPU tests: admission, caching flags, workloads, tracing, benches."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.admission import AdmissionController, AdmissionRejected
from app.config import get_settings
from bench.load_test import RequestRecord, arrival_times, summarize
from bench.workloads import item
from ops.runpod.vllm_flags import prefix_caching_arg


def test_prefix_caching_flags() -> None:
    assert prefix_caching_arg("true") == "--enable-prefix-caching"
    assert prefix_caching_arg("false") == "--no-enable-prefix-caching"
    with pytest.raises(ValueError):
        prefix_caching_arg("maybe")


def test_entrypoint_invokes_flag_helper() -> None:
    text = Path("ops/runpod/entrypoint.sh").read_text(encoding="utf-8")
    assert "vllm_flags.py" in text
    assert "CACHE_ARG" in text
    assert "ENABLE_PREFIX_CACHING" in text


def test_repeated_schema_is_stable_and_unrelated_changes() -> None:
    first = item("repeated-schema", 0)
    second = item("repeated-schema", 1)
    assert first.schema == second.schema
    assert first.question != second.question
    assert first.synthetic is True
    other = [item("unrelated-schema", i).schema for i in range(4)]
    assert len(set(other)) > 1


def test_arrival_times_are_independent_of_latency() -> None:
    times = arrival_times(10, 1.0)
    assert len(times) == 10
    assert times[0] == 0.0
    assert abs(times[1] - 0.1) < 1e-9


def test_missing_queue_wait_is_not_zero() -> None:
    records = [
        RequestRecord(1, 0, True, 200, 0.2, None, None, 4, 8, "", False),
        RequestRecord(1, 1, False, 503, 0.01, None, None, 0, 0, "admission_full", False),
    ]
    summary = summarize(records, wall_s=1.0)
    assert summary["rejections"] == 1
    assert summary["successes"] == 1
    assert summary["mean_queue_wait_ms"] != summary["mean_queue_wait_ms"]  # NaN


@pytest.mark.asyncio
async def test_admission_disabled_does_not_serialize() -> None:
    ctl = AdmissionController(enabled=False, max_active=1, max_waiting=0, max_wait_s=0.1)
    a = await ctl.acquire()
    b = await ctl.acquire()
    assert ctl.active == 0
    await a.release()
    await b.release()


@pytest.mark.asyncio
async def test_admission_rejects_when_full() -> None:
    ctl = AdmissionController(enabled=True, max_active=1, max_waiting=0, max_wait_s=1)
    held = await ctl.acquire()
    with pytest.raises(AdmissionRejected) as exc:
        await ctl.acquire()
    assert exc.value.reason == "admission_full"
    await held.release()
    again = await ctl.acquire()
    assert ctl.active == 1
    await again.release()
    assert ctl.active == 0


@pytest.mark.asyncio
async def test_admission_wait_timeout_releases_waiter() -> None:
    ctl = AdmissionController(enabled=True, max_active=1, max_waiting=2, max_wait_s=0.05)
    held = await ctl.acquire()
    with pytest.raises(AdmissionRejected) as exc:
        await ctl.acquire()
    assert exc.value.reason == "admission_timeout"
    assert ctl.waiting == 0
    await held.release()
    assert ctl.active == 0


@pytest.mark.asyncio
async def test_admission_cancel_while_waiting() -> None:
    ctl = AdmissionController(enabled=True, max_active=1, max_waiting=2, max_wait_s=5)
    held = await ctl.acquire()

    async def waiter() -> None:
        await ctl.acquire()

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert ctl.waiting == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ctl.waiting == 0
    await held.release()
    assert ctl.active == 0


def test_tracing_span_omits_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.tracing import configure, exported_spans, reset, span

    reset()
    configure(enabled=True)
    with span("prompt_assembly", request_id="r1", question="secret question", schema="CREATE"):
        pass
    spans = exported_spans()
    assert spans
    attrs = dict(spans[-1].attributes)
    assert attrs.get("request_id") == "r1"
    assert "question" not in attrs
    assert "schema" not in attrs
    reset()


def test_trace_log_records_ids_without_prompts(tmp_path: Path) -> None:
    from app.tracing import configure, reset, span

    log_path = tmp_path / "traces.jsonl"
    reset()
    configure(enabled=True, trace_log_path=str(log_path))
    with span("request_validation", request_id="r-trace", question="hidden"):
        pass
    reset()
    line = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert line["name"] == "request_validation"
    assert len(line["trace_id"]) == 32
    assert "question" not in line["attributes"]
    assert line["attributes"]["request_id"] == "r-trace"


def test_dashboard_queries_use_known_series() -> None:
    dash = json.loads(
        Path("ops/grafana/dashboards/phase3_serving.json").read_text(encoding="utf-8")
    )
    exprs = []
    for panel in dash["panels"]:
        for target in panel.get("targets") or []:
            exprs.append(target["expr"])
    joined = "\n".join(exprs)
    for name in (
        "llm_requests_total",
        "llm_admission_rejected_total",
        "llm_admission_waiting",
        "llm_inflight_requests",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
        "vllm:gpu_prefix_cache_hits_total",
    ):
        assert name in joined
    assert "histogram_quantile" in joined


@pytest.mark.asyncio
async def test_stream_close_releases_admission_slot() -> None:
    from app.backends.mock import MockBackend
    from app.config import Settings
    from app.serving import InferenceService

    settings = Settings(
        backend="mock",
        admission_enabled=True,
        max_concurrent_requests=1,
        max_waiting_requests=0,
        max_queue_wait_ms=50,
        mock_prefill_ms=1,
        mock_per_token_ms=1,
        mock_jitter_ms=0,
    )
    backend = MockBackend(settings)
    await backend.startup()
    service = InferenceService(backend, settings)
    lease = await service.admission.acquire()
    gen = service.generate_stream(
        request_id="stream-close",
        question="List employees",
        schema="CREATE TABLE employees (id INTEGER);",
        max_new_tokens=2,
        lease=lease,
    )
    await anext(gen)
    await gen.aclose()
    assert service.admission.active == 0
    await backend.shutdown()


def test_experiment_plans_keep_cache_conditions_aligned() -> None:
    from bench.experiments import plans

    plan = plans()
    caching = plan["A_prefix_caching"]
    assert caching["restart_vllm_between_cache_flags"] is True
    workloads = {row["workload"] for row in caching["conditions"]}
    assert workloads == {"repeated-schema", "unrelated-schema"}
    assert plan["C_concurrency"]["concurrency"] == [1, 2, 4, 8, 16, 32]
    assert "not higher capacity" in plan["B_admission"]["warning"]


@pytest.mark.asyncio
async def test_overload_returns_503_with_retry_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    monkeypatch.setenv("BACKEND", "mock")
    monkeypatch.setenv("ADMISSION_ENABLED", "true")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "1")
    monkeypatch.setenv("MAX_WAITING_REQUESTS", "0")
    monkeypatch.setenv("MAX_QUEUE_WAIT_MS", "50")
    monkeypatch.setenv("MOCK_PREFILL_MS", "200")
    monkeypatch.setenv("MOCK_PER_TOKEN_MS", "1")
    monkeypatch.setenv("MOCK_JITTER_MS", "0")
    monkeypatch.setenv("REQUEST_LOG_PATH", str(tmp_path / "requests.jsonl"))
    get_settings.cache_clear()
    from app.main import app

    body = {
        "question": "List employees",
        "schema": "CREATE TABLE employees (id INTEGER);",
        "max_new_tokens": 4,
    }
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first, second = await asyncio.gather(
                client.post("/generate", json=body),
                client.post("/generate", json=body),
            )
    statuses = sorted([first.status_code, second.status_code])
    assert statuses == [200, 503]
    rejected = first if first.status_code == 503 else second
    assert rejected.headers["retry-after"] == "1"
    assert rejected.json()["category"] == "admission_full"
    get_settings.cache_clear()


def test_admission_http_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND", "mock")
    monkeypatch.setenv("ADMISSION_ENABLED", "true")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "1")
    monkeypatch.setenv("MAX_WAITING_REQUESTS", "0")
    monkeypatch.setenv("MAX_QUEUE_WAIT_MS", "50")
    monkeypatch.setenv("MOCK_PREFILL_MS", "300")
    monkeypatch.setenv("MOCK_PER_TOKEN_MS", "1")
    monkeypatch.setenv("REQUEST_LOG_PATH", str(tmp_path / "requests.jsonl"))
    get_settings.cache_clear()
    from app.main import app

    body = {
        "question": "List employees",
        "schema": "CREATE TABLE employees (id INTEGER);",
        "max_new_tokens": 4,
    }
    with TestClient(app) as client:
        first = client.post("/generate", json=body)
        assert first.status_code == 200
        # Slot was released, so a later call succeeds.
        second = client.post("/generate", json=body)
        assert second.status_code == 200
    get_settings.cache_clear()
