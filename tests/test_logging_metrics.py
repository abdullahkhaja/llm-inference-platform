"""Request logging and Prometheus metrics tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_jsonl_request_logging(client: TestClient, tmp_log: Path) -> None:
    resp = client.post(
        "/generate",
        json={
            "question": "List all employees",
            "schema": "CREATE TABLE employees (id INTEGER);",
            "max_new_tokens": 8,
        },
    )
    assert resp.status_code == 200
    assert tmp_log.exists()
    lines = tmp_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    import json

    record = json.loads(lines[-1])
    assert record["request_id"] == resp.json()["request_id"]
    assert record["backend"] == "mock"
    assert record["simulated"] is True
    assert record["http_status"] == 200
    assert "queue_wait_ms" in record
    assert "inference_ms" in record
    assert "latency_ms" in record
    # Prompts/schemas must not appear by default
    assert "question" not in record
    assert "schema" not in record


def test_metrics_endpoint(client: TestClient) -> None:
    client.post(
        "/generate",
        json={
            "question": "Count rows",
            "schema": "CREATE TABLE t(id INT);",
            "max_new_tokens": 8,
        },
    )
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "llm_requests_total" in body
    assert "llm_request_latency_seconds" in body
    assert "llm_queue_wait_seconds" in body
    assert "llm_inference_duration_seconds" in body
    assert "llm_generated_tokens_total" in body
    assert "llm_queue_depth" in body
    assert "llm_requests_in_flight" in body
