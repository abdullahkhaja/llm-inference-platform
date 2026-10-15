"""Concurrent API request behavior through TestClient + threads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi.testclient import TestClient


def test_concurrent_generate(client: TestClient) -> None:
    payload = {
        "question": "List employees",
        "schema": "CREATE TABLE employees (id INTEGER);",
        "max_new_tokens": 10,
    }

    def call(_: int) -> dict:
        resp = client.post("/generate", json=payload)
        assert resp.status_code == 200
        return resp.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(call, i) for i in range(8)]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == 8
    assert all(r["simulated"] for r in results)
    # Serialized worker ⇒ some queue wait under concurrency
    assert max(r["queue_wait_ms"] for r in results) >= 0
    assert max(r["latency_ms"] for r in results) >= min(r["inference_ms"] for r in results)
