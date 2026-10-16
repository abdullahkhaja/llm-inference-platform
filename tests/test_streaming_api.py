"""SSE streaming endpoint tests (mock backend — CPU only)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_generate_stream_mock_sse(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/generate/stream",
        json={
            "question": "List all employees",
            "schema": "CREATE TABLE employees (id INTEGER);",
            "max_new_tokens": 8,
        },
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())

    assert "event: meta" in body
    assert "event: token" in body
    assert "event: usage" in body
    assert "event: done" in body
    assert "SIMULATED" in body


def test_generate_response_simulated_flag_from_backend(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={
            "question": "Count rows",
            "schema": "CREATE TABLE t(id INT);",
            "max_new_tokens": 8,
        },
    )
    data = resp.json()
    assert data["simulated"] is True
    assert data["metadata"]["measurements"] == "simulated"
    assert data["backend"] == "mock"
