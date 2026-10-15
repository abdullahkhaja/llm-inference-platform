"""API validation, health, generate, and error handling tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend"] == "mock"
    assert data["ready"] is True
    assert data["simulated"] is True


def test_root(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SIMULATED" in resp.text


def test_generate_success(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={
            "question": "List all employees earning more than 50000.",
            "schema": "CREATE TABLE employees (id INTEGER, name TEXT, salary INTEGER);",
            "max_new_tokens": 16,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["simulated"] is True
    assert data["backend"] == "mock"
    assert "SIMULATED" in data["sql"]
    assert data["output_tokens"] == 16
    assert data["latency_ms"] >= 0
    assert data["request_id"]


def test_generate_validation_empty(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={"question": "   ", "schema": "CREATE TABLE t(id INT);", "max_new_tokens": 8},
    )
    assert resp.status_code == 422


def test_generate_missing_fields(client: TestClient) -> None:
    resp = client.post("/generate", json={"question": "hi"})
    assert resp.status_code == 422


def test_generate_too_many_tokens(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={
            "question": "List names",
            "schema": "CREATE TABLE t(name TEXT);",
            "max_new_tokens": 10_000,
        },
    )
    assert resp.status_code == 400


def test_generate_payload_too_large(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MAX_QUESTION_CHARS", "20")
    get_settings.cache_clear()

    # Rebuild app client with new settings via lifespan
    from app.main import app

    with TestClient(app) as c:
        resp = c.post(
            "/generate",
            json={
                "question": "This question is definitely longer than twenty characters",
                "schema": "CREATE TABLE t(id INT);",
                "max_new_tokens": 8,
            },
        )
        assert resp.status_code == 413
    get_settings.cache_clear()


def test_request_id_header(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        headers={"x-request-id": "test-req-1"},
        json={
            "question": "Count employees",
            "schema": "CREATE TABLE employees (id INTEGER);",
            "max_new_tokens": 8,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["request_id"] == "test-req-1"
