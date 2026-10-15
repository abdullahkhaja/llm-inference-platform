"""Shared pytest fixtures for Phase 1 API tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Force fast mock timings for unit tests (still serialized).
os.environ["BACKEND"] = "mock"
os.environ["MOCK_PREFILL_MS"] = "5"
os.environ["MOCK_PER_TOKEN_MS"] = "1"
os.environ["MOCK_JITTER_MS"] = "0"
os.environ["MOCK_SEED"] = "123"
os.environ["MAX_NEW_TOKENS"] = "256"


@pytest.fixture()
def tmp_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("REQUEST_LOG_PATH", str(log_path))
    # Clear cached settings so env overrides apply
    from app.config import get_settings

    get_settings.cache_clear()
    yield log_path
    get_settings.cache_clear()


@pytest.fixture()
def client(tmp_log: Path):
    from app.main import app

    with TestClient(app) as c:
        yield c
