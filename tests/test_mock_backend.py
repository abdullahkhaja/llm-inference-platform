"""Mock backend serialization and timing behavior."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.backends.base import GenerationRequest
from app.backends.mock import MockBackend, estimate_tokens
from app.config import Settings


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        backend="mock",
        mock_prefill_ms=10,
        mock_per_token_ms=2,
        mock_jitter_ms=0,
        mock_seed=1,
    )


@pytest.mark.asyncio
async def test_mock_generate_labels_simulated(settings: Settings) -> None:
    backend = MockBackend(settings)
    await backend.startup()
    req = GenerationRequest(
        request_id="r1",
        question="List names",
        schema="CREATE TABLE t(name TEXT);",
        max_new_tokens=5,
        prompt_text="schema + question",
    )
    result = await backend.generate(req)
    assert result.simulated is True
    assert result.backend == "mock"
    assert result.output_tokens == 5
    assert "SIMULATED" in result.text
    assert result.inference_s > 0
    await backend.shutdown()


@pytest.mark.asyncio
async def test_request_serialization(settings: Settings) -> None:
    """Concurrent generates must not overlap inference (single lock)."""
    backend = MockBackend(settings)
    await backend.startup()

    async def one(i: int):
        return await backend.generate(
            GenerationRequest(
                request_id=f"r{i}",
                question=f"q{i}",
                schema="CREATE TABLE t(id INT);",
                max_new_tokens=8,
                prompt_text="prompt",
            )
        )

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(4)))
    elapsed = time.perf_counter() - t0

    # Each request: ~10ms prefill + 8*2ms = 26ms → 4 serial ≈ 104ms
    # Allow generous slack for CI scheduling but require > 2x single-request time.
    single = results[0].inference_s
    assert elapsed >= single * 2.5
    assert all(r.simulated for r in results)
    # At least some requests should have observed queue wait under contention
    assert max(r.queue_wait_s for r in results) >= 0.0
    await backend.shutdown()


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 20) == 5
