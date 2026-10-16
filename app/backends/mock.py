"""CPU mock inference backend with serialized single-worker timing."""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import TYPE_CHECKING, AsyncIterator

from app.backends.base import (
    Backend,
    GenerationRequest,
    GenerationResult,
    StreamEvent,
)

if TYPE_CHECKING:
    from app.config import Settings


def estimate_tokens(text: str) -> int:
    """Rough token estimate without loading a real tokenizer (~4 chars/token)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


class MockBackend(Backend):
    """Simulated text-to-SQL inference.

    A single asyncio.Lock serializes all inference so concurrent HTTP requests
    experience realistic queue wait. Timing is anchored to absolute deadlines
    (perf_counter) so sleep overshoot does not compound across tokens.

    All outputs are explicitly labeled as SIMULATED — not real model SQL.
    """

    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._waiting = 0
        self._rng = random.Random(settings.mock_seed)
        self._started = False

    async def startup(self) -> None:
        self._started = True

    async def shutdown(self) -> None:
        self._started = False

    @property
    def ready(self) -> bool:
        return self._started

    @property
    def queue_depth(self) -> int:
        return self._waiting

    def _build_simulated_sql(self, request: GenerationRequest, output_tokens: int) -> str:
        # Deterministic-looking placeholder SQL; clearly marked as mock.
        q = request.question.strip().replace("\n", " ")
        snippet = (q[:60] + "…") if len(q) > 60 else q
        # Pad with synthetic SELECT comment tokens to approximate token count.
        pad_words = max(0, output_tokens - 12)
        pad = " ".join(f"tok{i}" for i in range(pad_words))
        body = (
            f"-- [SIMULATED MOCK RESPONSE - NOT REAL MODEL OUTPUT]\n"
            f"-- question: {snippet}\n"
            f"SELECT * FROM mock_table WHERE 1=1"
        )
        if pad:
            body += f" /* {pad} */"
        body += ";"
        return body

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._started:
            raise RuntimeError("MockBackend.startup() was not called")

        self._waiting += 1
        enqueue_t = time.perf_counter()
        await self._lock.acquire()
        wait_s = time.perf_counter() - enqueue_t
        self._waiting -= 1
        try:
            input_tokens = estimate_tokens(request.prompt_text)
            output_tokens = max(1, request.max_new_tokens)

            # Timing model: prefill once + per-token decode. Single worker only.
            prefill_s = self._settings.mock_prefill_ms / 1000.0
            per_token_s = self._settings.mock_per_token_ms / 1000.0
            if self._settings.mock_jitter_ms > 0:
                jitter = self._rng.uniform(
                    -self._settings.mock_jitter_ms, self._settings.mock_jitter_ms
                ) / 1000.0
                prefill_s = max(0.0, prefill_s + jitter)

            t0 = time.perf_counter()
            await self._sleep_until(t0 + prefill_s)
            for step in range(output_tokens):
                await self._sleep_until(t0 + prefill_s + (step + 1) * per_token_s)

            inference_s = time.perf_counter() - t0
            text = self._build_simulated_sql(request, output_tokens)
            return GenerationResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                queue_wait_s=wait_s,
                inference_s=inference_s,
                prefill_s=prefill_s,
                decode_s=max(0.0, inference_s - prefill_s),
                simulated=True,
                backend=self.name,
                ttft_s=prefill_s,
                metadata={"note": "simulated_mock_inference", "measurements": "simulated"},
            )
        finally:
            self._lock.release()

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamEvent]:
        """Stream simulated tokens (CPU-only; for local SSE testing)."""
        result = await self.generate(request)
        # Emit the SQL in small chunks so clients can exercise SSE parsing.
        chunk_size = max(1, len(result.text) // max(1, min(result.output_tokens, 32)))
        text = result.text
        first = True
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size]
            if first:
                first = False
            yield StreamEvent(
                event="token",
                data={"text": piece, "request_id": request.request_id},
            )
            await asyncio.sleep(0)  # yield control
        yield StreamEvent(
            event="usage",
            data={
                "request_id": request.request_id,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "ttft_s": result.ttft_s,
                "inference_s": result.inference_s,
                "queue_wait_s": result.queue_wait_s,
                "simulated": True,
                "backend": self.name,
                "sql": result.text,
            },
        )
        yield StreamEvent(event="done", data={"request_id": request.request_id})

    @staticmethod
    async def _sleep_until(deadline: float) -> None:
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)
