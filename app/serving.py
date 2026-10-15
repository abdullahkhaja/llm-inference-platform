"""Inference orchestration — separated from HTTP routing."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from app.backends.base import Backend, GenerationRequest, GenerationResult

if TYPE_CHECKING:
    from app.config import Settings


SYSTEM_PROMPT = (
    "You translate a question into one read-only SQLite SELECT query. "
    "Return SQL only, no markdown or explanation."
)


def build_prompt(question: str, schema: str) -> str:
    """Chat-style prompt text used for token estimation and future real backends."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Database schema:\n{schema}\n\n"
        f"Question: {question}\nSQL:"
    )


class InferenceService:
    """Thin service wrapping a Backend with shared timing measurement."""

    def __init__(self, backend: Backend, settings: Settings) -> None:
        self.backend = backend
        self.settings = settings

    @property
    def queue_depth(self) -> int:
        return self.backend.queue_depth

    async def generate(
        self,
        *,
        request_id: str,
        question: str,
        schema: str,
        max_new_tokens: int,
    ) -> tuple[GenerationResult, float]:
        """Run inference and return (result, end_to_end_latency_s)."""
        prompt = build_prompt(question, schema)
        req = GenerationRequest(
            request_id=request_id,
            question=question,
            schema=schema,
            max_new_tokens=max_new_tokens,
            prompt_text=prompt,
        )
        t0 = time.perf_counter()
        result = await self.backend.generate(req)
        latency_s = time.perf_counter() - t0
        return result, latency_s
