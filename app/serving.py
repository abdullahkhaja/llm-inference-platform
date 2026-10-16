"""Inference orchestration — separated from HTTP routing.

Prompt construction matches training/finetune_spider.py (Llama chat messages
with the same system instruction and user schema/question/SQL: format).
"""

from __future__ import annotations

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


# Must stay identical to training/finetune_spider.py SYSTEM
SYSTEM_PROMPT = (
    "You translate a question into one read-only SQLite SELECT query. "
    "Return SQL only, no markdown or explanation."
)


def build_messages(question: str, schema: str) -> list[dict[str, str]]:
    """Chat messages matching finetune_spider.messages_for user/system roles."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Database schema:\n{schema}\n\n"
                f"Question: {question}\nSQL:"
            ),
        },
    ]


def build_prompt(question: str, schema: str) -> str:
    """Readable prompt text for logging / mock token estimates.

    Real vLLM serving applies the tokenizer chat template server-side from
    ``build_messages``; this string is a human-readable parallel, not a
    substitute for the chat template.
    """
    messages = build_messages(question, schema)
    parts = [f"[{m['role']}]\n{m['content']}" for m in messages]
    return "\n\n".join(parts)


class InferenceService:
    """Thin service wrapping a Backend with shared timing measurement."""

    def __init__(self, backend: Backend, settings: Settings) -> None:
        self.backend = backend
        self.settings = settings

    @property
    def queue_depth(self) -> int:
        return self.backend.queue_depth

    def _make_request(
        self,
        *,
        request_id: str,
        question: str,
        schema: str,
        max_new_tokens: int,
    ) -> GenerationRequest:
        messages = build_messages(question, schema)
        return GenerationRequest(
            request_id=request_id,
            question=question,
            schema=schema,
            max_new_tokens=max_new_tokens,
            prompt_text=build_prompt(question, schema),
            messages=messages,
        )

    async def generate(
        self,
        *,
        request_id: str,
        question: str,
        schema: str,
        max_new_tokens: int,
    ) -> tuple[GenerationResult, float]:
        """Run inference and return (result, end_to_end_latency_s)."""
        req = self._make_request(
            request_id=request_id,
            question=question,
            schema=schema,
            max_new_tokens=max_new_tokens,
        )
        t0 = time.perf_counter()
        result = await self.backend.generate(req)
        latency_s = time.perf_counter() - t0
        return result, latency_s

    async def generate_stream(
        self,
        *,
        request_id: str,
        question: str,
        schema: str,
        max_new_tokens: int,
    ) -> AsyncIterator[StreamEvent]:
        """Stream inference events from the backend."""
        req = self._make_request(
            request_id=request_id,
            question=question,
            schema=schema,
            max_new_tokens=max_new_tokens,
        )
        async for event in self.backend.generate_stream(req):
            yield event
