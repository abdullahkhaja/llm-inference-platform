"""Remote vLLM backend — OpenAI-compatible HTTP client (no local CUDA).

The FastAPI process never loads model weights. All GPU work happens on the
remote vLLM server (typically RunPod). This module is safe to import and test
on CPU-only machines.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import httpx

from app.backends.base import (
    Backend,
    GenerationRequest,
    GenerationResult,
    StreamEvent,
)

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class VLLMRemoteError(RuntimeError):
    """Remote vLLM / network failure."""

    def __init__(self, message: str, *, category: str = "remote_error") -> None:
        super().__init__(message)
        self.category = category


class VLLMBackend(Backend):
    """Async HTTP client for a remote OpenAI-compatible vLLM server."""

    name = "vllm"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Optional[httpx.AsyncClient] = None
        self._started = False
        self._ready = False
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def queue_depth(self) -> int:
        # Remote queue depth is not exposed; report local in-flight HTTP calls.
        return self._in_flight

    async def startup(self) -> None:
        timeout = httpx.Timeout(
            self._settings.vllm_timeout_s,
            connect=self._settings.vllm_connect_timeout_s,
        )
        self._client = httpx.AsyncClient(
            base_url=self._settings.vllm_base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._settings.vllm_api_key}",
                "Content-Type": "application/json",
            },
        )
        self._started = True
        await self._wait_until_ready()

    async def shutdown(self) -> None:
        self._ready = False
        self._started = False
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if not self._started or self._client is None:
            raise RuntimeError("VLLMBackend.startup() was not called")
        return self._client

    async def _wait_until_ready(self) -> None:
        deadline = time.perf_counter() + self._settings.vllm_ready_timeout_s
        last_error = "not checked"
        while time.perf_counter() < deadline:
            try:
                await self.check_ready()
                self._ready = True
                logger.info(
                    "vLLM ready at %s model=%s",
                    self._settings.vllm_base_url,
                    self._settings.vllm_model_name,
                )
                return
            except Exception as exc:  # noqa: BLE001 — retry until deadline
                last_error = str(exc)
                await asyncio.sleep(1.0)
        self._ready = False
        raise VLLMRemoteError(
            f"vLLM not ready within {self._settings.vllm_ready_timeout_s}s: {last_error}",
            category="not_ready",
        )

    async def check_ready(self) -> dict[str, Any]:
        """Query GET /models and confirm the configured model id is listed."""
        client = self._require_client()
        resp = await client.get("/models")
        if resp.status_code != 200:
            raise VLLMRemoteError(
                f"/models returned HTTP {resp.status_code}: {resp.text[:200]}",
                category="not_ready",
            )
        payload = resp.json()
        models = payload.get("data") or []
        ids = {m.get("id") for m in models if isinstance(m, dict)}
        expected = self._settings.vllm_model_name
        if expected not in ids:
            raise VLLMRemoteError(
                f"model {expected!r} not in /models ({sorted(ids)})",
                category="not_ready",
            )
        self._ready = True
        return payload

    def _chat_payload(
        self, request: GenerationRequest, *, stream: bool
    ) -> dict[str, Any]:
        messages = request.messages or [
            {"role": "user", "content": request.prompt_text},
        ]
        return {
            "model": self._settings.vllm_model_name,
            "messages": messages,
            "max_tokens": request.max_new_tokens,
            "temperature": self._settings.vllm_temperature,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
        }

    async def _post_chat_with_retries(
        self, payload: dict[str, Any]
    ) -> httpx.Response:
        client = self._require_client()
        # Drop null stream_options for non-stream
        body = {k: v for k, v in payload.items() if v is not None}
        attempts = self._settings.vllm_max_retries + 1
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                resp = await client.post("/chat/completions", json=body)
                if resp.status_code >= 500 and attempt < attempts - 1:
                    await asyncio.sleep(0.25 * (attempt + 1))
                    continue
                return resp
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    raise VLLMRemoteError(
                        f"vLLM request timed out: {exc}", category="timeout"
                    ) from exc
                await asyncio.sleep(0.25 * (attempt + 1))
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    raise VLLMRemoteError(
                        f"vLLM connection error: {exc}", category="connection"
                    ) from exc
                await asyncio.sleep(0.25 * (attempt + 1))
        raise VLLMRemoteError(
            f"vLLM request failed after retries: {last_exc}", category="connection"
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self._in_flight += 1
        t0 = time.perf_counter()
        try:
            payload = self._chat_payload(request, stream=False)
            resp = await self._post_chat_with_retries(payload)
            inference_s = time.perf_counter() - t0
            if resp.status_code != 200:
                raise VLLMRemoteError(
                    f"chat/completions HTTP {resp.status_code}: {resp.text[:300]}",
                    category="remote_http",
                )
            data = resp.json()
            text = self._extract_text(data)
            usage = data.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            return GenerationResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                queue_wait_s=None,
                inference_s=inference_s,
                prefill_s=None,
                decode_s=None,
                simulated=False,
                backend=self.name,
                ttft_s=None,
                metadata={
                    "measurements": "real",
                    "vllm_model": self._settings.vllm_model_name,
                    "finish_reason": self._finish_reason(data),
                    "remote_usage": usage,
                },
            )
        finally:
            self._in_flight -= 1

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamEvent]:
        client = self._require_client()
        self._in_flight += 1
        t0 = time.perf_counter()
        ttft_s: Optional[float] = None
        input_tokens = 0
        output_tokens = 0
        assembled: list[str] = []
        body = {
            k: v
            for k, v in self._chat_payload(request, stream=True).items()
            if v is not None
        }
        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    raw = (await resp.aread()).decode("utf-8", errors="replace")[:300]
                    yield StreamEvent(
                        event="error",
                        data={
                            "category": "remote_http",
                            "message": f"HTTP {resp.status_code}: {raw}",
                        },
                    )
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage")
                    if usage:
                        input_tokens = int(usage.get("prompt_tokens") or input_tokens)
                        output_tokens = int(
                            usage.get("completion_tokens") or output_tokens
                        )

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    token = delta.get("content") or ""
                    if token:
                        if ttft_s is None:
                            ttft_s = time.perf_counter() - t0
                        assembled.append(token)
                        yield StreamEvent(
                            event="token",
                            data={
                                "text": token,
                                "request_id": request.request_id,
                            },
                        )

            if output_tokens == 0:
                output_tokens = max(1, len(assembled)) if assembled else 0
            inference_s = time.perf_counter() - t0
            yield StreamEvent(
                event="usage",
                data={
                    "request_id": request.request_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "ttft_s": ttft_s,
                    "inference_s": inference_s,
                    "simulated": False,
                    "backend": self.name,
                    "sql": "".join(assembled),
                },
            )
            yield StreamEvent(event="done", data={"request_id": request.request_id})
        except httpx.TimeoutException as exc:
            yield StreamEvent(
                event="error",
                data={"category": "timeout", "message": str(exc)},
            )
        except httpx.TransportError as exc:
            yield StreamEvent(
                event="error",
                data={"category": "connection", "message": str(exc)},
            )
        except asyncio.CancelledError:
            yield StreamEvent(
                event="error",
                data={"category": "cancelled", "message": "client disconnected"},
            )
            raise
        finally:
            self._in_flight -= 1

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return (content or "").strip()

    @staticmethod
    def _finish_reason(data: dict[str, Any]) -> Optional[str]:
        choices = data.get("choices") or []
        if not choices:
            return None
        return choices[0].get("finish_reason")
