"""CPU-only tests for remote VLLMBackend using mocked HTTP (no CUDA / no weights)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.backends.base import GenerationRequest
from app.backends.vllm import VLLMBackend, VLLMRemoteError
from app.config import Settings
from app.serving import build_messages


def _settings(**overrides) -> Settings:
    base = dict(
        backend="vllm",
        vllm_base_url="http://vllm.test/v1",
        vllm_model_name="spider-sft",
        vllm_timeout_s=5.0,
        vllm_connect_timeout_s=2.0,
        vllm_max_retries=1,
        vllm_ready_timeout_s=2.0,
        vllm_temperature=0.0,
    )
    base.update(overrides)
    return Settings(**base)


def _request() -> GenerationRequest:
    q = "List employees earning more than 50000."
    schema = "CREATE TABLE employees (id INTEGER, name TEXT, salary INTEGER);"
    return GenerationRequest(
        request_id="req-1",
        question=q,
        schema=schema,
        max_new_tokens=32,
        prompt_text="unused",
        messages=build_messages(q, schema),
    )


@pytest.mark.asyncio
async def test_vllm_generate_parses_usage_and_marks_real() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "spider-sft"}, {"id": "meta-llama/Llama-3.1-8B-Instruct"}]},
            )
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content.decode())
        assert body["model"] == "spider-sft"
        assert body["temperature"] == 0.0
        assert body["messages"][0]["role"] == "system"
        assert "SQL:" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "SELECT name FROM employees WHERE salary > 50000;"
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 14, "total_tokens": 134},
            },
        )

    transport = httpx.MockTransport(handler)
    backend = VLLMBackend(_settings())
    backend._client = httpx.AsyncClient(
        base_url="http://vllm.test/v1",
        transport=transport,
        timeout=5.0,
    )
    backend._started = True
    backend._ready = True

    result = await backend.generate(_request())
    assert result.simulated is False
    assert result.backend == "vllm"
    assert result.input_tokens == 120
    assert result.output_tokens == 14
    assert result.queue_wait_s is None
    assert result.prefill_s is None
    assert "SELECT name FROM employees" in result.text
    await backend.shutdown()


@pytest.mark.asyncio
async def test_vllm_ready_requires_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "other-model"}]})

    backend = VLLMBackend(_settings())
    backend._client = httpx.AsyncClient(
        base_url="http://vllm.test/v1",
        transport=httpx.MockTransport(handler),
    )
    backend._started = True
    with pytest.raises(VLLMRemoteError, match="not in /models"):
        await backend.check_ready()
    await backend.shutdown()


@pytest.mark.asyncio
async def test_vllm_timeout_category() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    backend = VLLMBackend(_settings(vllm_max_retries=0))
    backend._client = httpx.AsyncClient(
        base_url="http://vllm.test/v1",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    backend._started = True
    backend._ready = True
    with pytest.raises(VLLMRemoteError) as exc:
        await backend.generate(_request())
    assert exc.value.category == "timeout"
    await backend.shutdown()


@pytest.mark.asyncio
async def test_vllm_stream_emits_tokens_usage_done() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"SELECT"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" 1;"}}]}\n\n',
        b'data: {"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "spider-sft"}]})
        return httpx.Response(200, content=b"".join(chunks), headers={"content-type": "text/event-stream"})

    backend = VLLMBackend(_settings())
    backend._client = httpx.AsyncClient(
        base_url="http://vllm.test/v1",
        transport=httpx.MockTransport(handler),
    )
    backend._started = True
    backend._ready = True

    events = []
    async for ev in backend.generate_stream(_request()):
        events.append(ev)

    kinds = [e.event for e in events]
    assert kinds[0] == "token"
    assert "usage" in kinds
    assert kinds[-1] == "done"
    usage = next(e for e in events if e.event == "usage")
    assert usage.data["input_tokens"] == 10
    assert usage.data["output_tokens"] == 2
    assert usage.data["simulated"] is False
    assert usage.data["sql"] == "SELECT 1;"
    await backend.shutdown()


def test_build_backend_vllm() -> None:
    from app.backends import build_backend

    b = build_backend(_settings())
    assert b.name == "vllm"
